/*
 * MIT License
 *
 * Copyright (c) 2024 CSCS, ETH Zurich, University of Basel, University of Zurich
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUTh WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUTh NOTh LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENTh SHALL THE
 * AUTHORS OR COPYRIGHTh HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORTh OR OTHERWISE, ARISING FROM,
 * OUTh OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file calculates dB/dt with the induction equation, as well as dissipation and correction terms
 *
 * @author Lukas Schmidt
 */

#pragma once

#include "cstone/cuda/annotation.hpp"
#include "cstone/sfc/box.hpp"

#include "sph/kernels.hpp"
#include "sph/table_lookup.hpp"

#include "resistivity.hpp"
#include "mhd_kernels.hpp"

namespace sph::magneto
{
// free parameters in Wissing et al. (2020) model for parabolic div-B cleaning
static constexpr float fclean  = 1.0;
static constexpr float sigma_c = 1.0;

template<size_t stride = 1, typename Tc, class T, class Tm>
HOST_DEVICE_FUN inline void
inductionAndDissipationJLoop(cstone::LocalIndex i, Tc K, Tc mu_0, const cstone::Box<Tc>& box,
                             const cstone::LocalIndex* neighbors, unsigned neighborsCount, const Tc* x, const Tc* y,
                             const Tc* z, const T* vx, const T* vy, const T* vz, const T* c, const Tc* Bx, const Tc* By,
                             const Tc* Bz, const T* h, const T* c11, const T* c12, const T* c13, const T* c22,
                             const T* c23, const T* c33, const T* wh, const T* xm, const T* kx, const T* gradh,
                             const Tm* m, const T* psi_ch, Tc* dBxi, Tc* dByi, Tc* dBzi, Tc* dui, const T* alpha_B,
                             const T* dBxdx, const T* dBxdy, const T* dBxdz, const T* dBydx, const T* dBydy,
                             const T* dBydz, const T* dBzdx, const T* dBzdy, const T* dBzdz, ResistivityScheme scheme)
{
    auto xi  = x[i];
    auto yi  = y[i];
    auto zi  = z[i];
    auto vxi = vx[i];
    auto vyi = vy[i];
    auto vzi = vz[i];
    auto ci  = c[i];

    auto hi     = h[i];
    auto mi     = m[i];
    auto xmassi = xm[i];
    auto kxi    = kx[i];
    auto gradhi = gradh[i];

    auto c11i = c11[i];
    auto c12i = c12[i];
    auto c13i = c13[i];
    auto c22i = c22[i];
    auto c23i = c23[i];
    auto c33i = c33[i];

    auto rhoi   = kxi * mi / xmassi;
    auto hiInv  = T(1) / hi;
    auto hiInv3 = hiInv * hiInv * hiInv;

    auto Bxi      = Bx[i];
    auto Byi      = By[i];
    auto Bzi      = Bz[i];
    auto psi_ch_i = psi_ch[i];
    auto alpha_Bi  = alpha_B[i];

    cstone::Vec3<T> gradBx_i{dBxdx[i], dBxdy[i], dBxdz[i]};
    cstone::Vec3<T> gradBy_i{dBydx[i], dBydy[i], dBydz[i]};
    cstone::Vec3<T> gradBz_i{dBzdx[i], dBzdy[i], dBzdz[i]};
    T eta_crit_i = std::cbrt(T(32) * T(M_PI) / T(3) / T(neighborsCount));

    cstone::Vec3<Tc> dB_diss    = {0.0, 0.0, 0.0};
    cstone::Vec3<Tc> divB_clean = {0.0, 0.0, 0.0};
    T                du_diss    = 0.0;

    // wave cleaning speed
    auto v_alfven2 = (Bxi * Bxi + Byi * Byi + Bzi * Bzi) / (mu_0 * rhoi);
    auto c_hi      = fclean * std::sqrt(ci * ci + v_alfven2);

    for (unsigned pj = 0; pj < neighborsCount; ++pj)
    {
        cstone::LocalIndex j = neighbors[stride * pj];

        auto mj     = m[j];
        auto xmassj = xm[j];
        auto rhoj   = kx[j] * mj / xmassj;

        T rx = xi - x[j];
        T ry = yi - y[j];
        T rz = zi - z[j];

        T vx_ij = vxi - vx[j];
        T vy_ij = vyi - vy[j];
        T vz_ij = vzi - vz[j];

        T hj    = h[j];
        T hjInv = T(1) / hj;

        applyPBC(box, T(2) * hi, rx, ry, rz);

        T r2   = rx * rx + ry * ry + rz * rz;
        T dist = std::sqrt(r2);

        T v1 = dist * hiInv;
        T v2 = dist * hjInv;

        T hjInv3 = hjInv * hjInv * hjInv;
        T Wi     = hiInv3 * lt::lookup(wh, v1);
        T Wj     = hjInv3 * lt::lookup(wh, v2);

        T termA1_i = -(c11i * rx + c12i * ry + c13i * rz) * Wi;
        T termA2_i = -(c12i * rx + c22i * ry + c23i * rz) * Wi;
        T termA3_i = -(c13i * rx + c23i * ry + c33i * rz) * Wi;

        auto c11j = c11[j];
        auto c12j = c12[j];
        auto c13j = c13[j];
        auto c22j = c22[j];
        auto c23j = c23[j];
        auto c33j = c33[j];

        T termA1_j = -(c11j * rx + c12j * ry + c13j * rz) * Wj;
        T termA2_j = -(c12j * rx + c22j * ry + c23j * rz) * Wj;
        T termA3_j = -(c13j * rx + c23j * ry + c33j * rz) * Wj;

        cstone::Vec3<T> vab_cross_rab{vy_ij * rz - vz_ij * ry, vz_ij * rx - vx_ij * rz, vx_ij * ry - vy_ij * rx};
        T               v_sigB = std::sqrt(norm2(vab_cross_rab) / r2);
        T alpha_B_avg = T(0.5) * (alpha_Bi + alpha_B[j]);

        cstone::Vec3<Tc> B_ab{Bxi - Bx[j], Byi - By[j], Bzi - Bz[j]};

        if (scheme == ResistivityScheme::SLR)
        {
            Tc              eta_ab = (v1 < v2) ? v1 : v2; // spacing in units of h
            cstone::Vec3<T> gradBx_j{dBxdx[j], dBxdy[j], dBxdz[j]};
            cstone::Vec3<T> gradBy_j{dBydx[j], dBydy[j], dBydz[j]};
            cstone::Vec3<T> gradBz_j{dBzdx[j], dBzdy[j], dBzdz[j]};
            B_ab += mhdSLRCorrection<Tc, T>({rx, ry, rz}, eta_ab, eta_crit_i, T(1), T(1), gradBx_i, gradBy_i, gradBz_i,
                                            gradBx_j, gradBy_j, gradBz_j);
        }

        // Conjugate-pair (non-symmetric) artificial resistivity (Price et al. 2018, eqs. 181-182)
        T grkern_i = (rx * termA1_i + ry * termA2_i + rz * termA3_i) / dist;
        T grkern_j = (rx * termA1_j + ry * termA2_j + rz * termA3_j) / dist;

        auto diss_op = T(0.5) * alpha_B_avg * v_sigB * mj * rhoi *
                       (grkern_i / (rhoi * rhoi) + grkern_j / (rhoj * rhoj));

        dB_diss += diss_op * B_ab;
        du_diss += diss_op * norm2(B_ab);

        // wave cleaning speed
        auto v_alfven2j  = (Bx[j] * Bx[j] + By[j] * By[j] + Bz[j] * Bz[j]) / (rhoj * mu_0);
        auto c_hj        = fclean * std::sqrt(c[j] * c[j] + v_alfven2j);

        // Conjugate-pair (non-symmetric) divergence cleaning (Price et al. 2018, eq. 172)
        cstone::Vec3<Tc> termA_i_vec{termA1_i, termA2_i, termA3_i};
        cstone::Vec3<Tc> termA_j_vec{termA1_j, termA2_j, termA3_j};

        divB_clean += mj * rhoi *
                      (psi_ch_i * c_hi / (gradhi * rhoi * rhoi) * termA_i_vec +
                       psi_ch[j] * c_hj / (gradh[j] * rhoj * rhoj) * termA_j_vec);
    }

    dB_diss *= K;

    divB_clean *= K;
    du_diss *= K / rhoi;

    *dBxi += dB_diss[0] - divB_clean[0];
    *dByi += dB_diss[1] - divB_clean[1];
    *dBzi += dB_diss[2] - divB_clean[2];

    *dui -= 0.5 * du_diss;
}
} // namespace sph::magneto
