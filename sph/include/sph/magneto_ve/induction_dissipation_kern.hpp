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
#include "cstone/traversal/ijloop/ijloop.hpp"

#include "sph/kernels.hpp"
#include "sph/table_lookup.hpp"

#include "resistivity.hpp"
#include "mhd_kernels.hpp"

namespace sph::magneto
{
// free parameters in Wissing et al. (2020) model for parabolic div-B cleaning
static constexpr float fclean  = 1.0;
static constexpr float sigma_c = 1.0;

template<class T>
struct InductionAndDissipationInteraction
{
    const T*          wh;
    T                 mu_0;
    ResistivityScheme scheme;

    template<class ParticleData, class Tc>
    constexpr auto operator()(const ParticleData& iData, const ParticleData& jData, cstone::Vec3<Tc> const& r_ij,
                              T r2) const
    {
        const auto [i, iPos, hi, vxi, vyi, vzi, ci, Bxi, Byi, Bzi, mi, xmassi, kxi, gradhi, c11i, c12i, c13i, c22i,
                    c23i, c33i, alpha_Bi, psi_ch_i, nci, dBxdxi, dBxdyi, dBxdzi, dBydxi, dBydyi, dBydzi, dBzdxi,
                    dBzdyi, dBzdzi, dvxdxi, dvxdyi, dvxdzi, dvydxi, dvydyi, dvydzi, dvzdxi, dvzdyi, dvzdzi, divBi,
                    dui] = iData;
        const auto [j, jPos, hj, vxj, vyj, vzj, cj, Bxj, Byj, Bzj, mj, xmassj, kxj, gradhj, c11j, c12j, c13j, c22j,
                    c23j, c33j, alpha_Bj, psi_ch_j, ncj, dBxdxj, dBxdyj, dBxdzj, dBydxj, dBydyj, dBydzj, dBzdxj,
                    dBzdyj, dBzdzj, dvxdxj, dvxdyj, dvxdzj, dvydxj, dvydyj, dvydzj, dvzdxj, dvzdyj, dvzdzj, divBj,
                    duj] = jData;

        T rhoi = kxi * mi / xmassi;
        T rhoj = kxj * mj / xmassj;

        T hiInv  = T(1) / hi;
        T hiInv3 = hiInv * hiInv * hiInv;
        T hjInv  = T(1) / hj;
        T hjInv3 = hjInv * hjInv * hjInv;

        T rx = r_ij[0];
        T ry = r_ij[1];
        T rz = r_ij[2];

        T dist    = std::sqrt(r2);
        T distInv = (i == j) ? T(0) : T(1) / dist;

        T v1 = dist * hiInv;
        T v2 = dist * hjInv;
        T Wi = hiInv3 * lt::lookup(wh, v1);
        T Wj = hjInv3 * lt::lookup(wh, v2);

        T termA1_i = -(c11i * rx + c12i * ry + c13i * rz) * Wi;
        T termA2_i = -(c12i * rx + c22i * ry + c23i * rz) * Wi;
        T termA3_i = -(c13i * rx + c23i * ry + c33i * rz) * Wi;

        T termA1_j = -(c11j * rx + c12j * ry + c13j * rz) * Wj;
        T termA2_j = -(c12j * rx + c22j * ry + c23j * rz) * Wj;
        T termA3_j = -(c13j * rx + c23j * ry + c33j * rz) * Wj;

        T vx_ij = vxi - vxj;
        T vy_ij = vyi - vyj;
        T vz_ij = vzi - vzj;

        cstone::Vec3<T> vab_cross_rab{vy_ij * rz - vz_ij * ry, vz_ij * rx - vx_ij * rz, vx_ij * ry - vy_ij * rx};
        T               v_sigB      = (i == j) ? T(0) : std::sqrt(norm2(vab_cross_rab) / r2);
        T               alpha_B_avg = T(0.5) * (alpha_Bi + alpha_Bj);

        cstone::Vec3<Tc> B_ab{Bxi - Bxj, Byi - Byj, Bzi - Bzj};

        if (scheme == ResistivityScheme::SLR)
        {
            T               eta_crit_i = std::cbrt(T(32) * T(M_PI) / T(3) / T(nci));
            Tc              eta_ab     = (v1 < v2) ? v1 : v2; // spacing in units of h
            cstone::Vec3<T> gradBx_i{dBxdxi, dBxdyi, dBxdzi};
            cstone::Vec3<T> gradBy_i{dBydxi, dBydyi, dBydzi};
            cstone::Vec3<T> gradBz_i{dBzdxi, dBzdyi, dBzdzi};
            cstone::Vec3<T> gradBx_j{dBxdxj, dBxdyj, dBxdzj};
            cstone::Vec3<T> gradBy_j{dBydxj, dBydyj, dBydzj};
            cstone::Vec3<T> gradBz_j{dBzdxj, dBzdyj, dBzdzj};
            B_ab += mhdSLRCorrection<Tc, T>({rx, ry, rz}, eta_ab, eta_crit_i, T(1), T(1), gradBx_i, gradBy_i, gradBz_i,
                                            gradBx_j, gradBy_j, gradBz_j);
        }

        // Conjugate-pair (non-symmetric) artificial resistivity (Price et al. 2018, eqs. 181-182)
        T grkern_i = (rx * termA1_i + ry * termA2_i + rz * termA3_i) * distInv;
        T grkern_j = (rx * termA1_j + ry * termA2_j + rz * termA3_j) * distInv;

        T diss_op =
            T(0.5) * alpha_B_avg * v_sigB * mj * rhoi * (grkern_i / (rhoi * rhoi) + grkern_j / (rhoj * rhoj));

        cstone::Vec3<Tc> dB_diss = diss_op * B_ab;
        T                du_diss = diss_op * norm2(B_ab);

        // wave cleaning speeds
        T v_alfven2i = (Bxi * Bxi + Byi * Byi + Bzi * Bzi) / (mu_0 * rhoi);
        T c_hi       = fclean * std::sqrt(ci * ci + v_alfven2i);
        T v_alfven2j = (Bxj * Bxj + Byj * Byj + Bzj * Bzj) / (rhoj * mu_0);
        T c_hj       = fclean * std::sqrt(cj * cj + v_alfven2j);

        // Non-symmetric constrained divB cleaning (Price et al. 2018, eq. 172)
        // VE-native conservative grad-psi: Lagrangian conjugate of the conservative divB
        cstone::Vec3<Tc> termA_i_vec{termA1_i, termA2_i, termA3_i};
        cstone::Vec3<Tc> termA_j_vec{termA1_j, termA2_j, termA3_j};

        cstone::Vec3<Tc> divB_clean = rhoi * mj *
                                      (psi_ch_i * c_hi * xmassi * xmassi / (kxi * mi * mi * gradhi) * termA_i_vec +
                                       psi_ch_j * c_hj * xmassj * xmassj / (kxj * mj * mj * gradhj) * termA_j_vec);

        return std::make_tuple(dB_diss[0], dB_diss[1], dB_diss[2], divB_clean[0], divB_clean[1], divB_clean[2],
                               du_diss);
    }
};

template<class T, class Tc>
struct InductionAndDissipationPostamble
{
    Tc K;
    T  mu_0;

    template<class ParticleData, class Result>
    constexpr auto operator()(const ParticleData& iData, const Result& result) const
    {
        const auto [i, iPos, hi, vxi, vyi, vzi, ci, Bxi, Byi, Bzi, mi, xmassi, kxi, gradhi, c11i, c12i, c13i, c22i,
                    c23i, c33i, alpha_Bi, psi_ch_i, nci, dBxdxi, dBxdyi, dBxdzi, dBydxi, dBydyi, dBydzi, dBzdxi,
                    dBzdyi, dBzdzi, dvxdxi, dvxdyi, dvxdzi, dvydxi, dvydyi, dvydzi, dvzdxi, dvzdyi, dvzdzi, divBi,
                    dui] = iData;
        auto [dB_diss_x, dB_diss_y, dB_diss_z, divB_clean_x, divB_clean_y, divB_clean_z, du_diss] = result;

        // Ideal induction equation
        Tc dBxi = -Bxi * (dvydyi + dvzdzi) + Byi * dvxdyi + Bzi * dvxdzi;
        Tc dByi = -Byi * (dvxdxi + dvzdzi) + Bxi * dvydxi + Bzi * dvydzi;
        Tc dBzi = -Bzi * (dvxdxi + dvydyi) + Bxi * dvzdxi + Byi * dvzdyi;

        dBxi += K * (dB_diss_x - divB_clean_x);
        dByi += K * (dB_diss_y - divB_clean_y);
        dBzi += K * (dB_diss_z - divB_clean_z);

        T  rhoi   = kxi * mi / xmassi;
        Tc du_out = dui - T(0.5) * K / rhoi * du_diss;

        // psi time differential (Wissing et al. 2020)
        T v_alfven2 = (Bxi * Bxi + Byi * Byi + Bzi * Bzi) / (mu_0 * rhoi);
        T ch        = fclean * std::sqrt(ci * ci + v_alfven2);
        T tau_Inv   = (sigma_c * ch) / hi;
        T d_psi_ch_out = -ch * divBi - psi_ch_i * (tau_Inv + (dvxdxi + dvydyi + dvzdzi) / T(2));

        return std::make_tuple(dBxi, dByi, dBzi, du_out, d_psi_ch_out);
    }
};

template<class Neighborhood, class Tc, class T, class Tm>
void inductionAndDissipationIjLoop(
    Neighborhood const& neighborhood, Tc K, Tc mu_0, ResistivityScheme scheme, const T* vx, const T* vy, const T* vz,
    const T* c, const Tc* Bx, const Tc* By, const Tc* Bz, const Tm* m, const T* xm, const T* kx, const T* gradh,
    const T* c11, const T* c12, const T* c13, const T* c22, const T* c23, const T* c33, const T* alpha_B,
    const T* psi_ch, const unsigned* nc, const T* dBxdx, const T* dBxdy, const T* dBxdz, const T* dBydx,
    const T* dBydy, const T* dBydz, const T* dBzdx, const T* dBzdy, const T* dBzdz, const T* dvxdx, const T* dvxdy,
    const T* dvxdz, const T* dvydx, const T* dvydy, const T* dvydz, const T* dvzdx, const T* dvzdy, const T* dvzdz,
    const T* divB, const T* wh, Tc* dBx_dt, Tc* dBy_dt, Tc* dBz_dt, Tc* du, T* d_psi_ch)
{
    const auto input =
        std::make_tuple(vx, vy, vz, c, Bx, By, Bz, m, xm, kx, gradh, c11, c12, c13, c22, c23, c33, alpha_B, psi_ch, nc,
                        dBxdx, dBxdy, dBxdz, dBydx, dBydy, dBydz, dBzdx, dBzdy, dBzdz, dvxdx, dvxdy, dvxdz, dvydx,
                        dvydy, dvydz, dvzdx, dvzdy, dvzdz, divB, du);
    const auto output = std::make_tuple(dBx_dt, dBy_dt, dBz_dt, du, d_psi_ch);
    neighborhood.ijLoop(input, output, InductionAndDissipationInteraction<T>{wh, T(mu_0), scheme},
                        InductionAndDissipationPostamble<T, Tc>{K, T(mu_0)});
}

} // namespace sph::magneto
