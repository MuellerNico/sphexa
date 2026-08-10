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
    T                 arFloor; // 1=disabled
    ResistivityScheme scheme;

    template<class ParticleData, class Tc>
    constexpr auto operator()(const ParticleData& iData, const ParticleData& jData, cstone::Vec3<Tc> const& r_ij,
                              T r2) const
    {
        const auto [i, iPos, hi, vxi, vyi, vzi, ci, Bxi, Byi, Bzi, mi, xmassi, kxi, gradhi, c11i, c12i, c13i, c22i,
                    c23i, c33i, alpha_Bi, psi_ch_i, nci, dBxdxi, dBxdyi, dBxdzi, dBydxi, dBydyi, dBydzi, dBzdxi,
                    dBzdyi, dBzdzi, dvxdxi, dvxdyi, dvxdzi, dvydxi, dvydyi, dvydzi, dvzdxi, dvzdyi, dvzdzi, divB_conj_i,
                    dui] = iData;
        const auto [j, jPos, hj, vxj, vyj, vzj, cj, Bxj, Byj, Bzj, mj, xmassj, kxj, gradhj, c11j, c12j, c13j, c22j,
                    c23j, c33j, alpha_Bj, psi_ch_j, ncj, dBxdxj, dBxdyj, dBxdzj, dBydxj, dBydyj, dBydzj, dBzdxj,
                    dBzdyj, dBzdzj, dvxdxj, dvxdyj, dvxdzj, dvydxj, dvydyj, dvydzj, dvzdxj, dvzdyj, dvzdzj, divB_conj_j,
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

        cstone::Vec3<Tc> termAi;
        termAi[0] = -(c11i * rx + c12i * ry + c13i * rz) * Wi;
        termAi[1] = -(c12i * rx + c22i * ry + c23i * rz) * Wi;
        termAi[2] = -(c13i * rx + c23i * ry + c33i * rz) * Wi;

        cstone::Vec3<Tc> termAj;
        termAj[0] = -(c11j * rx + c12j * ry + c13j * rz) * Wj;
        termAj[1] = -(c12j * rx + c22j * ry + c23j * rz) * Wj;
        termAj[2] = -(c13j * rx + c23j * ry + c33j * rz) * Wj;
        
        cstone::Vec3<Tc> v_ij          = {vxi - vxj, vyi - vyj, vzi - vzj};
        cstone::Vec3<Tc> vab_cross_rab = cross(v_ij, r_ij);

        T v_sigB      = (i == j) ? T(0) : T(std::sqrt(norm2(vab_cross_rab) / r2));
        T alpha_B_avg = T(0.5) * (alpha_Bi + alpha_Bj);

        cstone::Vec3<Tc> B_ab{Bxi - Bxj, Byi - Byj, Bzi - Bzj};
        T Lij = T(1);

        if (scheme == ResistivityScheme::SLR || scheme == ResistivityScheme::SLRB ||
            scheme == ResistivityScheme::SLRB2 || scheme == ResistivityScheme::SLRV ||
            scheme == ResistivityScheme::SLRV2 || scheme == ResistivityScheme::SLRC)
        {
            cstone::Vec3<T> gradBx_i{dBxdxi, dBxdyi, dBxdzi};
            cstone::Vec3<T> gradBy_i{dBydxi, dBydyi, dBydzi};
            cstone::Vec3<T> gradBz_i{dBzdxi, dBzdyi, dBzdzi};
            cstone::Vec3<T> gradBx_j{dBxdxj, dBxdyj, dBxdzj};
            cstone::Vec3<T> gradBy_j{dBydxj, dBydyj, dBydzj};
            cstone::Vec3<T> gradBz_j{dBzdxj, dBzdyj, dBzdzj};

            // SLR: pure reconstruction (balsi=balsj=1, Lij=1). SLRB/SLRB2: Balsara-like modulation.
            T balsi = T(1);
            T balsj = T(1);
            if (scheme == ResistivityScheme::SLRB || scheme == ResistivityScheme::SLRB2)
            {
                T B_norm_i     = std::sqrt(Bxi * Bxi + Byi * Byi + Bzi * Bzi);
                T B_norm_j     = std::sqrt(Bxj * Bxj + Byj * Byj + Bzj * Bzj);
                T gradB_norm_i = std::sqrt(norm2(gradBx_i) + norm2(gradBy_i) + norm2(gradBz_i));
                T gradB_norm_j = std::sqrt(norm2(gradBx_j) + norm2(gradBy_j) + norm2(gradBz_j));
                T modulator_i  = (B_norm_i > T(0)) ? hi * gradB_norm_i / B_norm_i : T(1);
                T modulator_j  = (B_norm_j > T(0)) ? hj * gradB_norm_j / B_norm_j : T(1);
                modulator_i    = stl::max(T(0), stl::min(modulator_i, T(1))); // clamp to [0,1]
                modulator_j    = stl::max(T(0), stl::min(modulator_j, T(1)));
                Lij            = stl::max(arFloor, T(0.5) * (modulator_i + modulator_j));
                if (scheme == ResistivityScheme::SLRB)
                {
                    balsi = T(1) - modulator_i;
                    balsj = T(1) - modulator_j;
                }
                else // SLRB2
                {
                    balsi = T(1) - modulator_i * modulator_i;
                    balsj = T(1) - modulator_j * modulator_j;
                }
            }
            T  eta_crit_i = std::cbrt(T(32) * T(M_PI) / T(3) / T(nci));
            Tc eta_ab     = (v1 < v2) ? v1 : v2; // spacing in units of h
            if (scheme == ResistivityScheme::SLRV)
            {
                B_ab += mhdSLRVCorrection<Tc, T>(r_ij, eta_ab, eta_crit_i, B_ab, gradBx_i, gradBy_i, gradBz_i,
                                                 gradBx_j, gradBy_j, gradBz_j);
            }
            else if (scheme == ResistivityScheme::SLRV2)
            {
                B_ab += mhdSLRV2Correction<Tc, T>(r_ij, eta_ab, eta_crit_i, gradBx_i, gradBy_i, gradBz_i, gradBx_j,
                                                  gradBy_j, gradBz_j);
            }
            else if (scheme == ResistivityScheme::SLRC)
            {
                B_ab += mhdSLRCCorrection<Tc, T>(r_ij, eta_ab, eta_crit_i, B_ab, gradBx_i, gradBy_i, gradBz_i,
                                                 gradBx_j, gradBy_j, gradBz_j);
            }
            else
            {
                B_ab += mhdSLRCorrection<Tc, T>(r_ij, eta_ab, eta_crit_i, balsi, balsj, gradBx_i, gradBy_i, gradBz_i,
                                                gradBx_j, gradBy_j, gradBz_j);
            }
        }

        // Conjugate-pair (non-symmetric) artificial resistivity (Price et al. 2018, eqs. 181-182)
        T grkern_i = dot(r_ij, termAi) * distInv / (rhoi * rhoi); // gradient kernel r̂·∇W/ρ²
        T grkern_j = dot(r_ij, termAj) * distInv / (rhoj * rhoj);
        T resistivity_ab = Lij * alpha_B_avg * v_sigB;
        T diss_op = T(0.5) * mj * rhoi * resistivity_ab * (grkern_i + grkern_j);

        cstone::Vec3<Tc> dB_diss = diss_op * B_ab;
        T                du_diss = diss_op * norm2(B_ab);

        // wave cleaning speeds
        T v_alfven2i = (Bxi * Bxi + Byi * Byi + Bzi * Bzi) / (mu_0 * rhoi);
        T c_hi       = fclean * std::sqrt(ci * ci + v_alfven2i);
        T v_alfven2j = (Bxj * Bxj + Byj * Byj + Bzj * Bzj) / (rhoj * mu_0);
        T c_hj       = fclean * std::sqrt(cj * cj + v_alfven2j);

        // constrained divB cleaning (Price et al. 2018, eq. 172), VE-native conjugate of divB_conj
        cstone::Vec3<Tc> divB_clean = rhoi * mj *
                                      (psi_ch_i * c_hi * xmassi * xmassi / (kxi * mi * mi * gradhi) * termAi +
                                       psi_ch_j * c_hj * xmassj * xmassj / (kxj * mj * mj * gradhj) * termAj);

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
                    dBzdyi, dBzdzi, dvxdxi, dvxdyi, dvxdzi, dvydxi, dvydyi, dvydzi, dvzdxi, dvzdyi, dvzdzi, divB_conj_i,
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
        T d_psi_ch_out = -ch * divB_conj_i - psi_ch_i * (tau_Inv + (dvxdxi + dvydyi + dvzdzi) / T(2));

        // Diagnostic outputs: resistive dB/dt and resistive heating
        Tc dB_diss_out_x = K * dB_diss_x;
        Tc dB_diss_out_y = K * dB_diss_y;
        Tc dB_diss_out_z = K * dB_diss_z;
        Tc du_diss_out   = -T(0.5) * K / rhoi * du_diss;

        return std::make_tuple(dBxi, dByi, dBzi, du_out, d_psi_ch_out, dB_diss_out_x, dB_diss_out_y, dB_diss_out_z,
                               du_diss_out);
    }
};

template<class Neighborhood, class Tc, class T, class Tm>
void inductionAndDissipationIjLoop(
    Neighborhood const& neighborhood, Tc K, Tc mu_0, ResistivityScheme scheme, T arFloor, const T* vx, const T* vy,
    const T* vz,
    const T* c, const Tc* Bx, const Tc* By, const Tc* Bz, const Tm* m, const T* xm, const T* kx, const T* gradh,
    const T* c11, const T* c12, const T* c13, const T* c22, const T* c23, const T* c33, const T* alpha_B,
    const T* psi_ch, const unsigned* nc, const T* dBxdx, const T* dBxdy, const T* dBxdz, const T* dBydx,
    const T* dBydy, const T* dBydz, const T* dBzdx, const T* dBzdy, const T* dBzdz, const T* dvxdx, const T* dvxdy,
    const T* dvxdz, const T* dvydx, const T* dvydy, const T* dvydz, const T* dvzdx, const T* dvzdy, const T* dvzdz,
    const T* divB_conj, const T* wh, Tc* dBx_dt, Tc* dBy_dt, Tc* dBz_dt, Tc* du, T* d_psi_ch, Tc* dB_diss_x,
    Tc* dB_diss_y, Tc* dB_diss_z, Tc* du_diss)
{
    const auto input =
        std::make_tuple(vx, vy, vz, c, Bx, By, Bz, m, xm, kx, gradh, c11, c12, c13, c22, c23, c33, alpha_B, psi_ch, nc,
                        dBxdx, dBxdy, dBxdz, dBydx, dBydy, dBydz, dBzdx, dBzdy, dBzdz, dvxdx, dvxdy, dvxdz, dvydx,
                        dvydy, dvydz, dvzdx, dvzdy, dvzdz, divB_conj, du);
    const auto output = std::make_tuple(dBx_dt, dBy_dt, dBz_dt, du, d_psi_ch, dB_diss_x, dB_diss_y, dB_diss_z, du_diss);
    neighborhood.ijLoop(input, output, InductionAndDissipationInteraction<T>{wh, T(mu_0), arFloor, scheme},
                        InductionAndDissipationPostamble<T, Tc>{K, T(mu_0)});
}

} // namespace sph::magneto
