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

/*! @file calculates the momentum and energy with the magnetic stress tensor
 *
 * @author Lukas Schmidt
 *
 */

#pragma once

#include "cstone/cuda/annotation.hpp"
#include "cstone/traversal/ijloop/ijloop.hpp"

#include "sph/kernels.hpp"
#include "sph/table_lookup.hpp"
#include "sph/hydro_ve/momentum_energy_kern.hpp" // avRvCorrection

namespace sph::magneto
{

template<bool SLR, class T>
struct MagneticMomentumAndEnergyInteraction
{
    const T* wh;
    T        mu_0, alpha_u, Atmin, Atmax, ramp, avFloor;

    template<class ParticleData, class Tc>
    constexpr auto operator()(const ParticleData& iData, const ParticleData& jData, cstone::Vec3<Tc> const& r_ij,
                              T r2) const
    {
        const auto [i, iPos, hi, vxi, vyi, vzi, mi, ci, ui, kxi, alpha_i, xmassi, pi, gradhi, c11i, c12i, c13i, c22i,
                    c23i, c33i, nci, Bxi, Byi, Bzi, dvxdxi, dvxdyi, dvxdzi, dvydxi, dvydyi, dvydzi, dvzdxi, dvzdyi,
                    dvzdzi, tdpdTrhoi, divvi, curlvi] = iData;
        const auto [j, jPos, hj, vxj, vyj, vzj, mj, cj, uj, kxj, alpha_j, xmassj, pj, gradhj, c11j, c12j, c13j, c22j,
                    c23j, c33j, ncj, Bxj, Byj, Bzj, dvxdxj, dvxdyj, dvxdzj, dvydxj, dvydyj, dvydzj, dvzdxj, dvzdyj,
                    dvzdzj, tdpdTrhoj, divvj, curlvj] = jData;

        T mu_0Inv = T(1) / mu_0;

        T Si_xx = T(0.5) * mu_0Inv * (Bxi * Bxi - Byi * Byi - Bzi * Bzi);
        T Si_xy = mu_0Inv * Bxi * Byi;
        T Si_xz = mu_0Inv * Bxi * Bzi;
        T Si_yy = T(0.5) * mu_0Inv * (-Bxi * Bxi + Byi * Byi - Bzi * Bzi);
        T Si_yz = mu_0Inv * Byi * Bzi;
        T Si_zz = T(0.5) * mu_0Inv * (-Bxi * Bxi - Byi * Byi + Bzi * Bzi);

        auto rhoi = kxi * mi / xmassi;
        auto proi = pi / (kxi * mi * mi * gradhi);
        auto voli = xmassi / kxi;

        T hiInv  = T(1) / hi;
        T hiInv3 = hiInv * hiInv * hiInv;

        T eta_crit = std::cbrt(T(32) * M_PI / T(3) / T(nci));

        T norm2_B          = Bxi * Bxi + Byi * Byi + Bzi * Bzi;
        T v_alfven2i       = norm2_B / (mu_0 * rhoi);
        T magneticVsignali = std::sqrt(ci * ci + v_alfven2i);

        [[maybe_unused]] util::array<T, 6> gradV_i;
        if constexpr (SLR)
        {
            gradV_i = {dvxdxi, dvxdyi + dvydxi, dvxdzi + dvzdxi, dvydyi, dvydzi + dvzdyi, dvzdzi};
        }

        T rx = r_ij[0];
        T ry = r_ij[1];
        T rz = r_ij[2];

        T dist    = std::sqrt(r2);
        T distInv = (i == j) ? T(0) : T(1) / dist;

        T ux_ij = rx * distInv;
        T uy_ij = ry * distInv;
        T uz_ij = rz * distInv;

        T vx_ij = vxi - vxj;
        T vy_ij = vyi - vyj;
        T vz_ij = vzi - vzj;

        T hjInv = T(1) / hj;

        T v1 = dist * hiInv;
        T v2 = dist * hjInv;

        T hjInv3 = hjInv * hjInv * hjInv;
        T Wi     = hiInv3 * lt::lookup(wh, v1);
        T Wj     = hjInv3 * lt::lookup(wh, v2);

        T termA1_i = -(c11i * rx + c12i * ry + c13i * rz) * Wi;
        T termA2_i = -(c12i * rx + c22i * ry + c23i * rz) * Wi;
        T termA3_i = -(c13i * rx + c23i * ry + c33i * rz) * Wi;

        T termA1_j = -(c11j * rx + c12j * ry + c13j * rz) * Wj;
        T termA2_j = -(c12j * rx + c22j * ry + c23j * rz) * Wj;
        T termA3_j = -(c13j * rx + c23j * ry + c33j * rz) * Wj;

        auto rhoj = kxj * mj / xmassj;
        auto proj = pj / (kxj * mj * mj * gradhj);
        auto volj = xmassj / kxj;

        T rv     = rx * vx_ij + ry * vy_ij + rz * vz_ij;
        T rv_slr = rv;
        T Lij    = T(1);

        if constexpr (SLR)
        {
            T eps_i   = T(1e-4) * ci * hiInv;
            T eps_j   = T(1e-4) * cj * hjInv;
            T denom_i = std::abs(divvi) + std::abs(curlvi) + eps_i;
            T denom_j = std::abs(divvj) + std::abs(curlvj) + eps_j;
            T f_i     = (denom_i > T(0)) ? std::abs(divvi) / denom_i : T(0); // per-particle Balsara modulators
            T f_j     = (denom_j > T(0)) ? std::abs(divvj) / denom_j : T(0);
            Lij       = stl::max(avFloor, T(0.5) * (f_i + f_j));
            T balsi   = T(1) - f_i * f_i;
            T balsj   = T(1) - f_j * f_j;
            rv_slr += avRvCorrection({rx, ry, rz}, stl::min(v1, v2), eta_crit, balsi, balsj, gradV_i,
                                     {dvxdxj, dvxdyj + dvydxj, dvxdzj + dvzdxj, dvydyj, dvydzj + dvzdyj, dvzdzj});
        }

        T v_alfven2j       = (Bxj * Bxj + Byj * Byj + Bzj * Bzj) / (rhoj * mu_0);
        T magneticVsignalj = std::sqrt(cj * cj + v_alfven2j);

        T wij             = rv * distInv;
        T wij_slr         = rv_slr * distInv;
        T delta_u         = ui - uj;
        T viscosity_ij = artificial_viscosity(alpha_i, alpha_j, magneticVsignali, magneticVsignalj, wij_slr, wij_slr, Lij);
        T heat_conduction = AV_heat_conduction(T(alpha_u), wij_slr, rhoi, rhoj, proi, proj, delta_u);

        // For time-step calculations
        T vijsignal = (i == j) ? T(0) : T(0.5) * (magneticVsignali + magneticVsignalj) - T(2) * wij_slr;

        T a_mom, b_mom;
        T Atwood = (std::abs(rhoi - rhoj)) / (rhoi + rhoj);
        if (Atwood < Atmin)
        {
            a_mom = xmassi * xmassi;
            b_mom = xmassj * xmassj;
        }
        else if (Atwood > Atmax)
        {
            a_mom = xmassi * xmassj;
            b_mom = a_mom;
        }
        else
        {
            T sigma_ij = ramp * (Atwood - Atmin);
            a_mom      = pow(xmassi, T(2) - sigma_ij) * pow(xmassj, sigma_ij);
            b_mom      = pow(xmassj, T(2) - sigma_ij) * pow(xmassi, sigma_ij);
        }

        auto a_visc        = mj / rhoi * viscosity_ij;
        auto b_visc        = mj / rhoj * viscosity_ij;
        T    a_visc_x      = T(0.5) * (a_visc * termA1_i + b_visc * termA1_j);
        T    a_visc_y      = T(0.5) * (a_visc * termA2_i + b_visc * termA2_j);
        T    a_visc_z      = T(0.5) * (a_visc * termA3_i + b_visc * termA3_j);
        T    a_visc_energy = a_visc_x * vx_ij + a_visc_y * vy_ij + a_visc_z * vz_ij;

        T a_heat      = voli * mj / mi * heat_conduction;
        T b_heat      = volj * heat_conduction;
        T a_heat_x    = T(0.5) * (a_heat * termA1_i + b_heat * termA1_j);
        T a_heat_y    = T(0.5) * (a_heat * termA2_i + b_heat * termA2_j);
        T a_heat_z    = T(0.5) * (a_heat * termA3_i + b_heat * termA3_j);
        T a_heat_cond = a_heat_x * ux_ij + a_heat_y * uy_ij + a_heat_z * uz_ij;

        T energy = mj * a_mom * (vx_ij * termA1_i + vy_ij * termA2_i + vz_ij * termA3_i);

        // gas pressure contributions
        a_mom /= kxi * mi * mi * gradhi; // fold normalization into a/b_mom. note: hydro equivalent uses precomputed prho
        b_mom /= kxj * mj * mj * gradhj;

        auto momentum_i = mj * pi * a_mom;
        auto momentum_j = mj * pj * b_mom;
        T    momentum_x = -(momentum_i * termA1_i + momentum_j * termA1_j);
        T    momentum_y = -(momentum_i * termA2_i + momentum_j * termA2_j);
        T    momentum_z = -(momentum_i * termA3_i + momentum_j * termA3_j);

        // magnetic pressure contributions
        T Sj_xx = T(0.5) * mu_0Inv * (Bxj * Bxj - Byj * Byj - Bzj * Bzj);
        T Sj_xy = mu_0Inv * Bxj * Byj;
        T Sj_xz = mu_0Inv * Bxj * Bzj;
        T Sj_yy = T(0.5) * mu_0Inv * (-Bxj * Bxj + Byj * Byj - Bzj * Bzj);
        T Sj_yz = mu_0Inv * Byj * Bzj;
        T Sj_zz = T(0.5) * mu_0Inv * (-Bxj * Bxj - Byj * Byj + Bzj * Bzj);

        auto momentum_xi = Si_xx * termA1_i + Si_xy * termA2_i + Si_xz * termA3_i;
        auto momentum_yi = Si_xy * termA1_i + Si_yy * termA2_i + Si_yz * termA3_i;
        auto momentum_zi = Si_xz * termA1_i + Si_yz * termA2_i + Si_zz * termA3_i;

        auto momentum_xj = Sj_xx * termA1_j + Sj_xy * termA2_j + Sj_xz * termA3_j;
        auto momentum_yj = Sj_xy * termA1_j + Sj_yy * termA2_j + Sj_yz * termA3_j;
        auto momentum_zj = Sj_xz * termA1_j + Sj_yz * termA2_j + Sj_zz * termA3_j;

        // tensile instability correction
        // auto rhosqinv = 1 / (rhoi * rhoj * gradhi);
        // f_i += 2 * mj * rhosqinv * (Bxi * termA1_i + Byi * termA2_i + Bzi * termA3_i); // SPHYNX
        // f_i += mj / (rhoi * rhoj) *
        //       ((Bxi + Bx[j]) * termA_avg[0] + (Byi + By[j]) * termA_avg[1] + (Bzi + Bz[j]) * termA_avg[2]); // GDSPH
        T f_i = mj * ((Bxi * termA1_i + Byi * termA2_i + Bzi * termA3_i) * a_mom +
                      (Bxj * termA1_j + Byj * termA2_j + Bzj * termA3_j) * b_mom); // PHANTOM (now using VE)

        momentum_x += mj * (a_mom * momentum_xi + b_mom * momentum_xj) - a_visc_x;
        momentum_y += mj * (a_mom * momentum_yi + b_mom * momentum_yj) - a_visc_y;
        momentum_z += mj * (a_mom * momentum_zi + b_mom * momentum_zj) - a_visc_z;

        return std::make_tuple(a_visc_energy, a_heat_cond, energy, momentum_x, momentum_y, momentum_z, f_i,
                               cstone::ijloop::symmetric::even(cstone::ijloop::reduction::max(vijsignal)));
    }
};

template<bool UseTdpdTrho, class T, class Tc>
struct MagneticMomentumAndEnergyPostamble
{
    Tc K, mu_0;

    template<class ParticleData, class Result>
    constexpr auto operator()(const ParticleData& iData, const Result& result) const
    {
        const auto [i, iPos, hi, vxi, vyi, vzi, mi, ci, ui, kxi, alpha_i, xmassi, pi, gradhi, c11i, c12i, c13i, c22i,
                    c23i, c33i, nci, Bxi, Byi, Bzi, dvxdxi, dvxdyi, dvxdzi, dvydxi, dvydyi, dvydzi, dvzdxi, dvzdyi,
                    dvzdzi, tdpdTrhoi, divvi, curlvi] = iData;
        auto [a_visc_energy, a_heat_cond, energy, momentum_x, momentum_y, momentum_z, f_i, maxvsignal] = result;

        T  mu_0Inv    = T(1) / mu_0;
        a_visc_energy = stl::max(T(0), a_visc_energy);
        T  proi       = pi / (kxi * mi * mi * gradhi);
        T  eCoeff     = UseTdpdTrho ? tdpdTrhoi : proi;
        Tc dui = K * (eCoeff * energy + T(0.5) * a_visc_energy + a_heat_cond); // factor of 2 already removed from 2P/rho

        // tensile instability correction factor, plasma beta dependent (Wissing et al.)
        T norm2_B = Bxi * Bxi + Byi * Byi + Bzi * Bzi;
        T beta    = T(2) * mu_0 * pi / norm2_B;
        T H       = T(0);
        // if (beta < 1) { H = 2.; } //SPHYNX
        // else if (beta <= 2) { H = 2 * (2. - beta) };
        if (beta < 2.) { H = 1.; } // PHANTOM
        else if (beta < 10.) { H = (10. - beta) / 8.; }
        // if (beta < T(1)) { H = T(1); } // Wissing
        // else if (beta < T(2)) { H = T(2) - beta; }

        // grad_P_xyz is stored as the acceleration, accel = -grad_P / rho
        return std::make_tuple(Tc(dui), T(K * (momentum_x - Bxi * f_i * H * mu_0Inv)),
                               T(K * (momentum_y - Byi * f_i * H * mu_0Inv)),
                               T(K * (momentum_z - Bzi * f_i * H * mu_0Inv)), maxvsignal);
    }
};

template<bool UseTdpdTrho, class T, class Tc>
struct MagneticMomentumAndEnergyPostambleWithDt : MagneticMomentumAndEnergyPostamble<UseTdpdTrho, T, Tc>
{
    Tc Kcour;

    MagneticMomentumAndEnergyPostambleWithDt(Tc K, Tc mu_0, Tc Kcour)
        : MagneticMomentumAndEnergyPostamble<UseTdpdTrho, T, Tc>{K, mu_0}
        , Kcour(Kcour)
    {
    }

    template<class ParticleData, class Result>
    constexpr auto operator()(const ParticleData& iData, const Result& result) const
    {
        const auto [du, grad_P_x, grad_P_y, grad_P_z, maxvsignal] =
            MagneticMomentumAndEnergyPostamble<UseTdpdTrho, T, Tc>::operator()(iData, result);
        const auto [i, iPos, hi, vxi, vyi, vzi, mi, ci, ui, kxi, alpha_i, xmassi, pi, gradhi, c11i, c12i, c13i, c22i,
                    c23i, c33i, nci, Bxi, Byi, Bzi, dvxdxi, dvxdyi, dvxdzi, dvydxi, dvydyi, dvydzi, dvzdxi, dvzdyi,
                    dvzdzi, tdpdTrhoi, divvi, curlvi] = iData;

        auto rhoi             = kxi * mi / xmassi;
        T    v_alfven2        = (Bxi * Bxi + Byi * Byi + Bzi * Bzi) / (this->mu_0 * rhoi);
        T    magneticVsignal  = std::sqrt(ci * ci + v_alfven2);

        // T dt = tsKCourant(maxvsignal, hi, magneticVsignal, Kcour); // less restrictive: ignores magneticVsignal when maxvsignal>0
        T dt = maxvsignal > T(0) ? stl::min(Kcour * hi / magneticVsignal, Kcour * hi / maxvsignal)
                                 : Kcour * hi / magneticVsignal;
        return std::make_tuple(du, grad_P_x, grad_P_y, grad_P_z, dt);
    }
};

template<bool SLR, class Neighborhood, class Tc, class T, class Tm, class Tm1>
void magneticMomentumAndEnergyIjLoop(Neighborhood const& neighborhood, Tc K, Tc Kcour, Tc mu_0, Tc alpha_u, T Atmin,
                                     T Atmax, T ramp, const T* vx, const T* vy, const T* vz, const Tm* m, const T* c,
                                     const Tc* u, const T* kx, const T* alpha, const T* xm, const T* p, const T* gradh,
                                     const T* c11, const T* c12, const T* c13, const T* c22, const T* c23, const T* c33,
                                     const unsigned* nc, const Tc* Bx, const Tc* By, const Tc* Bz, const T* dvxdx,
                                     const T* dvxdy, const T* dvxdz, const T* dvydx, const T* dvydy, const T* dvydz,
                                     const T* dvzdx, const T* dvzdy, const T* dvzdz, const T* tdpdTrho, const T* wh,
                                     T avFloor, Tm1* du, T* grad_P_x, T* grad_P_y, T* grad_P_z, const T* divv, const T* curlv,
                                     T* dt)
{
    if constexpr (!SLR) { dvxdx = dvxdy = dvxdz = dvydx = dvydy = dvydz = dvzdx = dvzdy = dvzdz = vx; }
    const auto input =
        std::make_tuple(vx, vy, vz, m, c, u, kx, alpha, xm, p, gradh, c11, c12, c13, c22, c23, c33, nc, Bx, By, Bz,
                        dvxdx, dvxdy, dvxdz, dvydx, dvydy, dvydz, dvzdx, dvzdy, dvzdz,
                        tdpdTrho ? tdpdTrho : vx /* pass random derefable array if tdpdTrho is null */,
                        divv, curlv);
    const auto output = std::make_tuple(du, grad_P_x, grad_P_y, grad_P_z, dt);
    if (tdpdTrho)
    {
        neighborhood.ijLoop(input, output,
                            MagneticMomentumAndEnergyInteraction<SLR, T>{wh, T(mu_0), T(alpha_u), Atmin, Atmax, ramp, avFloor},
                            MagneticMomentumAndEnergyPostambleWithDt<true, T, Tc>{K, mu_0, Kcour});
    }
    else
    {
        neighborhood.ijLoop(input, output,
                            MagneticMomentumAndEnergyInteraction<SLR, T>{wh, T(mu_0), T(alpha_u), Atmin, Atmax, ramp, avFloor},
                            MagneticMomentumAndEnergyPostambleWithDt<false, T, Tc>{K, mu_0, Kcour});
    }
}

} // namespace sph::magneto
