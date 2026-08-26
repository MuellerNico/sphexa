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
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file IAD + grad-h + velocity Jacobian + B Jacobian in a single neighbor loop
 *
 * Mirrors the hydro fusion in hydro_ve/iad_divv_curlv_kern.hpp. The per-pair terms accumulate
 * raw factors and the postamble applies the IAD matrix afterwards, which is exact because C is
 * constant per i and the neighbor sum is linear:
 *
 *     sum_j (C . r_ij) f_ij  ==  C . sum_j r_ij f_ij
 *
 * That is what allows c11..c33 and gradh to be produced by the same loop that consumes them,
 * instead of being carried in the input tuple for both i and j across three separate loops.
 *
 * @author Nicolas Müller
 */

#pragma once

#include "cstone/cuda/annotation.hpp"
#include "cstone/traversal/ijloop/ijloop.hpp"

#include "sph/kernels.hpp"
#include "sph/table_lookup.hpp"
#include "sph/hydro_ve/iad_gradh_kern.hpp"

namespace sph::magneto
{

template<class T>
struct IadDivvDivBInteraction
{
    const T *wh, *whd;

    template<class ParticleData, class Tc>
    constexpr auto operator()(const ParticleData& iData, const ParticleData& jData, cstone::Vec3<Tc> const& r_ij,
                              T r2) const
    {
        const auto [i, iPos, hi, vxi, vyi, vzi, Bxi, Byi, Bzi, mi, xmi, kxi, nci] = iData;
        const auto [j, jPos, hj, vxj, vyj, vzj, Bxj, Byj, Bzj, mj, xmj, kxj, ncj] = jData;

        auto iadResult = IADGradhInteraction<T>{wh, whd}(std::make_tuple(i, iPos, hi, mi, xmi, kxi, nci),
                                                         std::make_tuple(j, jPos, hj, mj, xmj, kxj, ncj), r_ij, r2);

        T hiInv = T(1) / hi;
        T dist  = std::sqrt(r2);
        T v1    = dist * hiInv;
        T Wi    = lt::lookup(wh, v1);

        // -r * W, the pair factor common to every gradient row. i == j needs no guard: r_ij is then zero.
        T mrx = -T(r_ij[0]) * Wi;
        T mry = -T(r_ij[1]) * Wi;
        T mrz = -T(r_ij[2]) * Wi;

        T vx_ji = vxj - vxi;
        T vy_ji = vyj - vyi;
        T vz_ji = vzj - vzi;

        T vxw = vx_ji * xmj;
        T vyw = vy_ji * xmj;
        T vzw = vz_ji * xmj;

        T Bx_ji = Bxj - Bxi;
        T By_ji = Byj - Byi;
        T Bz_ji = Bzj - Bzi;

        T Bxw = Bx_ji * xmj;
        T Byw = By_ji * xmj;
        T Bzw = Bz_ji * xmj;

        /* The conservative divB sums without xm[j], so it cannot be recovered from the xm-weighted
         * Jacobian factors above. Deferring C means accumulating the symmetrized outer product
         * B_ji (x) (-r W) instead of the contracted scalar: six sums rather than one.
         */
        return std::tuple_cat(
            iadResult, std::make_tuple(
                           // velocity Jacobian rows, pre-C
                           vxw * mrx, vxw * mry, vxw * mrz, //
                           vyw * mrx, vyw * mry, vyw * mrz, //
                           vzw * mrx, vzw * mry, vzw * mrz, //
                           // B Jacobian rows, pre-C
                           Bxw * mrx, Bxw * mry, Bxw * mrz, //
                           Byw * mrx, Byw * mry, Byw * mrz, //
                           Bzw * mrx, Bzw * mry, Bzw * mrz, //
                           // symmetrized B (x) (-r W), contracted with C in the postamble
                           Bx_ji * mrx,                     //
                           Bx_ji * mry + By_ji * mrx,       //
                           Bx_ji * mrz + Bz_ji * mrx,       //
                           By_ji * mry,                     //
                           By_ji * mrz + Bz_ji * mry,       //
                           Bz_ji * mrz));
    }
};

template<bool DoCurlv, bool SLR, class T, class Tc>
struct IadDivvDivBPostamble
{
    Tc K;

    template<class ParticleData, class Result>
    constexpr auto operator()(const ParticleData& iData, const Result& result) const
    {
        const auto [i, iPos, hi, vxi, vyi, vzi, Bxi, Byi, Bzi, mi, xmi, kxi, nci] = iData;
        const auto [tau11, tau12, tau13, tau22, tau23, tau33, whomegai, wrho0i, sum_error,                //
                    dVxX, dVxY, dVxZ, dVyX, dVyY, dVyZ, dVzX, dVzY, dVzZ,                                 //
                    dBxX, dBxY, dBxZ, dByX, dByY, dByZ, dBzX, dBzY, dBzZ,                                 //
                    q11, q12, q13, q22, q23, q33]                                 = result;

        const auto [c11i, c12i, c13i, c22i, c23i, c33i, gradhi] = IADGradhPostamble<T, Tc>{K}(
            std::make_tuple(i, iPos, hi, mi, xmi, kxi, nci),
            std::make_tuple(tau11, tau12, tau13, tau22, tau23, tau33, whomegai, wrho0i, sum_error));

        auto applyC = [&](T fx, T fy, T fz)
        {
            return cstone::Vec3<T>{c11i * fx + c12i * fy + c13i * fz, //
                                   c12i * fx + c22i * fy + c23i * fz, //
                                   c13i * fx + c23i * fy + c33i * fz};
        };

        const cstone::Vec3<T> dVx = applyC(dVxX, dVxY, dVxZ);
        const cstone::Vec3<T> dVy = applyC(dVyX, dVyY, dVyZ);
        const cstone::Vec3<T> dVz = applyC(dVzX, dVzY, dVzZ);

        const cstone::Vec3<T> dBx = applyC(dBxX, dBxY, dBxZ);
        const cstone::Vec3<T> dBy = applyC(dByX, dByY, dByZ);
        const cstone::Vec3<T> dBz = applyC(dBzX, dBzY, dBzZ);

        T hiInv    = T(1) / hi;
        T hiInv3   = hiInv * hiInv * hiInv;
        T norm_kxi = K * hiInv3 / (kxi * gradhi);

        T divv = norm_kxi * (dVx[0] + dVy[1] + dVz[2]);

        T dvxdx = norm_kxi * dVx[0];
        T dvxdy = norm_kxi * dVx[1];
        T dvxdz = norm_kxi * dVx[2];
        T dvydx = norm_kxi * dVy[0];
        T dvydy = norm_kxi * dVy[1];
        T dvydz = norm_kxi * dVy[2];
        T dvzdx = norm_kxi * dVz[0];
        T dvzdy = norm_kxi * dVz[1];
        T dvzdz = norm_kxi * dVz[2];

        T divB = norm_kxi * (dBx[0] + dBy[1] + dBz[2]);

        // energy-conjugate op for the cleaning psi source: exact transpose of grad-psi under the
        // magnetic-energy norm V = xm/kx. Not consistent for linear B, equal masses assumed.
        T divBcons  = c11i * q11 + c12i * q12 + c13i * q13 + c22i * q22 + c23i * q23 + c33i * q33;
        T divB_conj = K * hiInv3 * xmi / gradhi * divBcons;

        T curlB_x = norm_kxi * (dBz[1] - dBy[2]);
        T curlB_y = norm_kxi * (dBx[2] - dBz[0]);
        T curlB_z = norm_kxi * (dBy[0] - dBx[1]);

        T gradB_norm =
            norm_kxi * std::sqrt(dBx[0] * dBx[0] + dBx[1] * dBx[1] + dBx[2] * dBx[2] + dBy[0] * dBy[0] +
                                 dBy[1] * dBy[1] + dBy[2] * dBy[2] + dBz[0] * dBz[0] + dBz[1] * dBz[1] +
                                 dBz[2] * dBz[2]);

        auto iad = std::make_tuple(c11i, c12i, c13i, c22i, c23i, c33i, gradhi);

        auto velocity = [&]()
        {
            auto jacobian = std::make_tuple(dvxdx, dvxdy, dvxdz, dvydx, dvydy, dvydz, dvzdx, dvzdy, dvzdz);
            if constexpr (DoCurlv)
            {
                cstone::Vec3<T> curlV{dVz[1] - dVy[2], dVx[2] - dVz[0], dVy[0] - dVx[1]};
                T               curlv = norm_kxi * std::sqrt(norm2(curlV));
                return std::tuple_cat(std::make_tuple(divv, curlv), jacobian);
            }
            else { return std::tuple_cat(std::make_tuple(divv), jacobian); }
        }();

        auto magnetic = [&]()
        {
            auto common = std::make_tuple(divB, divB_conj, curlB_x, curlB_y, curlB_z, gradB_norm);
            if constexpr (SLR)
            {
                return std::tuple_cat(common, std::make_tuple(norm_kxi * dBx[0], norm_kxi * dBx[1], norm_kxi * dBx[2],
                                                              norm_kxi * dBy[0], norm_kxi * dBy[1], norm_kxi * dBy[2],
                                                              norm_kxi * dBz[0], norm_kxi * dBz[1], norm_kxi * dBz[2]));
            }
            else { return common; }
        }();

        return std::tuple_cat(iad, velocity, magnetic);
    }
};

template<bool SLR, class Neighborhood, class Tc, class Tm, class T>
void iadDivvDivBIjLoop(Neighborhood const& neighborhood, Tc K, const T* vx, const T* vy, const T* vz, const Tc* Bx,
                       const Tc* By, const Tc* Bz, const Tm* m, const T* xm, const T* kx, const unsigned* nc,
                       const T* wh, const T* whd, T* c11, T* c12, T* c13, T* c22, T* c23, T* c33, T* gradh, T* divv,
                       T* curlv, T* dvxdx, T* dvxdy, T* dvxdz, T* dvydx, T* dvydy, T* dvydz, T* dvzdx, T* dvzdy,
                       T* dvzdz, T* divB, T* divB_conj, T* curlB_x, T* curlB_y, T* curlB_z, T* gradB_norm, T* dBxdx,
                       T* dBxdy, T* dBxdz, T* dBydx, T* dBydy, T* dBydz, T* dBzdx, T* dBzdy, T* dBzdz)
{
    const auto input = std::make_tuple(vx, vy, vz, Bx, By, Bz, m, xm, kx, nc);

    const auto iad      = std::make_tuple(c11, c12, c13, c22, c23, c33, gradh);
    const auto jacobian = std::make_tuple(dvxdx, dvxdy, dvxdz, dvydx, dvydy, dvydz, dvzdx, dvzdy, dvzdz);
    const auto magneticCommon = std::make_tuple(divB, divB_conj, curlB_x, curlB_y, curlB_z, gradB_norm);
    const auto bJacobian =
        std::make_tuple(dBxdx, dBxdy, dBxdz, dBydx, dBydy, dBydz, dBzdx, dBzdy, dBzdz);

    auto run = [&](auto doCurlv, auto const& velocity)
    {
        constexpr bool DoCurlv = decltype(doCurlv)::value;
        if constexpr (SLR)
        {
            neighborhood.ijLoop(input, std::tuple_cat(iad, velocity, magneticCommon, bJacobian),
                                IadDivvDivBInteraction<T>{wh, whd},
                                IadDivvDivBPostamble<DoCurlv, true, T, Tc>{K});
        }
        else
        {
            neighborhood.ijLoop(input, std::tuple_cat(iad, velocity, magneticCommon),
                                IadDivvDivBInteraction<T>{wh, whd},
                                IadDivvDivBPostamble<DoCurlv, false, T, Tc>{K});
        }
    };

    if (curlv != nullptr) { run(std::true_type{}, std::tuple_cat(std::make_tuple(divv, curlv), jacobian)); }
    else { run(std::false_type{}, std::tuple_cat(std::make_tuple(divv), jacobian)); }
}

} // namespace sph::magneto
