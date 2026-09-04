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

/*! @file IAD + grad-h + velocity Jacobian in a single neighbor loop
 *
 * Mirrors hydro_ve/iad_divv_curlv_kern.hpp, differing only in keeping the velocity Jacobian
 * uncompressed: 9 components rather than the 6 symmetrized ones, because the induction equation
 * needs the antisymmetric part.
 *
 * The per-pair terms accumulate raw factors and the postamble applies the IAD matrix afterwards,
 * which is exact because C is constant per i and the neighbor sum is linear:
 *
 *     sum_j (C . r_ij) f_ij  ==  C . sum_j r_ij f_ij
 *
 * That is what lets c11..c33 and gradh be produced by the same loop that consumes them, instead of
 * being carried in the input tuple for both i and j across two loops.
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
struct IadFullDivvCurlvInteraction
{
    const T *wh, *whd;

    template<class ParticleData, class Tc>
    constexpr auto operator()(const ParticleData& iData, const ParticleData& jData, cstone::Vec3<Tc> const& r_ij,
                              T r2) const
    {
        const auto [i, iPos, hi, vxi, vyi, vzi, mi, xmi, kxi, nci] = iData;
        const auto [j, jPos, hj, vxj, vyj, vzj, mj, xmj, kxj, ncj] = jData;

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

        T vxw = (vxj - vxi) * xmj;
        T vyw = (vyj - vyi) * xmj;
        T vzw = (vzj - vzi) * xmj;

        return std::tuple_cat(iadResult, std::make_tuple(vxw * mrx, vxw * mry, vxw * mrz, //
                                                         vyw * mrx, vyw * mry, vyw * mrz, //
                                                         vzw * mrx, vzw * mry, vzw * mrz));
    }
};

template<bool DoCurlv, class T, class Tc>
struct IadFullDivvCurlvPostamble
{
    Tc K;

    template<class ParticleData, class Result>
    constexpr auto operator()(const ParticleData& iData, const Result& result) const
    {
        const auto [i, iPos, hi, vxi, vyi, vzi, mi, xmi, kxi, nci] = iData;
        const auto [tau11, tau12, tau13, tau22, tau23, tau33, whomegai, wrho0i, sum_error, //
                    dVxX, dVxY, dVxZ, dVyX, dVyY, dVyZ, dVzX, dVzY, dVzZ]                  = result;

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

        T hiInv    = T(1) / hi;
        T hiInv3   = hiInv * hiInv * hiInv;
        T norm_kxi = K * hiInv3 / (kxi * gradhi);

        T divv = norm_kxi * (dVx[0] + dVy[1] + dVz[2]);

        auto iad      = std::make_tuple(c11i, c12i, c13i, c22i, c23i, c33i, gradhi);
        auto jacobian = std::make_tuple(norm_kxi * dVx[0], norm_kxi * dVx[1], norm_kxi * dVx[2], //
                                        norm_kxi * dVy[0], norm_kxi * dVy[1], norm_kxi * dVy[2], //
                                        norm_kxi * dVz[0], norm_kxi * dVz[1], norm_kxi * dVz[2]);

        if constexpr (DoCurlv)
        {
            cstone::Vec3<T> curlV{dVz[1] - dVy[2], dVx[2] - dVz[0], dVy[0] - dVx[1]};
            T               curlv = norm_kxi * std::sqrt(norm2(curlV));
            return std::tuple_cat(iad, std::make_tuple(divv, curlv), jacobian);
        }
        else { return std::tuple_cat(iad, std::make_tuple(divv), jacobian); }
    }
};

template<class Neighborhood, class Tc, class Tm, class T>
void iadFullDivvCurlvIjLoop(Neighborhood const& neighborhood, Tc K, const T* vx, const T* vy, const T* vz, const Tm* m,
                            const T* xm, const T* kx, const unsigned* nc, const T* wh, const T* whd, T* c11, T* c12,
                            T* c13, T* c22, T* c23, T* c33, T* gradh, T* divv, T* curlv, T* dvxdx, T* dvxdy, T* dvxdz,
                            T* dvydx, T* dvydy, T* dvydz, T* dvzdx, T* dvzdy, T* dvzdz)
{
    const auto input    = std::make_tuple(vx, vy, vz, m, xm, kx, nc);
    const auto iad      = std::make_tuple(c11, c12, c13, c22, c23, c33, gradh);
    const auto jacobian = std::make_tuple(dvxdx, dvxdy, dvxdz, dvydx, dvydy, dvydz, dvzdx, dvzdy, dvzdz);

    if (curlv != nullptr)
    {
        neighborhood.ijLoop(input, std::tuple_cat(iad, std::make_tuple(divv, curlv), jacobian),
                            IadFullDivvCurlvInteraction<T>{wh, whd}, IadFullDivvCurlvPostamble<true, T, Tc>{K});
    }
    else
    {
        neighborhood.ijLoop(input, std::tuple_cat(iad, std::make_tuple(divv), jacobian),
                            IadFullDivvCurlvInteraction<T>{wh, whd}, IadFullDivvCurlvPostamble<false, T, Tc>{K});
    }
}

} // namespace sph::magneto
