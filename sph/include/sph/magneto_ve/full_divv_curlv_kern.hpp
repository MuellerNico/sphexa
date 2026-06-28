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
/*! @file
 * @brief Divergence of velocity vector field
 * Doesn't compress divv to six fields
 * @author Ruben Cabezon <ruben.cabezon@unibas.ch>
 */

#pragma once

#include "cstone/cuda/annotation.hpp"
#include "cstone/traversal/ijloop/ijloop.hpp"

#include "sph/kernels.hpp"
#include "sph/table_lookup.hpp"

namespace sph::magneto
{

template<class T>
struct FullDivvCurlvInteraction
{
    const T* wh;

    template<class ParticleData, class Tc>
    constexpr auto operator()(const ParticleData& iData, const ParticleData& jData, cstone::Vec3<Tc> const& r_ij,
                              T r2) const
    {
        const auto [i, iPos, hi, vxi, vyi, vzi, kxi, xmassi, c11i, c12i, c13i, c22i, c23i, c33i, gradhi] = iData;
        const auto [j, jPos, hj, vxj, vyj, vzj, kxj, xmassj, c11j, c12j, c13j, c22j, c23j, c33j, gradhj] = jData;

        T rx = r_ij[0];
        T ry = r_ij[1];
        T rz = r_ij[2];

        T hiInv = T(1) / hi;
        T dist  = std::sqrt(r2);
        T v1    = dist * hiInv;
        T Wi    = lt::lookup(wh, v1);

        T vx_ji = vxj - vxi;
        T vy_ji = vyj - vyi;
        T vz_ji = vzj - vzi;

        cstone::Vec3<T> termA;
        termA[0] = -(c11i * rx + c12i * ry + c13i * rz) * Wi;
        termA[1] = -(c12i * rx + c22i * ry + c23i * rz) * Wi;
        termA[2] = -(c13i * rx + c23i * ry + c33i * rz) * Wi;

        // per-pair contribution to the velocity-gradient rows dVx/dVy/dVz (summed over neighbors by the framework)
        cstone::Vec3<T> dVx = (vx_ji * xmassj) * termA;
        cstone::Vec3<T> dVy = (vy_ji * xmassj) * termA;
        cstone::Vec3<T> dVz = (vz_ji * xmassj) * termA;

        return std::make_tuple(dVx[0], dVx[1], dVx[2], dVy[0], dVy[1], dVy[2], dVz[0], dVz[1], dVz[2]);
    }
};

template<bool DoCurlv, class T, class Tc>
struct FullDivvCurlvPostamble
{
    Tc K;

    template<class ParticleData, class Result>
    constexpr auto operator()(const ParticleData& iData, const Result& result) const
    {
        const auto [i, iPos, hi, vxi, vyi, vzi, kxi, xmassi, c11i, c12i, c13i, c22i, c23i, c33i, gradhi] = iData;
        auto [dVxx, dVxy, dVxz, dVyx, dVyy, dVyz, dVzx, dVzy, dVzz]                                      = result;

        T hiInv    = T(1) / hi;
        T hiInv3   = hiInv * hiInv * hiInv;
        T norm_kxi = K * hiInv3 / (kxi * gradhi);

        T divv = norm_kxi * (dVxx + dVyy + dVzz);

        T dvxdx = norm_kxi * dVxx;
        T dvxdy = norm_kxi * dVxy;
        T dvxdz = norm_kxi * dVxz;
        T dvydx = norm_kxi * dVyx;
        T dvydy = norm_kxi * dVyy;
        T dvydz = norm_kxi * dVyz;
        T dvzdx = norm_kxi * dVzx;
        T dvzdy = norm_kxi * dVzy;
        T dvzdz = norm_kxi * dVzz;

        if constexpr (DoCurlv)
        {
            cstone::Vec3<T> curlV{dVzy - dVyz, dVxz - dVzx, dVyx - dVxy};
            T curlv  = norm_kxi * std::sqrt(norm2(curlV));
            return std::make_tuple(divv, curlv, dvxdx, dvxdy, dvxdz, dvydx, dvydy, dvydz, dvzdx, dvzdy, dvzdz);
        }
        else
        {
            return std::make_tuple(divv, dvxdx, dvxdy, dvxdz, dvydx, dvydy, dvydz, dvzdx, dvzdy, dvzdz);
        }
    }
};

template<class Neighborhood, class Tc, class T>
void fullDivvCurlvIjLoop(Neighborhood const& neighborhood, Tc K, const T* vx, const T* vy, const T* vz, const T* kx,
                         const T* xm, const T* c11, const T* c12, const T* c13, const T* c22, const T* c23,
                         const T* c33, const T* gradh, const T* wh, T* divv, T* curlv, T* dvxdx, T* dvxdy, T* dvxdz,
                         T* dvydx, T* dvydy, T* dvydz, T* dvzdx, T* dvzdy, T* dvzdz)
{
    const auto input = std::make_tuple(vx, vy, vz, kx, xm, c11, c12, c13, c22, c23, c33, gradh);
    if (curlv != nullptr)
    {
        const auto output =
            std::make_tuple(divv, curlv, dvxdx, dvxdy, dvxdz, dvydx, dvydy, dvydz, dvzdx, dvzdy, dvzdz);
        neighborhood.ijLoop(input, output, FullDivvCurlvInteraction<T>{wh}, FullDivvCurlvPostamble<true, T, Tc>{K});
    }
    else
    {
        const auto output = std::make_tuple(divv, dvxdx, dvxdy, dvxdz, dvydx, dvydy, dvydz, dvzdx, dvzdy, dvzdz);
        neighborhood.ijLoop(input, output, FullDivvCurlvInteraction<T>{wh}, FullDivvCurlvPostamble<false, T, Tc>{K});
    }
}

} // namespace sph::magneto
