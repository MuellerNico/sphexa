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
/*! @file spatial derivatives of the magnetic field
 *
 * @author Ruben Cabezon <ruben.cabezon@unibas.ch>
 * @author Lukas Schmidt
 */

#pragma once

#include "cstone/cuda/annotation.hpp"
#include "cstone/traversal/ijloop/ijloop.hpp"

#include "sph/kernels.hpp"
#include "sph/table_lookup.hpp"

#include "resistivity.hpp"

namespace sph::magneto
{

template<class T>
struct DivBCurlBInteraction
{
    const T* wh;

    template<class ParticleData, class Tc>
    constexpr auto operator()(const ParticleData& iData, const ParticleData& jData, cstone::Vec3<Tc> const& r_ij,
                              T r2) const
    {
        const auto [i, iPos, hi, Bxi, Byi, Bzi, kxi, xmassi, c11i, c12i, c13i, c22i, c23i, c33i, gradhi] = iData;
        const auto [j, jPos, hj, Bxj, Byj, Bzj, kxj, xmassj, c11j, c12j, c13j, c22j, c23j, c33j, gradhj] = jData;

        T rx = r_ij[0];
        T ry = r_ij[1];
        T rz = r_ij[2];

        T hiInv = T(1) / hi;
        T dist  = std::sqrt(r2);
        T v1    = dist * hiInv;
        T Wi    = lt::lookup(wh, v1);

        T Bx_ji = Bxj - Bxi;
        T By_ji = Byj - Byi;
        T Bz_ji = Bzj - Bzi;

        cstone::Vec3<T> termA;
        termA[0] = -(c11i * rx + c12i * ry + c13i * rz) * Wi;
        termA[1] = -(c12i * rx + c22i * ry + c23i * rz) * Wi;
        termA[2] = -(c13i * rx + c23i * ry + c33i * rz) * Wi;

        // per-pair contribution to the B-gradient rows dBx/dBy/dBz (summed over neighbors by the framework)
        cstone::Vec3<T> dBx = (Bx_ji * xmassj) * termA;
        cstone::Vec3<T> dBy = (By_ji * xmassj) * termA;
        cstone::Vec3<T> dBz = (Bz_ji * xmassj) * termA;

        // self-volume-weighted (no xm[j]) divergence sum for the conservative divB
        T divBcons = Bx_ji * termA[0] + By_ji * termA[1] + Bz_ji * termA[2];

        return std::make_tuple(dBx[0], dBx[1], dBx[2], dBy[0], dBy[1], dBy[2], dBz[0], dBz[1], dBz[2], divBcons);
    }
};

template<class T, class Tc>
struct DivBCurlBPostamble
{
    Tc                K;
    ResistivityScheme scheme;
    Tc                alpha_B_const;

    template<class ParticleData, class Result>
    constexpr auto operator()(const ParticleData& iData, const Result& result) const
    {
        constexpr T alpha_B_max = T(1.0); // temporary bounds for AR switch
        constexpr T alpha_B_min = T(0.05);

        const auto [i, iPos, hi, Bxi, Byi, Bzi, kxi, xmassi, c11i, c12i, c13i, c22i, c23i, c33i, gradhi] = iData;
        auto [dBxx, dBxy, dBxz, dByx, dByy, dByz, dBzx, dBzy, dBzz, divBcons]                            = result;

        T hiInv    = T(1) / hi;
        T hiInv3   = hiInv * hiInv * hiInv;
        T norm_kxi = K * hiInv3 / (kxi * gradhi);

        // Conservative (energy-conjugate) divergence for the constrained-cleaning psi source (exact
        // transpose of grad-psi operator under magnetic-energy norm V = xm/kx). Uses the self volume
        // element xm[i] and drops the 1/kx. Assumes equal particle masses.
        T divB = K * hiInv3 * xmassi / gradhi * divBcons;

        T curlB_x = norm_kxi * (dBzy - dByz);
        T curlB_y = norm_kxi * (dBxz - dBzx);
        T curlB_z = norm_kxi * (dByx - dBxy);

        T gradB_norm = norm_kxi * std::sqrt(dBxx * dBxx + dBxy * dBxy + dBxz * dBxz + dByx * dByx + dByy * dByy +
                                            dByz * dByz + dBzx * dBzx + dBzy * dBzy + dBzz * dBzz);

        T dBxdx = norm_kxi * dBxx;
        T dBxdy = norm_kxi * dBxy;
        T dBxdz = norm_kxi * dBxz;
        T dBydx = norm_kxi * dByx;
        T dBydy = norm_kxi * dByy;
        T dBydz = norm_kxi * dByz;
        T dBzdx = norm_kxi * dBzx;
        T dBzdy = norm_kxi * dBzy;
        T dBzdz = norm_kxi * dBzz;

        T alpha_B;
        if (scheme == ResistivityScheme::Constant) { alpha_B = alpha_B_const; }
        else if (scheme == ResistivityScheme::SLR || scheme == ResistivityScheme::SLRB ||
                 scheme == ResistivityScheme::SLRB2)
        {
            alpha_B = T(1);
        }
        else
        {
            // Switch (Tricco & Price 2013, eq. 16)
            T B_norm   = std::sqrt(Bxi * Bxi + Byi * Byi + Bzi * Bzi);
            T alpha_Bi = (B_norm > 0) ? hi * gradB_norm / B_norm : alpha_B_max;
            if (alpha_Bi > alpha_B_max) alpha_Bi = alpha_B_max;
            if (alpha_Bi < alpha_B_min) alpha_Bi = alpha_B_min;
            alpha_B = alpha_Bi;
        }

        return std::make_tuple(divB, curlB_x, curlB_y, curlB_z, gradB_norm, alpha_B, dBxdx, dBxdy, dBxdz, dBydx, dBydy,
                               dBydz, dBzdx, dBzdy, dBzdz);
    }
};

template<class Neighborhood, class Tc, class T>
void divBCurlBIjLoop(Neighborhood const& neighborhood, Tc K, const Tc* Bx, const Tc* By, const Tc* Bz, const T* kx,
                     const T* xm, const T* c11, const T* c12, const T* c13, const T* c22, const T* c23, const T* c33,
                     const T* gradh, const T* wh, T* divB, T* curlB_x, T* curlB_y, T* curlB_z, T* gradB_norm,
                     T* alpha_B, T* dBxdx, T* dBxdy, T* dBxdz, T* dBydx, T* dBydy, T* dBydz, T* dBzdx, T* dBzdy,
                     T* dBzdz, ResistivityScheme scheme, Tc alpha_B_const)
{
    const auto input  = std::make_tuple(Bx, By, Bz, kx, xm, c11, c12, c13, c22, c23, c33, gradh);
    const auto output = std::make_tuple(divB, curlB_x, curlB_y, curlB_z, gradB_norm, alpha_B, dBxdx, dBxdy, dBxdz,
                                        dBydx, dBydy, dBydz, dBzdx, dBzdy, dBzdz);
    neighborhood.ijLoop(input, output, DivBCurlBInteraction<T>{wh},
                        DivBCurlBPostamble<T, Tc>{K, scheme, alpha_B_const});
}

} // namespace sph::magneto
