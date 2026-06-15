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

/*! @file slope-limited reconstruction of the magnetic field for artificial resistivity
 *
 * @author Nicolas Müller
 */

#pragma once

#include <cmath>

#include "cstone/cuda/annotation.hpp"
#include "cstone/primitives/math.hpp"
#include "cstone/util/tuple.hpp"

namespace sph::magneto
{

//! @brief 3x3 matrix-vector product where the matrix is given as its three rows; result type follows @p vec
template<class Tv, class Tm>
HOST_DEVICE_FUN inline cstone::Vec3<Tv> matvec3(const cstone::Vec3<Tm>& row_x, const cstone::Vec3<Tm>& row_y,
                                                const cstone::Vec3<Tm>& row_z, const cstone::Vec3<Tv>& vec)
{
    cstone::Vec3<Tv> ret;
    ret[0] = row_x[0] * vec[0] + row_x[1] * vec[1] + row_x[2] * vec[2];
    ret[1] = row_y[0] * vec[0] + row_y[1] * vec[1] + row_y[2] * vec[2];
    ret[2] = row_z[0] * vec[0] + row_z[1] * vec[1] + row_z[2] * vec[2];
    return ret;
}

/*! @brief slope-limited reconstruction correction for the B-field difference B_ab = B_a - B_b
 *
 * Mirrors the velocity-side SLR scheme of García-Senz & Cabezón (2026). See PR Slr#597
 *
 * @param R           relative position vector (x_a - x_b)
 * @param eta_ab      min(|R|/h_a, |R|/h_b)  (q_ab in the paper, Eq. 15)
 * @param eta_crit    cbrt(32π / (3 n_b))    (q_crit, Eq. 16)
 * @param balsi       Balsara-like factor (1-B_a^p) for particle a (use 1 to disable)
 * @param balsj       Balsara-like factor (1-B_b^p) for particle b
 * @param gradBx_a    ∇Bx at particle a (∂Bx/∂x, ∂Bx/∂y, ∂Bx/∂z); same for By, Bz and particle b
 * @return            additive correction Δ such that B_ab_SLR = B_ab + Δ
 */
template<class Tc, class T>
HOST_DEVICE_FUN inline cstone::Vec3<Tc>
mhdSLRCorrection(cstone::Vec3<Tc> R, Tc eta_ab, T eta_crit, T balsi, T balsj, cstone::Vec3<T> gradBx_a,
                 cstone::Vec3<T> gradBy_a, cstone::Vec3<T> gradBz_a, cstone::Vec3<T> gradBx_b,
                 cstone::Vec3<T> gradBy_b, cstone::Vec3<T> gradBz_b)
{
    constexpr T q_fold_inv = 5.0f; // 1/q_fold with q_fold = 0.2 (Frontiere et al. 2017)
    
    cstone::Vec3<Tc> JBR_a = matvec3(gradBx_a, gradBy_a, gradBz_a, R);
    cstone::Vec3<Tc> JBR_b = matvec3(gradBx_b, gradBy_b, gradBz_b, R);

    // R · J_B · R
    T RJBR_a = T(dot(R, JBR_a));
    T RJBR_b = T(dot(R, JBR_b));

    // κ_ab (Eq. 14)
    T kappa_ab = T(1);
    // if (eta_ab < eta_crit) // force limiter to zero for anomalously close particle pairs (needed in mhd???)
    // {
    //     T etaDiff = T(q_fold_inv) * T(eta_ab - eta_crit);
    //     kappa_ab  = std::exp(-etaDiff * etaDiff);
    // }

    T F_ab   = (RJBR_b != T(0)) ? RJBR_a / RJBR_b : T(0); // Eq. 17
    T F_abp1 = T(1) + F_ab;

    // van Leer-like limiter (Eq. 13), with the 0.5 from Eqs. 11-12 absorbed into phi_ab
    T limiter = T(4) * F_ab / (F_abp1 * F_abp1);
    T vanLeer = (limiter < T(0)) ? T(0) : (limiter > T(1) ? T(1) : limiter);
    T phi_ab  = T(0.5) * kappa_ab * vanLeer;

    return Tc(-phi_ab) * (Tc(balsi) * JBR_a + Tc(balsj) * JBR_b);
}

} // namespace sph::magneto
