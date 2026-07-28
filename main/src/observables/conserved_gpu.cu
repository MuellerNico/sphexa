/*
 * MIT License
 *
 * Copyright (c) 2022 CSCS, ETH Zurich
 *               2022 University of Basel
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
 * @brief  Energy and momentum reductions on the GPU
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <thrust/execution_policy.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

#include "conserved_gpu.h"

namespace sphexa
{

using cstone::Vec3;
using thrust::get;

struct TuplePlus
{
    using type = thrust::tuple<double, Vec3<double>, Vec3<double>>;
    HOST_DEVICE_FUN type operator()(const type& a, const type& b) const
    {
        return type(get<0>(a) + get<0>(b), get<1>(a) + get<1>(b), get<2>(a) + get<2>(b));
    }
};

/*! @brief Functor to compute kinetic and internal energy and linear and angular momentum
 *
 * @tparam Tc   type of x,y,z coordinates
 * @tparam Tm   type of mass
 * @tparam Tv   type of velocities
 */
template<class Tc, class Tm, class Tv>
struct EMom
{
    /*! @brief compute energies and momenta for a single particle
     *
     * @param p   Tuple<x,y,z,m,vx,vy,vz,temp> with data for one particle
     * @return    Tuple<kinetic energy, internal energy, linear momentum, angular momentum>
     */
    HOST_DEVICE_FUN TuplePlus::type operator()(const thrust::tuple<Tc, Tc, Tc, Tm, Tv, Tv, Tv>& p)
    {
        Vec3<double> X{get<0>(p), get<1>(p), get<2>(p)};
        Vec3<double> V{get<4>(p), get<5>(p), get<6>(p)};
        Tm           m = get<3>(p);
        return {m * norm2(V), double(m) * V, double(m) * cross(X, V)};
    }
};

template<class Tc, class Tv, class Tt, class Tm>
std::tuple<double, double, Vec3<double>, Vec3<double>>
conservedQuantitiesGpu(double cv, const Tc* x, const Tc* y, const Tc* z, const Tv* vx, const Tv* vy, const Tv* vz,
                       const Tt* temp, const Tt* u, const Tm* m, size_t first, size_t last)
{
    auto it1 = thrust::make_zip_iterator(
        thrust::make_tuple(x + first, y + first, z + first, m + first, vx + first, vy + first, vz + first));
    auto it2 = thrust::make_zip_iterator(
        thrust::make_tuple(x + last, y + last, z + last, m + last, vx + last, vy + last, vz + last));

    //! apply EMom to each particle and reduce results into a single sum
    auto ret = thrust::transform_reduce(thrust::device, it1, it2, EMom<Tc, Tm, Tv>{}, TuplePlus::type{}, TuplePlus{});
    auto [eKin, linMom, angMom] = std::make_tuple(get<0>(ret), get<1>(ret), get<2>(ret));

    double eInt = 0.0;
    if (temp != nullptr)
    {
        eInt = cv * thrust::inner_product(thrust::device, m + first, m + last, temp + first, Tt(0.0));
    }
    else if (u != nullptr) { eInt = thrust::inner_product(thrust::device, m + first, m + last, u + first, Tt(0.0)); }

    return {0.5 * eKin, eInt, linMom, angMom};
}

#define CONSERVED_Q_GPU(Tc, Tv, Tt, Tm)                                                                                \
    template std::tuple<double, double, Vec3<double>, Vec3<double>> conservedQuantitiesGpu(                            \
        double cv, const Tc* x, const Tc* y, const Tc* z, const Tv* vx, const Tv* vy, const Tv* vz, const Tt* temp,    \
        const Tt* u, const Tm* m, size_t, size_t)

CONSERVED_Q_GPU(double, double, double, double);
CONSERVED_Q_GPU(double, double, double, float);
CONSERVED_Q_GPU(double, float, double, float);
CONSERVED_Q_GPU(float, float, float, float);

//! @brief magnetic energy density |B|² * V for a single particle, V = xm/kx
template<class Tc, class Th>
struct EMag
{
    //! @param p   Tuple<Bx, By, Bz, xm, kx> with data for one particle
    HOST_DEVICE_FUN double operator()(const thrust::tuple<Tc, Tc, Tc, Th, Th>& p)
    {
        const Vec3<double> B{get<0>(p), get<1>(p), get<2>(p)};
        return norm2(B) * get<3>(p) / get<4>(p);
    }
};

//! @brief per-particle div(B) error h*|div(B)|/|B|, guarded against zero field
template<class Tc, class Th>
struct DivBError
{
    //! @param p   Tuple<Bx, By, Bz, div(B), h> with data for one particle
    HOST_DEVICE_FUN double operator()(const thrust::tuple<Tc, Tc, Tc, Th, Th>& p)
    {
        const Vec3<double> B{get<0>(p), get<1>(p), get<2>(p)};
        auto               magB2 = norm2(B);
        return magB2 > 0 ? abs(get<3>(p)) * get<4>(p) / sqrt(magB2) : 0.0;
    }
};

template<class Tc, class Th>
std::tuple<double, double, double> magneticEnergyGpu(Tc mu_0, const Th* xm, const Th* kx, const Th* divB, const Th* h,
                                                     const Tc* Bx, const Tc* By, const Tc* Bz, size_t first,
                                                     size_t last)
{
    auto magIt1 = thrust::make_zip_iterator(
        thrust::make_tuple(Bx + first, By + first, Bz + first, xm + first, kx + first));
    auto magIt2 =
        thrust::make_zip_iterator(thrust::make_tuple(Bx + last, By + last, Bz + last, xm + last, kx + last));

    auto errIt1 = thrust::make_zip_iterator(
        thrust::make_tuple(Bx + first, By + first, Bz + first, divB + first, h + first));
    auto errIt2 =
        thrust::make_zip_iterator(thrust::make_tuple(Bx + last, By + last, Bz + last, divB + last, h + last));

    double BMag = thrust::transform_reduce(thrust::device, magIt1, magIt2, EMag<Tc, Th>{}, 0.0, thrust::plus<double>{});
    double cumulativeDivBError =
        thrust::transform_reduce(thrust::device, errIt1, errIt2, DivBError<Tc, Th>{}, 0.0, thrust::plus<double>{});
    double localMaxDivBError =
        thrust::transform_reduce(thrust::device, errIt1, errIt2, DivBError<Tc, Th>{}, 0.0, thrust::maximum<double>{});

    return {0.5 * BMag / mu_0, cumulativeDivBError, localMaxDivBError};
}

#define EMAG(Tc, Th)                                                                                                   \
    template std::tuple<double, double, double> magneticEnergyGpu(Tc mu_0, const Th* xm, const Th* kx, const Th* divB, \
                                                                  const Th* h, const Tc* Bx, const Tc* By,             \
                                                                  const Tc* Bz, size_t first, size_t last);

EMAG(double, double);
EMAG(double, float);
EMAG(float, float);

//! @brief total resistive heating rate Sum_i m_i * du_diss_i
template<class Tm, class Th>
double magneticDissipationGpu(const Tm* m, const Th* du_diss, size_t first, size_t last)
{
    return thrust::inner_product(thrust::device, m + first, m + last, du_diss + first, 0.0);
}

#define MAG_DISS(Tm, Th) \
    template double magneticDissipationGpu(const Tm* m, const Th* du_diss, size_t first, size_t last);

MAG_DISS(double, double);
MAG_DISS(float, double);
MAG_DISS(float, float);

} // namespace sphexa
