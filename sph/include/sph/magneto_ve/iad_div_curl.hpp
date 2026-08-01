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
 * @brief Integral-approach-to-derivative and divergence/curl i-loop driver
 *
 * @author Ruben Cabezon <ruben.cabezon@unibas.ch>
 */

#pragma once

#include "sph/sph_gpu.hpp"
#include "sph/hydro_ve/iad_gradh_kern.hpp"
#include "full_divv_curlv_kern.hpp"
#include "divB_curlB_kern.hpp"

namespace sph::magneto
{

//! @brief IAD coefficients + grad-h, reusing the hydro per-pair functors
template<class Neighborhood, class Tc, class Tm, class T>
void iadGradhIjLoop(Neighborhood const& neighborhood, Tc K, const Tm* m, const T* xm, const T* kx, const unsigned* nc,
                    const T* wh, const T* whd, T* c11, T* c12, T* c13, T* c22, T* c23, T* c33, T* gradh)
{
    neighborhood.ijLoop(std::make_tuple(m, xm, kx, nc), std::make_tuple(c11, c12, c13, c22, c23, c33, gradh),
                        IADGradhInteraction<T>{wh, whd}, IADGradhPostamble<T, Tc>{K});
}

template<class Tc, class SimulationData>
void computeIadFullDivvCurlv(const GroupView& grp, SimulationData& sim, const cstone::Box<Tc>& box)
{
    auto& d  = sim.hydro;
    auto& md = sim.magneto;

    if constexpr (cstone::HaveGpu<typename SimulationData::AcceleratorType>{})
    {
        cuda::computeIadFullDivvCurlv(grp, d, md, box);
    }
    else
    {
        auto* curlv = (d.x.size() == d.curlv.size()) ? d.curlv.data() : nullptr;

        iadGradhIjLoop(d.neighborhood, d.K, d.m.data(), d.xm.data(), d.kx.data(), d.nc.data(), d.wh.data(),
                       d.whd.data(), d.c11.data(), d.c12.data(), d.c13.data(), d.c22.data(), d.c23.data(),
                       d.c33.data(), d.gradh.data());

        fullDivvCurlvIjLoop(d.neighborhood, d.K, d.vx.data(), d.vy.data(), d.vz.data(), d.kx.data(), d.xm.data(),
                            d.c11.data(), d.c12.data(), d.c13.data(), d.c22.data(), d.c23.data(), d.c33.data(),
                            d.gradh.data(), d.wh.data(), d.divv.data(), curlv, md.dvxdx.data(), md.dvxdy.data(),
                            md.dvxdz.data(), md.dvydx.data(), md.dvydy.data(), md.dvydz.data(), md.dvzdx.data(),
                            md.dvzdy.data(), md.dvzdz.data());

        divBCurlBIjLoop(d.neighborhood, d.K, md.Bx.data(), md.By.data(), md.Bz.data(), d.kx.data(), d.xm.data(),
                        d.c11.data(), d.c12.data(), d.c13.data(), d.c22.data(), d.c23.data(), d.c33.data(),
                        d.gradh.data(), d.wh.data(), md.divB.data(), md.divB_conj.data(), md.curlB_x.data(),
                        md.curlB_y.data(), md.curlB_z.data(), md.gradB_norm.data(), md.alpha_B.data(),
                        md.dBxdx.data(), md.dBxdy.data(), md.dBxdz.data(), md.dBydx.data(), md.dBydy.data(),
                        md.dBydz.data(), md.dBzdx.data(),
                        md.dBzdy.data(), md.dBzdz.data(), md.resistivityScheme, md.alpha_B_const);
    }
}

} // namespace sph::magneto
