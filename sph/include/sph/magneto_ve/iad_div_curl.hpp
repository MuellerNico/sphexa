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
#include "iad_full_divv_curlv_kern.hpp"
#include "divB_curlB_kern.hpp"

namespace sph::magneto
{

template<bool SLR, class Tc, class SimulationData>
void computeIadFullDivvCurlv(const GroupView& grp, SimulationData& sim, const cstone::Box<Tc>& box)
{
    auto& d  = sim.hydro;
    auto& md = sim.magneto;

    if constexpr (cstone::HaveGpu<typename SimulationData::AcceleratorType>{})
    {
        cuda::computeIadFullDivvCurlv<SLR>(grp, d, md, box);
    }
    else
    {
        auto* curlv = (d.x.size() == d.curlv.size()) ? d.curlv.data() : nullptr;

        iadFullDivvCurlvIjLoop(d.neighborhood, d.K, d.vx.data(), d.vy.data(), d.vz.data(), d.m.data(), d.xm.data(),
                               d.kx.data(), d.nc.data(), d.wh.data(), d.whd.data(), d.c11.data(), d.c12.data(),
                               d.c13.data(), d.c22.data(), d.c23.data(), d.c33.data(), d.gradh.data(), d.divv.data(),
                               curlv, md.dvxdx.data(), md.dvxdy.data(), md.dvxdz.data(), md.dvydx.data(),
                               md.dvydy.data(), md.dvydz.data(), md.dvzdx.data(), md.dvzdy.data(), md.dvzdz.data());

        divBCurlBIjLoop<SLR>(d.neighborhood, d.K, md.Bx.data(), md.By.data(), md.Bz.data(), d.kx.data(), d.xm.data(),
                             d.c11.data(), d.c12.data(), d.c13.data(), d.c22.data(), d.c23.data(), d.c33.data(),
                             d.gradh.data(), d.wh.data(), md.divB.data(), md.divB_conj.data(), md.curlB_x.data(),
                             md.curlB_y.data(), md.curlB_z.data(), md.gradB_norm.data(), md.dBxdx.data(),
                             md.dBxdy.data(), md.dBxdz.data(), md.dBydx.data(), md.dBydy.data(), md.dBydz.data(),
                             md.dBzdx.data(), md.dBzdy.data(), md.dBzdz.data());
    }
}

} // namespace sph::magneto
