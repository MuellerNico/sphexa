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

#include "sph/sph_gpu.hpp"
#include "induction_dissipation_kern.hpp"

namespace sph::magneto
{

template<class Tc, class SimulationData>
void computeInductionAndDissipation(const GroupView& grp, SimulationData& sim, const cstone::Box<Tc>& box)
{
    auto& d  = sim.hydro;
    auto& md = sim.magneto;

    if constexpr (cstone::HaveGpu<typename SimulationData::AcceleratorType>{})
    {
        cuda::computeInductionAndDissipationGpu(grp, d, md, box);
    }
    else
    {
        inductionAndDissipationIjLoop(
            d.neighborhood, d.K, md.mu_0, md.resistivityScheme, d.vx.data(), d.vy.data(), d.vz.data(), d.c.data(),
            md.Bx.data(), md.By.data(), md.Bz.data(), d.m.data(), d.xm.data(), d.kx.data(), d.gradh.data(),
            d.c11.data(), d.c12.data(), d.c13.data(), d.c22.data(), d.c23.data(), d.c33.data(), md.alpha_B.data(),
            md.psi_ch.data(), d.nc.data(), md.dBxdx.data(), md.dBxdy.data(), md.dBxdz.data(), md.dBydx.data(),
            md.dBydy.data(), md.dBydz.data(), md.dBzdx.data(), md.dBzdy.data(), md.dBzdz.data(), md.dvxdx.data(),
            md.dvxdy.data(), md.dvxdz.data(), md.dvydx.data(), md.dvydy.data(), md.dvydz.data(), md.dvzdx.data(),
            md.dvzdy.data(), md.dvzdz.data(), md.divB.data(), d.wh.data(), md.dBx.data(), md.dBy.data(), md.dBz.data(),
            d.du.data(), md.d_psi_ch.data());
    }
}

} // namespace sph::magneto
