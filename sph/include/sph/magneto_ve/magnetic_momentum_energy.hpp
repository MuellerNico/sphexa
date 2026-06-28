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

#include "sph/sph_gpu.hpp"
#include "magnetic_momentum_energy_kern.hpp"

namespace sph::magneto
{

template<bool avClean, class T, class SimData>
void computeMomentumEnergy(const GroupView& grp, float* groupDt, SimData& sim, const cstone::Box<T>& box)
{
    auto& d  = sim.hydro;
    auto& md = sim.magneto;

    if constexpr (cstone::HaveGpu<typename SimData::AcceleratorType>{})
    {
        cuda::computeMagneticMomentumEnergy<avClean>(grp, groupDt, d, md, box);
    }
    else
    {
        magneticMomentumAndEnergyIjLoop<avClean>(
            d.neighborhood, d.K, d.Kcour, md.mu_0, md.alpha_u, d.Atmin, d.Atmax, d.ramp, d.vx.data(), d.vy.data(),
            d.vz.data(), d.m.data(), d.c.data(), d.u.data(), d.kx.data(), d.alpha.data(), d.xm.data(), d.p.data(),
            d.gradh.data(), d.c11.data(), d.c12.data(), d.c13.data(), d.c22.data(), d.c23.data(), d.c33.data(),
            d.nc.data(), md.Bx.data(), md.By.data(), md.Bz.data(), md.dvxdx.data(), md.dvxdy.data(), md.dvxdz.data(),
            md.dvydx.data(), md.dvydy.data(), md.dvydz.data(), md.dvzdx.data(), md.dvzdy.data(), md.dvzdz.data(),
            d.tdpdTrho.data(), d.wh.data(), d.du.data(), d.ax.data(), d.ay.data(), d.az.data(), d.dtCourant.data());

        auto minDt = std::numeric_limits<typename SimData::HydroType>::infinity();
#pragma omp parallel for reduction(min : minDt)
        for (auto i = grp.firstBody; i < grp.lastBody; ++i)
            minDt = std::min(minDt, d.dtCourant[i]);
        d.minDtCourant = minDt;
    }
}

} // namespace sph::magneto
