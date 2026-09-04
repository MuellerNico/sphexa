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

//! @brief SLR reconstructs B_ab directly and needs no amplitude; otherwise alpha_B is the --resistivity constant
template<bool SLR, class MagnetoData>
auto alphaB(const MagnetoData& md)
{
    return SLR ? decltype(md.alpha_B_const)(1) : md.alpha_B_const;
}


/*! @brief apply the per-particle induction terms on the host
 *
 * Reads the ijLoop's dB/dt (dissipation + cleaning) and du_diss in place and completes them.
 */
template<class MagnetoData, class HydroData>
void inductionPerParticleHost(size_t first, size_t last, HydroData& d, MagnetoData& md)
{
    using Tc = std::decay_t<decltype(*md.Bx.data())>;
    using T  = std::decay_t<decltype(*md.psi_ch.data())>;

    auto* dBx = md.dBx.data();
    auto* dBy = md.dBy.data();
    auto* dBz = md.dBz.data();
    auto* du  = d.du.data();
    const auto mu_0 = T(md.mu_0);

#pragma omp parallel for schedule(static)
    for (size_t i = first; i < last; ++i)
    {
        cstone::Vec3<Tc> dB{dBx[i], dBy[i], dBz[i]};
        T                d_psi_ch = 0;
        T                rhoi     = d.kx[i] * d.m[i] / d.xm[i];

        inductionPerParticle<Tc, T>(dB, d_psi_ch, md.Bx[i], md.By[i], md.Bz[i], md.dvxdx[i], md.dvxdy[i], md.dvxdz[i],
                                    md.dvydx[i], md.dvydy[i], md.dvydz[i], md.dvzdx[i], md.dvzdy[i], md.dvzdz[i],
                                    d.c[i], d.h[i], rhoi, md.psi_ch[i], md.divB_conj[i], mu_0);

        dBx[i]         = dB[0];
        dBy[i]         = dB[1];
        dBz[i]         = dB[2];
        md.d_psi_ch[i] = d_psi_ch;
        du[i] += md.du_diss[i];
    }
}

template<bool SLR, class Tc, class SimulationData>
void computeInductionAndDissipation(const GroupView& grp, SimulationData& sim, const cstone::Box<Tc>& box)
{
    auto& d  = sim.hydro;
    auto& md = sim.magneto;

    if constexpr (cstone::HaveGpu<typename SimulationData::AcceleratorType>{})
    {
        cuda::computeInductionAndDissipationGpu<SLR>(grp, d, md, box);
    }
    else
    {
        inductionAndDissipationIjLoop<SLR>(
            d.neighborhood, d.K, md.mu_0, alphaB<SLR>(md), d.vx.data(), d.vy.data(), d.vz.data(), d.c.data(),
            md.Bx.data(), md.By.data(), md.Bz.data(), d.m.data(), d.xm.data(), d.kx.data(), d.gradh.data(),
            d.c11.data(), d.c12.data(), d.c13.data(), d.c22.data(), d.c23.data(), d.c33.data(), md.psi_ch.data(),
            d.nc.data(), md.dBxdx.data(), md.dBxdy.data(), md.dBxdz.data(), md.dBydx.data(), md.dBydy.data(),
            md.dBydz.data(), md.dBzdx.data(), md.dBzdy.data(), md.dBzdz.data(), d.wh.data(), md.dBx.data(),
            md.dBy.data(), md.dBz.data(), md.dB_diss_x.data(), md.dB_diss_y.data(), md.dB_diss_z.data(),
            md.du_diss.data());

        inductionPerParticleHost(grp.firstBody, grp.lastBody, d, md);
    }
}

} // namespace sph::magneto
