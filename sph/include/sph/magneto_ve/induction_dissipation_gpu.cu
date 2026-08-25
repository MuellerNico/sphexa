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

/*! @file calculates dB/dt with the induction equation, as well as dissipation and correction terms
 *
 * @author Lukas Schmidt
 */

#include "cstone/cuda/cuda_utils.cuh"

#include "sph/neighborhood_gpu.hpp"
#include "sph/sph_gpu.hpp"
#include "sph/particles_data.hpp"
#include "sph/magneto_ve/magneto_data.hpp"
#include "induction_dissipation.hpp"
#include "induction_dissipation_kern.hpp"

namespace sph::magneto::cuda
{

template<bool SLR, class HydroData, class MagnetoData>
void computeInductionAndDissipationGpu(const GroupView& grp, HydroData& d, MagnetoData& m,
                                       const cstone::Box<typename HydroData::RealType>&)
{
    inductionAndDissipationIjLoop<SLR>(
        d.neighborhood, d.K, m.mu_0, alphaB<SLR>(m), rawPtr(d.vx), rawPtr(d.vy), rawPtr(d.vz), rawPtr(d.c),
        rawPtr(m.Bx), rawPtr(m.By), rawPtr(m.Bz), rawPtr(d.m), rawPtr(d.xm), rawPtr(d.kx), rawPtr(d.gradh),
        rawPtr(d.c11), rawPtr(d.c12), rawPtr(d.c13), rawPtr(d.c22), rawPtr(d.c23), rawPtr(d.c33),
        rawPtr(m.psi_ch), rawPtr(d.nc), rawPtr(m.dBxdx), rawPtr(m.dBxdy), rawPtr(m.dBxdz), rawPtr(m.dBydx),
        rawPtr(m.dBydy), rawPtr(m.dBydz), rawPtr(m.dBzdx), rawPtr(m.dBzdy), rawPtr(m.dBzdz), rawPtr(m.dvxdx),
        rawPtr(m.dvxdy), rawPtr(m.dvxdz), rawPtr(m.dvydx), rawPtr(m.dvydy), rawPtr(m.dvydz), rawPtr(m.dvzdx),
        rawPtr(m.dvzdy), rawPtr(m.dvzdz), rawPtr(m.divB_conj), rawPtr(d.wh), rawPtr(m.dBx), rawPtr(m.dBy),
        rawPtr(m.dBz), rawPtr(d.du), rawPtr(m.d_psi_ch), rawPtr(m.dB_diss_x), rawPtr(m.dB_diss_y),
        rawPtr(m.dB_diss_z), rawPtr(m.du_diss));

    checkGpuErrors(cudaDeviceSynchronize());
}

#define INDUCTION_DISSIPATION(slr)                                                                                    \
    template void computeInductionAndDissipationGpu<slr>(const GroupView& grp,                                        \
                                                         sphexa::ParticlesData<cstone::GpuTag>& d,                    \
                                                         sphexa::magneto::MagnetoData<cstone::GpuTag>& m,             \
                                                         const cstone::Box<SphTypes::CoordinateType>&)

INDUCTION_DISSIPATION(true);
INDUCTION_DISSIPATION(false);

} // namespace sph::magneto::cuda
