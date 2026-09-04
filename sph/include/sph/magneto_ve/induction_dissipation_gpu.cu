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

using cstone::GpuConfig;
using cstone::LocalIndex;

template<class Tc, class T, class Tm, class Tdu>
__global__ void inductionPerParticleKernel(GroupView grp, T mu_0, Tc* dBx, Tc* dBy, Tc* dBz, Tdu* du, T* d_psi_ch,
                                           const Tc* Bx, const Tc* By, const Tc* Bz, const T* dvxdx, const T* dvxdy,
                                           const T* dvxdz, const T* dvydx, const T* dvydy, const T* dvydz,
                                           const T* dvzdx, const T* dvzdy, const T* dvzdz, const T* c, const T* h,
                                           const T* kx, const Tm* m, const T* xm, const T* psi_ch, const T* divB_conj,
                                           const Tc* du_diss)
{
    LocalIndex laneIdx = threadIdx.x & (GpuConfig::warpSize - 1);
    LocalIndex warpIdx = (blockDim.x * blockIdx.x + threadIdx.x) >> GpuConfig::warpSizeLog2;
    if (warpIdx >= grp.numGroups) { return; }

    LocalIndex i = grp.groupStart[warpIdx] + laneIdx;
    if (i >= grp.groupEnd[warpIdx]) { return; }

    cstone::Vec3<Tc> dB{dBx[i], dBy[i], dBz[i]};
    T                psiDot = 0;
    T                rhoi   = kx[i] * m[i] / xm[i];

    inductionPerParticle<Tc, T>(dB, psiDot, Bx[i], By[i], Bz[i], dvxdx[i], dvxdy[i], dvxdz[i], dvydx[i], dvydy[i],
                                dvydz[i], dvzdx[i], dvzdy[i], dvzdz[i], c[i], h[i], rhoi, psi_ch[i], divB_conj[i],
                                mu_0);

    dBx[i]      = dB[0];
    dBy[i]      = dB[1];
    dBz[i]      = dB[2];
    d_psi_ch[i] = psiDot;
    du[i] += du_diss[i];
}

template<bool SLR, class HydroData, class MagnetoData>
void computeInductionAndDissipationGpu(const GroupView& grp, HydroData& d, MagnetoData& m,
                                       const cstone::Box<typename HydroData::RealType>&)
{
    inductionAndDissipationIjLoop<SLR>(
        d.neighborhood, d.K, m.mu_0, alphaB<SLR>(m), rawPtr(d.vx), rawPtr(d.vy), rawPtr(d.vz), rawPtr(d.c),
        rawPtr(m.Bx), rawPtr(m.By), rawPtr(m.Bz), rawPtr(d.m), rawPtr(d.xm), rawPtr(d.kx), rawPtr(d.gradh),
        rawPtr(d.c11), rawPtr(d.c12), rawPtr(d.c13), rawPtr(d.c22), rawPtr(d.c23), rawPtr(d.c33), rawPtr(m.psi_ch),
        rawPtr(d.nc), rawPtr(m.dBxdx), rawPtr(m.dBxdy), rawPtr(m.dBxdz), rawPtr(m.dBydx), rawPtr(m.dBydy),
        rawPtr(m.dBydz), rawPtr(m.dBzdx), rawPtr(m.dBzdy), rawPtr(m.dBzdz), rawPtr(d.wh), rawPtr(m.dBx),
        rawPtr(m.dBy), rawPtr(m.dBz), rawPtr(m.dB_diss_x), rawPtr(m.dB_diss_y), rawPtr(m.dB_diss_z),
        rawPtr(m.du_diss));

    unsigned numThreads       = 256;
    unsigned numWarpsPerBlock = numThreads / GpuConfig::warpSize;
    unsigned numBlocks        = (grp.numGroups + numWarpsPerBlock - 1) / numWarpsPerBlock;

    if (numBlocks > 0)
    {
        inductionPerParticleKernel<<<numBlocks, numThreads>>>(
            grp, typename MagnetoData::HydroType(m.mu_0), rawPtr(m.dBx), rawPtr(m.dBy), rawPtr(m.dBz), rawPtr(d.du),
            rawPtr(m.d_psi_ch), rawPtr(m.Bx), rawPtr(m.By), rawPtr(m.Bz), rawPtr(m.dvxdx), rawPtr(m.dvxdy),
            rawPtr(m.dvxdz), rawPtr(m.dvydx), rawPtr(m.dvydy), rawPtr(m.dvydz), rawPtr(m.dvzdx), rawPtr(m.dvzdy),
            rawPtr(m.dvzdz), rawPtr(d.c), rawPtr(d.h), rawPtr(d.kx), rawPtr(d.m), rawPtr(d.xm), rawPtr(m.psi_ch),
            rawPtr(m.divB_conj), rawPtr(m.du_diss));
    }

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
