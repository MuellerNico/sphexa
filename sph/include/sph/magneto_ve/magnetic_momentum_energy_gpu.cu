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

/*! @file calculates the momentum with the magnetic stress tensor
 *
 * @author Lukas Schmidt
 *
 */

#include <limits>

#include "cstone/primitives/warpscan.cuh"

#include "sph/neighborhood_gpu.hpp"
#include "sph/sph_gpu.hpp"
#include "sph/particles_data.hpp"
#include "sph/magneto_ve/magneto_data.hpp"
#include "sph/magneto_ve/magnetic_momentum_energy_kern.hpp"

namespace sph
{
namespace magneto::cuda
{

using cstone::GpuConfig;
using cstone::LocalIndex;

static __device__ float minDt_ve_device;

template<class T>
__global__ void reduceDt(const LocalIndex* __restrict__ grpStart, const LocalIndex* __restrict__ grpEnd,
                         const LocalIndex numGroups, const T* dtCourant, float* __restrict__ groupDt)
{
    unsigned laneIdx = threadIdx.x & (GpuConfig::warpSize - 1);
    unsigned grpIdx  = (blockDim.x * blockIdx.x + threadIdx.x) >> GpuConfig::warpSizeLog2;

    if (grpIdx >= numGroups) return;

    LocalIndex bodyBegin = grpStart[grpIdx];
    LocalIndex bodyEnd   = grpEnd[grpIdx];
    LocalIndex i         = bodyBegin + laneIdx;

    __shared__ float minBlockDt;
    if (threadIdx.x == 0) minBlockDt = std::numeric_limits<float>::infinity();
    __syncthreads();

    float dt         = i < bodyEnd ? dtCourant[i] : std::numeric_limits<T>::infinity();
    float minGroupDt = cstone::warpMin(dt);
    if (groupDt && laneIdx == 0) groupDt[grpIdx] = std::min(groupDt[grpIdx], minGroupDt);

    if (laneIdx == 0) cstone::atomicMinFloat(&minBlockDt, minGroupDt);
    __syncthreads();
    if (threadIdx.x == 0) cstone::atomicMinFloat(&minDt_ve_device, minBlockDt);
}

template<bool SLR, class HydroData, class MagnetoData>
void computeMagneticMomentumEnergy(const GroupView& grp, float* groupDt, HydroData& d, MagnetoData& m,
                                   const cstone::Box<typename HydroData::RealType>&)
{
    magneticMomentumAndEnergyIjLoop<SLR>(
        d.neighborhood, d.K, d.Kcour, m.mu_0, m.alpha_u, d.Atmin, d.Atmax, d.ramp, rawPtr(d.vx), rawPtr(d.vy),
        rawPtr(d.vz), rawPtr(d.m), rawPtr(d.c), rawPtr(d.u), rawPtr(d.kx), rawPtr(d.alpha), rawPtr(d.xm), rawPtr(d.p),
        rawPtr(d.gradh), rawPtr(d.c11), rawPtr(d.c12), rawPtr(d.c13), rawPtr(d.c22), rawPtr(d.c23), rawPtr(d.c33),
        rawPtr(d.nc), rawPtr(m.Bx), rawPtr(m.By), rawPtr(m.Bz), rawPtr(m.dvxdx), rawPtr(m.dvxdy), rawPtr(m.dvxdz),
        rawPtr(m.dvydx), rawPtr(m.dvydy), rawPtr(m.dvydz), rawPtr(m.dvzdx), rawPtr(m.dvzdy), rawPtr(m.dvzdz),
        rawPtr(d.tdpdTrho), rawPtr(d.wh), d.avFloor, rawPtr(d.du), rawPtr(d.ax), rawPtr(d.ay), rawPtr(d.az),
        rawPtr(d.divv), rawPtr(d.curlv), rawPtr(d.dtCourant));

    float minDt = std::numeric_limits<float>::infinity();
    checkGpuErrors(cudaMemcpyToSymbolAsync(GPU_SYMBOL(minDt_ve_device), &minDt, sizeof(minDt)));

    constexpr LocalIndex threads = 256;
    const LocalIndex     blocks  = cstone::iceil(grp.numGroups, threads / GpuConfig::warpSize);
    reduceDt<<<blocks, threads>>>(grp.groupStart, grp.groupEnd, grp.numGroups, rawPtr(d.dtCourant), groupDt);

    checkGpuErrors(cudaMemcpyFromSymbol(&minDt, GPU_SYMBOL(minDt_ve_device), sizeof(minDt)));
    d.minDtCourant = minDt;
}

#define MAG_MOM_ENERGY(avc)                                                                                             \
    template void computeMagneticMomentumEnergy<avc>(                                                                  \
        const GroupView& grp, float*, sphexa::ParticlesData<cstone::GpuTag>& d,                                        \
        sphexa::magneto::MagnetoData<cstone::GpuTag>& m, const cstone::Box<SphTypes::CoordinateType>&)

MAG_MOM_ENERGY(true);
MAG_MOM_ENERGY(false);

} // namespace magneto::cuda
} // namespace sph
