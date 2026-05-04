/*
 * MIT License
 *
 * Copyright (c) 2024 CSCS, ETH Zurich
 *               2024 University of Basel
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
 * @brief IAD and divergence/curl driver, not compacting the velocity jacobian
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 * @author Lukas Schmidt
 */

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/findneighbors.hpp"
#include "cstone/traversal/find_neighbors.cuh"

#include "sph/sph_gpu.hpp"
#include "sph/particles_data.hpp"
#include "sph/magneto_ve/magneto_data.hpp"
#include "iad_div_curl.hpp"
#include "full_divv_curlv_kern.hpp"
#include "sph/hydro_ve/iad_gradh_kern.hpp"
#include "sph/magneto_ve/divB_curlB_kern.hpp"

namespace sph::magneto::cuda
{

using cstone::GpuConfig;
using cstone::LocalIndex;
using cstone::TravConfig;
using cstone::TreeNodeIndex;

template<class Tc, class Tm, class T, class KeyType>
__global__ void
fullIadDivvCurlvGpu(Tc K, unsigned ngmax, const cstone::Box<Tc> box, const LocalIndex* grpStart,
                    const LocalIndex* grpEnd, LocalIndex numGroups, const cstone::OctreeNsView<Tc, KeyType> tree,
                    const Tc* x, const Tc* y, const Tc* z, const T* vx, const T* vy, const T* vz, const T* h,
                    const Tm* m, const T* wh, const T* whd, T* gradh, const T* xm, const T* kx, T* c11, T* c12, T* c13, T* c22,
                    T* c23, T* c33, T* divv, T* curlv, T* dvxdx, T* dvxdy, T* dvxdz, T* dvydx, T* dvydy, T* dvydz,
                    T* dvzdx, T* dvzdy, T* dvzdz, cstone::LocalIndex* nidx, TreeNodeIndex* globalPool)
{
    unsigned laneIdx     = threadIdx.x & (GpuConfig::warpSize - 1);
    unsigned targetIdx   = 0;
    unsigned warpIdxGrid = (blockDim.x * blockIdx.x + threadIdx.x) >> GpuConfig::warpSizeLog2;

    cstone::LocalIndex* neighborsWarp = nidx + ngmax * TravConfig::targetSize * warpIdxGrid;

    while (true)
    {
        // first thread in warp grabs next target
        if (laneIdx == 0) { targetIdx = atomicAdd(&cstone::targetCounterGlob, 1); }
        targetIdx = cstone::shflSync(targetIdx, 0);

        if (targetIdx >= numGroups) return;

        LocalIndex bodyBegin = grpStart[targetIdx];
        LocalIndex bodyEnd   = grpEnd[targetIdx];
        LocalIndex i         = bodyBegin + laneIdx;

        auto ncTrue = traverseNeighbors(bodyBegin, bodyEnd, x, y, z, h, tree, box, neighborsWarp, ngmax, globalPool);

        if (i >= bodyEnd) continue;

        unsigned ncCapped = stl::min(ncTrue[0], ngmax);
        IAD_gradhJLoop<TravConfig::targetSize>(i, K, box, neighborsWarp + laneIdx, ncCapped, x, y, z, h, m, wh, whd, xm, kx, c11,
                                         c12, c13, c22, c23, c33, gradh);
        full_divV_curlVJLoop<TravConfig::targetSize>(
            i, K, box, neighborsWarp + laneIdx, ncCapped, x, y, z, vx, vy, vz, h, c11, c12, c13, c22, c23, c33, wh, whd,
            gradh, kx, xm, divv, curlv, dvxdx, dvxdy, dvxdz, dvydx, dvydy, dvydz, dvzdx, dvzdy, dvzdz);
    }
}

template<class Tc, class T, class KeyType>
__global__ void
divBCurlBGpu(Tc K, unsigned ngmax, const cstone::Box<Tc> box, const LocalIndex* grpStart, const LocalIndex* grpEnd,
             LocalIndex numGroups, const cstone::OctreeNsView<Tc, KeyType> tree, const Tc* x, const Tc* y, const Tc* z,
             const Tc* Bx, const Tc* By, const Tc* Bz, const T* h, const T* wh, const T* gradh, const T* xm,
             const T* kx, const T* c11, const T* c12, const T* c13, const T* c22, const T* c23, const T* c33, T* divB,
             T* curlB_x, T* curlB_y, T* curlB_z, cstone::LocalIndex* nidx, TreeNodeIndex* globalPool)
{

    unsigned laneIdx     = threadIdx.x & (GpuConfig::warpSize - 1);
    unsigned targetIdx   = 0;
    unsigned warpIdxGrid = (blockDim.x * blockIdx.x + threadIdx.x) >> GpuConfig::warpSizeLog2;

    cstone::LocalIndex* neighborsWarp = nidx + ngmax * TravConfig::targetSize * warpIdxGrid;

    while (true)
    {
        // first thread in warp grabs next target
        if (laneIdx == 0) { targetIdx = atomicAdd(&cstone::targetCounterGlob, 1); }
        targetIdx = cstone::shflSync(targetIdx, 0);

        if (targetIdx >= numGroups) return;

        LocalIndex bodyBegin = grpStart[targetIdx];
        LocalIndex bodyEnd   = grpEnd[targetIdx];
        LocalIndex i         = bodyBegin + laneIdx;

        auto ncTrue = traverseNeighbors(bodyBegin, bodyEnd, x, y, z, h, tree, box, neighborsWarp, ngmax, globalPool);

        if (i >= bodyEnd) continue;

        unsigned ncCapped = stl::min(ncTrue[0], ngmax);
        divB_curlB_JLoop<TravConfig::targetSize>(i, K, box, neighborsWarp + laneIdx, ncCapped, x, y, z, Bx, By, Bz, h,
                                                 c11, c12, c13, c22, c23, c33, wh, gradh, kx, xm, divB, curlB_x,
                                                 curlB_y, curlB_z);
    }
}

template<class HydroData, class MagnetoData>
void computeIadFullDivvCurlv(const GroupView& grp, HydroData& d, MagnetoData& m,
                             const cstone::Box<typename HydroData::RealType>& box)
{
 
    auto [traversalPool, nidxPool] = cstone::allocateNcStacks(d.traversalStack, d.ngmax);
    cstone::resetTraversalCounters<<<1, 1>>>();

    auto* d_curlv = (d.x.size() == d.curlv.size()) ? rawPtr(d.curlv) : nullptr;

    fullIadDivvCurlvGpu<<<TravConfig::numBlocks(), TravConfig::numThreads>>>(
        d.K, d.ngmax, box, grp.groupStart, grp.groupEnd, grp.numGroups, d.treeView, rawPtr(d.x),
        rawPtr(d.y), rawPtr(d.z), rawPtr(d.vx), rawPtr(d.vy), rawPtr(d.vz),
        rawPtr(d.h), rawPtr(d.m), rawPtr(d.wh), rawPtr(d.whd), rawPtr(d.gradh), rawPtr(d.xm),
        rawPtr(d.kx), rawPtr(d.c11), rawPtr(d.c12), rawPtr(d.c13),
        rawPtr(d.c22), rawPtr(d.c23), rawPtr(d.c33), rawPtr(d.divv), d_curlv,
        rawPtr(m.dvxdx), rawPtr(m.dvxdy), rawPtr(m.dvxdz), rawPtr(m.dvydx),
        rawPtr(m.dvydy), rawPtr(m.dvydz), rawPtr(m.dvzdx), rawPtr(m.dvzdy),
        rawPtr(m.dvzdz), nidxPool, traversalPool);

    cstone::resetTraversalCounters<<<1, 1>>>();
    divBCurlBGpu<<<TravConfig::numBlocks(), TravConfig::numThreads>>>(
        d.K, d.ngmax, box, grp.groupStart, grp.groupEnd, grp.numGroups, d.treeView, rawPtr(d.x),
        rawPtr(d.y), rawPtr(d.z), rawPtr(m.Bx), rawPtr(m.By), rawPtr(m.Bz),
        rawPtr(d.h), rawPtr(d.wh), rawPtr(d.gradh), rawPtr(d.xm), rawPtr(d.kx),
        rawPtr(d.c11), rawPtr(d.c12), rawPtr(d.c13), rawPtr(d.c22),
        rawPtr(d.c23), rawPtr(d.c33), rawPtr(m.divB), rawPtr(m.curlB_x),
        rawPtr(m.curlB_y), rawPtr(m.curlB_z), nidxPool, traversalPool);

    checkGpuErrors(cudaDeviceSynchronize());
}

template void computeIadFullDivvCurlv(const GroupView& grp, sphexa::ParticlesData<cstone::GpuTag>& d,
                                      sphexa::magneto::MagnetoData<cstone::GpuTag>& m,
                                      const cstone::Box<SphTypes::CoordinateType>&);

} // namespace sph::magneto::cuda
