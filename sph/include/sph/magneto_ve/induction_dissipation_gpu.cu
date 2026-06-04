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

#include "cstone/cuda/cuda_utils.cuh"
#include "cstone/findneighbors.hpp"
#include "cstone/traversal/find_neighbors.cuh"

#include "sph/sph_gpu.hpp"
#include "sph/particles_data.hpp"
#include "sph/magneto_ve/magneto_data.hpp"
#include "induction_dissipation.hpp"
#include "induction_dissipation_kern.hpp"

namespace sph::magneto::cuda
{
using cstone::GpuConfig;
using cstone::LocalIndex;
using cstone::TravConfig;
using cstone::TreeNodeIndex;

template<class Tc, class T, class Tm, class KeyType>
__global__ void inductionDissipationGPU(
    Tc K, unsigned ngmax, const cstone::Box<Tc> box, const LocalIndex* grpStart, const LocalIndex* grpEnd,
    LocalIndex numGroups, const cstone::OctreeNsView<Tc, KeyType> tree, const Tc mu_0, const Tc* x, const Tc* y,
    const Tc* z, const T* vx, const T* vy, const T* vz, const T* c, const Tc* Bx, const Tc* By, const Tc* Bz,
    const T* h, const T* c11, const T* c12, const T* c13, const T* c22, const T* c23, const T* c33, const T* wh,
    const T* xm, const T* kx, const T* gradh, const Tm* m, const T* dvxdx, const T* dvxdy, const T* dvxdz,
    const T* dvydx, const T* dvydy, const T* dvydz, const T* dvzdx, const T* dvzdy, const T* dvzdz, T* psi_ch,
    const T* divB, Tc* dBx, Tc* dBy, Tc* dBz, Tc* du, T* alpha_B, T* d_psi_ch, const T* dBxdx, const T* dBxdy,
    const T* dBxdz, const T* dBydx, const T* dBydy, const T* dBydz, const T* dBzdx, const T* dBzdy, const T* dBzdz,
    ResistivityScheme scheme, LocalIndex* nidx, TreeNodeIndex* globalPool)
{
    unsigned laneIdx     = threadIdx.x & (GpuConfig::warpSize - 1);
    unsigned targetIdx   = 0;
    unsigned warpIdxGrid = (blockDim.x * blockIdx.x + threadIdx.x) >> GpuConfig::warpSizeLog2;

    LocalIndex* neighborsWarp = nidx + ngmax * TravConfig::targetSize * warpIdxGrid;

    while (true)
    {
        // first thread in warp grabs next target
        if (laneIdx == 0) { targetIdx = atomicAdd(&cstone::targetCounterGlob, 1); }
        targetIdx = cstone::shflSync(targetIdx, 0);

        if (targetIdx >= numGroups) return;

        LocalIndex bodyBegin = grpStart[targetIdx];
        LocalIndex bodyEnd   = grpEnd[targetIdx];
        LocalIndex i         = bodyBegin + laneIdx;

        // Induction Equation
        dBx[i] = -Bx[i] * (dvydy[i] + dvzdz[i]) + By[i] * dvxdy[i] + Bz[i] * dvxdz[i];
        dBy[i] = -By[i] * (dvxdx[i] + dvzdz[i]) + Bx[i] * dvydx[i] + Bz[i] * dvydz[i];
        dBz[i] = -Bz[i] * (dvxdx[i] + dvydy[i]) + Bx[i] * dvzdx[i] + By[i] * dvzdy[i];

        auto ncTrue = traverseNeighbors(bodyBegin, bodyEnd, x, y, z, h, tree, box, neighborsWarp, ngmax, globalPool);

        if (i >= bodyEnd) continue;

        unsigned ncCapped = stl::min(ncTrue[0], ngmax);
        inductionAndDissipationJLoop<TravConfig::targetSize>(
            i, K, mu_0, box, neighborsWarp + laneIdx, ncCapped, x, y, z, vx, vy, vz, c, Bx, By, Bz, h, c11, c12, c13,
            c22, c23, c33, wh, xm, kx, gradh, m, psi_ch, &dBx[i], &dBy[i], &dBz[i], &du[i], alpha_B, dBxdx, dBxdy,
            dBxdz, dBydx, dBydy, dBydz, dBzdx, dBzdy, dBzdz, scheme);

        // get psi time differential with the recipe of Wissing et al (2020)
        auto rho_i     = kx[i] * m[i] / xm[i];
        auto v_alfven2 = (Bx[i] * Bx[i] + By[i] * By[i] + Bz[i] * Bz[i]) / (mu_0 * rho_i);
        auto ch        = fclean * std::sqrt(c[i] * c[i] + v_alfven2);
        auto tau_Inv   = (sigma_c * ch) / h[i];
        d_psi_ch[i]    = -ch * divB[i] - psi_ch[i] * (tau_Inv + (dvxdx[i] + dvydy[i] + dvzdz[i]) / 2);
    }
}

template<class HydroData, class MagnetoData>
void computeInductionAndDissipationGpu(const GroupView& grp, HydroData& d, MagnetoData& m,
                                       const cstone::Box<typename HydroData::RealType>& box)
{

    auto [traversalPool, nidxPool] = cstone::allocateNcStacks(d.traversalStack, d.ngmax);
    cstone::resetTraversalCounters<<<1, 1>>>();

    inductionDissipationGPU<<<TravConfig::numBlocks(), TravConfig::numThreads>>>(
        d.K, d.ngmax, box, grp.groupStart, grp.groupEnd, grp.numGroups, d.treeView, m.mu_0, rawPtr(d.x),
        rawPtr(d.y), rawPtr(d.z), rawPtr(d.vx), rawPtr(d.vy), rawPtr(d.vz),
        rawPtr(d.c), rawPtr(m.Bx), rawPtr(m.By), rawPtr(m.Bz), rawPtr(d.h),
        rawPtr(d.c11), rawPtr(d.c12), rawPtr(d.c13), rawPtr(d.c22),
        rawPtr(d.c23), rawPtr(d.c33), rawPtr(d.wh), rawPtr(d.xm), rawPtr(d.kx),
        rawPtr(d.gradh), rawPtr(d.m), rawPtr(m.dvxdx), rawPtr(m.dvxdy),
        rawPtr(m.dvxdz), rawPtr(m.dvydx), rawPtr(m.dvydy), rawPtr(m.dvydz),
        rawPtr(m.dvzdx), rawPtr(m.dvzdy), rawPtr(m.dvzdz), rawPtr(m.psi_ch),
        rawPtr(m.divB), rawPtr(m.dBx), rawPtr(m.dBy), rawPtr(m.dBz),
        rawPtr(d.du), rawPtr(m.alpha_B), rawPtr(m.d_psi_ch), rawPtr(m.dBxdx), rawPtr(m.dBxdy), rawPtr(m.dBxdz),
        rawPtr(m.dBydx), rawPtr(m.dBydy), rawPtr(m.dBydz), rawPtr(m.dBzdx), rawPtr(m.dBzdy), rawPtr(m.dBzdz),
        m.resistivityScheme, nidxPool, traversalPool);
}

template void computeInductionAndDissipationGpu(const GroupView& grp, sphexa::ParticlesData<cstone::GpuTag>& d,
                                                sphexa::magneto::MagnetoData<cstone::GpuTag>& m,
                                                const cstone::Box<SphTypes::CoordinateType>&);
} // namespace sph::magneto::cuda
