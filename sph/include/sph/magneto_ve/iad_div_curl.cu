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

#include "sph/neighborhood_gpu.hpp"
#include "sph/sph_gpu.hpp"
#include "sph/particles_data.hpp"
#include "sph/magneto_ve/magneto_data.hpp"
#include "iad_div_curl.hpp"

namespace sph::magneto::cuda
{

template<bool SLR, class HydroData, class MagnetoData>
void computeIadFullDivvCurlv(const GroupView& grp, HydroData& d, MagnetoData& m,
                             const cstone::Box<typename HydroData::RealType>&)
{
    auto* curlv = (d.x.size() == d.curlv.size()) ? rawPtr(d.curlv) : nullptr;

    iadGradhIjLoop(d.neighborhood, d.K, rawPtr(d.m), rawPtr(d.xm), rawPtr(d.kx), rawPtr(d.nc), rawPtr(d.wh),
                   rawPtr(d.whd), rawPtr(d.c11), rawPtr(d.c12), rawPtr(d.c13), rawPtr(d.c22), rawPtr(d.c23),
                   rawPtr(d.c33), rawPtr(d.gradh));

    fullDivvCurlvIjLoop(d.neighborhood, d.K, rawPtr(d.vx), rawPtr(d.vy), rawPtr(d.vz), rawPtr(d.kx), rawPtr(d.xm),
                        rawPtr(d.c11), rawPtr(d.c12), rawPtr(d.c13), rawPtr(d.c22), rawPtr(d.c23), rawPtr(d.c33),
                        rawPtr(d.gradh), rawPtr(d.wh), rawPtr(d.divv), curlv, rawPtr(m.dvxdx), rawPtr(m.dvxdy),
                        rawPtr(m.dvxdz), rawPtr(m.dvydx), rawPtr(m.dvydy), rawPtr(m.dvydz), rawPtr(m.dvzdx),
                        rawPtr(m.dvzdy), rawPtr(m.dvzdz));

    divBCurlBIjLoop<SLR>(d.neighborhood, d.K, rawPtr(m.Bx), rawPtr(m.By), rawPtr(m.Bz), rawPtr(d.kx), rawPtr(d.xm),
                    rawPtr(d.c11), rawPtr(d.c12), rawPtr(d.c13), rawPtr(d.c22), rawPtr(d.c23), rawPtr(d.c33),
                    rawPtr(d.gradh), rawPtr(d.wh), rawPtr(m.divB), rawPtr(m.divB_conj), rawPtr(m.curlB_x),
                    rawPtr(m.curlB_y), rawPtr(m.curlB_z), rawPtr(m.gradB_norm), rawPtr(m.dBxdx), rawPtr(m.dBxdy),
                    rawPtr(m.dBxdz), rawPtr(m.dBydx), rawPtr(m.dBydy), rawPtr(m.dBydz), rawPtr(m.dBzdx),
                    rawPtr(m.dBzdy), rawPtr(m.dBzdz));

    checkGpuErrors(cudaDeviceSynchronize());
}

#define IAD_FULL_DIVV_CURLV(slr)                                                                                      \
    template void computeIadFullDivvCurlv<slr>(const GroupView& grp, sphexa::ParticlesData<cstone::GpuTag>& d,        \
                                               sphexa::magneto::MagnetoData<cstone::GpuTag>& m,                       \
                                               const cstone::Box<SphTypes::CoordinateType>&)

IAD_FULL_DIVV_CURLV(true);
IAD_FULL_DIVV_CURLV(false);

} // namespace sph::magneto::cuda
