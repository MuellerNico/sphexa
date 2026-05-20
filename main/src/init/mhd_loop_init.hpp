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

/*! @file Initialization of the MHD field-loop advection test (Gardiner & Stone 2005)
 */

#pragma once
#include "isim_init.hpp"
#include "cstone/sfc/box.hpp"
#include "sph/eos.hpp"
#include "early_sync.hpp"
#include "grid.hpp"
#include "utils.hpp"

namespace sphexa
{

InitSettings MhdLoopConstants()
{
    return {{"rhoIn", 1.0},
            {"rhoOut", 1.0},
            {"P", 1.0},
            {"A0", 1.0e-3},
            {"R0", 0.3},
            {"vx", 2.0},
            {"vy", 1.0},
            // Wissing script: 0.1*sqrt(5) ≈ 0.224. Paper text: 0.1/sqrt(5) ≈ 0.045. Canonical Gardiner-Stone: 0.
            {"vz", 0.1 * std::sqrt(5.0)},
            {"L", 1.0},
            {"gamma", 5.0 / 3.0},
            {"mui", 10.},
            {"Kcour", 0.2},
            {"ng0", 150},
            {"ngmax", 200},
            {"minDt", 1e-7},
            {"minDt_m1", 1e-7},
            {"gravConstant", 0.0},
            {"mhd-loop", 1.0}};
}

template<class T, class SimData>
void initMhdLoopFields(SimData& sim, const std::map<std::string, double>& constants, T massPart)
{
    auto& d  = sim.hydro;
    auto& md = sim.magneto;

    constexpr bool gpu = cstone::HaveGpu<typename SimData::AcceleratorType>{};
    using HT           = typename std::decay_t<decltype(d)>::HydroType;
    using RT           = typename std::decay_t<decltype(md)>::RealType;
    using XM           = typename std::decay_t<decltype(d)>::XM1Type;
    using Tmass        = typename std::decay_t<decltype(d)>::Tmass;

    T rhoIn  = constants.at("rhoIn");
    T rhoOut = constants.at("rhoOut");
    T P      = constants.at("P");
    T A0     = constants.at("A0");
    T R0     = constants.at("R0");
    T vx     = constants.at("vx");
    T vy     = constants.at("vy");
    T vz     = constants.at("vz");
    T gamma  = constants.at("gamma");
    T minDt  = constants.at("minDt");

    T hIn  = 0.5 * std::cbrt(3. * d.ng0 * massPart / 4. / M_PI / rhoIn);
    T hOut = 0.5 * std::cbrt(3. * d.ng0 * massPart / 4. / M_PI / rhoOut);

    auto cv      = sph::idealGasCv(d.muiConst, gamma);
    T    tempIn  = P / ((gamma - 1.) * rhoIn) / cv;
    T    tempOut = P / ((gamma - 1.) * rhoOut) / cv;

    cstone::fill<gpu>(d.m.begin(), d.m.end(), Tmass(massPart));
    cstone::fill<gpu>(d.du_m1.begin(), d.du_m1.end(), XM(0.0));
    cstone::fill<gpu>(d.mue.begin(), d.mue.end(), HT(2.0));
    cstone::fill<gpu>(d.mui.begin(), d.mui.end(), HT(constants.at("mui")));
    cstone::fill<gpu>(d.alpha.begin(), d.alpha.end(), HT(d.alphamin));

    cstone::fill<gpu>(d.vx.begin(), d.vx.end(), HT(vx));
    cstone::fill<gpu>(d.vy.begin(), d.vy.end(), HT(vy));
    cstone::fill<gpu>(d.vz.begin(), d.vz.end(), HT(vz));
    cstone::fill<gpu>(d.x_m1.begin(), d.x_m1.end(), XM(vx * minDt));
    cstone::fill<gpu>(d.y_m1.begin(), d.y_m1.end(), XM(vy * minDt));
    cstone::fill<gpu>(d.z_m1.begin(), d.z_m1.end(), XM(vz * minDt));

    cstone::fill<gpu>(md.Bz.begin(), md.Bz.end(), RT(0.0));
    cstone::fill<gpu>(md.dBx.begin(), md.dBx.end(), RT(0.0));
    cstone::fill<gpu>(md.dBy.begin(), md.dBy.end(), RT(0.0));
    cstone::fill<gpu>(md.dBz.begin(), md.dBz.end(), RT(0.0));
    cstone::fill<gpu>(md.dBx_m1.begin(), md.dBx_m1.end(), XM(0.0));
    cstone::fill<gpu>(md.dBy_m1.begin(), md.dBy_m1.end(), XM(0.0));
    cstone::fill<gpu>(md.dBz_m1.begin(), md.dBz_m1.end(), XM(0.0));
    cstone::fill<gpu>(md.psi_ch.begin(), md.psi_ch.end(), HT(0.0));
    cstone::fill<gpu>(md.d_psi_ch.begin(), md.d_psi_ch.end(), HT(0.0));
    cstone::fill<gpu>(md.d_psi_ch_m1.begin(), md.d_psi_ch_m1.end(), XM(0.0));

    auto&& x = toHost(d.x);
    auto&& y = toHost(d.y);

    const std::size_t N = d.x.size();
    std::vector<HT>   h(N);
    std::vector<RT>   Bx(N), By(N);
    std::vector<T>    temp(N);

    T R0sq = R0 * R0;

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < N; ++i)
    {
        T r2 = x[i] * x[i] + y[i] * y[i];
        if (r2 < R0sq)
        {
            if (r2 > T(0))
            {
                T r   = std::sqrt(r2);
                Bx[i] = RT(-A0 * y[i] / r);
                By[i] = RT(A0 * x[i] / r);
            }
            else
            {
                Bx[i] = RT(0);
                By[i] = RT(0);
            }
            h[i]    = HT(hIn);
            temp[i] = tempIn;
        }
        else
        {
            Bx[i]   = RT(0);
            By[i]   = RT(0);
            h[i]    = HT(hOut);
            temp[i] = tempOut;
        }
    }

    d.h    = std::move(h);
    md.Bx  = std::move(Bx);
    md.By  = std::move(By);
    d.temp = std::move(temp);
}

template<class SimData>
class MhdLoopGlass : public ISimInitializer<SimData>
{
protected:
    std::string          glassBlock;
    mutable InitSettings settings_;

public:
    MhdLoopGlass(std::string initBlock, std::string settingsFile, IFileReader* reader)
        : glassBlock(std::move(initBlock))
    {
        SimData d;
        settings_ = buildSettings(d, MhdLoopConstants(), settingsFile, reader);
    }

    cstone::Box<typename SimData::RealType> init(int rank, int numRanks, size_t cbrtNumPart, SimData& simData,
                                                 IFileReader* reader) const override
    {
        auto& d       = simData.hydro;
        auto& md      = simData.magneto;
        using KeyType = typename SimData::KeyType;
        using T       = typename SimData::RealType;
        auto pbc      = cstone::BoundaryType::periodic;

        std::vector<T> xBlock, yBlock, zBlock;
        readTemplateBlock(glassBlock, reader, xBlock, yBlock, zBlock);
        size_t blockSize = xBlock.size();

        T L      = settings_.at("L");
        T R0     = settings_.at("R0");
        T rhoIn  = settings_.at("rhoIn");
        T rhoOut = settings_.at("rhoOut");
        T densityRatio = rhoIn / rhoOut;

        int multi1D = std::rint(cbrtNumPart / std::cbrt(blockSize));
        T   Lz      = L / multi1D;

        cstone::Box<T>    globalBox(-L, L, -L / 2, L / 2, 0, Lz, pbc, pbc, pbc);
        cstone::Vec3<int> outerMulti = {2 * multi1D, multi1D, 1};

        auto [keyStart, keyEnd] = equiDistantSfcSegments<KeyType>(rank, numRanks, 100);

        std::vector<T> x, y, z;
        assembleCuboid<T>(keyStart, keyEnd, globalBox, outerMulti, xBlock, yBlock, zBlock, x, y, z);

        if (densityRatio != T(1))
        {
            // density-jump variant: carve a cylinder out of the outer assembly and refill it with
            // a denser/sparser blob. xy-only scaling preserves z spacing, so the inner glass becomes
            // anisotropic by sqrt(densityRatio) — acceptable for a z-invariant test.
            auto outsideCyl = [R0](auto u, auto v, auto) { return u * u + v * v >= R0 * R0; };
            selectParticles(x, y, z, outsideCyl);

            T halfA = T(0.5) / std::sqrt(densityRatio);
            if (halfA < R0)
            {
                throw std::runtime_error("mhd-loop: density ratio too large for current R0 (max ratio = " +
                                         std::to_string(1.0 / (4.0 * R0 * R0)) + ")\n");
            }

            cstone::Box<T>    innerBox(-halfA, halfA, -halfA, halfA, 0, Lz, pbc, pbc, pbc);
            cstone::Vec3<int> innerMulti = {multi1D, multi1D, 1};

            std::vector<T> xIn, yIn, zIn;
            assembleCuboid<T>(keyStart, keyEnd, innerBox, innerMulti, xBlock, yBlock, zBlock, xIn, yIn, zIn);

            auto keepCyl = [R0](auto u, auto v, auto) { return u * u + v * v < R0 * R0; };
            selectParticles(xIn, yIn, zIn, keepCyl);

            std::copy(xIn.begin(), xIn.end(), std::back_inserter(x));
            std::copy(yIn.begin(), yIn.end(), std::back_inserter(y));
            std::copy(zIn.begin(), zIn.end(), std::back_inserter(z));
        }

        d.x = x;
        d.y = y;
        d.z = z;

        size_t numParticlesGlobal = d.x.size();
        MPI_Allreduce(MPI_IN_PLACE, &numParticlesGlobal, 1, MpiType<size_t>{}, MPI_SUM, simData.comm);
        syncCoords<KeyType>(rank, numRanks, numParticlesGlobal, d.x, d.y, d.z, globalBox);

        d.resize(d.x.size());
        md.resize(d.x.size());

        settings_["numParticlesGlobal"] = double(numParticlesGlobal);
        BuiltinWriter attributeSetter(settings_);
        d.loadOrStoreAttributes(&attributeSetter);

        size_t numOuterAssembled = size_t(outerMulti[0]) * outerMulti[1] * outerMulti[2] * blockSize;
        T      outerVolume       = globalBox.lx() * globalBox.ly() * globalBox.lz();
        T      particleMass      = outerVolume * rhoOut / numOuterAssembled;

        initMhdLoopFields(simData, settings_, particleMass);

        return globalBox;
    }

    [[nodiscard]] const InitSettings& constants() const override { return settings_; }
};
} // namespace sphexa
