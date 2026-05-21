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

/*! @file Initialization of the MHD rotor test (Balsara & Spicer 1999)
 *
 * @author Nicolas Mueller <muellernico@outlook.com>
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

InitSettings MhdRotorConstants()
{
    return {{"rhoDisk", 10.0},
            {"rhoAmbient", 1.0},
            {"P", 1.0},
            {"v0", 2.0},
            {"R", 0.1},
            {"Bx", 5.0 / std::sqrt(4.0 * M_PI)},
            {"L", 0.5},
            {"gamma", 1.4},
            {"mui", 10.},
            {"Kcour", 0.2},
            {"ng0", 150},
            {"ngmax", 200},
            {"minDt", 1e-7},
            {"minDt_m1", 1e-7},
            {"gravConstant", 0.0},
            {"mhd-rotor", 1.0}};
}

template<class T, class SimData>
void initMhdRotorFields(SimData& sim, const std::map<std::string, double>& constants, T massPart)
{
    auto& d  = sim.hydro;
    auto& md = sim.magneto;

    constexpr bool gpu = cstone::HaveGpu<typename SimData::AcceleratorType>{};
    using HT           = typename std::decay_t<decltype(d)>::HydroType;
    using RT           = typename std::decay_t<decltype(md)>::RealType;
    using XM           = typename std::decay_t<decltype(d)>::XM1Type;
    using Tmass        = typename std::decay_t<decltype(d)>::Tmass;

    T rhoDisk    = constants.at("rhoDisk");
    T rhoAmbient = constants.at("rhoAmbient");
    T P          = constants.at("P");
    T v0         = constants.at("v0");
    T R          = constants.at("R");
    T Bx         = constants.at("Bx");
    T gamma      = constants.at("gamma");
    T minDt      = constants.at("minDt");

    T hDisk    = 0.5 * std::cbrt(3. * d.ng0 * massPart / 4. / M_PI / rhoDisk);
    T hAmbient = 0.5 * std::cbrt(3. * d.ng0 * massPart / 4. / M_PI / rhoAmbient);

    auto cv          = sph::idealGasCv(d.muiConst, gamma);
    T    tempDisk    = P / ((gamma - 1.) * rhoDisk) / cv;
    T    tempAmbient = P / ((gamma - 1.) * rhoAmbient) / cv;

    cstone::fill<gpu>(d.m.begin(), d.m.end(), Tmass(massPart));
    cstone::fill<gpu>(d.du_m1.begin(), d.du_m1.end(), XM(0.0));
    cstone::fill<gpu>(d.mue.begin(), d.mue.end(), HT(2.0));
    cstone::fill<gpu>(d.mui.begin(), d.mui.end(), HT(constants.at("mui")));
    cstone::fill<gpu>(d.alpha.begin(), d.alpha.end(), HT(d.alphamax));

    cstone::fill<gpu>(d.vz.begin(), d.vz.end(), HT(0.0));
    cstone::fill<gpu>(d.z_m1.begin(), d.z_m1.end(), XM(0.0));

    cstone::fill<gpu>(md.Bx.begin(), md.Bx.end(), RT(Bx));
    cstone::fill<gpu>(md.By.begin(), md.By.end(), RT(0.0));
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
    std::vector<HT>   h(N), vx(N), vy(N);
    std::vector<T>    temp(N);
    std::vector<XM>   x_m1(N), y_m1(N);

    T Rsq = R * R;

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < N; ++i)
    {
        T r2 = x[i] * x[i] + y[i] * y[i];
        if (r2 <= Rsq)
        {
            // solid-body rotation: |v| grows linearly with radius, v0 at the rim
            vx[i]   = HT(-v0 * y[i] / R);
            vy[i]   = HT(v0 * x[i] / R);
            h[i]    = HT(hDisk);
            temp[i] = tempDisk;
        }
        else
        {
            vx[i]   = HT(0.0);
            vy[i]   = HT(0.0);
            h[i]    = HT(hAmbient);
            temp[i] = tempAmbient;
        }
        x_m1[i] = XM(vx[i] * minDt);
        y_m1[i] = XM(vy[i] * minDt);
    }

    d.h    = std::move(h);
    d.vx   = std::move(vx);
    d.vy   = std::move(vy);
    d.temp = std::move(temp);
    d.x_m1 = std::move(x_m1);
    d.y_m1 = std::move(y_m1);
}

template<class SimData>
class MhdRotorGlass : public ISimInitializer<SimData>
{
protected:
    std::string          glassBlock;
    mutable InitSettings settings_;

public:
    MhdRotorGlass(std::string initBlock, std::string settingsFile, IFileReader* reader)
        : glassBlock(std::move(initBlock))
    {
        SimData d;
        settings_ = buildSettings(d, MhdRotorConstants(), settingsFile, reader);
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

        T L            = settings_.at("L");
        T R            = settings_.at("R");
        T rhoDisk      = settings_.at("rhoDisk");
        T rhoAmbient   = settings_.at("rhoAmbient");
        T densityRatio = rhoDisk / rhoAmbient;

        int multi1D = std::rint(cbrtNumPart / std::cbrt(blockSize));
        T   Lz      = 2 * L / multi1D;

        cstone::Box<T>    globalBox(-L, L, -L, L, 0, Lz, pbc, pbc, pbc);
        cstone::Vec3<int> outerMulti = {multi1D, multi1D, 1};

        auto [keyStart, keyEnd] = equiDistantSfcSegments<KeyType>(rank, numRanks, 100);

        std::vector<T> x, y, z;
        assembleCuboid<T>(keyStart, keyEnd, globalBox, outerMulti, xBlock, yBlock, zBlock, x, y, z);

        if (densityRatio != T(1))
        {
            // carve the rotor disk out of the ambient assembly and refill it with a denser blob.
            // xy-only scaling preserves z spacing; the inner glass becomes anisotropic by
            // sqrt(densityRatio) -- acceptable for a z-invariant test.
            auto outsideDisk = [R](auto u, auto v, auto) { return u * u + v * v >= R * R; };
            selectParticles(x, y, z, outsideDisk);

            T halfA = L / std::sqrt(densityRatio);
            if (halfA < R)
            {
                throw std::runtime_error("mhd-rotor: density ratio too large for current R (max ratio = " +
                                         std::to_string((L / R) * (L / R)) + ")\n");
            }

            cstone::Box<T>    innerBox(-halfA, halfA, -halfA, halfA, 0, Lz, pbc, pbc, pbc);
            cstone::Vec3<int> innerMulti = {multi1D, multi1D, 1};

            std::vector<T> xIn, yIn, zIn;
            assembleCuboid<T>(keyStart, keyEnd, innerBox, innerMulti, xBlock, yBlock, zBlock, xIn, yIn, zIn);

            auto keepDisk = [R](auto u, auto v, auto) { return u * u + v * v < R * R; };
            selectParticles(xIn, yIn, zIn, keepDisk);

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
        T      particleMass      = outerVolume * rhoAmbient / numOuterAssembled;

        initMhdRotorFields(simData, settings_, particleMass);

        return globalBox;
    }

    [[nodiscard]] const InitSettings& constants() const override { return settings_; }
};
} // namespace sphexa
