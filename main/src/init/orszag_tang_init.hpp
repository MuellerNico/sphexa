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

/*! @file Initialization of the Orszag-Tang vortex test
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

InitSettings OrszagTangConstants()
{
    return {{"rho", 25. / (36. * M_PI)},
            {"P", 5. / (12. * M_PI)},
            {"B0", 1. / std::sqrt(4. * M_PI)},
            {"v0", 1.},
            {"L", 1.},
            {"gamma", 5. / 3.},
            {"mui", 10.},
            {"Kcour", 0.2},
            {"ng0", 150},
            {"ngmax", 200},
            {"minDt", 1e-7},
            {"minDt_m1", 1e-7},
            {"gravConstant", 0.0},
            {"orszag-tang", 1.0}};
}

template<class T, class SimData>
void initOrszagTangFields(SimData& sim, const std::map<std::string, double>& constants, T massPart)
{
    auto& d  = sim.hydro;
    auto& md = sim.magneto;

    constexpr bool gpu = cstone::HaveGpu<typename SimData::AcceleratorType>{};
    using HT           = typename std::decay_t<decltype(d)>::HydroType;
    using RT           = typename std::decay_t<decltype(md)>::RealType;
    using XM           = typename std::decay_t<decltype(d)>::XM1Type;
    using Tmass        = typename std::decay_t<decltype(d)>::Tmass;

    T rho   = constants.at("rho");
    T p     = constants.at("P");
    T B0    = constants.at("B0");
    T v0    = constants.at("v0");
    T L     = constants.at("L");
    T gamma = constants.at("gamma");

    T h = 0.5 * std::cbrt(3. * d.ng0 * massPart / 4. / M_PI / rho);

    auto cv   = sph::idealGasCv(d.muiConst, gamma);
    T    temp = p / ((gamma - 1.) * rho) / cv;

    cstone::fill<gpu>(d.m.begin(), d.m.end(), Tmass(massPart));
    cstone::fill<gpu>(d.du_m1.begin(), d.du_m1.end(), XM(0.0));
    cstone::fill<gpu>(d.mue.begin(), d.mue.end(), HT(2.0));
    cstone::fill<gpu>(d.mui.begin(), d.mui.end(), HT(constants.at("mui")));
    cstone::fill<gpu>(d.alpha.begin(), d.alpha.end(), HT(d.alphamin));
    cstone::fill<gpu>(d.temp.begin(), d.temp.end(), T(temp));
    cstone::fill<gpu>(d.h.begin(), d.h.end(), HT(h));

    cstone::fill<gpu>(md.dBx.begin(), md.dBx.end(), RT(0.0));
    cstone::fill<gpu>(md.dBy.begin(), md.dBy.end(), RT(0.0));
    cstone::fill<gpu>(md.dBz.begin(), md.dBz.end(), RT(0.0));
    cstone::fill<gpu>(md.dBx_m1.begin(), md.dBx_m1.end(), XM(0.0));
    cstone::fill<gpu>(md.dBy_m1.begin(), md.dBy_m1.end(), XM(0.0));
    cstone::fill<gpu>(md.dBz_m1.begin(), md.dBz_m1.end(), XM(0.0));

    cstone::fill<gpu>(md.psi_ch.begin(), md.psi_ch.end(), HT(0.0));
    cstone::fill<gpu>(md.d_psi_ch.begin(), md.d_psi_ch.end(), HT(0.0));
    cstone::fill<gpu>(md.d_psi_ch_m1.begin(), md.d_psi_ch_m1.end(), XM(0.0));

    T k = 2. * M_PI / L;

    auto&& x = toHost(d.x);
    auto&& y = toHost(d.y);

    const std::size_t N = d.x.size();
    std::vector<HT>   vx(N), vy(N), vz(N);
    std::vector<RT>   Bx(N), By(N), Bz(N);
    std::vector<XM>   x_m1(N), y_m1(N), z_m1(N);

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < N; ++i)
    {
        vx[i] = -v0 * std::sin(k * y[i]);
        vy[i] = v0 * std::sin(k * x[i]);
        vz[i] = HT(0.0);

        Bx[i] = -B0 * std::sin(k * y[i]);
        By[i] = B0 * std::sin(2. * k * x[i]);
        Bz[i] = RT(0.0);

        x_m1[i] = vx[i] * d.minDt;
        y_m1[i] = vy[i] * d.minDt;
        z_m1[i] = vz[i] * d.minDt;
    }

    d.vx   = std::move(vx);
    d.vy   = std::move(vy);
    d.vz   = std::move(vz);
    md.Bx  = std::move(Bx);
    md.By  = std::move(By);
    md.Bz  = std::move(Bz);
    d.x_m1 = std::move(x_m1);
    d.y_m1 = std::move(y_m1);
    d.z_m1 = std::move(z_m1);
}

template<class SimData>
class OrszagTangGlass : public ISimInitializer<SimData>
{
protected:
    std::string          glassBlock;
    mutable InitSettings settings_;

public:
    OrszagTangGlass(std::string initBlock, std::string settingsFile, IFileReader* reader)
        : glassBlock(std::move(initBlock))
    {
        SimData d;
        settings_ = buildSettings(d, OrszagTangConstants(), settingsFile, reader);
    }

    cstone::Box<typename SimData::RealType> init(int rank, int numRanks, size_t cbrtNumPart, SimData& simData,
                                                 IFileReader* reader) const override
    {
        auto& d       = simData.hydro;
        auto& md      = simData.magneto;
        using KeyType = typename SimData::KeyType;
        using T       = typename SimData::RealType;

        std::vector<T> xBlock, yBlock, zBlock;
        readTemplateBlock(glassBlock, reader, xBlock, yBlock, zBlock);
        size_t blockSize = xBlock.size();

        int multi1D = std::rint(cbrtNumPart / std::cbrt(blockSize));

        // Orszag-Tang is z-invariant: tile a single glass block in z so the slab thickness
        // Lz = L / multi1D keeps the inter-particle spacing cubic.
        cstone::Vec3<int> multiplicity       = {multi1D, multi1D, 1};
        size_t            numParticlesGlobal = size_t(multi1D) * multi1D * blockSize;

        T    L   = settings_.at("L");
        T    Lz  = L / multi1D;
        auto pbc = cstone::BoundaryType::periodic;
        cstone::Box<T> globalBox(0, L, 0, L, 0, Lz, pbc, pbc, pbc);

        auto [keyStart, keyEnd] = equiDistantSfcSegments<KeyType>(rank, numRanks, 100);
        std::vector<T> x, y, z;
        assembleCuboid<T>(keyStart, keyEnd, globalBox, multiplicity, xBlock, yBlock, zBlock, x, y, z);
        d.x = x; // uploads to GPU if active
        d.y = y;
        d.z = z;
        d.resize(d.x.size());
        md.resize(d.x.size());

        settings_["numParticlesGlobal"] = double(numParticlesGlobal);
        BuiltinWriter attributeSetter(settings_);
        d.loadOrStoreAttributes(&attributeSetter);

        T volume       = globalBox.lx() * globalBox.ly() * globalBox.lz();
        T particleMass = volume * settings_.at("rho") / numParticlesGlobal;

        initOrszagTangFields(simData, settings_, particleMass);

        return globalBox;
    }

    [[nodiscard]] const InitSettings& constants() const override { return settings_; }
};
} // namespace sphexa
