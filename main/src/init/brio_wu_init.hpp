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

/*! @file Initialization of the Brio-Wu MHD shock tube test
 *  @author Nicolas Müller 
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

InitSettings BrioWuConstants()
{
    return {{"rhoL", 1.0},      {"rhoR", 0.125},
            {"pL", 1.0},        {"pR", 0.1},
            {"Bx", 0.75},       {"ByL", 1.0},          {"ByR", -1.0},
            {"L", 2.0},         {"gamma", 2.0},
            {"mui", 10.},       {"Kcour", 0.2},
            {"ng0", 150},       {"ngmax", 200},
            {"minDt", 1e-7},    {"minDt_m1", 1e-7},
            {"gravConstant", 0.0}, {"brio-wu", 1.0}};
}

template<class T, class SimData>
void initBrioWuFields(SimData& sim, const std::map<std::string, double>& constants, T massPart)
{
    auto& d  = sim.hydro;
    auto& md = sim.magneto;

    constexpr bool gpu = cstone::HaveGpu<typename SimData::AcceleratorType>{};
    using HT           = typename std::decay_t<decltype(d)>::HydroType;
    using RT           = typename std::decay_t<decltype(md)>::RealType;
    using XM           = typename std::decay_t<decltype(d)>::XM1Type;
    using Tmass        = typename std::decay_t<decltype(d)>::Tmass;

    T rhoL  = constants.at("rhoL");
    T rhoR  = constants.at("rhoR");
    T pL    = constants.at("pL");
    T pR    = constants.at("pR");
    T Bx    = constants.at("Bx");
    T ByL   = constants.at("ByL");
    T ByR   = constants.at("ByR");
    T gamma = constants.at("gamma");

    T hL = 0.5 * std::cbrt(3. * d.ng0 * massPart / 4. / M_PI / rhoL);
    T hR = 0.5 * std::cbrt(3. * d.ng0 * massPart / 4. / M_PI / rhoR);

    auto cv    = sph::idealGasCv(d.muiConst, gamma);
    T    tempL = pL / ((gamma - 1.) * rhoL) / cv;
    T    tempR = pR / ((gamma - 1.) * rhoR) / cv;

    cstone::fill<gpu>(d.m.begin(), d.m.end(), Tmass(massPart));
    cstone::fill<gpu>(d.du_m1.begin(), d.du_m1.end(), XM(0.0));
    cstone::fill<gpu>(d.mue.begin(), d.mue.end(), HT(2.0));
    cstone::fill<gpu>(d.mui.begin(), d.mui.end(), HT(constants.at("mui")));
    cstone::fill<gpu>(d.alpha.begin(), d.alpha.end(), HT(d.alphamax));

    cstone::fill<gpu>(d.vx.begin(), d.vx.end(), HT(0.0));
    cstone::fill<gpu>(d.vy.begin(), d.vy.end(), HT(0.0));
    cstone::fill<gpu>(d.vz.begin(), d.vz.end(), HT(0.0));
    cstone::fill<gpu>(d.x_m1.begin(), d.x_m1.end(), XM(0.0));
    cstone::fill<gpu>(d.y_m1.begin(), d.y_m1.end(), XM(0.0));
    cstone::fill<gpu>(d.z_m1.begin(), d.z_m1.end(), XM(0.0));

    cstone::fill<gpu>(md.Bx.begin(), md.Bx.end(), RT(Bx));
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

    const std::size_t N = d.x.size();
    std::vector<HT>   h(N);
    std::vector<RT>   By(N);
    std::vector<T>    temp(N);

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < N; ++i)
    {
        bool left = x[i] < T(0);
        h[i]      = left ? HT(hL) : HT(hR);
        By[i]     = left ? RT(ByL) : RT(ByR);
        temp[i]   = left ? tempL : tempR;
    }

    d.h    = std::move(h);
    md.By  = std::move(By);
    d.temp = std::move(temp);
}

template<class SimData>
class BrioWuGlass : public ISimInitializer<SimData>
{
protected:
    std::string          glassBlock;
    mutable InitSettings settings_;

public:
    BrioWuGlass(std::string initBlock, std::string settingsFile, IFileReader* reader)
        : glassBlock(std::move(initBlock))
        , ISimInitializer<SimData>(settingsFile)
    {
        SimData d;
        settings_ = buildSettings(d, BrioWuConstants(), settingsFile, reader);
    }

    cstone::Box<typename SimData::RealType> initImpl(int rank, int numRanks, size_t cbrtNumPart, SimData& simData,
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

        // Wissing & Shen (2020) sect. 3.3: tube spans x in [-L, L] with the discontinuity at
        // x = 0 and a second one at the periodic boundary, so only the middle is physical.
        // Thin box: Lt shrinks with resolution so the transverse particle count stays fixed,
        // mirroring the paper's [1024, 24, 24] / [512, 12, 12] layout at feasible cost.
        T L = settings_.at("L");

        int multi1D = std::max(1, (int)std::rint(cbrtNumPart / std::cbrt(blockSize)));
        T   Lt      = L / (4 * multi1D);

        cstone::Box<T> globalBox(-L, L, 0, Lt, 0, Lt, pbc, pbc, pbc);
        cstone::Box<T> leftBox(-L, 0, 0, Lt, 0, Lt, pbc, pbc, pbc);
        cstone::Box<T> rightBox(0, L, 0, Lt, 0, Lt, pbc, pbc, pbc);

        // equal particle mass with rhoL/rhoR = 8: the dense (left) half gets 2x the
        // multiplicity per dimension, halving the inter-particle spacing
        cstone::Vec3<int> leftMulti  = {8 * multi1D, 2, 2};
        cstone::Vec3<int> rightMulti = {4 * multi1D, 1, 1};

        auto [keyStart, keyEnd] = equiDistantSfcSegments<KeyType>(rank, numRanks, 100);

        std::vector<T> x, y, z;
        assembleCuboid<T>(keyStart, keyEnd, leftBox, leftMulti, xBlock, yBlock, zBlock, x, y, z);
        assembleCuboid<T>(keyStart, keyEnd, rightBox, rightMulti, xBlock, yBlock, zBlock, x, y, z);

        d.x = x; // uploads to GPU if active
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

        size_t numLeft      = size_t(leftMulti[0]) * leftMulti[1] * leftMulti[2] * blockSize;
        T      leftVolume   = L * Lt * Lt;
        T      particleMass = leftVolume * settings_.at("rhoL") / numLeft;

        initBrioWuFields(simData, settings_, particleMass);

        return globalBox;
    }

    [[nodiscard]] const InitSettings& constants() const override { return settings_; }
};

template<class SimData>
class BrioWuGrid : public ISimInitializer<SimData>
{
    mutable InitSettings settings_;

public:
    BrioWuGrid(std::string settingsFile, IFileReader* reader)
        : ISimInitializer<SimData>(settingsFile)
    {
        SimData d;
        settings_ = buildSettings(d, BrioWuConstants(), settingsFile, reader);
    }

    cstone::Box<typename SimData::RealType> initImpl(int rank, int numRanks, size_t cbrtNumPart, SimData& simData,
                                                 IFileReader*) const override
    {
        auto& d       = simData.hydro;
        auto& md      = simData.magneto;
        using KeyType = typename SimData::KeyType;
        using T       = typename SimData::RealType;
        auto pbc      = cstone::BoundaryType::periodic;

        T L = settings_.at("L");

        // -n is the number of lattice planes along x in the low-density (right) half, so -n 512
        // reproduces Wissing & Shen (2020) [1024, 24, 24] / [512, 12, 12] exactly. The dense left
        // half gets half the lattice spacing (rhoL/rhoR = 8 with equal particle mass), and the
        // transverse extent is fixed at 12 right-half spacings, shrinking Lt as resolution grows.
        size_t nxRight = cbrtNumPart;
        size_t nt      = 12;
        T      dxRight = L / nxRight;
        T      Lt      = nt * dxRight;

        cstone::Box<T> globalBox(-L, L, 0, Lt, 0, Lt, pbc, pbc, pbc);
        cstone::Box<T> leftBox(-L, 0, 0, Lt, 0, Lt, pbc, pbc, pbc);
        cstone::Box<T> rightBox(0, L, 0, Lt, 0, Lt, pbc, pbc, pbc);

        cstone::Vec3<size_t> leftSide{2 * nxRight, 2 * nt, 2 * nt};
        cstone::Vec3<size_t> rightSide{nxRight, nt, nt};

        size_t numLeft  = leftSide[0] * leftSide[1] * leftSide[2];
        size_t numRight = rightSide[0] * rightSide[1] * rightSide[2];

        auto [firstLeft, lastLeft]   = partitionRange(numLeft, rank, numRanks);
        auto [firstRight, lastRight] = partitionRange(numRight, rank, numRanks);

        std::vector<T> x(lastLeft - firstLeft), y(lastLeft - firstLeft), z(lastLeft - firstLeft);
        regularGrid(leftBox, leftSide, firstLeft, lastLeft, x, y, z);

        std::vector<T> xR(lastRight - firstRight), yR(lastRight - firstRight), zR(lastRight - firstRight);
        regularGrid(rightBox, rightSide, firstRight, lastRight, xR, yR, zR);

        x.insert(x.end(), xR.begin(), xR.end());
        y.insert(y.end(), yR.begin(), yR.end());
        z.insert(z.end(), zR.begin(), zR.end());

        d.x = x; // uploads to GPU if active
        d.y = y;
        d.z = z;

        size_t numParticlesGlobal = numLeft + numRight;
        syncCoords<KeyType>(rank, numRanks, numParticlesGlobal, d.x, d.y, d.z, globalBox);

        d.resize(d.x.size());
        md.resize(d.x.size());

        settings_["numParticlesGlobal"] = double(numParticlesGlobal);
        BuiltinWriter attributeSetter(settings_);
        d.loadOrStoreAttributes(&attributeSetter);

        T leftVolume   = L * Lt * Lt;
        T particleMass = leftVolume * settings_.at("rhoL") / numLeft;

        initBrioWuFields(simData, settings_, particleMass);

        return globalBox;
    }

    [[nodiscard]] const InitSettings& constants() const override { return settings_; }
};
} // namespace sphexa
