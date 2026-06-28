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

/*! @file Initialization of the travelling Alfvèn-Wave test
 *
 * @author Lukas Schmidt
 */

#pragma once
#include "isim_init.hpp"
#include "cstone/sfc/box.hpp"
#include "early_sync.hpp"
#include "grid.hpp"
#include "utils.hpp"

namespace sphexa
{

InitSettings ALfvenWaveConstants()
{
    return {{"rho", 1.},           {"L", 3.},
            {"lambda", 1},         {"P", 0.1},
            {"sinA", 2. / 3.},     {"sinB", 2. / std::sqrt(5)},
            {"gamma", 5. / 3.},    {"Kcour", 0.2},
            {"ng0", 150},          {"ngmax", 200},
            {"minDt", 1e-7},       {"minDt_m1", 1e-7},
            {"gravConstant", 0.0}, {"alfven-wave", 1.0}};
}

template<class T>
static cstone::Vec3<cstone::Vec3<T>> coordinateTransformToCartesian(T sinA, T sinB)
{
    T               cosA = std::sqrt(1. - sinA * sinA);
    T               cosB = std::sqrt(1. - sinB * sinB);
    cstone::Vec3<T> a = {cosA * cosB, -sinB, -sinA * cosB}, b = {cosA * sinB, cosB, -sinA * sinB}, c = {sinA, 0, cosA};
    return {a, b, c};
}

template<class T>
static cstone::Vec3<cstone::Vec3<T>> coordinateTransformToRotated(T sinA, T sinB)
{
    T cosA = std::sqrt(1. - sinA * sinA);
    T cosB = std::sqrt(1. - sinB * sinB);

    // Inverse matrix of coordinateTransformToCartesian.
    // Note that since this is a rotation matrix, its inverse is the transpose
    cstone::Vec3<T> a = {cosA * cosB, cosA * sinB, sinA}, b = {-sinB, cosB, 0}, c = {-sinA * cosB, -sinA * sinB, cosA};
    return {a, b, c};
}

template<class T, class SimData>
void initAlfvenWaveFields(SimData& sim, const std::map<std::string, double>& constants, T massPart)
{
    auto& d  = sim.hydro;
    auto& md = sim.magneto;

    constexpr bool gpu = cstone::HaveGpu<typename SimData::AcceleratorType>{};
    using HT           = typename std::decay_t<decltype(d)>::HydroType;
    using RT           = typename std::decay_t<decltype(md)>::RealType;
    using XM           = typename std::decay_t<decltype(d)>::XM1Type;
    using Tmass        = typename std::decay_t<decltype(d)>::Tmass;

    T sinA   = constants.at("sinA");
    T sinB   = constants.at("sinB");
    T cosA   = std::sqrt(1. - sinA * sinA);
    T cosB   = std::sqrt(1. - sinB * sinB);
    T lambda = constants.at("lambda");
    T p      = constants.at("P");
    T gamma  = constants.at("gamma");
    T rho    = constants.at("rho");

    T h = 0.5 * std::cbrt(3. * d.ng0 * massPart / 4. / M_PI / rho);

    auto cv   = sph::idealGasCv(d.muiConst, gamma);
    T    temp = p / ((gamma - 1.) * rho) / cv;

    cstone::fill<gpu>(d.m.begin(), d.m.end(), Tmass(massPart));
    cstone::fill<gpu>(d.du_m1.begin(), d.du_m1.end(), XM(0.0));
    cstone::fill<gpu>(d.mue.begin(), d.mue.end(), HT(2.0));
    cstone::fill<gpu>(d.mui.begin(), d.mui.end(), HT(10.0));
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

    T k = 2 * M_PI / lambda;

    // estimate max timestep from Alfvèn speed
    auto v_alfven = sqrt((1 + 0.01) / md.mu_0 * rho);
    d.maxDt       = 0.1 * lambda / v_alfven;

    auto&& x = toHost(d.x);
    auto&& y = toHost(d.y);
    auto&& z = toHost(d.z);

    const std::size_t  N = d.x.size();
    std::vector<HT>    vx(N), vy(N), vz(N);
    std::vector<RT>    Bx(N), By(N), Bz(N);
    std::vector<XM>    x_m1(N), y_m1(N), z_m1(N);

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < N; ++i)
    {
        T               x1     = dot(coordinateTransformToRotated(sinA, sinB)[0], {x[i], y[i], z[i]});
        T               sinKx1 = std::sin(k * x1);
        T               cosKx1 = std::cos(k * x1);
        cstone::Vec3<T> Vrot   = {0, 0.1 * sinKx1, 0.1 * cosKx1};
        cstone::Vec3<T> Brot   = {1, 0.1 * sinKx1, 0.1 * cosKx1};

        vx[i] = dot(coordinateTransformToCartesian(sinA, sinB)[0], Vrot);
        vy[i] = dot(coordinateTransformToCartesian(sinA, sinB)[1], Vrot);
        vz[i] = dot(coordinateTransformToCartesian(sinA, sinB)[2], Vrot);

        Bx[i] = dot(coordinateTransformToCartesian(sinA, sinB)[0], Brot);
        By[i] = dot(coordinateTransformToCartesian(sinA, sinB)[1], Brot);
        Bz[i] = dot(coordinateTransformToCartesian(sinA, sinB)[2], Brot);

        x_m1[i] = vx[i] * d.minDt;
        y_m1[i] = vy[i] * d.minDt;
        z_m1[i] = vz[i] * d.minDt;
    }

    d.vx    = std::move(vx);
    d.vy    = std::move(vy);
    d.vz    = std::move(vz);
    md.Bx   = std::move(Bx);
    md.By   = std::move(By);
    md.Bz   = std::move(Bz);
    d.x_m1  = std::move(x_m1);
    d.y_m1  = std::move(y_m1);
    d.z_m1  = std::move(z_m1);
}

template<class SimData>
class AlfvenGlass : public ISimInitializer<SimData>
{
protected:
    std::string          glassBlock;
    mutable InitSettings settings_;

public:
    AlfvenGlass(std::string initBlock, std::string settingsFile, IFileReader* reader)
        : glassBlock(std::move(initBlock))
        , ISimInitializer<SimData>(settingsFile)
    {
        SimData d;
        settings_ = buildSettings(d, ALfvenWaveConstants(), settingsFile, reader);
    }

    cstone::Box<typename SimData::RealType> initImpl(int rank, int numRanks, size_t cbrtNumPart, SimData& simData,
                                                 IFileReader* reader) const override
    {
        auto& d       = simData.hydro;
        auto& md      = simData.magneto;
        using KeyType = typename SimData::KeyType;
        using T       = typename SimData::RealType;

        std::vector<T> xBlock, yBlock, zBlock;
        readTemplateBlock(glassBlock, reader, xBlock, yBlock, zBlock);
        size_t blockSize = xBlock.size();

        int               multi1D            = std::rint(cbrtNumPart / std::cbrt(blockSize));
        cstone::Vec3<int> multiplicity       = {2 * multi1D, multi1D, multi1D};
        size_t            numParticlesGlobal = 2 * multi1D * multi1D * multi1D * blockSize;

        T              L   = settings_.at("L");
        auto           pbc = cstone::BoundaryType::periodic;
        cstone::Box<T> globalBox(0, L, 0, L / 2., 0, L / 2., pbc, pbc, pbc);

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

        initAlfvenWaveFields(simData, settings_, particleMass);

        return globalBox;
    }

    [[nodiscard]] const InitSettings& constants() const override { return settings_; }
};
} // namespace sphexa
