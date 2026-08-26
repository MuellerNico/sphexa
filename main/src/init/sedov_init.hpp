/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
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
 * @brief Sedov blast simulation data initialization
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#pragma once

#include <map>

#include "cstone/primitives/primitives_acc.hpp"
#include "cstone/sfc/box.hpp"
#include "sph/eos.hpp"

#include "isim_init.hpp"
#include "sedov_constants.hpp"
#include "early_sync.hpp"
#include "grid.hpp"
#include "utils.hpp"

namespace sphexa
{

template<class Dataset>
void initSedovFields(Dataset& d, const InitSettings& constants)
{
    constexpr bool gpu = cstone::HaveGpu<typename Dataset::AcceleratorType>{};
    using T            = Dataset::RealType;

    double r           = constants.at("r1");
    double totalVolume = std::pow(2 * r, 3);
    double hInit       = std::cbrt(3.0 / (4 * M_PI) * d.ng0 * totalVolume / d.numParticlesGlobal) * 0.5;

    double mPart  = constants.at("mTotal") / d.numParticlesGlobal;
    double width  = T(2) * hInit;
    double width2 = width * width;

    // We distribute energy as exp(-r2 / width2), with width taken as the current 2h, so that the enery
    // is deposited in about ng0 neighbors.
    // ener0 is the constant that should multiply the Gaussian so that its integral equals energytotal
    double ener0 = constants.at("energyTotal") / std::pow(M_PI, 1.5) / width2 / width;

    initFieldsAtRest(d, mPart);
    cstone::fill<gpu>(d.h.begin(), d.h.end(), hInit);

    auto cv = sph::idealGasCv(d.muiConst, d.gamma);

    // If temperature is not allocated, we can still use this initializer for just the coordinates
    if (d.temp.empty() && d.u.empty()) { return; }

    auto&& x = toHost(d.x);
    auto&& y = toHost(d.y);
    auto&& z = toHost(d.z);

    std::vector<T> u(d.x.size());
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < d.x.size(); i++)
    {
        T r2 = norm2(cstone::Vec3<T>{x[i], y[i], z[i]});
        u[i] = ener0 * exp(-(r2 / width2)) + constants.at("u0");
    }
    if (!d.temp.empty())
    {
        std::for_each(u.begin(), u.end(), [cvm1 = 1.0 / cv](auto& t) { t *= cvm1; });
        d.temp = std::move(u);
    }
    else { d.u = std::move(u); }
}

template<class Dataset>
class SedovGrid : public ISimInitializer<Dataset>
{
    mutable InitSettings settings_;

public:
    SedovGrid()
        : ISimInitializer<Dataset>({})
    {
        Dataset d;
        settings_ = buildSettings(d, sedovConstants(), {}, nullptr);
    }

    cstone::Box<typename Dataset::RealType> initImpl(int rank, int numRanks, size_t cubeSide, Dataset& simData,
                                                     IFileReader*) const override
    {
        auto& d                   = simData.hydro;
        using KeyType             = typename Dataset::KeyType;
        using T                   = typename Dataset::RealType;
        size_t numParticlesGlobal = cubeSide * cubeSide * cubeSide;

        auto [first, last] = partitionRange(numParticlesGlobal, rank, numRanks);
        d.resize(last - first);

        T              r = settings_.at("r1");
        cstone::Box<T> globalBox(-r, r, cstone::BoundaryType::periodic);
        std::vector<T> x(last - first), y(last - first), z(last - first);
        regularGrid(r, cubeSide, first, last, x, y, z);
        d.x = x; // uploads to GPU if active
        d.y = y;
        d.z = z;
        syncCoords<KeyType>(rank, numRanks, numParticlesGlobal, d.x, d.y, d.z, globalBox);
        d.resize(d.x.size());

        settings_["numParticlesGlobal"] = double(numParticlesGlobal);
        BuiltinWriter attributeSetter(settings_);
        d.loadOrStoreAttributes(&attributeSetter);

        initSedovFields(d, settings_);

        return globalBox;
    }

    [[nodiscard]] const InitSettings& constants() const override { return settings_; }
};

template<class Dataset>
class SedovGlass : public ISimInitializer<Dataset>
{
protected:
    std::string          glassBlock;
    mutable InitSettings settings_;

public:
    SedovGlass(std::string initBlock, std::string settingsFile, IFileReader* reader)
        : glassBlock(std::move(initBlock))
        , ISimInitializer<Dataset>(settingsFile)
    {
        Dataset d;
        settings_ = buildSettings(d, sedovConstants(), settingsFile, reader);
    }

    /*! @brief initialize particle data with a constant density cube
     *
     * @param[in]    rank             MPI rank ID
     * @param[in]    numRanks         number of MPI ranks
     * @param[in]    cbrtNumPart      the cubic root of the global number of particles to generate
     * @param[inout] simData          particle dataset
     * @param[in]    reader           loads input files
     * @return                        the global coordinate bounding box
     */
    cstone::Box<typename Dataset::RealType> initImpl(int rank, int numRanks, size_t cbrtNumPart, Dataset& simData,
                                                     IFileReader* reader) const override
    {
        auto& d       = simData.hydro;
        using KeyType = typename Dataset::KeyType;
        using T       = typename Dataset::RealType;

        std::vector<T> xBlock, yBlock, zBlock;
        readTemplateBlock(glassBlock, reader, xBlock, yBlock, zBlock);
        size_t blockSize = xBlock.size();

        int               multi1D            = std::rint(cbrtNumPart / std::cbrt(blockSize));
        cstone::Vec3<int> multiplicity       = {multi1D, multi1D, multi1D};
        size_t            numParticlesGlobal = multi1D * multi1D * multi1D * blockSize;

        T              r = settings_.at("r1");
        cstone::Box<T> globalBox(-r, r, cstone::BoundaryType::periodic);

        auto [keyStart, keyEnd] = equiDistantSfcSegments<KeyType>(rank, numRanks, 100);

        std::vector<T> x, y, z;
        assembleCuboid<T>(keyStart, keyEnd, globalBox, multiplicity, xBlock, yBlock, zBlock, x, y, z);
        d.x = x; // uploads to GPU if active
        d.y = y;
        d.z = z;
        d.resize(d.x.size());

        settings_["numParticlesGlobal"] = double(numParticlesGlobal);
        BuiltinWriter attributeSetter(settings_);
        d.loadOrStoreAttributes(&attributeSetter);

        initSedovFields(d, settings_);

        return globalBox;
    }

    const InitSettings& constants() const override { return settings_; }
};

template<class HydroData, class MagnetoData>
void initMagnetoFields(MagnetoData& md, HydroData& d, const std::map<std::string, double>& constants)
{
    constexpr bool gpu = cstone::HaveGpu<typename HydroData::AcceleratorType>{};

    auto Bmag = constants.at("Bmag");
    using T   = typename HydroData::RealType;
    using HT  = typename HydroData::HydroType;
    using XM  = typename HydroData::XM1Type;

    cstone::fill<gpu>(md.Bx.begin(), md.Bx.end(), T(Bmag / sqrt(2.)));
    cstone::fill<gpu>(md.By.begin(), md.By.end(), T(0.0));
    cstone::fill<gpu>(md.Bz.begin(), md.Bz.end(), T(Bmag / sqrt(2.)));

    cstone::fill<gpu>(md.dBx.begin(), md.dBx.end(), T(0.0));
    cstone::fill<gpu>(md.dBy.begin(), md.dBy.end(), T(0.0));
    cstone::fill<gpu>(md.dBz.begin(), md.dBz.end(), T(0.0));
    cstone::fill<gpu>(md.dBx_m1.begin(), md.dBx_m1.end(), XM(0.0));
    cstone::fill<gpu>(md.dBy_m1.begin(), md.dBy_m1.end(), XM(0.0));
    cstone::fill<gpu>(md.dBz_m1.begin(), md.dBz_m1.end(), XM(0.0));

    cstone::fill<gpu>(md.psi_ch.begin(), md.psi_ch.end(), HT(0.0));
    cstone::fill<gpu>(md.d_psi_ch.begin(), md.d_psi_ch.end(), HT(0.0));
    cstone::fill<gpu>(md.d_psi_ch_m1.begin(), md.d_psi_ch_m1.end(), XM(0.0));

    auto cv       = sph::idealGasCv(d.muiConst, d.gamma);
    T    p_in     = 100.;
    T    p_out    = 1.;
    T    temp_in  = p_in / ((d.gamma - 1.) * constants.at("rho0")) / cv;
    T    temp_out = p_out / ((d.gamma - 1.) * constants.at("rho0")) / cv;

    auto&& x = toHost(d.x);
    auto&& y = toHost(d.y);
    auto&& z = toHost(d.z);

    std::vector<T> temp(d.x.size());
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < d.x.size(); ++i)
    {
        T r     = sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);
        temp[i] = (r <= 0.125) ? temp_in : temp_out;
    }
    d.temp = std::move(temp);
}

template<class SimData>
class SedovMagnetoGrid : public SedovGrid<SimData>
{
    mutable InitSettings settings_;

public:
    SedovMagnetoGrid()
        : SedovGrid<SimData>()
    {
        SimData sim;
        settings_ = buildSettings(sim, magneticSedovConstants(), {}, nullptr);
    }

    cstone::Box<typename SimData::RealType> initImpl(int rank, int numRanks, size_t cubeSide, SimData& simData,
                                                     IFileReader* reader) const override
    {
        auto box = SedovGrid<SimData>::initImpl(rank, numRanks, cubeSide, simData, reader);
        auto& md = simData.magneto;
        md.resize(simData.hydro.x.size());
        initMagnetoFields(md, simData.hydro, settings_);

        settings_["numParticlesGlobal"] = double(simData.hydro.numParticlesGlobal);
        BuiltinWriter attributeSetter(settings_);
        simData.hydro.loadOrStoreAttributes(&attributeSetter);
        return box;
    }

    [[nodiscard]] const InitSettings& constants() const override { return settings_; }
};

template<class SimData>
class SedovMagneto : public SedovGlass<SimData>
{
    std::string          glassBlock = SedovGlass<SimData>::glassBlock;
    mutable InitSettings settings_;

public:
    SedovMagneto(std::string initBlock, std::string settingsFile, IFileReader* reader)
        : SedovGlass<SimData>(initBlock, settingsFile, reader)
    {
        SimData sim;
        settings_ = buildSettings(sim, magneticSedovConstants(), settingsFile, reader);
    }

    cstone::Box<typename SimData::RealType> initImpl(int rank, int numRanks, size_t cbrtNumPart, SimData& simData,
                                                 IFileReader* reader) const override
    {
        auto  box = SedovGlass<SimData>::initImpl(rank, numRanks, cbrtNumPart, simData, reader);
        auto& md  = simData.magneto;
        md.resize(simData.hydro.x.size());
        initMagnetoFields(md, simData.hydro, settings_);

        settings_["numParticlesGlobal"] = double(simData.hydro.numParticlesGlobal);
        BuiltinWriter attributeSetter(settings_);
        simData.hydro.loadOrStoreAttributes(&attributeSetter);
        return box;
    }
};

} // namespace sphexa
