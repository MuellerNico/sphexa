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

/*! @file
 * @brief artificial resistivity scheme selector for MHD dissipation
 *
 * @author Nicolas Mueller
 */

#pragma once

namespace sph::magneto
{

enum class ResistivityScheme : int
{
    Constant = 0, //!< spatially uniform alpha_B
    Switch   = 1, //!< Tricco & Price (2013) switch
    SLR      = 2, //!< slope-limited reconstruction of B (García-Senz & Cabezón 2026 analogue), reconstruction only
    SLRB     = 3, //!< SLR + Balsara-like modulation (1 - modulator)
    SLRB2    = 4, //!< SLR + Balsara-like modulation (1 - modulator^2)
};

//! @brief SLR-family schemes reconstruct B_ab and need the B-field Jacobian at i and j
constexpr bool usesSLR(ResistivityScheme scheme)
{
    return scheme == ResistivityScheme::SLR || scheme == ResistivityScheme::SLRB || scheme == ResistivityScheme::SLRB2;
}

} // namespace sph::magneto
