/*
 * Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of Matthieu FAESSEL, or ARMINES nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS AND CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _DCPUID_H
#define _DCPUID_H

#include "DTypes.h"

#include <string>
#include <vector>

namespace smil
{
    struct SIMD_Instructions
    {
        bool MMX;
        bool SSE;
        bool SSE2;
        bool SSE3;
        bool SSSE3;
        bool SSE41;
        bool SSE42;
        bool AES;
        bool AVX;

        // added 15/10/2021
        bool AVX2;
        bool AVX512;
        bool FMA;
    };

    // Associativity.
    enum { WDISABLED, W1, W2, W4=4, W8=6, W16=8, W32=10, W48, W64, W96, W128, WFULL };

    // Data cache information.
    struct Cache_Descriptors
    {
        int type; // 1 : data, 2 : instructions, 3 : unified
        int size;
        int sets; // ???
        int associativity;
        int lines_per_tag;
        int line_size;
    };

    class CpuID
    {

      public:

        CpuID();

        string getVendor() const { return vendor; }
        unsigned getCores() const { return cores; }
        unsigned getLogical() const { return logical; }
        bool isHyperThreated() const { return hyperThreaded; }
        const SIMD_Instructions &getSimdInstructions() const { return simdInstructions; }
        const std::vector<Cache_Descriptors> &getCaches() const {
            return L;
        }
        unsigned int getNbrCacheLevel() const { return L.size (); }

      protected:
        UINT32 regs[4];
        UINT32 &eax, &ebx, &ecx, &edx;
        unsigned eaxFeatures, edxFeatures, ecxFeatures, ebxFeatures;

        unsigned cores;
        unsigned logical;
        string vendor;
        bool hyperThreaded;
        SIMD_Instructions simdInstructions;
        std::vector<Cache_Descriptors> L;

        void load(unsigned i);

    };
} // namespace smil

#endif // _DCPUID_H
