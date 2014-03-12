/*
 * Copyright (c) 2011, Matthieu FAESSEL and ARMINES
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

#include "DCpuID.h"

#ifdef USE_OPEN_MP
#include <omp.h>
#endif

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif // _MSC_VER

using namespace smil;


CpuID::CpuID()
  : eax(regs[0]),
    ebx(regs[1]),
    ecx(regs[2]),
    edx(regs[3]),
    cores(0), 
    logical(0)
{
    eax = ebx = ecx = edx = 0;
    
    // Get vendor
    load(0);
    vendor += string((const char *)&ebx, 4);
    vendor += string((const char *)&edx, 4);
    vendor += string((const char *)&ecx, 4);

    // Get CPU features
    // See http://en.wikipedia.org/wiki/CPUID
    load(1);
    edxFeatures = edx;
    ecxFeatures = ecx;
    ebxFeatures = ebx;

    // HTT
    hyperThreaded =  (edxFeatures & (1 << 28))!=0;
    
    // SIMD
    simdInstructions.MMX = (edxFeatures & (1 << 23))!=0;
    simdInstructions.SSE = (edxFeatures & (1 << 25))!=0;
    simdInstructions.SSE2 = (edxFeatures & (1 << 26))!=0;
    simdInstructions.SSE3 = (ecxFeatures & (1 << 0))!=0;
    simdInstructions.SSSE3 = (ecxFeatures & (1 << 9))!=0;
    simdInstructions.SSE41 = (ecxFeatures & (1 << 19))!=0;
    simdInstructions.SSE42 = (ecxFeatures & (1 << 20))!=0;
    simdInstructions.AES = (ecxFeatures & (1 << 25))!=0;
    simdInstructions.AVX = (ecxFeatures & (1 << 28))!=0;
    
    
#ifdef USE_OPEN_MP
    #pragma omp parallel
    {
      cores = omp_get_num_procs();
      logical = omp_get_max_threads();
    }
    if (hyperThreaded)
      cores /= 2;
#else // USE_OPEN_MP
    cores = 1;
    logical = ebxFeatures & 0x00FF0000;
#endif // USE_OPEN_MP
    cores = max(cores, 1U);
}


void CpuID::load(unsigned i) 
{
#ifdef _MSC_VER
    __cpuid((int *)regs, i);
#else
    __cpuid(i, eax, ebx, ecx, edx);
#endif // _MSC_VER
}


