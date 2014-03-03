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

#ifdef WIN32
#include <intrin.h>
#endif // WIN32

using namespace smil;


CpuID::CpuID()
  : cores(0), 
    logical(0),
    eax(regs[0]),
    ebx(regs[1]),
    ecx(regs[2]),
    edx(regs[3])
{
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
    hyperThreaded =  edxFeatures & (1 << 28);
    
    // SIMD
    simdInstructions.MMX = edxFeatures & (1 << 23);
    simdInstructions.SSE = edxFeatures & (1 << 25);
    simdInstructions.SSE2 = edxFeatures & (1 << 26);
    simdInstructions.SSE3 = ecxFeatures & (1 << 0);
    simdInstructions.SSSE3 = ecxFeatures & (1 << 9);
    simdInstructions.SSE41 = ecxFeatures & (1 << 19);
    simdInstructions.SSE42 = ecxFeatures & (1 << 20);
    simdInstructions.AES = ecxFeatures & (1 << 25);
    simdInstructions.AVX = ecxFeatures & (1 << 28);
    
    
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
#ifdef _WIN32
    __cpuid((int *)regs, (int)i);
#else
    asm volatile
      ("cpuid" : "=a" (eax), "=b" (ebx), "=c" (ecx), "=d" (edx)
      : "a" (i), "c" (0));
    // ECX is set to zero for CPUID function 4
#endif
}


