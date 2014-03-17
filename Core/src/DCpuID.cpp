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
#include <iostream>

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
   
    L[0].size = 0;
    L[0].associativity = 0;
    L[0].lines_per_tag = 0;
    L[0].line_size = 0;
    L[1].size = 0;
    L[1].associativity = 0;
    L[1].lines_per_tag = 0;
    L[1].line_size = 0;
    L[2].size = 0;
    L[2].associativity = 0;
    L[2].lines_per_tag = 0;
    L[2].line_size = 0;

    // CPUID leaves with cache information:
    // 2 : Cache descriptors (AMD has zero cache descriptors)
    // 4 : Newer intel CPUS
    // 0x80000005 : AMD only
    // 0X80000006 : AMD only (L2 infos given for intel CPUS)
    // 0x8000001D : AMD only (used on Bulldozer CPUS, and can contradict leaf 0x80000006)
    if (vendor == "GenuineIntel") {
        load(2);
        nbr_cache_level = 3;
        int ways;
        int partitions;
        int line_size;
        int sets;

        for (int i=0; i<nbr_cache_level; ++i) { 
            __asm__ (
                "mov $0x04, %%eax\n\t"
                "mov %1, %%ecx\n\t"
                "cpuid\n\t"
                "mov %%ebx, %0\n\t"
                "mov %%ecx, %1"
                :"=b"(ebxFeatures),"=c"(ecxFeatures)
                :"r"(i)
                :"%eax","%edx"
            );

            ways = (ebxFeatures & 0xFFC0000000) >> 22;
            partitions = (ebxFeatures & 0x003FF000) >> 12;
            line_size = (ebxFeatures & 0x00000FFF) ;
            sets = ecxFeatures;
            L[i].size = (ways+1)*(partitions+1)*(line_size+1)*(sets+1);
            L[i].line_size = line_size+1;
            L[i].associativity = ways+1;
            L[i].lines_per_tag = partitions+1;
        } 

    } else {
        nbr_cache_level = 3;
        // In case of AMD 
        load(0x80000005);
        ecxFeatures = ecx;
        L[0].size = (ecxFeatures & 0xFF000000) >> 24; // In KBs
        L[0].associativity = (ecxFeatures & 0x00FF0000) >> 16; // FFh = full
        L[0].lines_per_tag = (ecxFeatures & 0x0000FF00) >> 8;
        L[0].line_size = (ecxFeatures & 0x000000FF); // In bytes

        // Not sure if L2 and L3 will turn out to be of any use.
        load(0x80000006);
        ecxFeatures = ecx;
        L[1].size = (ecxFeatures & 0xFFFF0000) >> 16; // In KBs
        L[1].associativity = (ecxFeatures & 0x0000F000) >> 12; 
        L[1].lines_per_tag = (ecxFeatures & 0x00000F00) >> 8 ;
        L[1].line_size = (ecxFeatures & 0x000000FF); // In bytes
        edxFeatures = edx;
        L[2].size = ((edxFeatures & 0xFFFC0000) >> 18)*512; // in KBs
        L[2].associativity = (edxFeatures & 0x0000F000) >> 12; 
        L[2].lines_per_tag = (edxFeatures & 0x00000F00) >> 8;
        L[2].line_size = (edxFeatures & 0x000000FF);
        // End of AMD
    } 

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


