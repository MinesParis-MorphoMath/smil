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
   
    // CPUID leaves with cache information:
    // 2 : Cache descriptors (AMD has zero cache descriptors)
    // 4 : Newer intel CPUS
    // 0x80000005 : AMD only
    // 0X80000006 : AMD only (L2 infos given for intel CPUS)
    // 0x8000001D : AMD only (used on Bulldozer CPUS, and can contradict leaf 0x80000006)
    Cache_Descriptors tmp;
    if (vendor == "GenuineIntel") {
        do {
            cerr << i << endl; 
            __asm__ (
                "mov $0x04, %%eax\n\t"
                "mov %3, %%ecx\n\t"
                "cpuid\n\t"
                "mov %%eax, %0\n\t"
                "mov %%ebx, %1\n\t"
                "mov %%ecx, %2"
                :"=a"(eaxFeatures), "=b"(ebxFeatures),"=c"(ecxFeatures)
                :"r"(i)
                :"%edx"
            );
            tmp.type = (eaxFeatures & (0x0000000F));
            tmp.associativity = ((ebxFeatures & 0xFFC00000) >> 22) +1;
            tmp.lines_per_tag = ((ebxFeatures & 0x003FF000) >> 12) +1;
            tmp.line_size = (ebxFeatures & 0x00000FFF) +1;
            tmp.sets = ecxFeatures+1;
            tmp.size = (tmp.associativity)*(tmp.lines_per_tag)*(tmp.line_size)*(tmp.sets);
            if (tmp.type != 0)
                L.push_back (tmp);
            ++i;
        } while (tmp.type != 0) ;

    } else {
        // In case of AMD 
        load(0x80000005);
        ecxFeatures = ecx;
        tmp.size = (ecxFeatures & 0xFF000000) >> 24; // In KBs
        tmp.associativity = (ecxFeatures & 0x00FF0000) >> 16; // FFh = full
        tmp.lines_per_tag = (ecxFeatures & 0x0000FF00) >> 8;
        tmp.line_size = (ecxFeatures & 0x000000FF); // In bytes
        tmp.type = 1;
        L.push_back (tmp);

        // Not sure if L2 and L3 will turn out to be of any use.
        load(0x80000006);
        ecxFeatures = ecx;
        tmp.size = (ecxFeatures & 0xFFFF0000) >> 16; // In KBs
        tmp.associativity = (ecxFeatures & 0x0000F000) >> 12; 
        tmp.lines_per_tag = (ecxFeatures & 0x00000F00) >> 8 ;
        tmp.line_size = (ecxFeatures & 0x000000FF); // In bytes
        tmp.type = 3;
        L.push_back (tmp);

        edxFeatures = edx;
        tmp.size = ((edxFeatures & 0xFFFC0000) >> 18)*512; // in KBs
        tmp.associativity = (edxFeatures & 0x0000F000) >> 12; 
        tmp.lines_per_tag = (edxFeatures & 0x00000F00) >> 8;
        tmp.line_size = (edxFeatures & 0x000000FF);
        tmp.type = 3;
        L.push_back (tmp); 

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


