/*
 * Copyright (c) 2011-2015, Matthieu FAESSEL and ARMINES
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



#include "Core/include/DCore.h"
#include "DMorpho.h"

#if _OPENMP >= 201107 // ( >= 3.1 )
// #include "UserModules/Chabardes/include/private/DMinima.hpp"
#endif 


using namespace smil;

int main()
{
    Image<UINT8> im1("http://smil.cmm.mines-paristech.fr/images/barbara.png");
    Image<UINT8> im2(im1);
    Image<UINT8> im3(im1);
    Image<UINT8> im4(im1);
    Image<UINT16> imLbl(im1);
    Image<UINT16> imLbl2(im1);
    
    
    UINT BENCH_NRUNS = 50;
    
    sup(im1, UINT8(30), im2);
    BENCH_IMG(build, im2, im1, im3);
    
    gradient(im1, im2);
    
    BENCH_IMG(minima, im2, im3, sSE());
    
#if _OPENMP >= 201107 // ( >= 3.1 )
    BENCH_IMG(fastMinima, im2, im3, sSE());
#endif 
    
    label(im3, imLbl);
    
    BENCH_IMG(basins, im2, imLbl, imLbl2);
    
    BENCH_IMG(watershed, im2, imLbl, im4);
    
    BENCH_NRUNS = 10;
    BENCH_IMG(watershedExtinction, im2, imLbl, im4);
        
}

