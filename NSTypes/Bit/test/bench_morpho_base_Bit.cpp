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



#include "DBit.h"
#include "DCore.h"
#include "DMorpho.h"
#include "DGui.h"
#include "DIO.h"

using namespace smil;

int main(int argc, char *argv[])
{
    typedef Image<UINT8> imType1;
    typedef Image<Bit> imType2;
    
    imType1 im1(1024, 1024);
    imType1 im2(im1);
    imType2 b1(im1);
    imType2 b2(im1);
    
    UINT BENCH_NRUNS = 1E3;
    BENCH_IMG_STR(dilate, "hSE", im1, im2, hSE());
    BENCH_IMG_STR(dilate, "hSE", b1, b2, hSE());
    
    BENCH_IMG_STR(dilate, "sSE", im1, im2, sSE());
    BENCH_IMG_STR(dilate, "sSE", b1, b2, sSE());

    BENCH_IMG_STR(open, "hSE", im1, im2, hSE());
    BENCH_IMG_STR(open, "hSE", b1, b2, hSE());
    
    BENCH_IMG_STR(open, "sSE", im1, im2, sSE());
    BENCH_IMG_STR(open, "sSE", b1, b2, sSE());
    
    return 0;
}

