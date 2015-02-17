/*
 * Smil
 * Copyright (c) 2011-2015 Matthieu Faessel
 *
 * This file is part of Smil.
 *
 * Smil is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * Smil is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Smil.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 */


#include <stdio.h>
#include <time.h>


#include "Core/include/DCore.h"
#include "Base/include/DBase.h"

using namespace smil;

#ifdef __SSE__

#include <emmintrin.h>

void SSE_INT_Sup(Image_UINT8 &im1, Image_UINT8 &im2, Image_UINT8 &im3)
{
    __m128i r0,r1;
    int size = im1.getWidth();
    int nlines = im1.getLineCount();
    
    UINT8 *p1 = im1.getPixels();
    UINT8 *p2 = im2.getPixels();
    UINT8 *p3 = im3.getPixels();

//     addLine<UINT8> al;

    for (int l=0;l<nlines;l++)
    {
        __m128i *l1 = (__m128i*)p1;
        __m128i *l2 = (__m128i*)p2;
        __m128i *l3 = (__m128i*)p3;
        
        for(int i=0 ; i<size ; i+=16, l1++, l2++, l3++)
        {
            r0 = _mm_load_si128(l1);
            r1 = _mm_load_si128(l2);
            r1 = _mm_max_epu8(r0, r1);
            _mm_store_si128(l3, r1);
        }
        
        p1 += size;
        p2 += size;
        p3 += size;
    }
};

#endif // __SSE__

void SSE_AV_Sup(Image_UINT8 &im1, Image_UINT8 &im2, Image_UINT8 &im3)
{
    int size = im1.getWidth();
    int nlines = im1.getLineCount();
    
    UINT8 *p1 = im1.getPixels();
    UINT8 *p2 = im2.getPixels();
    UINT8 *p3 = im3.getPixels();

    for (int l=0;l<nlines;l++)
    {
        UINT8 *l1 = p1;
        UINT8 *l2 = p2;
        UINT8 *l3 = p3;
        
        for(int i=0 ; i<size ; i++)
            l3[i] = (l1[i] > l2[i]) ? l1[i] : l2[i];
        
        p1 += size;
        p2 += size;
        p3 += size;
    }
};


void bench_INT_vs_AV()
{
    cout << "---- Intrinsic SIMD vs AV ----" << endl;
    
    int sx = 1024;
    int sy = 1024;
    
    Image_UINT8 im1(sx, sy);
    Image_UINT8 im2(im1);
    Image_UINT8 im3(im1);

    double BENCH_NRUNS = 1E4;

#ifdef __SSE__
    BENCH_IMG(SSE_INT_Sup, im1, im2, im3);
#endif // __SSE__
    BENCH_IMG(SSE_AV_Sup, im1, im2, im3);
    BENCH_IMG(sup, im1, im2, im3);
    
    cout << endl;
}

void bench_NCores()
{
    cout << endl << "---- Nbr Threads ----" << endl;
    
    int sx = 1024;
    int sy = 1024;
    
    Image_UINT8 im1(sx, sy);
    Image_UINT8 im2(im1);
    Image_UINT8 im3(im1);

    double BENCH_NRUNS = 1E4;

    Core *core = Core::getInstance();

    for (UINT i=1; i<=core->getMaxNumberOfThreads(); i++)
    {
        core->setNumberOfThreads(i);
        cout << i << ": ";
        BENCH_IMG(sup, im1, im2, im3);
    }    
    
    cout << endl;
}

void bench_Size()
{
    cout << endl << "---- Size ----" << endl;
    
    int sy = 1024;
    
//     Core::getInstance()->setNumberOfThreads(1);
    cout << Core::getInstance()->getNumberOfThreads() << " thread(s)" << endl;
    
    for (size_t sx=100;sx<1E6;sx*=2)
    {
        Image_UINT8 im1(sx, sy);
        Image_UINT8 im2(im1);
        Image_UINT8 im3(im1);

        double BENCH_NRUNS = 1E7 / sx;

        BENCH_IMG(sup, im1, im2, im3);
    }
    
    cout << endl;
}

int main(int argc, char *argv[])
{
    bench_INT_vs_AV();
    bench_NCores();
    bench_Size();
    
    return 0;
}

