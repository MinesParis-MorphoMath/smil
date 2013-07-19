/*
 * Smil
 * Copyright (c) 2010 Matthieu Faessel
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

//#include <boost/signal.hpp>
//#include <boost/bind.hpp>

#include "DCore.h"
#include "DBase.h"

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



int main(int argc, char *argv[])
{
    int sx = 1024; //24;
    int sy = 1024;
    
    Image_UINT8 im1(sx, sy);
    Image_UINT8 im2(im1);
    Image_UINT8 im3(im1);

    Image_UINT16 im4(im1);

//     sx = 40;
//     sy = 20;

    UINT8 val = 10;
    double BENCH_NRUNS = 1E4;
    
    BENCH_IMG(fill, im1, val);
    BENCH_IMG(copy, im1, im3);
    BENCH_CROSS_IMG(copy, im1, im4);
    BENCH_IMG(inv, im1, im2);
    BENCH_IMG(inf, im1, im2, im3);
    BENCH_IMG(inf, im1, val, im3);
    BENCH_IMG(sup, im1, im2, im3);
    BENCH_IMG_STR(sup, "val", im1, val, im3);
    BENCH_IMG(add, im1, im2, im3);
    BENCH_IMG(addNoSat, im1, im2, im3);
    BENCH_IMG(add, im1, val, im3);
    BENCH_IMG(sub, im1, im2, im3);
    BENCH_IMG(subNoSat, im1, im2, im3);
    BENCH_IMG(sub, im1, val, im3);
    BENCH_IMG(grt, im1, im2, im3);
    BENCH_IMG(div, im1, im2, im3);
    BENCH_IMG(mul, im1, im2, im3);
    BENCH_IMG(mul, im1, val, im3);
    BENCH_IMG(mulNoSat, im1, im2, im3);
    BENCH_IMG(mulNoSat, im1, val, im3);


//      supLine<UINT8> f;
//       unaryMorphImageFunction<UINT8, supLine<UINT8> > mf;
//       BENCH_IMG(volIm, (im1));
//       im6.show();

//       add(im1, im2, im5);
//       im5.printSelf(sx < 50);
//       cout << im5;

//       im5.show();

//       qapp.Exec();

//       fill(im1, UINT8(100));
//       fill(im3, UINT8(0));


}

