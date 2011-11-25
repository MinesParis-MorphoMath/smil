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

#include "DImage.h"
#include "DImageArith.hpp"
#include "DBaseLineOperations.hpp"

#define bench(func, args) \
      t1 = clock(); \
      for (int i=0;i<nRuns;i++) \
	func args; \
        cout << #func << ": " << 1E3 * double(clock() - t1) / CLOCKS_PER_SEC / nRuns << " ms" << endl;


#ifdef __SSE__

#include <emmintrin.h>

void testAdd(Image_UINT8 &im1, Image_UINT8 &im2, Image_UINT8 &im3)
{
    __m128i r0,r1;
    int size = im1.getWidth();
    int nlines = im1.getLineCount();

//     addLine<UINT8> al;

    for (int l=0;l<nlines;l++)
    {
        UINT8 *lineIn1 = im1.getLines()[l];
        UINT8 *lineIn2 = im2.getLines()[l];
        UINT8 *lineOut = im3.getLines()[l];

//     al._exec(lineIn1, lineIn2, size, lineOut);
        for (int i=0 ; i<size+16 ; i+=16)
        {
            r0 = _mm_load_si128((__m128i *) lineIn1);
            r1 = _mm_load_si128((__m128i *) lineIn2);
            _mm_add_epi8(r0, r1);
// 	  _mm_adds_epi8(r0, r1);
            _mm_store_si128((__m128i *) lineOut,r1);

            lineIn1 += 16;
            lineIn2 += 16;
            lineOut += 16;
        }
    }
};

#endif // __SSE__



int main(int argc, char *argv[])
{
    Image_UINT8 im1(10,10);
    Image_UINT8 im2;
    Image_UINT8 im3;

    Image_UINT16 im4;

    int sx = 1000; //24;
    int sy = 1024;
//     sx = 40;
//     sy = 20;

    im1.setSize(sx, sy);
    im2.setSize(sx, sy);
    im3.setSize(sx, sy);
    im4.setSize(sx, sy);

    fill(im1, UINT8(100));
    fill(im2, UINT8(5));

    int t1;
    int nRuns = (int)1E3;
    UINT8 val = 10;

    bench(fill, (im3, val));
    bench(copy, (im1, im3));
//     bench(copy, (im1, im4));
    bench(inv, (im1, im2));
    bench(inf, (im1, im2, im3));
//     bench(inf, (im1, val, im3));
    bench(sup, (im1, im2, im3));
//     bench(sup, (im1, val, im3));
    bench(add, (im1, im2, im3));
    bench(addNoSat, (im1, im2, im3));
//     bench(add, (im1, val, im3));
    bench(sub, (im1, im2, im3));
    bench(subNoSat, (im1, im2, im3));
//     bench(sub, (im1, val, im3));
    bench(grt, (im1, im2, im3));
    bench(div, (im1, im2, im3));
    bench(mul, (im1, im2, im3));
//     bench(mul, (im1, val, im3));
    bench(mulNoSat, (im1, im2, im3));
//     bench(mulNoSat, (im1, val, im3));

//     bench(testAdd, (im1, im2, im3));


//      supLine<UINT8> f;
//       unaryMorphImageFunction<UINT8, supLine<UINT8> > mf;
//       bench(volIm, (im1));
//       im6.show();

//       add(im1, im2, im5);
//       im5.printSelf(sx < 50);
//       cout << im5;

//       im5.show();

//       qapp.Exec();

//       fill(im1, UINT8(100));
//       fill(im3, UINT8(0));


}

