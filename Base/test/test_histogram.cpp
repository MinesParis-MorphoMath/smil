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
#include "DImageHistogram.hpp"

#include "DGui.h"

#define bench(func, args) \
      t1 = clock(); \
      for (int i=0;i<nRuns;i++) \
	func args; \
        cout << #func << ": " << 1E3 * double(clock() - t1) / CLOCKS_PER_SEC / nRuns << " ms" << endl;


int testStretchHist()
{
    Image_UINT8 im1(4,4);
    Image_UINT8 im2(4,4);
    Image_UINT8 im3(4,4);

    UINT8 vec1[16] = { 50, 51, 52, 50,
                       50, 55, 60, 45,
                       98, 54, 65, 50,
                       35, 59, 20, 48
                     };

    UINT8 vec2[16] = { 98,   101,  104,  98,
                       98,   114,  130,  81,
                       255,  111,  147,  98,
                       49,   127,  0,    91
                     };

    im1 << vec1;
    im2 << vec2;

    stretchHist(im1, im3);

    return im2==im3;
}

int main(int argc, char *argv[])
{
//     Image_UINT8 im1(4,4);
//     Image_UINT8 im2(4,4);
//     Image_UINT8 im3(4,4);
//     Image_UINT8 im4(4,4);
// 
//     UINT8 vec1[16] = { 50, 51, 52, 50, \
//                        50, 55, 60, 45, \
//                        98, 54, 65, 50, \
//                        35, 59, 20, 48
//                      };
// 
//     UINT8 vec2[16] = { 10, 51, 20, 10, \
//                        40, 15, 10, 15, \
//                        58, 24, 25, 50, \
//                        15, 29, 10, 48
//                      };
// 
//     UINT8 vec3[16] = { 50, 51, 52, 50, \
//                        50, 55, 58, 45, \
//                        58, 54, 58, 50, \
//                        35, 58, 20, 48
//                      };
// 
//     im1 << vec1;
// //       im2 << vec2;
// //       inf(im1, im2, im3);
//     thresh(im1, UINT8(100), im4);
// //       inf(im1, im2, im3);
// 
//     testStretchHist();
// 
//     im4.printSelf(1);

    Image_UINT8 im1("/media/DELLO/ESRF/S1P1V1_1.png");
    
    vector<UINT> tvals = otsuThresholdValues(im1, 2);
    for (vector<UINT>::iterator it=tvals.begin();it!=tvals.end();it++)
      cout << *it << endl;
    
}

