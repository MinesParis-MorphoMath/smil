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

#include "DBit.h"



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

    Image<Bit> b1(sx, sy), b2(b1), b3(b1);
    
    UINT8 val = 10;
    UINT BENCH_NRUNS = 1E4;
    
    inf(b1, b2, b3);
//     BENCH_IMG(inv, im1, im2);
    BENCH_IMG(inv, b1, b2);
    BENCH_IMG(fill, b1, Bit(0));
    BENCH_IMG(inf, b1, b2, b3);
//     BENCH_IMG(inf, b1, Bit(1), im3);
    BENCH_IMG(sup, b1, b2, b3);
    return 1;
    
//     BENCH_IMG(sup, b1, b2, b3);
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

