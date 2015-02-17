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

#include "DBit.h"
#include "DCore.h"
#include "DBase.h"


using namespace smil;

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
    UINT BENCH_NRUNS = 1E3;
    
    BENCH_IMG(fill, im1, UINT8(0));
    BENCH_IMG(fill, b1, Bit(0));
    
    BENCH_IMG(copy, im1, im2);
    BENCH_IMG(copy, b1, b2);
    
    BENCH_IMG(inv, im1, im2);
    BENCH_IMG(inv, b1, b2);
    
    BENCH_IMG(inf, im1, im2, im3);
    BENCH_IMG(inf, b1, b2, b3);
    
    BENCH_IMG(add, im1, im2, im3);
    BENCH_IMG(add, b1, b2, b3);

    BENCH_IMG(grt, im1, im2, im3);
    BENCH_IMG(grt, b1, b2, b3);
    
    return 0;
}

