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


int main(void)
{
    int sx = 1024; //24;
    int sy = 1024;
    
    Image_UINT8 im1(sx, sy);
    Image_UINT8 im2(im1);
    Image_UINT8 im3(im1);

    Image_UINT16 im4(im1);

    double BENCH_NRUNS = 1E4;
    
    UINT8 val = 127;
    
    BENCH_IMG(fill, im1, val);
    BENCH_IMG(copy, im1, im3);
    BENCH_CROSS_IMG(copy, im1, im4);
    BENCH_IMG(inv, im1, im2);
    BENCH_IMG(inf, im1, im2, im3);
    BENCH_IMG_STR(inf, "val", im1, val, im3);
    BENCH_IMG(sup, im1, im2, im3);
    BENCH_IMG_STR(sup, "val", im1, val, im3);
    BENCH_IMG(add, im1, im2, im3);
    BENCH_IMG(addNoSat, im1, im2, im3);
    BENCH_IMG_STR(add, "val", im1, val, im3);
    BENCH_IMG(sub, im1, im2, im3);
    BENCH_IMG(subNoSat, im1, im2, im3);
    BENCH_IMG_STR(sub, "val", im1, val, im3);
    BENCH_IMG(grt, im1, im2, im3);
    BENCH_IMG(div, im1, im2, im3);
    BENCH_IMG(mul, im1, im2, im3);
    BENCH_IMG_STR(mul, "val", im1, val, im3);
    BENCH_IMG(mulNoSat, im1, im2, im3);
    BENCH_IMG_STR(mulNoSat, "val", im1, val, im3);



}

