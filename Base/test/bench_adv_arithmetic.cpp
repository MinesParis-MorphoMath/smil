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


int main(int argc, char *argv[])
{
    int sx = 1024; //24;
    int sy = 1024;
    
    Image_UINT8 im1(sx, sy);
    Image_UINT8 im2(im1);
    Image_UINT8 im3(im1);

    Image_UINT16 im4(im1);

    double BENCH_NRUNS = 1E2;
    
    UINT8 val = 127;
    std::map<UINT8, UINT8> lut;
    for (int i=0;i<256;i++)
      lut[i] = 255-i;
    
    BENCH_IMG(randFill, im1);
    BENCH_IMG(applyLookup, im1, lut, im2);

}

