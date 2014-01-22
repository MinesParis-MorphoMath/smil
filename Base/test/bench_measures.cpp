/*
 * Smil
 * Copyright (c) 2011 Matthieu Faessel
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

#include "DCore.h"
#include "DBase.h"
#include "DMeasures.hpp"
#include "DLabelMeasures.hpp"

using namespace smil;


int main(int argc, char *argv[])
{
    UINT BENCH_NRUNS = 1E2;
    
    Image_UINT8 im(1024,1024);
    Image_UINT8::lineType pixels = im.getPixels();
    
    randFill(im);
    BENCH_IMG(area, im);
    
    
    fill(im, UINT8(0));
    drawRectangle(im, 200,200,512,512,UINT8(127), 1);

    
    BENCH_IMG(measAreas, im);

}

