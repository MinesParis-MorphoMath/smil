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


#include <cstdio>
#include <ctime>

#include "Core/include/DCore.h"
#include "Base/include/private/DBlobMeasures.hpp"
#include "Morpho/include/private/DMorphoLabel.hpp"
#include "Morpho/include/private/DMorphImageOperations.hxx"

using namespace smil;


int main()
{
    UINT BENCH_NRUNS = 1E3;
    
    Image<UINT8> im1("https://smil.cmm.minesparis.psl.eu/images/barbara.png");
    Image<UINT8> im2(im1);
    Image<UINT8> im3(im1);
    threshold(im1, im2);
    label(im2, im3);
    
    BENCH_IMG(blobsArea, im2);
    
    BENCH_IMG(computeBlobs, im3);
    map<UINT8, Blob> blobs = computeBlobs(im3);
    
    BENCH_STR(blobsArea, "Blobs", blobs);
    
    BENCH_IMG(blobsVolume, im1, blobs);

}

