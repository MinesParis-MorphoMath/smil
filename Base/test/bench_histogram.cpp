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
#include "DImageDraw.hpp"
#include "DMeasures.hpp"
#include "DBlobMeasures.hpp"
#include "Smil-build.h"

using namespace smil;

int main(void)
{
  UINT BENCH_NRUNS = 1E2;

  Image_UINT8 im(1024, 1024);

  randFill(im);
  BENCH_IMG(histogram, im);
  BENCH_IMG(histogramMap, im);

  char *path = pathTestImage("barbara.png");
  Image_UINT8 imb(path);
  BENCH_IMG(histogram, imb);
  BENCH_IMG(histogramMap, imb);  
}
