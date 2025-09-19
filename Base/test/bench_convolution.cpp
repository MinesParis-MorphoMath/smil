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
#include "DImageConvolution.hpp"

using namespace smil;

int main(void)
{
  UINT BENCH_NRUNS = 100;

  Image<UINT8> im1(1024, 1024);
  Image<UINT8> im2(im1);

  BENCH_IMG_STR(gaussianFilter, "size 2", im1, 2, im2);
  BENCH_IMG_STR(gaussianFilter, "size 20", im1, 20, im2);

  return 0;
}
