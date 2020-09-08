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
#include "DImageMatrix.hpp"

using namespace smil;

int main(void)
{
  int sx = 1024; // 24;
  int sy = 1024;

  Image_UINT8 im1(sx, sy);
  Image_UINT8 im2(im1);
  Image_UINT8 im3(im1);

  UINT BENCH_NRUNS = 100;

  BENCH_IMG(matTranspose, im1, im2);
  BENCH_IMG(matMultiply, im1, im2, im3);
}
