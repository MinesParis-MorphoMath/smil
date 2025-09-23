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
#include "DZhangSkel.hpp"
#include "Smil-build.h"

using namespace smil;

int main()
{
  UINT BENCH_NRUNS = 1E2;

  char *path = pathTestImage("bw/H4skeleton.png");

  Image<UINT8> im1(path);
  Image<UINT8> im2(im1);

  BENCH_IMG(zhangSkeleton, im1, im2);
  BENCH_IMG(zhangThinning, im1, im2);
  BENCH_IMG(imageThinning, im1, im2, "Zhang");
  BENCH_IMG(imageThinning, im1, im2, "DongLinHuang");
}
