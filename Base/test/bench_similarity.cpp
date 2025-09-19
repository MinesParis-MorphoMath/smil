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
#include "Base/include/DBase.h"

using namespace smil;

int main(void)
{
  const size_t dim = 256;

  Image<UINT8> imGt(dim, dim);
  Image<UINT8> imIn(dim, dim);

  off_t m, M;

  fill(imGt, UINT8(0));
  m = dim / 8;
  M = 5 * dim / 8;
  for (off_t j = m; j < M; j++)
    for (off_t i = m; i < M; i++)
      imGt.setPixel(i, j, 255);

  fill(imIn, UINT8(0));
  m = 2 * dim / 8;
  M = 7 * dim / 8;
  for (off_t j = m; j < M; j++)
    for (off_t i = m; i < M; i++)
      imIn.setPixel(i, j, 255);

  UINT BENCH_NRUNS = 100;

  BENCH_IMG(indexJaccard, imGt, imIn);
  BENCH_IMG(indexRuzicka, imGt, imIn);
  BENCH_IMG(indexAccuracy, imGt, imIn);
  BENCH_IMG(indexPrecision, imGt, imIn);
  BENCH_IMG(indexRecall, imGt, imIn);
  BENCH_IMG(indexFScore, imGt, imIn);
  BENCH_IMG(indexSensitivity, imGt, imIn);
  BENCH_IMG(indexSpecificity, imGt, imIn);
  BENCH_IMG(indexFallOut, imGt, imIn);
  BENCH_IMG(indexMissRate, imGt, imIn);
  BENCH_IMG(indexOverlap, imGt, imIn);

  BENCH_IMG(distanceHamming, imGt, imIn);
  BENCH_IMG(distanceHausdorff, imGt, imIn);

  // BENCH_IMG(isBinary, imGt);
  // BENCH_IMG(valueList, imGt);
}
