/*
 * Copyright (c) 2011-2015, Matthieu FAESSEL and ARMINES
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of Matthieu FAESSEL, or ARMINES nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS AND CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "Core/include/DCore.h"
#include "Morpho/include/DMorpho.h"
#include "DAdvanced.h"
#include "Smil-build.h"

using namespace smil;

int main(int argc, char *argv[])
{
  Image<UINT8> imIn;
  if (argc > 1) {
    read(argv[1], imIn);
  } else {
    char *path = pathTestImage("bw/balls.png");
    imIn = Image<UINT8>(path);
  }
  Image<UINT8> imOut(imIn);

  int length = 15;

  UINT BENCH_NRUNS = 200;

  for (int angle = 0; angle < 90; angle += 15) {
    cout << "* angle " << angle << " " << endl;
    BENCH_IMG(lineDilate, imIn, imOut, length, angle);
    BENCH_IMG(lineErode, imIn, imOut, length, angle);
    BENCH_IMG(lineOpen, imIn, imOut, length, angle);
    BENCH_IMG(lineClose, imIn, imOut, length, angle);
    cout << endl;
    BENCH_IMG(oldLineDilate, imIn, angle, length, imOut);
    BENCH_IMG(oldLineErode, imIn, angle, length, imOut);
    BENCH_IMG(oldLineOpen, imIn, angle, length, imOut);
    BENCH_IMG(oldLineClose, imIn, angle, length, imOut);
    cout << endl;
    BENCH_IMG(imFastLineOpen, imIn, angle, length, imOut);
    cout << endl;
  }
  cout << endl;

  BENCH_IMG(squareDilate, imIn, imOut, 2 * length);
  BENCH_IMG(squareErode,  imIn, imOut, 2 * length);
  BENCH_IMG(squareOpen,   imIn, imOut, 2 * length);
  BENCH_IMG(squareClose,  imIn, imOut, 2 * length);
  cout << endl;

  BENCH_IMG(oldSquareDilate, imIn, length, imOut);
  BENCH_IMG(oldSquareErode,  imIn, length, imOut);
  BENCH_IMG(oldSquareOpen,   imIn, length, imOut);
  BENCH_IMG(oldSquareClose,  imIn, length, imOut);
  cout << endl;

  BENCH_IMG(circleDilate, imIn, imOut, length);
  BENCH_IMG(circleErode,  imIn, imOut, length);
  BENCH_IMG(circleOpen,   imIn, imOut, length);
  BENCH_IMG(circleClose,  imIn, imOut, length);
  cout << endl;

  BENCH_IMG(oldCircleDilate, imIn, length, imOut);
  BENCH_IMG(oldCircleErode,  imIn, length, imOut);
  BENCH_IMG(oldCircleOpen,   imIn, length, imOut);
  BENCH_IMG(oldCircleClose,  imIn, length, imOut);
  cout << endl;
}
