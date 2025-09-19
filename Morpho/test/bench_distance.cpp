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
#include "DMorpho.h"

using namespace smil;

class Test_Distance : public TestCase
{
  virtual void run()
  {
    Image<UINT8> im1(40, 40);
    Image<UINT8> im2(im1);
    Image<UINT8> im3(im1);

    fill(im1, UINT8(255));
    im1.setPixel(10, 10, UINT(0));
    drawLine(im1, 30, 10, 3, 3, UINT8(0));
    drawRectangle(im1, 10, 30, 3, 3, UINT8(0), true);
    im1.setPixel(30, 30, UINT(0));
    distV0(im1, im3, CrossSE());

    UINT BENCH_NRUNS = 100;
    BENCH_IMG(dist, im1, im2, CrossSE());
    TEST_ASSERT(im2 == im3);

    if (retVal != RES_OK)
      im2.printSelf(1);
  }
};

int main()
{
  TestSuite ts;
  ADD_TEST(ts, Test_Distance);
  return ts.run();
}
