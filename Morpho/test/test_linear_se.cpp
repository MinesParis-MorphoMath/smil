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
#include "Base/include/DBase.h"
#include "DMorpho.h"

using namespace smil;

StrElt generateLinearSE(int length, int theta, int zeta = 0)
{
  StrElt se;

  int xf = length * cos(theta * PI / 180.);
  int yf = length * sin(theta * PI / 180.);

  vector<Point<int>> v;

  v = bresenhamPoints(0, 0, xf, yf);
  for (size_t i = 0; i < v.size(); i++)
    se.addPoint(v[i].x, v[i].y, v[i].z);
  return se;
}

class Test_LinearSE : public TestCase
{
  virtual void run()
  {
    vector<Point<int>> v;

    int    length = 20;
    StrElt se;

    se = generateLinearSE(length, 0);
    se.setName("0 degres");
    se.printSelf();

    se = generateLinearSE(length, 30);
    se.setName("30 degres");
    se.printSelf();

    se = generateLinearSE(length, 45);
    se.setName("45 degres");
    se.printSelf();

    se = generateLinearSE(length, 90);
    se.setName("90 degres");
    se.printSelf();

    se = generateLinearSE(length, 180);
    se.setName("180 degres");
    se.printSelf();

    se = generateLinearSE(length, 270);
    se.setName("270 degres");
    se.printSelf();

    TEST_ASSERT(true);
  }
};

int main()
{
  TestSuite ts;
  ADD_TEST(ts, Test_LinearSE);
  return ts.run();
}
