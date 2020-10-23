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
#include "DMorphoExtras.h"
#include "Smil-build.h"

using namespace smil;

class TestAreaOpening03 : public TestCase
{
  virtual void run()
  {
    char *path;

    path = pathTestImage("test-results/zquares-y.png");
    Image_UINT8 imIn(path);
    Image_UINT8 imOut(imIn);

    path = pathTestImage("test-results/zquares-y-03.png");
    Image_UINT8 imTruth(path);
    areaOpening(imIn, 3, imOut);
    TEST_ASSERT(imOut == imTruth);
    if (retVal != RES_OK)
      imOut.printSelf();
  }
};

class TestAreaOpening06 : public TestCase
{
  virtual void run()
  {
    char *path;

    path = pathTestImage("test-results/zquares-y.png");
    Image_UINT8 imIn(path);
    Image_UINT8 imOut(imIn);

    path = pathTestImage("test-results/zquares-y-06.png");
    Image_UINT8 imTruth(path);
    areaOpening(imIn, 6, imOut);
    TEST_ASSERT(imOut == imTruth);
    if (retVal != RES_OK)
      imOut.printSelf();
  }
};

class TestAreaOpening12 : public TestCase
{
  virtual void run()
  {
    char *path;

    path = pathTestImage("test-results/zquares-y.png");
    Image_UINT8 imIn(path);
    Image_UINT8 imOut(imIn);
    path = pathTestImage("test-results/zquares-y-12.png");
    Image_UINT8 imTruth(path);

    Morpho::setDefaultSE(CrossSE());
    areaOpening(imIn, 12, imOut);
    TEST_ASSERT(imOut == imTruth);
    if (retVal != RES_OK)
      imOut.printSelf();
  }
};

class TestAreaOpening20 : public TestCase
{
  virtual void run()
  {
    char *path;

    path = pathTestImage("test-results/zquares-y.png");
    Image_UINT8 imIn(path);
    Image_UINT8 imOut(imIn);
    path = pathTestImage("test-results/zquares-y-20.png");
    Image_UINT8 imTruth(path);

    Morpho::setDefaultSE(CrossSE());
    areaOpening(imIn, 20, imOut);
    TEST_ASSERT(imOut == imTruth);
    if (retVal != RES_OK)
      imOut.printSelf();
  }
};

class TestAreaOpening30 : public TestCase
{
  virtual void run()
  {
    char *path;

    path = pathTestImage("test-results/zquares-y.png");
    Image_UINT8 imIn(path);
    Image_UINT8 imOut(imIn);
    path = pathTestImage("test-results/zquares-y-30.png");
    Image_UINT8 imTruth(path);

    Morpho::setDefaultSE(CrossSE());
    areaOpening(imIn, 30, imOut);
    TEST_ASSERT(imOut == imTruth);
    if (retVal != RES_OK)
      imOut.printSelf();
  }
};

int main()
{
  TestSuite ts;
  ADD_TEST(ts, TestAreaOpening03);
  ADD_TEST(ts, TestAreaOpening06);
  ADD_TEST(ts, TestAreaOpening12);
  ADD_TEST(ts, TestAreaOpening20);
  ADD_TEST(ts, TestAreaOpening30);
  return ts.run();
}
