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
#include "IO/include/private/DImageIO_RAW.hpp"

#ifdef SMIL_WRAP_RGB
#include "NSTypes/RGB/include/DRGB.h"
#endif // SMIL_WRAP_RGB

#include "Smil-build.h"

using namespace smil;

class Test_RW_RAW : public TestCase
{
  virtual void run()
  {
    typedef UINT8 T;
    const char *fName = "_smil_io_tmp.raw";

    Image<T> im1(3, 3, 2);
    T tab[] = {
      28, 2, 3,
      2, 5, 6,
      3, 8, 9,
      4, 11, 12,
      5, 15, 16,
      6, 18, 19
    };
    im1 << tab;
    TEST_ASSERT(writeRAW(im1, fName) == RES_OK);

    Image<T> im2;
    TEST_ASSERT(readRAW(fName, 3, 3, 2, im2) == RES_OK);
    TEST_ASSERT(im1 == im2);
  }
};

#ifdef USE_PNG
class Test_RW_PNG : public TestCase
{
  virtual void run()
  {
    const char *fName = "_smil_io_tmp.png";
    Image<UINT8> im1(3, 3);
    UINT8 pix[] = {28, 2, 3, 2, 5, 6, 3, 8, 9};
    im1 << pix;
    TEST_ASSERT(write(im1, fName) == RES_OK);
    Image<UINT8> im2;
    TEST_ASSERT(read(fName, im2) == RES_OK);
    TEST_ASSERT(im1 == im2);
    BaseImage *im3 = createFromFile(fName);
    TEST_ASSERT(im3 != NULL);
    delete im3;
  }
};

#ifdef USE_CURL
class Test_Curl : public TestCase
{
  virtual void run()
  {
    Image<UINT8> im1;

    const char *bImage = SmilWebImages "/barbara.png";
    const char *aImage = SmilWebImages "/arearea.png";

    read(bImage, im1);
    TEST_ASSERT(im1.isAllocated());
    Image<UINT8> im2(bImage);
    TEST_ASSERT(im2.isAllocated());
    BaseImage *im0 = createFromFile(aImage);
    TEST_ASSERT(im0 != NULL);
    delete im0;
  }
};
#endif // USE_CURL
#endif // USE_PNG

#ifdef USE_TIFF
class Test_RW_TIFF : public TestCase
{
  virtual void run()
  {
    const char *fName = "_smil_io_tmp.tiff";
    Image<UINT8> im1(3, 3);
    UINT8 pix[] = {28, 2, 3, 2, 5, 6, 3, 8, 9};
    im1 << pix;
    TEST_ASSERT(write(im1, fName) == RES_OK);
    Image<UINT8> im2;
    TEST_ASSERT(read(fName, im2) == RES_OK);
    TEST_ASSERT(im1 == im2);
    BaseImage *im3 = createFromFile(fName);
    TEST_ASSERT(im3 != NULL);
    delete im3;
  }
};
#endif // USE_TIFF

// TODO: Test JPEG format

class Test_RW_PGM : public TestCase
{
  virtual void run()
  {
    const char *fName = "_smil_io_tmp.pgm";
    Image<UINT8> im1(3, 3);
    UINT8 pix[] = {28, 2, 3, 2, 5, 6, 3, 8, 9};
    im1 << pix;
    TEST_ASSERT(write(im1, fName) == RES_OK);
    Image<UINT8> im2;
    TEST_ASSERT(read(fName, im2) == RES_OK);
    TEST_ASSERT(im1 == im2);
    BaseImage *im3 = createFromFile(fName);
    TEST_ASSERT(im3 != NULL);
    delete im3;
  }
};

class Test_RW_BMP : public TestCase
{
  virtual void run()
  {
    const char *fName = "_smil_io_tmp.bmp";
    Image<UINT8> im1(3, 3);
    UINT8 pix[] = {28, 2, 3, 2, 5, 6, 3, 8, 9};
    im1 << pix;
    TEST_ASSERT(write(im1, fName) == RES_OK);
    Image<UINT8> im2;
    TEST_ASSERT(read(fName, im2) == RES_OK);
    TEST_ASSERT(im1 == im2);
    BaseImage *im3 = createFromFile(fName);
    TEST_ASSERT(im3 != NULL);
    delete im3;
  }
};

int main(void)
{
  TestSuite ts;

  ADD_TEST(ts, Test_RW_RAW);
#ifdef USE_PNG
  ADD_TEST(ts, Test_RW_PNG);
#ifdef USE_CURL
  ADD_TEST(ts, Test_Curl);
#endif // USE_CURL
#endif // USE_PNG
#ifdef USE_TIFF
  ADD_TEST(ts, Test_RW_TIFF);
#endif // USE_TIFF
  ADD_TEST(ts, Test_RW_PGM);
  ADD_TEST(ts, Test_RW_BMP);

  return ts.run();
}
