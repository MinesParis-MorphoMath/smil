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

#include "Core/include/DCore.h"
#include "DImageArith.hpp"

using namespace smil;

class Test_Cast : public TestCase
{
  virtual void run()
  {
    INT16 vec1[20] = {-32768, 2,  -12532, 32767, -5, -3024L, 2042L, -8, 9,  10,
                      -11,    12, 13,     14,    15, 16,     17,    18, 19, 20};
    UINT16 vecTruth[20] = {
        0,     32770, 20236, 65535, 32763, 29744, 34810, 32760, 32777, 32778,
        32757, 32780, 32781, 32782, 32783, 32784, 32785, 32786, 32787, 32788,
    };

    Image<INT16> im1(4, 5);
    Image<UINT16> im2(im1);
    Image<UINT16> imTruth(im1);

    im1 << vec1;
    imTruth << vecTruth;

    TEST_ASSERT(cast(im1, im2) == RES_OK);
    TEST_ASSERT(equ(im2, imTruth));

    if (retVal != RES_OK) {
      im2.printSelf(1);
      imTruth.printSelf(1);
    }
  }
};

class Test_Fill : public TestCase
{
  virtual void run()
  {
    UINT8 vec1[20]    = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                      11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    UINT8 vecFill[20] = {127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
                         127, 127, 127, 127, 127, 127, 127, 127, 127, 127};

    Image<UINT8> im1(4, 5);
    Image<UINT8> imTruth(4, 5);

    im1 << vec1;
    imTruth << vecFill;

    TEST_ASSERT(fill(im1, UINT8(127)) == RES_OK);

    TEST_ASSERT(equ(im1, imTruth));
  }
};

class Test_Equal : public TestCase
{
  virtual void run()
  {
    UINT8 vec1[20]   = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                      11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    UINT8 vec2[20]   = {1,  4,  5,  4,  9,  5,  7,  8,  2,  10,
                      13, 20, 13, 15, 15, 16, 17, 18, 19, 20};
    UINT8 vecEqu[20] = {255, 0, 0,   255, 0,   0,   255, 255, 0,   255,
                        0,   0, 255, 0,   255, 255, 255, 255, 255, 255};

    Image<UINT8> im1(4, 5);
    Image<UINT8> im2(im1);
    Image<UINT8> im3(im1);
    Image<UINT8> imTruth(im1);

    UINT8 *pix;

    im1 << vec1;
    im2 << vec2;

    equ(im1, im2, im3);
    pix = im3.getPixels();
    for (UINT i = 0; i < im1.getPixelCount(); i++)
      TEST_ASSERT(pix[i] == vecEqu[i]);

    imTruth << vecEqu;
    TEST_ASSERT(im3 == imTruth);

    TEST_ASSERT(!(im1 == im2));
  }
};

class Test_Bit : public TestCase
{
  virtual void run()
  {
    UINT8 vec1[20] = {
        97,  223, 13,  127, 229, 210, 57, 114, 248, 104,
        182, 67,  194, 251, 31,  69,  92, 79,  250, 114,
    };
    UINT8 vec2[20] = {
        229, 131, 91, 79,  226, 139, 162, 39,  226, 59,
        230, 230, 86, 100, 176, 158, 122, 132, 213, 219,
    };

    Image<UINT8> im1(4, 5);
    Image<UINT8> im2(im1);
    Image<UINT8> im3(im1);
    Image<UINT8> imTruth(im1);

    im1 << vec1;
    im2 << vec2;

    UINT8 vecAnd[20] = {
        97,  131, 9,  79, 224, 130, 32, 34, 224, 40,
        166, 66,  66, 96, 16,  4,   88, 4,  208, 82,
    };

    bitAnd(im1, im2, im3);
    imTruth << vecAnd;

    TEST_ASSERT(im3 == imTruth);
    if (retVal != RES_OK) {
      im3.printSelf(1);
      imTruth.printSelf(1);
    }
  }
};

class Test_ApplyLookup : public TestCase
{
  virtual void run()
  {
    UINT8 vec1[20] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                      11, 12, 13, 14, 15, 16, 17, 18, 19, 20};

    map<UINT8, UINT8> lut;
    lut[2]  = 5;
    lut[6]  = 255;
    lut[11] = 7;

    UINT8 vec2[20] = {
        0, 5, 0, 0, 0, 255, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };

    Image<UINT8> im1(4, 5);
    Image<UINT8> im2(im1);
    Image<UINT8> imTruth(4, 5);

    im1 << vec1;
    imTruth << vec2;

    TEST_ASSERT(applyLookup(im1, lut, im2) == RES_OK);
    TEST_ASSERT(equ(im2, imTruth));

    if (retVal != RES_OK) {
      im2.printSelf(1);
      imTruth.printSelf(1);
    }
  }
};

int main(void)
{
  TestSuite ts;

  ADD_TEST(ts, Test_Cast);
  ADD_TEST(ts, Test_Fill);
  ADD_TEST(ts, Test_Equal);
  ADD_TEST(ts, Test_Bit);
  ADD_TEST(ts, Test_ApplyLookup);

  return ts.run();
}
