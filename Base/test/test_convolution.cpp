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
#include "DImageConvolution.hpp"

using namespace smil;

class Test_ConvolHoriz : public TestCase
{
  virtual void run()
  {
    Image<UINT8> im1(10, 5);
    Image<UINT8> im2(im1);
    Image<UINT8> im3(im1);

    UINT8 vec1[] = {
        59,  39,  15,  156, 35,  75,  132, 123, 25,  88,  66,  188, 77,
        125, 121, 45,  249, 155, 112, 252, 9,   128, 74,  99,  239, 186,
        35,  186, 11,  124, 219, 70,  163, 234, 226, 199, 54,  104, 67,
        80,  192, 133, 13,  15,  4,   135, 61,  254, 36,  173,
    };
    im1 << vec1;

    UINT8 vecTruth[] = {
        48,  44,  58,  81,  78,  86,  104, 96,  72,  68,  109, 124, 117,
        111, 107, 123, 161, 166, 167, 195, 55,  81,  98,  133, 171, 157,
        118, 103, 86,  89,  162, 141, 164, 203, 208, 166, 111, 86,  79,
        77,  157, 110, 52,  24,  42,  84,  121, 142, 129, 131,
    };
    im3 << vecTruth;

    double kern[] = {0.0545, 0.2442, 0.4026, 0.2442, 0.0545};
    horizConvolve(im1, vector<double>(kern, kern + 5), im2);
    TEST_ASSERT(im2 == im3);

    if (retVal != RES_OK) {
      im1.printSelf(1);
      im2.printSelf(1);
    }
  }
};

class Test_ConvolVert : public TestCase
{
  virtual void run()
  {
    Image<UINT8> im1(10, 5);
    Image<UINT8> im2(im1);
    Image<UINT8> im3(im1);

    UINT8 vec1[] = {
        59,  39,  15,  156, 35,  75,  132, 123, 25,  88,  66,  188, 77,
        125, 121, 45,  249, 155, 112, 252, 9,   128, 74,  99,  239, 186,
        35,  186, 11,  124, 219, 70,  163, 234, 226, 199, 54,  104, 67,
        80,  192, 133, 13,  15,  4,   135, 61,  254, 36,  173,
    };
    im1 << vec1;

    UINT8 vecTruth[] = {
        57,  97,  41,  140, 80,  73,  165, 139, 54,  147, 58,  127, 65,
        132, 135, 98,  152, 151, 60,  166, 86,  123, 89,  136, 183, 145,
        98,  158, 51,  145, 148, 108, 96,  136, 165, 170, 62,  166, 47,
        125, 187, 110, 69,  97,  99,  161, 56,  196, 44,  136,
    };
    im3 << vecTruth;

    double kern[] = {0.0545, 0.2442, 0.4026, 0.2442, 0.0545};
    vertConvolve(im1, vector<double>(kern, kern + 5), im2);
    TEST_ASSERT(im2 == im3);

    if (retVal != RES_OK)
      im2.printSelf(1);
  }
};

class Test_GaussianFilter : public TestCase
{
  virtual void run()
  {
    Image<UINT8> im1(10, 5);
    Image<UINT8> im2(im1);
    Image<UINT8> im3(im1);

    UINT8 vec1[] = {
        59,  39,  15,  156, 35,  75,  132, 123, 25,  88,  66,  188, 77,
        125, 121, 45,  249, 155, 112, 252, 9,   128, 74,  99,  239, 186,
        35,  186, 11,  124, 219, 70,  163, 234, 226, 199, 54,  104, 67,
        80,  192, 133, 13,  15,  4,   135, 61,  254, 36,  173,
    };
    im1 << vec1;

    UINT8 vecTruth[] = {
        69,  74,  81,  95,  95,  104, 124, 120, 106, 113, 82,  93,  99,
        114, 121, 124, 132, 127, 116, 128, 99,  105, 114, 135, 152, 143,
        126, 115, 105, 113, 130, 116, 115, 133, 149, 140, 118, 108, 98,
        100, 150, 118, 94,  94,  108, 118, 117, 119, 108, 108,
    };

    im3 << vecTruth;

    gaussianFilter(im1, 2, im2);

    Image<UINT8> ims(im1);
    sub(im2, im3, ims);

    TEST_ASSERT(maxVal(ims) <= 1);

    if (retVal != RES_OK) {
      im2.printSelf(1);
      im3.printSelf(1);
      ims.printSelf(1);
    }
  }
};

int main(void)
{
  TestSuite ts;

  ADD_TEST(ts, Test_ConvolHoriz);
  ADD_TEST(ts, Test_ConvolVert);
  ADD_TEST(ts, Test_GaussianFilter);

  return ts.run();
}
