/*
 * Smil
 * Copyright (c) 2011 Matthieu Faessel
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

#include "DImage.h"
#include "DImageConvolution.hpp"
#include "DTest.h"

using namespace smil;


class Test_Convol : public TestCase
{
  virtual void run()
  {
      Image_UINT8 im1(10,5);
      Image_UINT8 im2(im1);
      Image_UINT8 im3(im1);
      
      UINT8 vec1[] = 
      {
	 59,  39,  15, 156,  35,  75, 132, 123,  25,  88,
         66, 188,  77, 125, 121,  45, 249, 155, 112, 252,
	  9, 128,  74,  99, 239, 186,  35, 186,  11, 124,
	219,  70, 163, 234, 226, 199,  54, 104,  67,  80,
	192, 133,  13,  15,   4, 135,  61, 254,  36, 173,
      };
      im1 << vec1;
      
      UINT8 vecTruth1[] = 
      {
	  34,  42,  58,  81,  78,  86, 104,  96,  68,  48,
	  76, 117, 117, 111, 107, 123, 161, 166, 158, 137,
	  38,  77,  98, 133, 171, 157, 118, 103,  82,  62,
	114, 134, 164, 203, 208, 166, 111,  86,  74,  54,
	110, 104,  52,  24,  42,  84, 121, 142, 122,  92,
      };
      im3 << vecTruth1;
      
      float kern[] = { 0.0545, 0.2442, 0.4026, 0.2442, 0.0545 };
      horizConvolve(im1, kern, 2, im2);
      TEST_ASSERT(im2==im3);
      
      if(retVal!=RES_OK)
         im2.printSelf(1);
      
      UINT8 vecTruth2[] = 
      {
	  40,  68,  28,  98,  56,  51, 115,  97,  38, 103,
	  55, 120,  61, 125, 127,  92, 143, 143,  57, 157,
	  86, 123,  89, 136, 183, 145,  98, 158,  51, 145,
	140, 102,  91, 128, 156, 160,  58, 157,  44, 118,
	131,  77,  49,  68,  69, 113,  39, 137,  31,  95,
      };
      im3 << vecTruth2;
      fill(im2, UINT8(0));
      vertConvolve(im1, kern, 2, im2);
      TEST_ASSERT(im2==im3);
      
      if(retVal!=RES_OK)
         im2.printSelf(1);
  }
};

class Test_GaussianFilter : public TestCase
{
  virtual void run()
  {
      Image_UINT8 im1(10,5);
      Image_UINT8 im2(im1);
      Image_UINT8 im3(im1);
      
      UINT8 vec1[] = 
      {
	 59,  39,  15, 156,  35,  75, 132, 123,  25,  88,
         66, 188,  77, 125, 121,  45, 249, 155, 112, 252,
	  9, 128,  74,  99, 239, 186,  35, 186,  11, 124,
	219,  70, 163, 234, 226, 199,  54, 104,  67,  80,
	192, 133,  13,  15,   4, 135,  61, 254,  36, 173,
      };
      im1 << vec1;
      
      UINT8 vecTruth[] = 
      {
	34,  49,  57,  66,  66,  72,  87,  84,  70,  56,
	54,  83,  94, 107, 115, 117, 124, 119, 104,  84,
	69,  99, 113, 135, 152, 143, 126, 115,  99,  79,
	86, 104, 108, 126, 141, 132, 111, 103,  88,  66,
	74,  78,  66,  66,  76,  83,  82,  83,  71,  53,
      };
      im3 << vecTruth;
      
      gaussianFilter(im1, 2, im2);
      TEST_ASSERT(im2==im3);
      
      if(retVal!=RES_OK)
         im2.printSelf(1);
  }
};

int main(int argc, char *argv[])
{
      TestSuite ts;

      ADD_TEST(ts, Test_Convol);
      ADD_TEST(ts, Test_GaussianFilter);
      
      return ts.run();
}

