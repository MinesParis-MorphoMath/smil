/*
 * Smil
 * Copyright (c) 2010 Matthieu Faessel
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
#include "DImageArith.hpp"
#include "DTest.h"

using namespace smil;

class Test_Fill : public TestCase
{
  virtual void run()
  {
      UINT8 vec1[20] 	= {   1, 2, 3,   4, 5, 6,   7,   8, 9,  10, 11, 12,  13, 14,  15,  16,  17,  18,  19,  20 };
      UINT8 vecFill[20] = { 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127 };
      
      
      Image_UINT8 im1(4,5);
      Image_UINT8 imTruth(4,5);
      
      im1 << vec1;
      imTruth << vecFill;
      
      TEST_ASSERT(fill(im1, UINT8(127))==RES_OK);
      
      TEST_ASSERT(equ(im1, imTruth));
  }
};

class Test_Equal : public TestCase
{
  virtual void run()
  {
      UINT8 vec1[20] 	= {   1, 2, 3,   4, 5, 6,   7,   8, 9,  10, 11, 12,  13, 14,  15,  16,  17,  18,  19,  20 };
      UINT8 vec2[20] 	= {   1, 4, 5,   4, 9, 5,   7,   8, 2,  10, 13, 20,  13, 15,  15,  16,  17,  18,  19,  20 };
      UINT8 vecEqu[20] 	= { 255, 0, 0, 255, 0, 0, 255, 255, 0, 255,  0,  0, 255,  0, 255, 255, 255, 255, 255, 255 };
      
      
      Image_UINT8 im1(4,5);
      Image_UINT8 im2(im1);
      Image_UINT8 im3(im1);
      Image_UINT8 imTruth(im1);
      
      UINT8 *pix;

      im1 << vec1;
      im2 << vec2;
      
      equ(im1, im2, im3);
      pix = im3.getPixels();
      for (UINT i=0;i<im1.getPixelCount();i++)
	TEST_ASSERT(pix[i]==vecEqu[i]);
      
      imTruth << vecEqu;
      TEST_ASSERT(im3==imTruth);
      
      TEST_ASSERT(!(im1==im2));
  }
};

int main(int argc, char *argv[])
{
      TestSuite ts;

      ADD_TEST(ts, Test_Fill);
      ADD_TEST(ts, Test_Equal);
      
      return ts.run();
}

