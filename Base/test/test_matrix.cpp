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
#include "DImageMatrix.hpp"
#include "DTest.h"

using namespace smil;

class Test_Transpose : public TestCase
{
  virtual void run()
  {
      UINT8 vec1[20] 	= {   
	1, 2, 3, 4, 
	5, 6, 7, 8,
	9, 10, 11, 12,
	13, 14,  15,  16,
	17,  18,  19,  20 
      };
      UINT8 vecTrans[20] = {
	1, 5, 9, 13, 17, 
	2, 6, 10, 14, 18, 
	3, 7, 11, 15, 19, 
	4, 8, 12, 16, 20
      };
      
      
      Image_UINT8 im1(4,5);
      Image_UINT8 im2(im1);
      Image_UINT8 imTrans(5,4);
      
      im1 << vec1;
      
      matTrans(im1, im2);
      imTrans << vecTrans;
      TEST_ASSERT(im2==imTrans);
      
//       im2.printSelf(1);
  }
};

class Test_Multiply : public TestCase
{
  virtual void run()
  {
      UINT8 vec1[8] = 
      {   
	1, 2, 
	3, 4, 
	5, 6, 
	7, 8
      };
      
      UINT8 vec2[6] = 
      {   
	1, 2, 3, 
	4, 5, 6, 
      };
      
      UINT8 vecMul[12] = {
	9, 12, 15, 
	19, 26, 33, 
	29, 40, 51, 
	39, 54, 69, 
      };
      
      
      Image_UINT8 im1(2,4);
      Image_UINT8 im2(3,2);
      Image_UINT8 im3;
      Image_UINT8 imMul(3,4);
      
      im1 << vec1;
      im2 << vec2;
      
      matMul(im1, im2, im3);
      imMul << vecMul;
      TEST_ASSERT(im3==imMul);
      
//       im3.printSelf(1);
  }
};

int main(int argc, char *argv[])
{
      TestSuite ts;

      ADD_TEST(ts, Test_Transpose);
      ADD_TEST(ts, Test_Multiply);
      
      return ts.run();
}

