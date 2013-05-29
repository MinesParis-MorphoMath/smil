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
#include "DImageTransform.hpp"
#include "DTest.h"

using namespace smil;


class Test_Resize : public TestCase
{
  virtual void run()
  {

      Image_UINT8 im1(5,5);
      Image_UINT8 im2(10,10);
      Image_UINT8 imRef(10,10);
      
      UINT8 vec1[25] = 
      {   
	1, 2, 3, 4, 5,
	5, 8, 10, 36, 63,
	105, 200, 36, 33, 125,
	6, 23, 125, 66, 124,
	25, 215, 104, 225, 23
      };
      
      UINT8 vecref[100] = 
      {   
	1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 
	2, 3, 4, 4, 5, 5, 10, 14, 19, 23, 
	4, 5, 6, 7, 7, 8, 17, 25, 33, 42, 
	25, 33, 42, 40, 27, 15, 23, 31, 43, 59, 
	65, 88, 111, 103, 64, 25, 29, 32, 47, 73, 
	105, 143, 181, 167, 101, 36, 34, 33, 51, 88, 
	65, 90, 116, 117, 94, 71, 61, 51, 61, 93, 
	25, 38, 51, 68, 87, 107, 88, 68, 72, 98, 
	9, 30, 51, 73, 97, 120, 111, 102, 99, 101, 
	17, 65, 114, 133, 122, 112, 132, 151, 141, 102, 
      };
      
      im1 << vec1;
      
      resize(im1, 10, 10, im2);
      imRef << vecref;
//       im2.printSelf(1);
      TEST_ASSERT(im2==imRef);
      
      scale(im1, 2, 2, im2);
      imRef << vecref;
//       im2.printSelf(1);
      TEST_ASSERT(im2==imRef);
      
      
  }
};

int main(int argc, char *argv[])
{
      TestSuite ts;

      ADD_TEST(ts, Test_Resize);
      
      return ts.run();
}

