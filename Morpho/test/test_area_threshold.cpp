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
#include "Base/include/DBase.h"
// #include "Base/include/private/DBlobOperations.hpp"
#include "DMorpho.h"


using namespace smil;

class Test_Area_Threshold : public TestCase
{
  virtual void run()
  {
    Image_UINT8 im1(16, 16);
    Image_UINT8 im2(16, 16);
    Image_UINT8 imRef(16, 16);

    UINT8 vec1[256] = {
   0,   0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,
   0, 255, 255, 255, 255, 255,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,
   0, 255, 255, 255, 255, 255,   0, 255, 255,   0,   0, 255, 255, 255,   0,   0,
   0, 255, 255, 255, 255, 255,   0, 255, 255,   0, 255, 255, 255, 255, 255,   0,
   0, 255, 255, 255, 255, 255,   0,   0, 255,   0, 255, 255, 255, 255, 255,   0,
   0, 255, 255, 255, 255, 255,   0,   0, 255,   0, 255, 255, 255, 255, 255,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255, 255,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0, 255, 255, 255, 255,   0,   0, 255, 255,   0,   0, 255, 255,   0,   0,
   0,   0, 255, 255, 255, 255,   0,   0, 255, 255,   0,   0,   0,   0,   0,   0,
   0,   0, 255, 255, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255, 255,   0,
   0,   0, 255, 255, 255, 255,   0,   0,   0,   0,   0,   0, 255, 255, 255,   0,
   0,   0,   0,   0,   0,   0,   0,   0, 255,   0,   0,   0, 255, 255, 255,   0,
   0,   0,   0, 255, 255, 255,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,
   0,   0, 255, 255, 255, 255, 255,   0, 255,   0,   0,   0,   0,   0,   0,   0,
   0,   0, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   };

    UINT8 vecref[256] = {
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0, 255, 255, 255,   0,   0,
   0, 255, 255, 255, 255, 255,   0,   0,   0,   0, 255, 255, 255, 255, 255,   0,
   0, 255, 255, 255, 255, 255,   0,   0,   0,   0, 255, 255, 255, 255, 255,   0,
   0, 255, 255, 255, 255, 255,   0,   0,   0,   0, 255, 255, 255, 255, 255,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 255, 255,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0, 255, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0, 255, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0, 255, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0, 255, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0,   0, 255, 255, 255, 255, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   };
    
    im1 << vec1;
    imRef << vecref;

    // remove all zones with area smaller than 10 pixels
    areaThreshold(im1, 10, true, im2);

    TEST_ASSERT(im2 == imRef);
    if (!(im2 == imRef)) {
      cout << "* im1" << endl; 
      im1.printSelf("im1");
      cout << "* imRef" << endl; 
      imRef.printSelf("imRef");
      cout << "* im2" << endl; 
      im2.printSelf("im2");
    }
  }
};

int main(void)
{
  TestSuite ts;

  ADD_TEST(ts, Test_Area_Threshold);

  return ts.run();
}
