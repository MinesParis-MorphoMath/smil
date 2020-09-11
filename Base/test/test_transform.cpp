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
#include "DImageTransform.hpp"

using namespace smil;

class Test_Resize : public TestCase
{
  virtual void run()
  {
    Image_UINT8 im1(5, 5);
    Image_UINT8 im2(10, 10);
    Image_UINT8 imRef(10, 10);

    UINT8 vec1[25] = {
        1,   2,   3,   4,   5,
        5,   8,  10,  36,  63,
      105, 200,  36,  33, 125,
        6,  23, 125,  66, 124,
       25, 215, 104, 225,  23};

    /* previous expected result - wrong
    UINT8 vecref[100] = {
        1,   1,   1,   2,   2,   3,   3,   3,   4,   4,
        2,   3,   4,   4,   5,   5,  10,  14,  19,  23,
        4,   5,   6,   7,   7,   8,  17,  25,  33,  42,
       25,  33,  42,  40,  27,  15,  23,  31,  43,  59,
       65,  88, 111, 103,  64,  25,  29,  32,  47,  73,
      105, 143, 181, 167, 101,  36,  34,  33,  51,  88,
       65,  90, 116, 117,  94,  71,  61,  51,  61,  93,
       25,  38,  51,  68,  87, 107,  88,  68,  72,  98,
        9,  30,  51,  73,  97, 120, 111, 102,  99, 101,
       17,  65, 114, 133, 122, 112, 132, 151, 141, 102,
    };
    */
    
    /* new result with new functions - looks better but not the best yet */
    UINT8 vecref[100] = {
        1,   1,   2,   2,   3,   3,   4,   4,   5,   5,
        3,   4,   5,   5,   6,  13,  20,  27,  34,  34,
        5,   6,   8,   9,  10,  23,  36,  49,  63,  63,
       55,  79, 104,  63,  23,  28,  34,  64,  94,  94,
      105, 152, 200, 118,  36,  34,  33,  79, 125, 125,
       55,  83, 111,  96,  80,  65,  49,  87, 124, 124,
        6,  14,  23,  74, 125,  95,  66,  95, 124, 124,
       15,  67, 119, 116, 114, 130, 145, 109,  73,  73,
       25, 120, 215, 159, 104, 164, 225, 124,  23,  23,
       25, 120, 215, 159, 104, 164, 225, 124,  23,  23,
    };

    im1 << vec1;

    resize(im1, 10, 10, im2);
    imRef << vecref;
    // im2.printSelf(1);
    TEST_ASSERT(im2 == imRef);
    if (!(im2 == imRef)) {
      cout << "* im1" << endl; 
      im1.printSelf("im1");
      cout << "* imRef" << endl; 
      imRef.printSelf("imRef");
      cout << "* im2" << endl; 
      im2.printSelf("im2");
    }

    scale(im1, 2, 2, im2);
    imRef << vecref;
    // im2.printSelf(1);
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

  ADD_TEST(ts, Test_Resize);

  return ts.run();
}
