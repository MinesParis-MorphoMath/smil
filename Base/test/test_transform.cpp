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
#include "Smil-build.h"

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

    /* previous expected result - wrong */
    /*
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
    /*
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
    */

    UINT8 vecref[100] = {
        1,   1,   1,   2,   2,   3,   3,   4,   4,   5,
        2,   3,   4,   5,   5,   8,  14,  19,  25,  30,
        4,   5,   7,   7,   8,  14,  24,  35,  45,  56,
       38,  53,  68,  54,  30,  22,  29,  40,  62,  83,
       82, 115, 149, 114,  58,  30,  32,  42,  76, 111,
       82, 117, 152, 125,  79,  52,  45,  49,  87, 124,
       39,  58,  77,  86,  92,  86,  68,  62,  93, 124,
        8,  24,  40,  70, 105, 113,  96,  86,  99, 112,
       16,  66, 117, 124, 116, 122, 140, 144, 106,  67,
       25, 109, 193, 178, 128, 130, 184, 202, 112,  23,
    };

    im1 << vec1;
    imRef << vecref;

    resize(im1, 10, 10, im2, "bilinear");
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

class Test_Resize_Closest : public TestCase
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
    
    /* new result with new functions - looks better but not the best yet */
    /*
    UINT8 vecref[100] = {
        1,   2,   2,   3,   3,   4,   4,   5,   5,   5,
        5,   8,   8,  10,  10,  36,  36,  63,  63,  63,
        5,   8,   8,  10,  10,  36,  36,  63,  63,  63,
      105, 200, 200,  36,  36,  33,  33, 125, 125, 125, 
      105, 200, 200,  36,  36,  33,  33, 125, 125, 125,
        6,  23,  23, 125, 125,  66,  66, 124, 124, 124,
        6,  23,  23, 125, 125,  66,  66, 124, 124, 124,
       25, 215, 215, 104, 104, 225, 225,  23,  23,  23, 
       25, 215, 215, 104, 104, 225, 225,  23,  23,  23,
       25, 215, 215, 104, 104, 225, 225,  23,  23,  23,
    };
    */

    UINT8 vecref[100] = {
        1,   1,   2,   2,   3,   3,   4,   4,   5,   5, 
        1,   1,   2,   2,   3,   3,   4,   4,   5,   5,
        5,   5,   8,   8,  10,  10,  36,  36,  63,  63,
        5,   5,   8,   8,  10,  10,  36,  36,  63,  63,
      105, 105, 200, 200,  36,  36,  33,  33, 125, 125,
      105, 105, 200, 200,  36,  36,  33,  33, 125, 125,
        6,   6,  23,  23, 125, 125,  66,  66, 124, 124,
        6,   6,  23,  23, 125, 125,  66,  66, 124, 124,
       25,  25, 215, 215, 104, 104, 225, 225,  23,  23,
       25,  25, 215, 215, 104, 104, 225, 225,  23,  23,
    };
    im1 << vec1;
    imRef << vecref;

    resize(im1, 10, 10, im2, "closest");
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

class Test_Scale : public TestCase
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
    /*
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
    */

    UINT8 vecref[100] = { 
        1,   1,   1,   2,   2,   3,   3,   4,   4,   5,
        2,   3,   4,   5,   5,   8,  14,  19,  25,  30,
        4,   5,   7,   7,   8,  14,  24,  35,  45,  56,
       38,  53,  68,  54,  30,  22,  29,  40,  62,  83,
       82, 115, 149, 114,  58,  30,  32,  42,  76, 111,
       82, 117, 152, 125,  79,  52,  45,  49,  87, 124,
       39,  58,  77,  86,  92,  86,  68,  62,  93, 124,
        8,  24,  40,  70, 105, 113,  96,  86,  99, 112,
       16,  66, 117, 124, 116, 122, 140, 144, 106,  67,
       25, 109, 193, 178, 128, 130, 184, 202, 112,  23,
    };

    im1 << vec1;
    imRef << vecref;

    scale(im1, 2, 2, im2);
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

class Test_Rotate090 : public TestCase
{
  virtual void run()
  {
    Image_UINT8 imIn(10,10);
    Image_UINT8 imOut(imIn);
    Image_UINT8 imRef(imIn);

    UINT8 vIn[100] = {
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 2, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 3, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 4, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 5, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    UINT8 vRef[100] = {
      0,   0,   0,   0,   0,   0,   0,   0,   0,   1,
      0,   0,   0,   0,   0,   0,   0,   0,   2,   0,
      0,   0,   0,   0,   0,   0,   0,   3,   0,   0,
      0,   0,   0,   0,   0,   0,   4,   0,   0,   0,
      0,   0,   0,   0,   0,   5,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    };

    imIn << vIn;
    imRef << vRef;

    rotateX90(imIn, 1, imOut);

    TEST_ASSERT(imOut == imRef);
    if (retVal != RES_OK) {
      cout << "* imRef" << endl; 
      imRef.printSelf("imRef");
      cout << "* imOut" << endl; 
      imOut.printSelf("imOut");
    }
  }
};

class Test_Rotate180 : public TestCase
{
  virtual void run()
  {
    Image_UINT8 imIn(10,10);
    Image_UINT8 imOut(imIn);
    Image_UINT8 imRef(imIn);

    UINT8 vIn[100] = {
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 2, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 3, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 4, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 5, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    UINT8 vRef[100] = {
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   5,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   4,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   3,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   2,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   1,
    };

    imIn << vIn;
    imRef << vRef;

    rotateX90(imIn, 2, imOut);

    TEST_ASSERT(imOut == imRef);
    if (retVal != RES_OK) {
      cout << "* imRef" << endl; 
      imRef.printSelf("imRef");
      cout << "* imOut" << endl; 
      imOut.printSelf("imOut");
    }
  }
};

class Test_Rotate270 : public TestCase
{
  virtual void run()
  {
    Image_UINT8 imIn(10,10);
    Image_UINT8 imOut(imIn);
    Image_UINT8 imRef(imIn);

    UINT8 vIn[100] = {
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 2, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 3, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 4, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 5, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    UINT8 vRef[100] = {
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
       0,   0,   0,   0,   5,   0,   0,   0,   0,   0,
       0,   0,   0,   4,   0,   0,   0,   0,   0,   0,
       0,   0,   3,   0,   0,   0,   0,   0,   0,   0,
       0,   2,   0,   0,   0,   0,   0,   0,   0,   0,
       1,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    };

    imIn << vIn;
    imRef << vRef;

    rotateX90(imIn, 3, imOut);

    TEST_ASSERT(imOut == imRef);
    if (retVal != RES_OK) {
      cout << "* imRef" << endl; 
      imRef.printSelf("imRef");
      cout << "* imOut" << endl; 
      imOut.printSelf("imOut");
    }
  }
};

class Test_Rotate360 : public TestCase
{
  virtual void run()
  {
    Image_UINT8 imIn(10,10);
    Image_UINT8 imOut(imIn);
    Image_UINT8 imRef(imIn);

    UINT8 vIn[100] = {
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 2, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 3, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 4, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 5, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    UINT8 vRef[100] = {
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 2, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 3, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 4, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 5, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    imIn << vIn;
    imRef << vRef;

    rotateX90(imIn, 4, imOut);

    TEST_ASSERT(imOut == imRef);
    if (retVal != RES_OK) {
      cout << "* imRef" << endl; 
      imRef.printSelf("imRef");
      cout << "* imOut" << endl; 
      imOut.printSelf("imOut");
    }
  }
};


class Test_HorizFlip : public TestCase
{
  virtual void run()
  {
    Image_UINT8 imIn(10,10);
    Image_UINT8 imOut(imIn);
    Image_UINT8 imRef(imIn);

    UINT8 vIn[100] = {
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 2, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 3, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 4, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 5, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    UINT8 vRef[100] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
      0, 0, 0, 0, 0, 0, 0, 0, 2, 0,
      0, 0, 0, 0, 0, 0, 0, 3, 0, 0,
      0, 0, 0, 0, 0, 0, 4, 0, 0, 0,
      0, 0, 0, 0, 0, 5, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    imIn << vIn;
    imRef << vRef;

    horizFlip(imIn, imOut);

    TEST_ASSERT(imOut == imRef);
    if (retVal != RES_OK) {
      cout << "* imRef" << endl; 
      imRef.printSelf("imRef");
      cout << "* imOut" << endl; 
      imOut.printSelf("imOut");
    }
  }
};

class Test_VertFlip : public TestCase
{
  virtual void run()
  {
    Image_UINT8 imIn(10,10);
    Image_UINT8 imOut(imIn);
    Image_UINT8 imRef(imIn);

    UINT8 vIn[100] = {
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 2, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 3, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 4, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 5, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    UINT8 vRef[100] = {
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 5, 0, 0, 0, 0, 0,
      0, 0, 0, 4, 0, 0, 0, 0, 0, 0,
      0, 0, 3, 0, 0, 0, 0, 0, 0, 0,
      0, 2, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };

    imIn << vIn;
    imRef << vRef;

    vertFlip(imIn, imOut);

    TEST_ASSERT(imOut == imRef);
    if (retVal != RES_OK) {
      cout << "* imRef" << endl; 
      imRef.printSelf("imRef");
      cout << "* imOut" << endl; 
      imOut.printSelf("imOut");
    }
  }
};

int main(void)
{
  TestSuite ts;

  ADD_TEST(ts, Test_Resize);
  ADD_TEST(ts, Test_Resize_Closest);
  ADD_TEST(ts, Test_Scale);

  ADD_TEST(ts, Test_Rotate090);
  ADD_TEST(ts, Test_Rotate180);
  ADD_TEST(ts, Test_Rotate270);  
  ADD_TEST(ts, Test_Rotate360);  

  ADD_TEST(ts, Test_HorizFlip);  
  ADD_TEST(ts, Test_VertFlip);  

  return ts.run();
}
