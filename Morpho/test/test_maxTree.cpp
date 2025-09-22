/*
 * Copyright (c) 201255, Matthieu FAESSEL and ARMINES
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS AND CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#include "Core/include/DCore.h"
#include "DMorphoMaxTree.hpp"

using namespace smil;



class Test_MaxTree : public TestCase
{
  virtual void run()
  {
      typedef UINT8 dataType1;
      typedef UINT16 dataType2;
      
      typedef Image<dataType1> imType1;
      typedef Image<dataType2> imType2;
      
      imType1 im1(10,10);
      imType1 im2(im1);
      imType1 im3(im1);
      
      imType2 imLbl(im1);
      
      dataType1 vec1[] = 
      {
         50,  30,   0,   0,   0,   0,   0,   0,   0, 255,
         20,  20,   0,   0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   5,   5,   0,
          0,   0,   0,   0,   0,   0,   7,   6,   5,   0,
          0,  10,  10,  30,  10,   0,   0,   0,   6,   0,
          0,  10,  10,  30,  10,   0,   0,   0,   0,   0,
          0,  10,  30,  30,  30,  10,   0,   0,   0,   0,
          0,  10,  10,  10,  10,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0,   0,   0,  20,   0,
          5,   0,   0,   0,   0,   0,   0,   0,  20,  20,
      };
      
      im1 << vec1;
      
      ultimateOpen(im1, im2, imLbl, CrossSE());
      
      dataType1 vecTrans[] =
      {
          30, 20, 0, 0, 0, 0, 0, 0, 0, 255, 
          20, 20, 0, 0, 0, 0, 0, 0, 0, 0, 
          0, 0, 0, 0, 0, 0, 0, 5, 5, 0, 
          0, 0, 0, 0, 0, 0, 5, 5, 5, 0, 
          0, 10, 10, 20, 10, 0, 0, 0, 5, 0, 
          0, 10, 10, 20, 10, 0, 0, 0, 0, 0, 
          0, 10, 20, 20, 20, 10, 0, 0, 0, 0, 
          0, 10, 10, 10, 10, 0, 0, 0, 0, 0, 
          0, 0, 0, 0, 0, 0, 0, 0, 20, 0, 
          5, 0, 0, 0, 0, 0, 0, 0, 20, 20, 
      };
      
      dataType2 vecIndic[] =
      {
          2, 3, 0, 0, 0, 0, 0, 0, 0, 2, 
          3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 
          0, 0, 0, 0, 0, 0, 0, 4, 4, 0, 
          0, 0, 0, 0, 0, 0, 4, 4, 4, 0, 
          0, 5, 5, 4, 5, 0, 0, 0, 4, 0, 
          0, 5, 5, 4, 5, 0, 0, 0, 0, 0, 
          0, 5, 4, 4, 4, 5, 0, 0, 0, 0, 
          0, 5, 5, 5, 5, 0, 0, 0, 0, 0, 
          0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 
          2, 0, 0, 0, 0, 0, 0, 0, 3, 3, 
      };
      
      imType1 imTrans(im1);
      imType2 imIndic(imLbl);
      
      imTrans << vecTrans;
      TEST_ASSERT(im2==imTrans);
      
      imIndic << vecIndic;
      TEST_ASSERT(imLbl==imIndic);
      
      if (retVal!=RES_OK)
      {
        im2.printSelf(1);
        imLbl.printSelf(1);
      }
  }
};

class Test_MaxTree_3D : public TestCase
{
  virtual void run()
  {
      typedef UINT8 dataType1;
      typedef UINT16 dataType2;
      
      typedef Image<dataType1> imType1;
      typedef Image<dataType2> imType2;
      
      imType1 im1(5,5,4);
      imType1 im2(im1);
      imType1 im3(im1);
      
      imType2 imLbl(im1);
      
      dataType1 vec1[] = 
      {
          50,  30,   0,   0,   0,
          0,   0,   0,   0, 255,
          20,  20,   0,   0,   0,
          0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,

          0,   0,   5,   5,   0,
          0,   0,   0,   0,   0,
          0,   7,   6,   5,   0,
          0,  10,  10,  30,  10,
          0,   0,   0,   6,   0,

          0,  10,  10,  30,  10,
          0,   0,   0,   0,   0,
          0,  10,  30,  30,  30,
          10,   0,   0,   0,   0,
          0,  10,  10,  10,  10,

          0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,
          0,   0,   0,  20,   0,
          5,   0,   0,   0,   0,
          0,   0,   0,  20,  20,
      };
      
      im1 << vec1;
      
      ultimateOpen(im1, im2, imLbl, Cross3DSE());
      
      dataType1 vecTrans[] =
      {
        50,  30,   0,   0,   0,
        0,   0,   0,   0, 255,
        13,  13,   0,   0,   0,
        0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,

        0,   0,   5,   5,   0,
        0,   0,   0,   0,   0,
        0,   6,   6,   5,   0,
        0,   6,   6,  23,   6,
        0,   0,   0,   6,   0,

        0,  10,  10,  30,  10,
        0,   0,   0,   0,   0,
        0,   6,  23,  23,  23,
        10,   0,   0,   0,   0,
        0,   6,   6,   6,   6,

        0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,
        0,   0,   0,  13,   0,
        5,   0,   0,   0,   0,
        0,   0,   0,  14,  14,
      };
      
      dataType2 vecIndic[] =
      {
        2,     2,     0,     0,     0,
        0,     0,     0,     0,     2,
        2,     2,     0,     0,     0,
        0,     0,     0,     0,     0,
        0,     0,     0,     0,     0,

        0,     0,     2,     2,     0,
        0,     0,     0,     0,     0,
        0,     4,     4,     4,     0,
        0,     4,     4,     2,     4,
        0,     0,     0,     4,     0,

        0,     2,     2,     2,     2,
        0,     0,     0,     0,     0,
        0,     4,     2,     2,     2,
        2,     0,     0,     0,     0,
        0,     4,     4,     4,     4,

        0,     0,     0,     0,     0,
        0,     0,     0,     0,     0,
        0,     0,     0,     2,     0,
        2,     0,     0,     0,     0,
        0,     0,     0,     2,     2,
      };
      
      imType1 imTrans(im1);
      imType2 imIndic(imLbl);
      
      imTrans << vecTrans;
      TEST_ASSERT(im2==imTrans);
      
      if (retVal!=RES_OK)
      {
        im2.printSelf(1);
        imTrans.printSelf(1);
      }
      
      imIndic << vecIndic;
      TEST_ASSERT(imLbl==imIndic);
      
      if (retVal!=RES_OK)
      {
        imLbl.printSelf(1);
        imIndic.printSelf(1);
      }
  }
};

class Test_DeltaUO : public TestCase
{
  virtual void run()
  {
      typedef UINT8 dataType1;
      typedef UINT16 dataType2;
      
      typedef Image<dataType1> imType1;
      typedef Image<dataType2> imType2;
      
      imType1 im1(8,8);
      imType1 im2(im1);
      imType1 im3(im1);
      
      imType2 imLbl(im1);
      
      dataType1 vec1[] = 
      {
        0,   0,   0,   0,   0,   0,   0,   0,
        0,  10,  10,  10,  10,  10,  10,   0,
        0,  10,  40,  40,  40,  40,  10,   0,
        0,  10,  40,  50,  50,  50,  10,   0,
        0,  10,  40,  50,  60,  50,  10,   0,
        0,  10,  40,  50,  50,  50,  10,   0,
        0,  10,  10,  10,  10,  10,  10,   0,
        0,   0,   0,   0,   0,   0,   0,   0,
      };
      
      im1 << vec1;
      
      ultimateOpen(im1, im2, imLbl, CrossSE(), (dataType2)7, 1);
      
      dataType1 vecTrans[] =
      {
        0,   0,   0,   0,   0,   0,   0,   0,
        0,  10,  10,  10,  10,  10,  10,   0,
        0,  10,  30,  30,  30,  30,  10,   0,
        0,  10,  30,  40,  40,  40,  10,   0,
        0,  10,  30,  40,  40,  40,  10,   0,
        0,  10,  30,  40,  40,  40,  10,   0,
        0,  10,  10,  10,  10,  10,  10,   0,
        0,   0,   0,   0,   0,   0,   0,   0,
      };
      
      dataType2 vecIndic[] =
      {
        0,     0,     0,     0,     0,     0,     0,     0,
        0,     7,     7,     7,     7,     7,     7,     0,
        0,     7,     5,     5,     5,     5,     7,     0,
        0,     7,     5,     5,     5,     5,     7,     0,
        0,     7,     5,     5,     5,     5,     7,     0,
        0,     7,     5,     5,     5,     5,     7,     0,
        0,     7,     7,     7,     7,     7,     7,     0,
        0,     0,     0,     0,     0,     0,     0,     0,
      };
      
      imType1 imTrans(im1);
      imType2 imIndic(imLbl);
      
      imTrans << vecTrans;
      TEST_ASSERT(im2==imTrans);
      
      imIndic << vecIndic;
      TEST_ASSERT(imLbl==imIndic);
      
      if (retVal!=RES_OK)
      {
        im2.printSelf(1);
        imLbl.printSelf(1);
      }
  }
};

class Test_UO_MSER : public TestCase
{
  virtual void run()
  {
      typedef UINT8 dataType1;
      typedef UINT16 dataType2;
      
      typedef Image<dataType1> imType1;
      typedef Image<dataType2> imType2;
      
      imType1 im1(5,4);
      imType1 im2(im1);
      imType1 im3(im1);
      
      imType2 imLbl(im1);
      
      dataType1 vec1[] = 
      {
        5, 110, 110, 0, 140,
        5, 170, 170, 0, 0,
        5, 170, 170, 0, 255,
         0,   0,   0, 0,  9,
      };
      
      im1 << vec1;
      
      ultimateOpenMSER(im1, im2, imLbl, CrossSE(),(dataType2)4, 0,3);
      
      dataType1 vecTrans[] =
      {
        0, 103, 103, 0, 130,
        0, 103, 103, 0, 0,
        0, 103, 103, 0, 241,
        0, 0, 0, 0, 0,
      };
      
      dataType2 vecIndic[] =
      {
        0, 4, 4, 0, 2,
        0, 4, 4, 0, 0,
        0, 4, 4, 0, 2,
        0, 0, 0, 0, 0,
      };
      
      imType1 imTrans(im1);
      imType2 imIndic(imLbl);
      
      imTrans << vecTrans;
      TEST_ASSERT(im2==imTrans);
      
      if (retVal!=RES_OK)
      {
        im2.printSelf(1);
        imTrans.printSelf(1);
      }
      
      imIndic << vecIndic;
      TEST_ASSERT(imLbl==imIndic);
      
      if (retVal!=RES_OK)
      {
        imLbl.printSelf(1);
        imIndic.printSelf(1);
      }
  }
};

class Test_AttributeOpening : public TestCase
{
  virtual void run()
  {
      typedef UINT8 dataType;
      typedef Image<dataType> imType;
      
      imType im1(8,8);
      imType im2(im1);
      
      dataType vec1[] = 
      {
        1,   1,   1,   1,   1,   1,   1,   1,
        1,  10,  10,  10,  10,  10,  10,   1,
        1,  10,  40,  40,  40,  40,  10,   1,
        1,  10,  40,  50,  50,  50,  10,   1,
        1,  10,  40,  50,  60,  50,  10,   1,
        1,  10,  40,  50,  50,  50,  10,   1,
        1,  10,  10,  10,  10,  10,  10,   1,
        1,   1,   1,   1,   1,   1,   1,   1,
      };
      
      im1 << vec1;
      
      areaOpen(im1, 10, im2);
      
      dataType vec2[] =
      {
        1,   1,   1,   1,   1,   1,   1,   1,
        1,  10,  10,  10,  10,  10,  10,   1,
        1,  10,  40,  40,  40,  40,  10,   1,
        1,  10,  40,  40,  40,  40,  10,   1,
        1,  10,  40,  40,  40,  40,  10,   1,
        1,  10,  40,  40,  40,  40,  10,   1,
        1,  10,  10,  10,  10,  10,  10,   1,
        1,   1,   1,   1,   1,   1,   1,   1,
      };
      
      imType imTruth(im1);
      
      imTruth << vec2;
      TEST_ASSERT(im2==imTruth);
      
      if (retVal!=RES_OK)
      {
        im2.printSelf(1);
        imTruth.printSelf(1);
      }
  }
};


int main()
{
      TestSuite ts;
      ADD_TEST(ts, Test_MaxTree);
      ADD_TEST(ts, Test_MaxTree_3D);
      ADD_TEST(ts, Test_DeltaUO);
      ADD_TEST(ts, Test_UO_MSER);
      ADD_TEST(ts, Test_AttributeOpening);
      
      return ts.run();
}

