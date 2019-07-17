/*
 * Copyright (c) 2011-2015, Matthieu FAESSEL and ARMINES
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
#include "DMorpho.h"
#include "DMorphoWatershed.hpp"
#include "DMorphoWatershedExtinction.hpp"

using namespace smil;

class Test_Basins : public TestCase
{
  virtual void run()
  {
      typedef UINT8 dtType;
      typedef UINT16 dtType2;
      
      dtType vecIn[] = { 
        2, 2, 2, 2, 2, 2,
         7, 7, 7, 7, 7, 7,
        2, 7, 5, 6, 2, 2,
         2, 6, 5, 6, 2, 2,
        2, 2, 6, 4, 3, 2,
         2, 2, 3, 4, 2, 2,
        2, 2, 2, 2, 4, 2
      };
      
      dtType2 vecMark[] = { 
        1, 1, 1, 1, 1, 1,
         0, 0, 0, 0, 0, 0,
        2, 0, 0, 0, 3, 3,
         2, 0, 0, 0, 3, 3,
        2, 2, 0, 0, 0, 3,
         2, 2, 0, 0, 3, 3,
        2, 2, 2, 2, 0, 3
      };
      
      Image<dtType> imIn(6,7);
      Image<dtType2> imMark(imIn);
      Image<dtType2> imLbl(imIn);

      imIn << vecIn;
      imMark << vecMark;
      
      StrElt se = hSE();
      
      basins(imIn, imMark, imLbl, se);
      
      dtType2 vecLblTruth[] = { 
       1,       1,       1,       1,       1,       1,
           1,       1,       1,       1,       1,       1,
       2,       2,       3,       3,       3,       3,
           2,       2,       3,       3,       3,       3,
       2,       2,       2,       3,       3,       3,
           2,       2,       2,       3,       3,       3,
       2,       2,       2,       2,       3,       3,
/* Previous truth XXX JOE
       1,       1,       1,       1,       1,       1,
           1,       1,       1,       1,       1,       1,
       2,       3,       3,       3,       3,       3,
           2,       3,       3,       3,       3,       3,
       2,       2,       3,       3,       3,       3,
           2,       2,       2,       3,       3,       3,
       2,       2,       2,       2,       3,       3,
*/
      };
      
      Image<dtType2> imLblTruth(imIn);
      
      imLblTruth << vecLblTruth;
      
      TEST_ASSERT(imLbl==imLblTruth);
      
      if (retVal!=RES_OK)
      {
        imLbl.printSelf(1, true);
        imLblTruth.printSelf(1, true);
      }
  }
};


class Test_Basins_Plateaus : public TestCase
{
  virtual void run()
  {
      typedef UINT8 dtType;
      typedef UINT16 dtType2;
      
      dtType vecIn[] = { 
        0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0
      };
      
      dtType2 vecMark[] = { 
        1, 1, 1, 1, 1, 1,
         0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
         3, 3, 3, 3, 3, 3,
        0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0,
        2, 2, 2, 2, 2, 2
      };
      
      Image<dtType> imIn(6,7);
      Image<dtType2> imMark(imIn);
      Image<dtType2> imLbl(imIn);

      imIn << vecIn;
      imMark << vecMark;
      
      StrElt se = hSE();
      
      basins(imIn, imMark, imLbl, se);
      
      dtType2 vecLblTruth[] = { 
        1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1,
        3, 3, 3, 3, 3, 3,
         3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3,
         2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2
      };
      
      Image<dtType2> imLblTruth(imIn);
      
      imLblTruth << vecLblTruth;
      
      TEST_ASSERT(imLbl==imLblTruth);
      
      if (retVal!=RES_OK)
      {
        imLbl.printSelf(1, true);
        imLblTruth.printSelf(1, true);
      }
  }
};


class Test_ProcessWatershedHierarchicalQueue : public TestCase
{
  virtual void run()
  {
      UINT8 vecIn[] = { 
        2, 2, 2, 2, 2, 2,
        7, 7, 7, 7, 7, 7,
        2, 7, 5, 6, 2, 2,
        2, 6, 5, 6, 2, 2,
        2, 2, 6, 4, 3, 2,
        2, 2, 3, 4, 2, 2,
        2, 2, 2, 2, 4, 2
      };
      
      UINT8 vecLbl[] = { 
        1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0,
        2, 0, 0, 0, 3, 3,
        2, 0, 0, 0, 3, 3,
        2, 2, 0, 0, 0, 3,
        2, 2, 0, 0, 3, 3,
        2, 2, 2, 2, 0, 3
      };
      
      Image_UINT8 imIn(6,7);
      Image_UINT8 imLbl(imIn);
      Image_UINT8 imWS(imIn);

      imIn << vecIn;
      imLbl << vecLbl;
      
      HierarchicalQueue<UINT8> pq;
      StrElt se = hSE();
      
      WatershedFlooding<UINT8,UINT8> flooding;
      flooding.initialize(imIn, imLbl, imWS, se);
      flooding.processImage(imIn, imLbl, se);

      UINT8 vecLblTruth[] = { 
    1,    1,    1,    1,    1,    1,
       1,    1,    1,    1,    1,    1,
    2,    3,    3,    3,    3,    3,
       2,    3,    3,    3,    3,    3,
    2,    2,    3,    3,    3,    3,
       2,    2,    2,    3,    3,    3,
    2,    2,    2,    2,    3,    3,
/* previous truth XXX JOE
        1,    1,    1,    1,    1,    1,
          1,    1,    1,    1,    1,    1,
        2,    2,    3,    3,    3,    3,
          2,    2,    3,    3,    3,    3,
        2,    2,    2,    3,    3,    3,
          2,    2,    2,    3,    3,    3,
        2,    2,    2,    2,    3,    3,
*/
      };
      
      UINT8 vecStatusTruth[] = { 
        1,    1,    1,    1,    1,    1,
        255,  255,  255,  255,  255,  255,
        1,  255,    1,    1,    1,    1,
          1,  255,    1,    1,    1,    1,
        1,    1,  255,  255,    1,    1,
          1,    1,    1,  255,    1,    1,
        1,    1,    1,    1,  255,    1,
      };
      
      Image_UINT8 imLblTruth(imIn);
      Image_UINT8 imStatusTruth(imIn);
      
      imLblTruth << vecLblTruth;
      imStatusTruth << vecStatusTruth;
      
      TEST_ASSERT(imLbl==imLblTruth);
      
      if (retVal!=RES_OK)
      {
        imLbl.printSelf(1, true);
        imLblTruth.printSelf(1, true);
      }
      
      TEST_ASSERT(*(flooding.imgWS)==imStatusTruth);
      
      if (retVal!=RES_OK)
      {
        flooding.imgWS->printSelf(1, true);
        imStatusTruth.printSelf(1, true);
      }
  }
};



template <class dtType=UINT8>
class Test_Watershed : public TestCase
{
  virtual void run()
  {
      dtType vecIn[] = { 
        2, 2, 2, 2, 2, 2,
         7, 7, 7, 7, 7, 7,
        2, 7, 5, 6, 2, 2,
         2, 6, 5, 6, 2, 2,
        2, 2, 6, 4, 3, 2,
         2, 2, 3, 4, 2, 2,
        2, 2, 2, 2, 4, 2
      };
      
      dtType vecMark[] = { 
        1, 1, 1, 1, 1, 1,
         0, 0, 0, 0, 0, 0,
        2, 0, 0, 0, 3, 3,
         2, 0, 0, 0, 3, 3,
        2, 2, 0, 0, 0, 3,
         2, 2, 0, 0, 3, 3,
        2, 2, 2, 2, 0, 3
      };
      
      Image<dtType> imIn(6,7);
      Image<dtType> imMark(imIn);
      Image<dtType> imWs(imIn);
      Image<dtType> imLbl(imIn);

      imIn << vecIn;
      imMark << vecMark;
      
      StrElt se = hSE();
      
      watershed(imIn, imMark, imWs, imLbl, se);
      
      dtType vecLblTruth[] = { 
        1,    1,    1,    1,    1,    1,
          1,    1,    1,    1,    1,    1,
        2,    3,    3,    3,    3,    3,
          2,    3,    3,    3,    3,    3,
        2,    2,    3,    3,    3,    3,
          2,    2,    2,    3,    3,    3,
        2,    2,    2,    2,    3,    3,
      };
      
      dtType maxV = ImDtTypes<dtType>::max();
      
      dtType vecWsTruth[] = { 
        0,    0,    0,    0,    0,    0,
        maxV,  maxV,  maxV,  maxV,  maxV,  maxV,
        0,  maxV,    0,    0,    0,    0,
          0,  maxV,    0,    0,    0,    0,
        0,    0,  maxV,  maxV,    0,    0,
          0,    0,    0,  maxV,    0,    0,
        0,    0,    0,    0,  maxV,    0,
      };
      
      Image<dtType> imLblTruth(imIn);
      Image<dtType> imWsTruth(imIn);
      
      imLblTruth << vecLblTruth;
      imWsTruth << vecWsTruth;
      
      TEST_ASSERT(imLbl==imLblTruth);
      TEST_ASSERT(imWs==imWsTruth);
      
      if (retVal!=RES_OK)
      {
        imLbl.printSelf(1, true);
        imWs.printSelf(1, true);
      }
      
      // Test idempotence
      
      Image<dtType> imWs2(imIn);
      
      watershed(imWs, imMark, imWs2, imLbl, se);
      TEST_ASSERT(imWs2==imWs);
  }
};

class Test_Watershed_Plateaus : public TestCase
{
  virtual void run()
  {
      typedef UINT8 dtType;
      typedef UINT16 dtType2;
      
      dtType vecIn[] = { 
        0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0
      };
      
      dtType2 vecMark[] = { 
        1, 1, 1, 1, 1, 0,
         0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
         3, 3, 3, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 2,
        2, 2, 2, 2, 2, 2
      };
      
      Image<dtType> imIn(6,7);
      Image<dtType2> imMark(imIn);
      Image<dtType> imWS(imIn);
      Image<dtType2> imLbl(imIn);

      imIn << vecIn;
      imMark << vecMark;
      
      StrElt se = hSE();
      
      watershed(imIn, imMark, imWS, imLbl, se);
      
      dtType vecWSTruth[] = { 
        0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,
      255,  255,  255,  255,  255,    0,
          0,    0,    0,    0,  255,  255,
        0,    0,    0,    0,  255,    0,
        255,  255,  255,  255,    0,    0,
        0,    0,    0,    0,    0,    0,
      };
      
      Image<dtType> imWSTruth(imIn);
      imWSTruth << vecWSTruth;
      
      dtType2 vecLblTruth[] = { 
        1,       1,       1,       1,       1,       1,
            1,       1,       1,       1,       1,       1,
        1,       1,       1,       1,       1,       1,
            3,       3,       3,       3,       1,       1,
        3,       3,       3,       3,       2,       2,
            3,       3,       3,       2,       2,       2,
        2,       2,       2,       2,       2,       2,
      };
      
      Image<dtType2> imLblTruth(imIn);
      imLblTruth << vecLblTruth;
      
      TEST_ASSERT(imWS==imWSTruth);
      if (retVal!=RES_OK)
      {
        imWS.printSelf(1, true);
        imWSTruth.printSelf(1, true);
      }
      
      TEST_ASSERT(imLbl==imLblTruth);
      if (retVal!=RES_OK)
      {
        imLbl.printSelf(1, true);
        imLblTruth.printSelf(1, true);
      }
  }
};


class Test_Watershed_Indempotence : public TestCase
{
  virtual void run()
  {
      UINT8 vecIn[] = { 
          98,   81,   45,  233,  166,  112,  100,   20,  176,   79,
              4,   11,   57,  246,  137,   90,   69,  212,   16,  219,
          131,  165,   20,    4,  201,  100,  166,   57,  144,  104,
            143,  242,  185,  188,  221,   97,   46,   66,  117,  222,
          146,  121,  234,  204,  113,  116,   40,  183,   74,   56,
            147,  205,  221,  168,  210,  168,   14,  122,  226,  158,
          226,  114,  146,  157,   48,  112,  254,   94,  179,  117,
              61,   71,  238,   40,   20,   97,  157,   60,   25,  231,
          116,  173,  181,   83,   86,  137,  252,  100,    4,  223,
              4,  231,   83,  150,  133,  131,    8,  133,  226,  187,
      };
      
      UINT8 vecMark[] = { 
          1,    1,    0,    0,    0,    2,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,    0,    0,    3,    0,    0,
          4,    0,    0,    0,    5,    0,    0,    0,    0,    0,
            0,    0,    6,    0,    0,    0,    0,    0,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    7,    0,    0,    8,    0,    0,
          0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
            0,    0,    0,    0,    9,    0,    0,    0,    0,   10,
      };
      
      Image_UINT8 imIn(10,10);
      Image_UINT8 imMark(imIn);
      Image_UINT8 imWs(imIn);
      Image_UINT8 imWs2(imIn);

      imIn << vecIn;
      imMark << vecMark;

      watershed(imIn, imMark, imWs, hSE());
      watershed(imWs, imMark, imWs2, hSE());
      
      TEST_ASSERT(imWs==imWs2);
      
      if (retVal!=RES_OK)
      {
          imWs.printSelf(1, true);
          imWs2.printSelf(1, true);
      }
  }
};



int main()
{
      TestSuite ts;
      
      ADD_TEST(ts, Test_Basins);
      ADD_TEST(ts, Test_Basins_Plateaus);
      ADD_TEST(ts, Test_ProcessWatershedHierarchicalQueue);

      typedef Test_Watershed<UINT8> Test_WS_UINT8;
      typedef Test_Watershed<UINT16> Test_WS_UINT16;
      ADD_TEST(ts, Test_WS_UINT8);
      ADD_TEST(ts, Test_WS_UINT16);
      
      ADD_TEST(ts, Test_Watershed_Plateaus);

      ADD_TEST(ts, Test_Watershed_Indempotence);
      
      return ts.run();
      
}


