/*
 * Copyright (c) 2011, Matthieu FAESSEL and ARMINES
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


#include "DCore.h"
#include "DMorpho.h"
#include "DGui.h"
#include "DIO.h"

using namespace smil;

class Test_Dilate_2Points : public TestCase
{
  virtual void run()
  {
      typedef UINT8 dataType;
      typedef Image<dataType> imType;
      
      imType im1(7,7);
      imType im2(im1);
      imType im3(im1);
      
      dataType vec1[] = {
	114, 133,  74, 160,  57,  25,  37,
	 23,  73,   9, 196, 118,  23, 110,
	154, 248, 165, 159, 210,  47,  58,
	213,  74,   8, 163,   3, 240, 213,
	158,  67,  52, 103, 163, 158,   9,
	 85,  36, 124,  12,   7,  56, 253,
	214, 148,  20, 200,  53,  10,  58
      };
      
      im1 << vec1;
      
      StrElt dse;
      dse.addPoint(0,0);
      dse.addPoint(1,1);
      
      dilate(im1, im2, dse());

      dataType dilateVec[] = {
	114, 133,  74, 160,  57,  25,  37,
	  23, 114, 133, 196, 160,  57, 110,
	154, 248, 165, 159, 210, 118,  58,
	213, 154, 248, 165, 159, 240, 213,
	158, 213,  74, 103, 163, 158, 240,
	  85, 158, 124,  52, 103, 163, 253,
	214, 148,  36, 200,  53,  10,  58,
      };
      im3 << dilateVec;
      
      
      TEST_ASSERT(im2==im3);      
      
      if (retVal!=RES_OK)
	im2.printSelf(1);
  }
};

class Test_Dilate_Hex : public TestCase
{
  virtual void run()
  {
      typedef UINT8 dataType;
      typedef Image<dataType> imType;
      
      imType im1(7,7);
      imType im2(im1);
      imType im3(im1);
      
      dataType vec1[] = {
	114, 133,  74, 160,  57,  25,  37,
	 23,  73,   9, 196, 118,  23, 110,
	154, 248, 165, 159, 210,  47,  58,
	213,  74,   8, 163,   3, 240, 213,
	158,  67,  52, 103, 163, 158,   9,
	 85,  36, 124,  12,   7,  56, 253,
	214, 148,  20, 200,  53,  10,  58
      };
      
      im1 << vec1;
      
      dataType dilateHexVec[] = {
	133, 133, 160, 196, 196, 118, 110, 
	248, 248, 196, 210, 210, 118, 110, 
	248, 248, 248, 210, 210, 240, 240, 
	248, 248, 165, 210, 240, 240, 240, 
	213, 213, 124, 163, 163, 240, 253, 
	214, 148, 200, 200, 163, 253, 253, 
	214, 214, 200, 200, 200, 58, 253
      };
      im3 << dilateHexVec;
      
      // The specialized way
      dilate(im1, im2, hSE());
      TEST_ASSERT(im2==im3);      
      
      // The generic way
      StrElt se;
      se.points = hSE().points;
      se.odd = true;
      dilate(im1, im2, se);
      TEST_ASSERT(im2==im3);      
      
      // With an homothetic SE
      dilate(im1, im3, hSE(3));
      dilate(im1, im2, hSE().homothety(3));
      TEST_ASSERT(im2==im3);
      
  }
};


class Test_Dilate_Squ : public TestCase
{
  virtual void run()
  {
      typedef UINT8 dataType;
      typedef Image<dataType> imType;
      
      imType im1(7,7);
      imType im2(im1);
      imType im3(im1);
      
      dataType vec1[] = {
	114, 133,  74, 160,  57,  25,  37,
	 23,  73,   9, 196, 118,  23, 110,
	154, 248, 165, 159, 210,  47,  58,
	213,  74,   8, 163,   3, 240, 213,
	158,  67,  52, 103, 163, 158,   9,
	 85,  36, 124,  12,   7,  56, 253,
	214, 148,  20, 200,  53,  10,  58
      };
      
      im1 << vec1;
      
      dataType dilateSquVec[] = {
	133, 133, 196, 196, 196, 118, 110, 
	248, 248, 248, 210, 210, 210, 110, 
	248, 248, 248, 210, 240, 240, 240, 
	248, 248, 248, 210, 240, 240, 240, 
	213, 213, 163, 163, 240, 253, 253, 
	214, 214, 200, 200, 200, 253, 253, 
	214, 214, 200, 200, 200, 253, 253
      };
      im3 << dilateSquVec;

      // The specialized way
      dilate(im1, im2, sSE());
      TEST_ASSERT(im2==im3);      
      
      // The generic way
      StrElt se;
      se.points = sSE().points;
      dilate(im1, im2, se);
      TEST_ASSERT(im2==im3);      
//       im1.printSelf(1);
//       im2.printSelf(1);
//       im3.printSelf(1);
  }
};


class Test_Dilate_3D : public TestCase
{
  virtual void run()
  {
      typedef UINT8 dataType;
      typedef Image<dataType> imType;
      
      imType im1(5,5,5);
      imType im2(im1);
      imType im3(im1);
      
      dataType vec1[] = {
	207, 170, 100, 107, 141,
	230, 233,  99, 245, 115,
	  71, 112, 121, 177, 141,
	155, 153, 109,  85, 134,
	147, 122, 106,  72, 173,

	204, 221, 116,  19,  91,
	231, 227,   6,  77,  80,
	148,  53,  58, 248,  43,
	174,  64, 156,  41, 241,
	  42, 197, 139, 152,  27,

	  19,  45, 149, 125, 118,
	  67,  75,  84, 183,  95,
	176, 160,  67, 183, 238,
	148,  76,  36, 206,  69,
	  80, 125, 134, 236, 167,

	120,  24, 109,   5, 176,
	136,  24, 222,  31, 149,
	  85,  99, 224, 170,  27,
	  65,  91, 188, 132,  20,
	172,  25,  96, 208, 232,

	166,  33, 103,  45,  15,
	  15, 166,  39, 125, 171,
	216,   6, 195, 184,  37,
	  90,  14, 136,  60, 184,
	164, 125,  21,  98,   3,
      };
      
      im1 << vec1;
      
      dataType dilateVec[] = {
	233, 233, 245, 245, 245,
	233, 233, 248, 248, 248,
	233, 233, 248, 248, 248,
	197, 197, 248, 248, 248,
	197, 197, 197, 241, 241,

	233, 233, 245, 245, 245,
	233, 233, 248, 248, 248,
	233, 233, 248, 248, 248,
	197, 197, 248, 248, 248,
	197, 197, 236, 241, 241,

	231, 231, 227, 222, 183,
	231, 231, 248, 248, 248,
	231, 231, 248, 248, 248,
	197, 224, 248, 248, 248,
	197, 197, 236, 241, 241,

	166, 222, 222, 222, 183,
	216, 224, 224, 238, 238,
	216, 224, 224, 238, 238,
	216, 224, 236, 238, 238,
	172, 188, 236, 236, 236,

	166, 222, 222, 222, 176,
	216, 224, 224, 224, 184,
	216, 224, 224, 224, 184,
	216, 224, 224, 232, 232,
	172, 188, 208, 232, 232,
      };
      im3 << dilateVec;

      // The specialized way
      dilate(im1, im2, CubeSE());
      TEST_ASSERT(im2==im3);      
  }
};



int main(int argc, char *argv[])
{
      TestSuite ts;
      ADD_TEST(ts, Test_Dilate_2Points);
      ADD_TEST(ts, Test_Dilate_Hex);
      ADD_TEST(ts, Test_Dilate_Squ);
      ADD_TEST(ts, Test_Dilate_3D);
      
//       UINT BENCH_NRUNS = 5E3;
      Image_UINT8 im1(1024, 1024), im2(im1);
//       BENCH_IMG_STR(dilate, "hSE", im1, im2, hSE());
//       BENCH_IMG_STR(dilate, "sSE", im1, im2, sSE());
// cout << endl;
//       tc(im1, im2, sSE());
      return ts.run();
  
}

