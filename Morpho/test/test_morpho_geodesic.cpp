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



#include "DMorphoGeodesic.hpp"

using namespace smil;

class TestDistanceSquare : public TestCase
{
  virtual void run()
  {
      Image_UINT8 im1(10,10);
      Image_UINT8 im2(im1);
      Image_UINT8 imTruth(im1);
      
      UINT8 vec1[] = 
      { 
	  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
	  0, 255, 255, 255, 255, 255,   0,   0,   0,   0,
	  0, 255, 255, 255, 255, 255,   0,   0,   0,   0,
	  0, 255, 255, 255, 255, 255,   0, 255, 255, 255,
	  0, 255, 255, 255, 255, 255,   0,   0, 255,   0,
	  0,   0,   0,   0,   0,   0,   0,   0, 255,   0,
	  0, 255,   0,   0,   0,   0,   0,   0, 255,   0,
	  0, 255, 255,   0,   0, 255,   0,   0, 255,   0,
	  0, 255, 255,   0,   0,   0,   0,   0,   0,   0,
	  0,   0, 255,   0,   0,   0,   0,   0,   0,   0,
      };
      im1 << vec1;
      
      UINT8 vecTruth[] = 
      { 
	  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
	  0,   1,   1,   1,   1,   1,   0,   0,   0,   0,
	  0,   1,   2,   2,   2,   1,   0,   0,   0,   0,
	  0,   1,   2,   2,   2,   1,   0,   1,   1,   1,
	  0,   1,   1,   1,   1,   1,   0,   0,   1,   0,
	  0,   0,   0,   0,   0,   0,   0,   0,   1,   0,
	  0,   1,   0,   0,   0,   0,   0,   0,   1,   0,
	  0,   1,   1,   0,   0,   1,   0,   0,   1,   0,
	  0,   1,   1,   0,   0,   0,   0,   0,   0,   0,
	  0,   0,   1,   0,   0,   0,   0,   0,   0,   0,
      };
      imTruth << vecTruth;
      
      dist(im1, im2, sSE());
      TEST_ASSERT(im2==imTruth);
      if (retVal!=RES_OK)
	im2.printSelf(1);
  }
};

class TestDistanceCross : public TestCase
{
  virtual void run()
  {
      Image_UINT8 im1(8,8);
      Image_UINT8 im2(im1);
      Image_UINT8 imTruth(im1);
      
      UINT8 vec1[] = 
      { 
	  255, 255, 255, 255, 255, 255, 255, 255,
	  255,   0, 255, 255, 255, 255, 255, 255,
	  255, 255, 255,   0,   0, 255, 255, 255,
	  255, 255, 255,   0, 255, 255, 255, 255,
	  255, 255, 255, 255, 255,   0,   0,   0,
	  255, 255, 255, 255, 255,   0, 255,   0,
	  255, 255, 255, 255, 255,   0,   0,   0,
	  255, 255, 255, 255, 255, 255, 255, 255,  
      };
      im1 << vec1;
      
      UINT8 vecTruth[] = 
      { 
	  2,   1,   2,   2,   2,   3,   4,   4,
	  1,   0,   1,   1,   1,   2,   3,   3,
	  2,   1,   1,   0,   0,   1,   2,   2,
	  3,   2,   1,   0,   1,   1,   1,   1,
	  4,   3,   2,   1,   1,   0,   0,   0,
	  5,   4,   3,   2,   1,   0,   1,   0,
	  5,   4,   3,   2,   1,   0,   0,   0,
	  6,   5,   4,   3,   2,   1,   1,   1,      };
      imTruth << vecTruth;
      
      dist(im1, im2, cSE());
      TEST_ASSERT(im2==imTruth);
      if (retVal!=RES_OK)
	im2.printSelf(1);
  }
};

int main(int argc, char *argv[])
{
      TestSuite ts;
      ADD_TEST(ts, TestDistanceSquare);
      ADD_TEST(ts, TestDistanceCross);      
      return ts.run();
      
}

