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


#include "DTest.h"
#include "DCompositeSE.h"
#include "DHitOrMiss.hpp"

using namespace smil;

class Test_Thin : public TestCase
{
  virtual void run()
  {
      typedef UINT8 dataType;
      typedef Image<dataType> imType;
      
      imType im1(10,10);
      imType im2(im1);
      
      dataType vec1[] = 
      {
	255,   0, 255,   0,   0, 255, 255,   0, 255, 255,
	  0,   0, 255,   0, 255,   0, 255,   0, 255,   0,
	  0,   0,   0,   0,   0,   0, 255, 255,   0, 255,
	  0, 255, 255, 255,   0,   0, 255, 255,   0,   0,
	  0,   0, 255,   0,   0,   0,   0,   0,   0,   0,
	  0,   0,   0,   0,   0,   0,   0, 255, 255, 255,
	  0,   0, 255,   0,   0, 255,   0,   0,   0, 255,
	255, 255,   0, 255, 255,   0, 255,   0, 255, 255,
	255, 255,   0, 255,   0,   0, 255,   0, 255, 255,
	  0, 255, 255, 255,   0,   0,   0,   0,   0, 255
      };
      
      im1 << vec1;
      
      
      
      dataType hmtLVec[] = 
      {
	  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 
	  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 
	  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 
	  0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 
	  0,   0, 255,   0,   0,   0,   0,   0,   0,   0, 
	  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 
	  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 
	  0,   0,   0,   0,   0,   0,   0,   0, 255,   0, 
	  0,   0,   0,   0, 255,   0,   0,   0, 255,   0, 
	  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 
      };
      imType imHmtl(im1);
      imHmtl << hmtLVec;
      
      CompStrEltList sel = HMT_sL1(4);
      hitOrMiss(im1, sel, im2);
      TEST_ASSERT(im2==imHmtl);
      if (retVal!=RES_OK)
	im2.printSelf(1);
      
      
      return;
      
      dataType thinLVec[] = 
      {
	255,   0, 255,   0,   0, 255, 255,   0, 255, 255,
	  0,   0, 255,   0, 255,   0, 255,   0, 255,   0,
	  0,   0,   0,   0,   0,   0, 255, 255,   0, 255,
	  0, 255, 255, 255,   0,   0, 255, 255,   0,   0,
	  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
	  0,   0,   0,   0,   0,   0,   0, 255, 255, 255,
	  0,   0, 255,   0,   0, 255,   0,   0,   0, 255,
	255, 255,   0, 255, 255,   0, 255,   0,   0, 255,
	255, 255,   0, 255,   0,   0, 255,   0,   0, 255,
	  0, 255, 255, 255,   0,   0,   0,   0,   0, 255
      };
      imType imThin(im1);
      imThin << thinLVec;
      
      thin(im1, sel, im2);
      im2.printSelf(1);
      TEST_ASSERT(im2==imThin);
      if (retVal!=RES_OK)
	im2.printSelf(1);
      
      
      
      dataType thickVec[] = 
      {
	255,   0, 255,   0,   0, 255, 255,   0, 255, 255,
	  0,   0, 255,   0, 255,   0, 255,   0, 255,   0,
	  0,   0,   0,   0,   0,   0, 255, 255,   0, 255,
	  0, 255, 255, 255,   0,   0, 255, 255,   0,   0,
	  0,   0, 255,   0,   0,   0,   0,   0,   0,   0,
	  0,   0,   0,   0,   0,   0,   0, 255, 255, 255,
	  0,   0, 255,   0,   0, 255,   0,   0,   0, 255,
	255, 255,   0, 255, 255,   0, 255,   0, 255, 255,
	255, 255,   0, 255, 255,   0, 255,   0, 255, 255,
	  0, 255, 255, 255,   0,   0,   0,   0,   0, 255
      };
      imType imThick(im1);
      imThick << thickVec;
      
      thick(im1, sel, im2);
      TEST_ASSERT(im2==imThick);
      if (retVal!=RES_OK)
	im2.printSelf(1);
  }
};

class Test_FullThin : public TestCase
{
  virtual void run()
  {
      typedef UINT8 dataType;
      typedef Image<dataType> imType;
      
      imType im1(10,10);
      imType im2(im1);
      imType im3(im1);
      
      dataType vec1[] = 
      {
	255,   0, 255,   0,   0, 255, 255,   0, 255, 255,
	  0,   0, 255,   0, 255,   0, 255,   0, 255,   0,
	  0,   0,   0,   0,   0,   0, 255, 255,   0, 255,
	  0, 255, 255, 255,   0,   0, 255, 255,   0,   0,
	  0,   0, 255,   0,   0,   0,   0,   0,   0,   0,
	  0,   0,   0,   0,   0,   0,   0, 255, 255, 255,
	  0,   0, 255,   0,   0, 255,   0,   0,   0, 255,
	255, 255,   0, 255, 255,   0, 255,   0, 255, 255,
	255, 255,   0, 255,   0,   0, 255,   0, 255, 255,
	  0, 255, 255, 255,   0,   0,   0,   0,   0, 255
      };
      
      im1 << vec1;
      
      dataType hmtLVec[] = 
      {
	  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 
	  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 
	  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 
	  0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 
	  0,   0, 255,   0,   0,   0,   0,   0,   0,   0, 
	  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 
	  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 
	  0,   0,   0,   0,   0,   0,   0,   0, 255,   0, 
	  0,   0,   0,   0, 255,   0,   0,   0, 255,   0, 
	  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 
      };
      im3 << hmtLVec;
      
      CompStrEltList sel = HMT_sL1(4);
      hitOrMiss(im1, sel, im2);
      TEST_ASSERT(im2==im3);
  }
};

class Test_LineJunc : public TestCase
{
  virtual void run()
  {
      typedef UINT8 dataType;
      typedef Image<dataType> imType;
      
      imType im1(10,10);
      imType im2(im1);
      imType im3(im1);
      
      dataType vec1[] = 
      {
	255,   0, 255,   0,   0, 255, 255,   0, 255, 255,
	  0,   0, 255,   0, 255,   0, 255,   0, 255,   0,
	  0,   0,   0,   0,   0,   0, 255, 255,   0, 255,
	  0, 255, 255, 255,   0,   0, 255, 255,   0,   0,
	  0,   0, 255,   0,   0,   0,   0, 255,   0,   0,
	  0,   0,   0,   0,   0,   0,   0, 255, 255, 255,
	  0,   0, 255,   0,   0, 255,   0,   0, 255,   0,
	255, 255,   0, 255, 255,   0, 255,   0, 255, 255,
	255, 255,   0, 255,   0,   0, 255,   0, 255, 255,
	  0, 255, 255, 255,   0,   0,   0,   0,   0, 255
      };
      
      im1 << vec1;
      
      dataType hmtLVec[] = 
      {
	  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 
	  0,   0,   0,   0,   0,   0,   0,   0, 255,   0, 
	  0,   0,   0,   0,   0,   0,   0, 255,   0,   0, 
	  0,   0, 255,   0,   0,   0,   0,   0,   0,   0, 
	  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 
	  0,   0,   0,   0,   0,   0,   0,   0, 255,   0, 
	  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 
	  0,   0,   0, 255,   0,   0,   0,   0,   0,   0, 
	  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 
	  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 
      };
      im3 << hmtLVec;
      
      CompStrEltList sel = HMT_sLineJunc(8);
      hitOrMiss(im1, sel, im2);
//       im2.printSelf(1);
      TEST_ASSERT(im2==im3);
  }
};

int main(int argc, char *argv[])
{
      TestSuite ts;
      ADD_TEST(ts, Test_Thin);
      ADD_TEST(ts, Test_LineJunc);
      
      return ts.run();
}

