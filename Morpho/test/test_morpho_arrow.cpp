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



#include "DMorpho.h"

class TestArrow : public TestCase
{
  virtual void run()
  {
      Image_UINT8 im1(5,5);
      Image_UINT8 im2(5,5);
      Image_UINT8 imTruth(5,5);
      
      UINT8 vec1[] = { 
	1, 3, 10, 2, 9, 
	5, 5, 5, 9, 3, 
	3, 5, 7, 5, 5, 
	8, 7, 4, 1, 1, 
	4, 10, 1, 6, 0
      };
      
      im1 << vec1;

      arrowGrt(im1, im2, sSE0(), UINT8(255));
      UINT8 vecGrt[] = { 
	0,  16, 31,  0, 20, 	// 16 = 0b00010000
	196, 104, 160, 95, 32, 
	0, 18, 119, 142, 76, 
	197, 107, 5, 2, 4, 
	0, 241, 0, 241, 0
      };
      imTruth << vecGrt;
      TEST_ASSERT(im2==imTruth);

      arrowLow(im1, im2, sSE0(), UINT8(255));
      UINT8 vecLow[] = { 
	255, 239, 224, 255, 227, 
	56, 130, 69, 32, 223, 
	255, 13, 128, 80, 163, 
	58, 20, 250, 244, 235, 
	255, 14, 127, 14, 255
      };
      imTruth << vecLow;
      TEST_ASSERT(im2==imTruth);

//       im2.printSelf(1);
    
  }
};


#include "DCore.h"
#include "DGui.h"
#include "DMorphoBase.hpp"

int main(int argc, char *argv[])
{
      TestSuite ts;
      ADD_TEST(ts, TestArrow);
      
      return ts.run();
      
}

