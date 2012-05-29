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
 *     * Neither the name of the University of California, Berkeley nor the
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
      
      UINT8 vecTruth[] = { 
	0,  16, 241,  0, 80, 	// 16 = 0b00010000
	70, 44, 10, 245, 8, 
	0, 144, 221, 226, 100, 
	71, 173, 65, 128, 64, 
	0, 31, 0, 31, 0, 
      };
      
//       im2.printSelf(1);
      imTruth << vecTruth;
      TEST_ASSERT(im2==imTruth);
  }
};


#include "DCore.h"
#include "DMorphoBase.hpp"

int main(int argc, char *argv[])
{
//     B b;
//      expose2(b);
      Core::initialize();
	
      Image_UINT8 im1(5,5);
      if (read("/home/faessel/src/morphee/trunk/utilities/Images/Gray/akiyo_y.png", im1)!=RES_OK)
	  read("/home/mat/src/morphee/trunk/utilities/Images/Gray/akiyo_y.png", im1);
      Image_UINT8 im2(im1);
      thresh(im1, (UINT8)127, im1);
      
//       testLabel(im1, im2, sSE());
      
      int BENCH_NRUNS = 1E2;
      BENCH_IMG(label, im1, im2, hSE());
      im2 += ((im2>(UINT8)0) & (UINT8)100);
//       BENCH_IMG(dilate, im1, im2, sSE);
      
      im2.show();
      Core::execLoop();
      
      TestSuite ts;
//       ADD_TEST(ts, TestArrow);
      
      
      return ts.run();
      
}

