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

class Test_Dilate_Hex : public TestCase
{
  virtual void run()
  {
      typedef UINT8 dataType;
      typedef Image<dataType> imType;
      
      imType im1(5,5);
      imType im2(im1);
      imType im3(im1);
      
      dataType vec1[] = {
	114, 133, 74, 160, 57, 
	23, 73, 9, 196, 118, 
	154, 248, 165, 159, 210, 
	213, 74, 8, 163, 3, 
	158, 67, 52, 103, 163
      };
      
      im1 << vec1;
      
      dataType dilateHexVec[] = {
	133, 133, 160, 196, 196, 
	248, 248, 196, 210, 210, 
	248, 248, 248, 210, 210, 
	248, 248, 165, 210, 210, 
	213, 213, 103, 163, 163
      };
      im3 << dilateHexVec;
      dilate(im1, im2, hSE());
//       im2.printSelf(1);
//       im3.printSelf(1);
      TEST_ASSERT(im2==im3);      
  }
};

class Test_Dilate_Squ : public TestCase
{
  virtual void run()
  {
      typedef UINT8 dataType;
      typedef Image<dataType> imType;
      
      imType im1(5,5);
      imType im2(im1);
      imType im3(im1);
      
      dataType vec1[] = {
	114, 133, 74, 160, 57, 
	23, 73, 9, 196, 118, 
	154, 248, 165, 159, 210, 
	213, 74, 8, 163, 3, 
	158, 67, 52, 103, 163
      };
      
      im1 << vec1;
      
      dataType dilateSquVec[] = {
	133, 133, 196, 196, 196, 
	248, 248, 248, 210, 210, 
	248, 248, 248, 210, 210, 
	248, 248, 248, 210, 210, 
	213, 213, 163, 163, 163
      };
      im3 << dilateSquVec;
      dilate(im1, im2, sSE());
      TEST_ASSERT(im2==im3);      
//       im1.printSelf(1);
//       im2.printSelf(1);
//       im3.printSelf(1);
  }
};



int main(int argc, char *argv[])
{
      TestSuite ts;
      ADD_TEST(ts, Test_Dilate_Hex);
      ADD_TEST(ts, Test_Dilate_Squ);
      
      UINT BENCH_NRUNS = 1E3;
      Image_UINT8 im1(1024, 1024), im2(im1);
      BENCH_IMG_STR(dilate, "hSE", im1, im2, hSE());
      BENCH_IMG_STR(dilate, "sSE", im1, im2, sSE());

      return ts.run();
  
}

