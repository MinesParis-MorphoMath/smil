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
#include "DMorphoFilter.hpp"

using namespace smil;

class Test_Mean : public TestCase
{
  virtual void run()
  {
      UINT8 vec1[20] = 
      {
         1,  2,  3,  4, 
         5,  6,  7,  8, 
         9, 10, 11, 12,
        13, 14, 15, 16,
        17, 18, 19, 20 
      };
      UINT8 vecMean[20]        =
      {
        3, 4, 5, 5, 
        5, 6, 7, 7, 
        9, 10, 11, 11, 
        13, 14, 15, 15, 
        15, 16, 17, 17, 
      };
      
      
      Image_UINT8 im1(4,5);
      Image_UINT8 im2(im1);
      Image_UINT8 imTruth(im1);
      
      im1 << vec1;
      
      mean(im1, im2);
      
      imTruth << vecMean;
      TEST_ASSERT(im2==imTruth);
      
      if (retVal!=RES_OK)
      {
        imTruth.printSelf(1);
        im2.printSelf(1);
      }
  }
};

class Test_Median : public TestCase
{
  virtual void run()
  {
      UINT8 vec1[20] = 
      {
         1,  2,  3,  4, 
         5,  6,  7,  8, 
         9, 10, 11, 12,
        13, 14, 15, 16,
        17, 18, 19, 20 
      };
      UINT8 vecMedian[20]        =
      {
          5,   5,   6,   7,
          6,   6,   7,   8,
          10,  10,  11,  12,
          14,  14,  15,  16,
          17,  17,  18,  19,
      };
      
      
      Image_UINT8 im1(4,5);
      Image_UINT8 im2(im1);
      Image_UINT8 imTruth(im1);
      
      im1 << vec1;
      
      median(im1, im2, sSE());
      
      imTruth << vecMedian;
      TEST_ASSERT(im2==imTruth);
      
      if (retVal!=RES_OK)
      {
        imTruth.printSelf(1);
        im2.printSelf(1);
      }
  }
};


int main(int argc, char *argv[])
{
      TestSuite ts;
      ADD_TEST(ts, Test_Mean);
      ADD_TEST(ts, Test_Median);
      
      return ts.run();
}

