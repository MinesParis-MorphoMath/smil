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



#include <cstdio>
#include <ctime>

#include "Core/include/DCore.h"
#include "DMorpho.h"



using namespace smil;


class Test_MinMax : public TestCase
{
  virtual void run()
  {
      typedef UINT8 dataType;
      typedef Image<dataType> imType;
      
      imType im1(7,7);
      imType im2(im1);
      imType imTruth(im1);
      
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
      
      minima(im1, im2, sSE());

      dataType truthVecMin[] = {
          0,   0,   0,   0,   0,   0,   0,
        255,   0, 255,   0,   0, 255,   0,
          0,   0,   0,   0,   0,   0,   0,
          0,   0, 255,   0, 255,   0,   0,
          0,   0,   0,   0,   0,   0, 255,
          0,   0,   0,   0, 255,   0,   0,
          0,   0,   0,   0,   0,   0,   0,
      };
      imTruth << truthVecMin;
      
      
      TEST_ASSERT(im2==imTruth);      
      
      if (retVal!=RES_OK)
        im2.printSelf(1);
      
      
      maxima(im1, im2, sSE());
      
      dataType truthVecMax[] = {
          0, 255,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0, 255,
          0, 255,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0, 255,   0,
          0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0, 255,
        255,   0,   0, 255,   0,   0,   0,
      };
      imTruth << truthVecMax;
      
      
      TEST_ASSERT(im2==imTruth);      
      
      if (retVal!=RES_OK)
        im2.printSelf(1);

  }
};

class Test_HMinMax : public TestCase
{
  virtual void run()
  {
      typedef UINT8 dataType;
      typedef Image<dataType> imType;
      
      imType im1(7,7);
      imType im2(im1);
      imType imTruth(im1);
      
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
      
      hMinima(im1, UINT8(20), im2, sSE());

      dataType truthVecMin[] = {
          0,   0,   0,   0,   0, 255, 255,
        255,   0, 255,   0,   0, 255,   0,
          0,   0,   0,   0,   0,   0,   0,
          0,   0, 255,   0, 255,   0,   0,
          0,   0,   0,   0,   0,   0, 255,
          0,   0,   0, 255, 255,   0,   0,
          0,   0, 255,   0,   0, 255,   0,
      };
      imTruth << truthVecMin;
      
      
      TEST_ASSERT(im2==imTruth);      
      
      if (retVal!=RES_OK)
        im2.printSelf(1);
      
      
      hMaxima(im1, UINT8(50), im2, sSE());
      
      dataType truthVecMax[] = {
        255, 255,   0,   0,   0,   0,   0,
          0,   0,   0, 255,   0,   0, 255,
          0, 255,   0,   0, 255,   0,   0,
        255,   0,   0,   0,   0, 255, 255,
          0,   0,   0,   0,   0,   0,   0,
          0,   0,   0,   0,   0,   0, 255,
        255,   0,   0, 255,   0,   0,   0,
      };
      imTruth << truthVecMax;
      
      
      TEST_ASSERT(im2==imTruth);      
      
      if (retVal!=RES_OK)
        im2.printSelf(1);

  }
};


int main()
{
      TestSuite ts;
      ADD_TEST(ts, Test_MinMax);
      ADD_TEST(ts, Test_HMinMax);
      
      return ts.run();
    
}

