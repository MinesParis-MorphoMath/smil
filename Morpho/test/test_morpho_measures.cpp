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



#include "DMorphoMeasures.hpp"

using namespace smil;

class TestGranulometry : public TestCase
{
  virtual void run()
  {
      Image<UINT8> im1(10,10);
      Image<UINT8> im2(im1);
      Image<UINT8> imTruth(im1);
      
      UINT8 vec1[] = 
      { 
          0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
          0, 255, 255, 255, 255, 255,   0,   0,   0,   0,
          0, 255, 255, 255, 255, 255,   0,   0,   0,   0,
          0, 255, 255, 255, 255, 255,   0, 255, 255, 255,
          0, 255, 255, 255, 255, 255,   0,   0, 255,   0,
          0, 255, 255, 255, 255, 255,   0,   0, 255,   0,
          0, 255, 255, 255, 255, 255,   0,   0, 255,   0,
          0, 255, 255,   0,   0, 255,   0,   0, 255,   0,
          0, 255, 255,   0,   0,   0,   0,   0,   0,   0,
          0,   0, 255,   0,   0,   0,   0,   0,   0,   0,
      };
      im1 << vec1;
      
      vector<double> granulo = measGranulometry(im1, hSE(), 1, false);

      TEST_ASSERT(granulo[0]==3060 && granulo[1]==1785 && granulo[2]==6120);      
      // TEST_ASSERT(granulo[0]==12 && granulo[1]==7 && granulo[2]==24);
      if (retVal!=RES_OK)
        for (UINT i=0;i<granulo.size();i++)
          cout << i << " " << granulo[i] << endl;
  }
};


int main()
{
      TestSuite ts;
      ADD_TEST(ts, TestGranulometry);
      
      return ts.run();
      
}

