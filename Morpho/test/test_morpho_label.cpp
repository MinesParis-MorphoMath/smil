/*
 * Copyright (c) 2011-2014, Matthieu FAESSEL and ARMINES
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

using namespace smil;

class Test_Label : public TestCase
{
  virtual void run()
  {
      typedef UINT16 dataType;
      typedef Image<dataType> imType;
      
      imType im1(7,7);
      imType im2(im1);
      imType im3(im1);
      
      dataType vec1[] = {
	0, 0, 0, 0, 0, 0, 1,
	0, 0, 0, 0, 1, 1, 1,
	0, 1, 0, 0, 1, 1, 1,
	1, 1, 0, 0, 0, 1, 0,
	1, 0, 0, 0, 0, 0, 0,
	1, 0, 1, 0, 0, 1, 0,
	0, 0, 1, 0, 1, 1, 0
      };
      
      im1 << vec1;
      
      label(im1, im2, sSE());
//       im2.printSelf(1);
      
//       im2.show();
//       Gui::execLoop();
      dataType vec3[] = {
	0, 0, 0, 0, 0, 0, 1, 
	0, 0, 0, 0, 1, 1, 1, 
	0, 2, 0, 0, 1, 1, 1, 
	2, 2, 0, 0, 0, 1, 0, 
	2, 0, 0, 0, 0, 0, 0, 
	2, 0, 3, 0, 0, 4, 0, 
	0, 0, 3, 0, 4, 4, 0
      };
      im3 << vec3;
      
      TEST_ASSERT(im2==im3);
      if (retVal!=RES_OK)
	im2.printSelf(1);
  }
};

class Test_LabelLambdaFlatZones : public TestCase
{
  virtual void run()
  {
      typedef UINT16 dataType;
      typedef Image<dataType> imType;
      
      imType im1(7,7);
      imType im2(im1);
      imType im3(im1);
      
      dataType vec1[] = {
	  39, 239, 224,  19, 147, 186,  13,
	 157,  30,  29, 190, 140,  80, 250,
	  45,  86, 117,  43,  28,  133,  67,
	  41,  46,  49, 232, 128, 167, 197,
	 116,  72,  37, 156,  135,   5, 186,
	 193, 192, 168, 104, 162,  60,  39,
	 202, 161,  33, 160, 228, 150, 203,
      };
      
      im1 << vec1;
      
      lambdaLabel(im1, (UINT16)10, im2, sSE());

      dataType vec3[] = {
        1,     2,     3,     1,     4,     5,     6,
        7,     1,     1,     8,     4,     9,    10,
        11,    12,    13,    11,    14,     4,    15,
        11,    11,    11,    16,     4,    17,    18,
        19,    20,    11,    21,     4,    22,    23,
        24,    24,    21,    25,    21,    26,    27,
        24,    21,    28,    21,    29,    30,    31,
      };
      im3 << vec3;
      
      TEST_ASSERT(im2==im3);
      if (retVal!=RES_OK)
	im2.printSelf(1);
  }
};


class Test_Label_Mosaic : public TestCase
{
  virtual void run()
  {
      typedef UINT16 dataType;
      typedef Image<dataType> imType;
      
      imType im1(7,7);
      imType im2(im1);
      imType im3(im1);
      
      dataType vec1[] = {
	1, 1, 1, 2, 2, 2, 2,
	1, 1, 1, 2, 2, 2, 2,
	1, 1, 1, 2, 1, 2, 2,
	3, 3, 1, 1, 1, 1, 1,
	3, 3, 3, 3, 4, 4, 4,
	3, 3, 3, 4, 4, 4, 4,
	3, 3, 3, 4, 4, 4, 4
      };
      
      im1 << vec1;
      
      label(im1, im2, sSE());
      
      TEST_ASSERT(im2==im1);
      if (retVal!=RES_OK)
	im2.printSelf(1);
  }
};

class Test_LabelWithArea : public TestCase
{
  virtual void run()
  {
      typedef UINT16 dataType;
      typedef Image<dataType> imType;
      
      imType im1(7,7);
      imType im2(im1);
      imType im3(im1);
      
      dataType vec1[] = {
	0, 0, 0, 0, 0, 0, 1,
	0, 0, 0, 0, 1, 1, 1,
	0, 1, 0, 0, 1, 1, 1,
	1, 1, 0, 0, 0, 1, 0,
	1, 0, 0, 0, 0, 0, 0,
	1, 0, 1, 0, 0, 1, 0,
	0, 0, 1, 0, 1, 1, 0
      };
      
      im1 << vec1;
      
      labelWithArea(im1, im2, sSE());
//       im2.printSelf(1);
      
//       im2.show();
//       Gui::execLoop();
      dataType vec3[] = {
	0,     0,     0,     0,     0,     0,     8,
	0,     0,     0,     0,     8,     8,     8,
	0,     5,     0,     0,     8,     8,     8,
	5,     5,     0,     0,     0,     8,     0,
	5,     0,     0,     0,     0,     0,     0,
	5,     0,     2,     0,     0,     3,     0,
	0,     0,     2,     0,     3,     3,     0,
      };
      im3 << vec3;
      
      TEST_ASSERT(im2==im3);
  }
};


class Test_LabelNeighbors : public TestCase
{
  virtual void run()
  {
      typedef UINT16 dataType;
      typedef Image<dataType> imType;
      
      imType im1(7,7);
      imType im2(im1);
      imType im3(im1);
      
      dataType vec1[] = {
	0, 0, 0, 0, 0, 0, 1, 
	0, 0, 0, 0, 1, 1, 1, 
	0, 2, 0, 0, 1, 1, 1, 
	2, 2, 0, 0, 0, 1, 0, 
	2, 0, 0, 0, 0, 0, 0, 
	2, 0, 3, 0, 0, 4, 0, 
	0, 0, 3, 0, 4, 4, 0
      };
      
      im1 << vec1;
      
      neighbors(im1, im2, sSE());
//       im2.printSelf(1);
      
      dataType vec3[] = {
	1, 1, 1, 2, 2, 2, 2, 
	2, 2, 2, 2, 2, 2, 2, 
	2, 2, 2, 2, 2, 2, 2, 
	2, 2, 2, 2, 2, 2, 2, 
	2, 3, 3, 2, 3, 3, 3, 
	2, 3, 2, 3, 2, 2, 2, 
	2, 3, 2, 3, 2, 2, 2, 
      };
      im3 << vec3;
      
      TEST_ASSERT(im2==im3);
  }
};


int main(int argc, char *argv[])
{
      TestSuite ts;
      ADD_TEST(ts, Test_Label);
      ADD_TEST(ts, Test_LabelLambdaFlatZones);
      ADD_TEST(ts, Test_Label_Mosaic);
      ADD_TEST(ts, Test_LabelWithArea);
      ADD_TEST(ts, Test_LabelNeighbors);
      
      return ts.run();
  
}

