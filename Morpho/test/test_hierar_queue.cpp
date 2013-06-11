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
#include "DGui.h"
#include "DMorpho.h"
#include "DMorphoWatershed.hpp"

using namespace smil;


class Test_HierarchicalQueue : public TestCase
{
  virtual void run()
  {
      Image_UINT8 img(6,1);
      UINT8 vals[] = { 0, 0, 0, 2, 2, 2 };
      img << vals;
      
      HierarchicalQueue<UINT8> pq;
      pq.initialize(img);
      pq.push(2, 15);
      pq.push(2, 11);
      pq.push(2, 10);
      pq.push(0, 9);
      pq.push(0, 12);
      pq.push(0, 8);
      
      TEST_ASSERT(pq.pop()==9);
      TEST_ASSERT(pq.pop()==12);
      TEST_ASSERT(pq.pop()==8);
      TEST_ASSERT(pq.pop()==15);
      TEST_ASSERT(pq.pop()==11);
      TEST_ASSERT(pq.pop()==10);
  }
};

class Test_InitHierarchicalQueue : public TestCase
{
  virtual void run()
  {
      UINT8 vecIn[] = { 
	2, 2, 2, 2, 2, 2,
	7, 7, 7, 7, 7, 7,
	2, 7, 5, 6, 2, 2,
	2, 6, 5, 6, 2, 2,
	2, 2, 6, 4, 3, 2,
	2, 2, 3, 4, 2, 2,
	2, 2, 2, 2, 4, 2
      };
      
      UINT8 vecLbl[] = { 
	1, 1, 1, 1, 1, 1,
	0, 0, 0, 0, 0, 0,
	2, 0, 0, 0, 3, 3,
	2, 0, 0, 0, 3, 3,
	2, 2, 0, 0, 0, 3,
	2, 2, 0, 0, 3, 3,
	2, 2, 2, 2, 0, 3
      };
      
      
      Image_UINT8 imIn(6,7);
      Image_UINT8 imLbl(imIn);
      Image_UINT8 imStatus(imIn);

      imIn << vecIn;
      imLbl << vecLbl;
      
      HierarchicalQueue<UINT8> pq;
      
      StrElt se = hSE();
      
      initWatershedHierarchicalQueue(imIn, imLbl, imStatus, pq);
      
  }
};

class Test_Build : public TestCase
{
  virtual void run()
  {
      UINT8 vecIn[] = { 
	1, 2, 0, 5, 5, 5, 3, 3, 3, 1, 1
      };
      
      UINT8 vecMark[] = { 
	0, 0, 0, 0, 4, 1, 1, 2, 0, 0, 0
      };
      
      Image_UINT8 imIn(11,1);
      Image_UINT8 imMark(imIn);
      Image_UINT8 imBuild(imIn);

      imIn << vecIn;
      imMark << vecMark;
      
      dualBuild(imIn, imMark, imBuild, sSE());
      
      UINT8 vecTruth[] = { 
	0, 0, 0, 0, 4, 2, 2, 2, 1, 1, 1
      };
      
      Image_UINT8 imTruth(imIn);
      
      imTruth << vecTruth;
      
      TEST_ASSERT(imBuild==imTruth);
      
      if (retVal!=RES_OK)
	imBuild.printSelf(1);
  }
};


int main(int argc, char *argv[])
{
      TestSuite ts;
      ADD_TEST(ts, Test_HierarchicalQueue);
      ADD_TEST(ts, Test_InitHierarchicalQueue);
      ADD_TEST(ts, Test_Build);
      
      return ts.run();
      
}


