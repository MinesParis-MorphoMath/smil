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
#include "DMorphoHierarQ.hpp"

class Test_HierarchicalQueue : public TestCase
{
  virtual void run()
  {
      PriorityQueue<UINT8> pq;
      pq.push(2, 10);
      pq.push(2, 11);
      pq.push(2, 15);
      pq.push(0, 9);
      pq.push(0, 8);
      pq.push(0, 12);
      
      TEST_ASSERT(pq.top().value==0 && pq.top().offset==9);
      pq.pop();
      TEST_ASSERT(pq.top().value==0 && pq.top().offset==8);
      pq.pop();
      TEST_ASSERT(pq.top().value==0 && pq.top().offset==12);
      pq.pop();
      TEST_ASSERT(pq.top().value==2 && pq.top().offset==10);
      pq.pop();
      TEST_ASSERT(pq.top().value==2 && pq.top().offset==11);
      pq.pop();
      TEST_ASSERT(pq.top().value==2 && pq.top().offset==15);
      pq.pop();
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
      
      PriorityQueue<UINT8> pq;
      
      StrElt se = sSE();
      
      initPriorityQueue(imIn, imLbl, imStatus, &pq);
      imStatus.printSelf(1);
      processPriorityQueue(imIn, imLbl, imStatus, &pq, &se);
      pq.printSelf();
      
      imStatus.printSelf(1);
      imLbl.printSelf(1);
  }
};


int main(int argc, char *argv[])
{
      TestSuite ts;
//       ADD_TEST(ts, Test_HierarchicalQueue);
      ADD_TEST(ts, Test_InitHierarchicalQueue);
      
      
      return ts.run();
      
}

