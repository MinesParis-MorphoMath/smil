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


#include "DBit.h"
#include "DCore.h"
#include "DBase.h"
#include "DMorpho.h"
#include "DGui.h"
#include "DIO.h"

using namespace smil;

class Test_Copy : public TestCase
{
  virtual void run()
  {
      typedef Bit dataType;
      typedef Image<dataType> imType;
      
      imType im1(7,7);
      imType im2(im1);
      
       bool vec1[] = {
	0, 0, 0, 0, 0, 1, 1,
	1, 1, 0, 0, 0, 1, 0,
	0, 0, 0, 0, 0, 1, 0,
	0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 1, 0, 0,
	0, 0, 0, 0, 0, 0, 1,
	0, 1, 0, 0, 0, 1, 1,
      };
      
      im1 << BitArray(vec1, 49);
      
      copy(im1, im2);
      
      TEST_ASSERT(im1==im2);
      if (retVal==RES_ERR)
      {
	  im1.printSelf(1);
	  im2.printSelf(1);
      }
  }
};

class Test_Trans : public TestCase
{
  virtual void run()
  {
      typedef Bit dataType;
      typedef Image<dataType> imType;
      
      imType im1(7,7);
      imType im2(im1);
      imType im3(im1);
      
      bool vec1[] = {
	0, 0, 0, 0, 0, 1, 1,
	1, 1, 0, 0, 0, 1, 0,
	0, 0, 0, 0, 0, 1, 0,
	0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 1, 0, 0,
	0, 0, 0, 0, 0, 0, 1,
	0, 1, 0, 0, 0, 1, 1,
      };
      im1 << BitArray(vec1, 49);
      
      BitArray bit(vec1, 49);
      BitArray bit2(vec1, 49);
      
      trans(im1, 2, -2, im2);
      
      bool vec3[] = {
	0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 
	0, 0, 1, 1, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 1, 
      };
      im3 << BitArray(vec3, 49);
      
      TEST_ASSERT(im2==im3);
      if (retVal==RES_ERR)
      {
	  im3.printSelf(1);
	  im2.printSelf(1);
      }
  }
};

class Test_Sup : public TestCase
{
  virtual void run()
  {
      typedef Bit dataType;
      typedef Image<dataType> imType;
      
      imType im1(7,7);
      imType im2(im1);
      imType im3(im1);
      
      bool vec1[] = {
	0, 0, 0, 0, 0, 1, 1,
	1, 1, 0, 0, 0, 1, 0,
	0, 0, 0, 0, 0, 1, 0,
	0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 1, 0, 0,
	0, 0, 0, 0, 0, 0, 1,
	0, 1, 0, 0, 0, 1, 1,
      };
      im1 << BitArray(vec1, 49);
      
      bool vec2[] = {
	0, 1, 1, 0, 0, 1, 0,
	1, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0,
	0, 1, 0, 1, 0, 0, 1,
	0, 0, 0, 0, 0, 0, 0,
	1, 1, 1, 0, 0, 0, 1,
	0, 1, 0, 0, 1, 0, 0,
      };
      im2 << BitArray(vec2, 49);
      
      bool vec3[] = {
	0, 1, 1, 0, 0, 1, 1, 
	1, 1, 0, 0, 0, 1, 0, 
	0, 0, 0, 0, 0, 1, 0, 
	0, 1, 0, 1, 0, 0, 1, 
	0, 0, 0, 0, 1, 0, 0, 
	1, 1, 1, 0, 0, 0, 1, 
	0, 1, 0, 0, 1, 1, 1
      };
      
      sup(im1, im2, im3);
      
      im2 << BitArray(vec3, 49);
      
      TEST_ASSERT(im2==im3);
      if (retVal==RES_ERR)
      {
	  im2.printSelf(1);
	  im3.printSelf(1);
      }
      
  }
};



int main(int argc, char *argv[])
{
      TestSuite ts;
      ADD_TEST(ts, Test_Copy);
      ADD_TEST(ts, Test_Trans);
      ADD_TEST(ts, Test_Sup);
      
      Image<Bit> im1(256,256);
      fill(im1, Bit(1));
      im1.show();
      Gui::execLoop();
      return ts.run();
  
}

