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
#include "IO/include/DIO.h"
#include "Gui/include/DGui.h"

using namespace smil;


class Test_VTK_RW : public TestCase
{
  virtual void run()
  {
    typedef UINT8 T;
    const char *fName = "_smil_io_tmp.vtk";
    
    Image<T> im1(3, 3, 2);
    T tab[] = { 28, 2, 3,
                 2, 5, 6,
                 3, 8, 9,
                 4, 11, 12,
                 5, 15, 16,
                 6, 18, 19 };
    im1 << tab;
    TEST_ASSERT( write(im1, fName)==RES_OK );
    
    Image<T> im2;
    
    TEST_ASSERT( read(fName, im2)==RES_OK );
    
    TEST_ASSERT(im1==im2);
  }
};

int main(void)
{
      TestSuite ts;

      ADD_TEST(ts, Test_VTK_RW);
      
//       createFromFile("/home/faessel/src/divers/2012-MSME/tmp.vtk");
      
      return ts.run();
      
}


