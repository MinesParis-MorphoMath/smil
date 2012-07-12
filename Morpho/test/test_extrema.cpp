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



#include <stdio.h>
#include <time.h>

//#include <boost/signal.hpp>
//#include <boost/bind.hpp>

#include "DCore.h"
#include "DMorpho.h"
#include "DIO.h"

#include "DGui.h"


#define bench(func, args) \
      t1 = clock(); \
      for (int i=0;i<nRuns;i++) \
	func args; \
        cout << #func << ": " << 1E3 * double(clock() - t1) / CLOCKS_PER_SEC / nRuns << " ms" << endl;




int main(int argc, char *argv[])
{

    Core::initialize();
    
//      int c;
    Image_UINT8 im1(4,4);
    Image_UINT8 im2(4,4);
    Image_UINT8 im3(4,4);

    UINT8 vec1[16] = { 50, 51, 52, 50, \
                       50, 55, 60, 45, \
                       98, 54, 65, 50, \
                       35, 59, 20, 48
                     };

    UINT8 vec2[16] = { 10, 51, 20, 10, \
                       40, 15, 10, 15, \
                       58, 24, 25, 50, \
                       15, 29, 10, 48
                     };

    UINT8 vec3[16] = { 50, 51, 52, 50, \
                       50, 55, 58, 45, \
                       58, 54, 58, 50, \
                       35, 58, 20, 48
                     };

//       im1 << vec1;
//       im2 << vec2;
//       inf(im1, im2, im3);
    build(im2, im1, im3);
//       inf(im1, im2, im3);

    im3.printSelf(1);
    im3.show();
    
    Gui::execLoop();
}

