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
#include "DIO.h"

#include "DGui.h"

#include "Core/include/private/DMultichannelTypes.hpp"
#include "Core/include/DColor.h"

using namespace smil;

namespace smil 
{
    template <>
    void Image< MultichannelType<UINT8,3> >::init()
    {
    }
}

int main(int argc, char *argv[])
{
    Image_UINT8 im1;
    Image_UINT8 im2;
    Image_UINT8 im3;

//     readPNG("http://cmm.ensmp.fr/~faessel/smil/images/barbara.png");

//       im1 << "/home/faessel/src/morphee/trunk/utilities/Images/Gray/akiyo_y.png";
//     if(read("/home/faessel/src/ivp/faessel/DATA/BANQUE_IMAGES/IVP024-1/Bon/C0603_C1_1_20100326-105102/1.bmp", im1)!=RES_OK)
// 	read("/home/mat/src/ivp/faessel/DATA/BANQUE_IMAGES/IVP024-1/Bon/C0603_C1_1_20100326-105102/1.bmp", im1);
// //     im1 >> "/home/faessel/tmp/tmp.bmp";
//     cout << endl;
    string str = im2.getName();
//     im2 << "http://cmm.ensmp.fr/~faessel/smil/images/barbara.png";
//     im2 << "/home/mat/src/morphee/trunk/utilities/Images/Gray/akiyo_y.png";
    im2 << "/home/mat/tmp/akiyo_y.bmp";
    im2 << "/home/mat/tmp/1.bmp";
    im2 << "/home/mat/tmp/arearea.bmp";
//     im2.show();
    
//     Image<RGB> rgbIm;
//     rgbIm << "/home/mat/src/morphee/trunk/utilities/Images/Color/arearea.png";
//     rgbIm >> "/home/mat/tmp/arearea.png";
    
//     rgbIm.show();
    Gui::execLoop();
    
}

