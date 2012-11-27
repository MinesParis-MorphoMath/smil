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
#include "DBase.h"
#include "DMorpho.h"
#include "DGui.h"
#include "DIO.h"

// #include "Addons/MorphM/include/private/DMorphMImage.hpp"


// #include "DGraph.hpp"

#include <vector>

void mafunc(const Image<UINT8>& im)
{
}

int main(int argc, char *argv[])
{


    Image_UINT8 im1(256,256);
//     Image_UINT8 im2;

//     im2 << ( (im1>UINT8(100)) & im1 );
//     return 1;
//     if (read("/home/faessel/src/morphee/trunk/utilities/Images/Gray/DNA_small.png", im1)!=RES_OK)
//       read("/home/mat/src/morphee/trunk/utilities/Images/Gray/DNA_small.png", im1);

   read("http://cmm.ensmp.fr/~faessel/smil/images/lena.png", im1);
//     read("lena.png", im1);

//     morphee::Image<UINT8> *mIm = new morphee::Image<UINT8>(512,512);
//     mIm->allocateImage();
//     morphee::ImageInterface *imInt = (morphee::ImageInterface*)(mIm);
//     
// //     dilate((Image<UINT8>)morphIm, im1);
//     
//     ExtImage<UINT8> im2 = morphmImage<UINT8>(*mIm);
//     ExtImage<UINT8> *im3 = new morphmImage<UINT8>(*imInt);
//     fill(*im3, UINT8(127));
//     
//     im3->printSelf();
    im1.show();
//     
//     Gui::execLoop();

}

