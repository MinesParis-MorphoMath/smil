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


/*
Démarrage : /home/faessel/src/Smil/prev-build/bin/test_morpho_base
/home/faessel/src/Smil/prev-build/bin/test_morpho_base 
Hello from thread 0 out of 1
Width: 1024
vol	UINT8	1024x1024	0 secs
sup	UINT8	1024x1024	210 µsecs
sup	BIN	1024x1024	30 µsecs
dilate hSE	UINT8	1024x1024	840 µsecs
dilate hSE	BIN	1024x1024	180 µsecs
dilate sSE	UINT8	1024x1024	2.92 msecs
dilate sSE	BIN	1024x1024	570 µsecs
erode hSE	UINT8	1024x1024	880 µsecs
erode hSE	BIN	1024x1024	200 µsecs
erode sSE	UINT8	1024x1024	3 msecs
erode sSE	BIN	1024x1024	550 µsecs
*/

#include <stdio.h>
#include <time.h>

#include <cassert>

//#include <boost/signal.hpp>
//#include <boost/bind.hpp>

#include "DCore.h"
#include "DMorpho.h"
#include "DIO.h"

#include "DBitArray.h"

void func(StrElt *se)
{
    cout << "se base" << endl;
}

void func(StrElt &se)
{
    cout << "hSE" << endl;
}



// class test_base_BIN : public TestCase
// {
//     void run()
//     {
// 	im1.setSize(1024,1024);
// 	im2.setSize(1024,1024);
// 	fill(im1, true);
// 	fill(im2, true);
// 	
// 	TEST_ASSERT(vol(im1)==im1.getWidth()*im1.getHeight());
// 	TEST_ASSERT(equ(im1,im2));
// 
// 	dilate(im1, im2);
//     }
//     Image<bool> im1, im2;
// };






// template <>
// class Image<Bit> : public baseImage
// {
// public:
//     void printSelf(bool displayPixVals = false) { Image<Bit>::printSelf(); }
//     virtual void* getVoidPointer() {}
//     virtual void modified() {}
// };

inline void copyBits(BitArray &bArr, UINT pos, BitArray::INT_TYPE &intOut)
{
    BitArray::INT_TYPE *intIn = bArr.intArray + pos/BitArray::INT_TYPE_SIZE;
    UINT rPos = pos%BitArray::INT_TYPE_SIZE;
    if (rPos==0)
	intOut = *intIn;
    else
	intOut = (*intIn >> rPos) | (*(intIn+1) << (BitArray::INT_TYPE_SIZE-rPos));
}

// void cpy(BitArray r1, int size, BitArray r2)
// {
//     int intSize = BitArray::INT_SIZE(size);
//     int curSize = 0;
//     
//     int startX = r1.index/BitArray::INT_TYPE_SIZE;
//     int startY = r2.index/BitArray::INT_TYPE_SIZE;
//     int endX = (r1.index+size)/BitArray::INT_TYPE_SIZE;
//     int endY = (r2.index+size)/BitArray::INT_TYPE_SIZE;
//     
//     BitArray::INT_TYPE *b1 = r1.intArray + startX;
//     BitArray::INT_TYPE *b2 = r2.intArray + startY;
//     
//     int startx = r1.index % BitArray::INT_TYPE_SIZE;
//     int endx = MIN(startx + size, BitArray::INT_TYPE_SIZE);
//     
//     BitArray::INT_TYPE maskIn, maskOut;
//     BitArray::INT_TYPE intMax = BitArray::INT_TYPE_MAX();
//     maskIn = (intMax << startx) & (intMax >> (BitArray::INT_TYPE_SIZE-endx));
//     
//     *b2 = maskIn;
// }

// void cpy(Image<Bit> &imIn, Image<Bit> &imOut)
// {
//     typename ImDtTypes<Bit>::sliceType lIn = imIn.getLines();
//     typename ImDtTypes<Bit>::sliceType lOut = imOut.getLines();
//     UINT realWidth = BitArray::INT_SIZE(imIn.getWidth());
//     UINT64 *pixIn = imIn.getPixels().intArray;
//     UINT64 *pixOut = imOut.getPixels().intArray;
// 
//     for (int i=0;i<imIn.getLineCount();i++)
//     {
//       UINT64 *pIn = lIn[i].intArray;
//       UINT64 *pOut = lOut[i].intArray;
//       memcpy(pOut, pIn, realWidth*sizeof(UINT64));
//       
// //       memcpy(pixOut, pixIn, realWidth*sizeof(UINT64));
// //       pixIn+=realWidth;
// //       pixOut+=realWidth;
//     }
// }
template<class T = float, int i = 5> class A
{
   public:
      A();
      int value;
};

template<> class A<> { public: A(); };
template<> class A<double, 10> { public: A(); };

template<class T, int i> A<T, i>::A() : value(i) {
   cout << "Primary template, "
        << "non-type argument is " << value << endl;
}

A<>::A() {
   cout << "Explicit specialization "
        << "default arguments" << endl;
}

A<double, 10>::A() {
   cout << "Explicit specialization "
        << "<double, 10>" << endl;
}


int main(int argc, char *argv[])
{
  
  Image_UINT8 im1(5,5);
  Image_UINT8 im2(im1);
  
  UINT8 vec1[] = {
    0, 0, 1, 1, 0,
    0, 1, 0, 0, 0,
    1, 1, 0, 0, 0,
    1, 0, 1, 0, 1,
    0, 0, 1, 1, 1
  };
  
  im1 << vec1;
  
  label(im1, im2, sSE());
  im2.printSelf(1);
//     
// //     BENCH_IMG(copy, imb1, imb2);
// //     BENCH_IMG(copy, bim1, bim2);
// //     BENCH_IMG(cpy, bim1, bim2);
//     
// //     return 0;
//     
//     BENCH_IMG(vol, im1);
//     BENCH_IMG(vol, bim1);
//     BENCH_IMG(sup, im1, im2, im3);
//     BENCH_IMG(sup, bim1, bim2, bim3);
//     BENCH_IMG(sup, imb1, imb2, imb3);
//     
//     BENCH_IMG_STR(dilate, "hSE", im1, im3, hSE());
//     BENCH_IMG_STR(dilate, "hSE", bim1, bim2, hSE());
//     BENCH_IMG_STR(dilate, "hSE", imb1, imb2, hSE());
//     BENCH_IMG_STR(dilate, "sSE", im1, im3, sSE());
//     BENCH_IMG_STR(dilate, "sSE", bim1, bim2, sSE());
//     BENCH_IMG_STR(dilate, "sSE", imb1, imb2, sSE());
//     
//     BENCH_IMG_STR(erode, "hSE", im1, im3, hSE());
//     BENCH_IMG_STR(erode, "hSE", bim1, bim3, hSE());
//     BENCH_IMG_STR(erode, "hSE", imb1, imb3, hSE());
//     BENCH_IMG_STR(erode, "sSE", im1, im3, sSE());
//     BENCH_IMG_STR(erode, "sSE", bim1, bim3, sSE());
//     BENCH_IMG_STR(erode, "sSE", imb1, imb3, sSE());
//     
}

