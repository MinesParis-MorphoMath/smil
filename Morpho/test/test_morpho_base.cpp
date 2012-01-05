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



#include <stdio.h>
#include <time.h>

#include <cassert>

//#include <boost/signal.hpp>
//#include <boost/bind.hpp>

#include "DImage.h"
#include "DImageArith.hpp"
#include "DLineArith_BIN.hxx"
#include "DMorpho.h"
#include "DImageIO.h"
#include "DTest.h"
#include "DBench.h"

#ifdef BUILD_GUI
#include "DGui.h"
#include <QApplication>
#endif // BUILD_GUI



void func(StrElt *se)
{
    cout << "se base" << endl;
}

void func(hSE *se)
{
    cout << "hSE" << endl;
}



class test_base_BIN : public TestCase
{
    void run()
    {
	im1.setSize(1024,1024);
	im2.setSize(1024,1024);
	fill(im1, true);
	fill(im2, true);
	
	TEST_ASSERT(vol(im1)==im1.getWidth()*im1.getHeight());
	TEST_ASSERT(equ(im1,im2));

	dilate(im1, im2);
    }
    Image<bool> im1, im2;
};

#include "DLineArith_BIN.hxx"

class Bit;

class BitArray
{
public:
  BitArray() : index(0) {}
  BIN_TYPE *array;
  UINT width;
  UINT height;
  UINT bitWidth;
  UINT curX, curY;
  UINT padX;
  UINT index;
  
  inline bool getValue(UINT ind)
  {
      int Y = ind / bitWidth;
      int X = (ind + Y*padX) / BIN::SIZE;
      int x = ind % BIN::SIZE;
      return (array[X] & (1UL << x))!=0;
  }
  inline void setValue(UINT ind, bool val)
  {
      int Y = ind / bitWidth;
      int X = (ind + Y*padX) / BIN::SIZE;
      int x = ind % BIN::SIZE;
      if (val)
	array[X] |= (1UL << x);
      else array[X] &= ~(1UL << x);
  }
  inline Bit operator [] (UINT i);
  inline Bit operator * ();
  inline BitArray& operator ++ (int dummy);
  inline BitArray& operator ++ ();
};

class Bit
{
public:
  Bit() : bitArray(NULL), value(false) {}
  Bit(bool v) : bitArray(NULL), value(v) {}
  BitArray *bitArray;
  UINT index;
  bool value;
  inline operator bool()
  {
      return bitArray->getValue(index);
  }
  inline Bit& operator = (bool v)
  {
      if (bitArray)
	bitArray->setValue(index, v);
      else value = v;
      return *this;
  }
  inline Bit& operator = (Bit &src)
  {
      if (bitArray)
      {
	if (src.bitArray)
	  bitArray->setValue(index, src.bitArray->getValue(index));
	else
	  bitArray->setValue(index, src.value);
      }
      else
      {
	if (src.bitArray)
	  value = src.bitArray->getValue(index);
	else
	  value = src.value;
      }
      return *this;
  }
};

Bit BitArray::operator [] (UINT i)
{
  Bit b;
  b.bitArray = this;
  b.index = i;
  return b;
}

Bit BitArray::operator * ()
{
  Bit b;
  b.bitArray = this;
  b.index = index;
  return b;
}

BitArray& BitArray::operator++(int dummy)
{
  index++;
  return *this;
}

BitArray& BitArray::operator++()
{
  index++;
  return *this;
}

int main(int argc, char *argv[])
{
#ifdef BUILD_GUI
    QApplication qapp(argc, argv);
#endif // BUILD_GUI

   BitArray b;
   b.width = 2;
   b.height = 2;
   b.array = new BIN_TYPE[4];
   for (int i=0;i<4;i++)
     b.array[i] = 0;
   b.bitWidth = 70;
   b.padX = b.width*BIN::SIZE - b.bitWidth;
   
   cout << b.getValue(74) << endl;
   Bit bit = 1;
   *b = bit;
   *++b = *b;
   
   for (int i=0;i<10;i++,++b)
     *b = 1;
   
   for (int i=0;i<140;i++)
      cout << (bool)b[i] << " ";
   cout <<  endl;
   
   
   return 0;
   
    for (int i=0;i<argc;i++)
      cout << argv[i] << " ";
    cout << endl;
    int BENCH_NRUNS = 1E3;

    TestSuite t;
    ADD_TEST(t, test_base_BIN);
    
    t.run();
    
  int iam = 0, np = 1;

  #pragma omp parallel private(iam, np)
  {
    #if defined (_OPENMP)
      np = omp_get_num_threads();
      iam = omp_get_thread_num();
    #endif
    printf("Hello from thread %d out of %d\n", iam, np);
  }
   
    UINT w = 1024, h = 1024, d = 1;
//     UINT w = 768, h = 576;
    
    typedef Image<bool> imType;
    
    imType bim1(w, h, d);
    imType bim2(bim1);
    imType bim3(bim1);
    
    cout << "Width: " << w << endl;
    cout << "Line count: " << bim1.getLineCount() << endl;
        
    fill(bim1, true);
    fill(bim2, false);
    
    Image_UINT8 im1(w,h);
    Image_UINT8 im2(im1);
    Image_UINT8 im3(im1);

    fill(im1, UINT8(100));
    fill(im2, UINT8(5));
    
    sup(bim1, bim2, bim3);
    dilate(bim1, bim2);

    equ(im1, im2);
    
    BENCH_IMG(vol, im1);
    BENCH_IMG(vol, bim1);
    BENCH_IMG(sup, im1, im2, im3);
    BENCH_IMG(sup, bim1, bim2, bim3);
    BENCH_IMG_STR(dilate, "hSE", im1, im3, hSE());
    BENCH_IMG_STR(dilate, "hSE", bim1, bim2, hSE());
    BENCH_IMG_STR(erode, "hSE", im1, im3, hSE());
    BENCH_IMG_STR(erode, "hSE", bim1, bim3, hSE());
    
    
// cout << "err: " << __FILE__ << __LINE__ << __FUNCTION__ << endl;
    return 0;
    
//     Image_UINT16 im4;
// 
//     int sx = 1024;
//     int sy = 1024;
//     /*      sx = 40;
//           sy = 20;*/
// 
//     im1.setSize(sx, sy);
//     im2.setSize(sx, sy);
//     im3.setSize(sx, sy);
//     im4.setSize(sx, sy);
// 
//     fill(im1, UINT8(100));
//     fill(im2, UINT8(5));
// 
// 
//     UINT8 val = 10;
// 
// //       BENCH(fill, (im3, val));
// //       BENCH(copy, (im1, im3));
// //       BENCH(copy, (im1, im4));
// //       BENCH(inv, (im1, im2));
// //       BENCH(inf, (im1, im2, im3));
// //       BENCH(inf, (im1, val, im3));
// //       BENCH(sup, (im1, im2, im3));
// //       BENCH(sup, (im1, val, im3));
// //       BENCH(add, (im1, im2, im3));
// //       BENCH(addNoSat, (im1, im2, im3));
// //       BENCH(add, (im1, val, im3));
// //       BENCH(sub, (im1, im2, im3));
// //       BENCH(sub, (im1, val, im3));
// //       BENCH(grt, (im1, im2, im3));
// //       BENCH(div, (im1, im2, im3));
// //       BENCH(mul, (im1, im2, im3));
// //       BENCH(mul, (im1, val, im3));
// //       BENCH(mulNoSat, (im1, im2, im3));
// //       BENCH(mulNoSat, (im1, val, im3));
// 
// // 	BENCH(testAdd, (im1, im2, im3));
// //       BENCH(sup, (im1, im2, im3));
// 
//     im3.printSelf(sx < 50);
// 
//     /*      fill((UINT8)1, im1);
//           fill((UINT8)2, im2);*/
// 
//     Image_UINT8 im5(50,50), im6(50,50);
// //       fill(UINT8(5), im6);
// //       im5 = im1 + im2;
// 
// //       fill(im5, UINT8(100));
//     StrElt se = hSE();
// 
//     im5 << UINT8(127);
// //     erode(im5, im6, sSE(5));
// //       im5.show();
//     im6.show();

//       se.addPoint(5,5);
//       se.addPoint(5,0);
    /*       se.addPoint(0,0);
          se.addPoint(1,0);
          se.addPoint(1,1);
          se.addPoint(0,1);
          se.addPoint(-1,1);
          se.addPoint(-1,0);
          se.addPoint(-1,-1);
          se.addPoint(0,-1);*/

//      supLine<UINT8> f;
//       unaryMorphImageFunction<UINT8, supLine<UINT8> > mf;
//       BENCH(dilate, (im1, im3, se));
//     BENCH(erode, (im1, im3, se));
//       BENCH(volIm, (im1));
//       im6.show();

//       add(im1, im2, im5);
//       im5.printSelf(sx < 50);
//       cout << im5;

//       im5.show();

//       qapp.Exec();

//       fill(im1, UINT8(100));
//       fill(im3, UINT8(0));

//       readPNGFile("/home/faessel/src/morphee/trunk/utilities/Images/Gray/akiyo_y.png", &im1);
//       im1 << "/home/faessel/src/morphee/trunk/utilities/Images/Gray/akiyo_y.png";
//       dilate(im1, im3, se);

//       im1.show();
//       im3.show();
#ifdef BUILD_GUI
//     qapp.exec();
#endif // BUILD_GUI

//       baseImage *im = createImage(c);
//       copy(im, im);

//       maFunc<UINT8> fi;

//       fi.test((UINT8)5);
}

