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

//#include <boost/signal.hpp>
//#include <boost/bind.hpp>

#include "DImage.h"
#include "DImageArith.hpp"
#include "DLineArith_BIN.hxx"
#include "DMorpho.h"
#include "DImageIO.h"

#ifdef BUILD_GUI
#include "DGui.h"
#include <QApplication>
#endif // BUILD_GUI


#define bench(func, args) \
      t1 = clock(); \
      for (int i=0;i<nRuns;i++) \
	func args; \
        cout << #func << ": " << 1E3 * double(clock() - t1) / CLOCKS_PER_SEC / nRuns << " ms" << endl;



void func(StrElt *se)
{
    cout << "se base" << endl;
}

void func(hSE *se)
{
    cout << "hSE" << endl;
}




#include "DLineArith_BIN.hxx"

int main(int argc, char *argv[])
{
#ifdef BUILD_GUI
    QApplication qapp(argc, argv);
#endif // BUILD_GUI

    int t1 = clock();
    int nRuns = (int)1E3;

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
    
    Image_UINT8 im1(w,h);
    Image_UINT8 im2(im1);
    Image_UINT8 im3(im1);

    fill(im1, UINT8(100));
    fill(im2, UINT8(5));
    
    bench(sup, (bim1, bim2, bim3));
    bench(sup, (im1, im2, im3));
    bench(dilate, (bim1, bim3, hSE()));
    bench(dilate, (im1, im3, hSE()));
//     bench(erode, (bim1, bim3, hSE()));
//     bench(erode, (im1, im3, hSE()));
    

    return 0;
    
    Image_UINT16 im4;

    int sx = 1024;
    int sy = 1024;
    /*      sx = 40;
          sy = 20;*/

    im1.setSize(sx, sy);
    im2.setSize(sx, sy);
    im3.setSize(sx, sy);
    im4.setSize(sx, sy);

    fill(im1, UINT8(100));
    fill(im2, UINT8(5));


    UINT8 val = 10;

//       bench(fill, (im3, val));
//       bench(copy, (im1, im3));
//       bench(copy, (im1, im4));
//       bench(inv, (im1, im2));
//       bench(inf, (im1, im2, im3));
//       bench(inf, (im1, val, im3));
//       bench(sup, (im1, im2, im3));
//       bench(sup, (im1, val, im3));
//       bench(add, (im1, im2, im3));
//       bench(addNoSat, (im1, im2, im3));
//       bench(add, (im1, val, im3));
//       bench(sub, (im1, im2, im3));
//       bench(sub, (im1, val, im3));
//       bench(grt, (im1, im2, im3));
//       bench(div, (im1, im2, im3));
//       bench(mul, (im1, im2, im3));
//       bench(mul, (im1, val, im3));
//       bench(mulNoSat, (im1, im2, im3));
//       bench(mulNoSat, (im1, val, im3));

// 	bench(testAdd, (im1, im2, im3));
//       bench(sup, (im1, im2, im3));

    im3.printSelf(sx < 50);

    /*      fill((UINT8)1, im1);
          fill((UINT8)2, im2);*/

    Image_UINT8 im5(50,50), im6(50,50);
//       fill(UINT8(5), im6);
//       im5 = im1 + im2;

//       fill(im5, UINT8(100));
    StrElt se = hSE();

    im5 << UINT8(127);
//     erode(im5, im6, sSE(5));
//       im5.show();
    im6.show();

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
//       bench(dilate, (im1, im3, se));
//     bench(erode, (im1, im3, se));
//       bench(volIm, (im1));
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

