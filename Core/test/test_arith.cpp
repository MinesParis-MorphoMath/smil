
#include <stdio.h>
#include <time.h>

//#include <boost/signal.hpp>
//#include <boost/bind.hpp>

#include "DImage.h"
#include "DImageArith.hpp"

#define bench(func, args) \
      t1 = clock(); \
      for (int i=0;i<nRuns;i++) \
	func args; \
        cout << #func << ": " << 1E3 * double(clock() - t1) / CLOCKS_PER_SEC / nRuns << " ms" << endl;




int main(int argc, char *argv[])
{
      Image_UINT8 im1(10,10);
      Image_UINT8 im2;
      Image_UINT8 im3;
      
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
     
      int t1 = clock();

      int nRuns = (int)1E3;
      UINT8 val = 10;
      
//       bench(fill, (im3, val));
//       bench(copy, (im1, im3));
//       bench(copy, (im1, im4));
      bench(inv, (im1, im2));
//       bench(inf, (im1, im2, im3));
//       bench(inf, (im1, val, im3));
//       bench(sup, (im1, im2, im3));
//       bench(sup, (im1, val, im3));
      bench(add, (im1, im2, im3));
      bench(addNoSat, (im1, im2, im3));
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
      
      Image_UINT8 im5(10,10), im6(10,10);
//       fill(UINT8(5), im6);
//       im5 = im1 + im2;

//       fill(im5, UINT8(100));
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
//       bench(volIm, (im1));
//       im6.show();
      
//       add(im1, im2, im5);
      im5.printSelf(sx < 50);
      cout << im5;

//       im5.show();
      
//       qapp.Exec();
      
//       fill(im1, UINT8(100));
//       fill(im3, UINT8(0));
      
  
}

