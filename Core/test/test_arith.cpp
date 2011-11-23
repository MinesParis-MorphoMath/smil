
#include <stdio.h>
#include <time.h>

//#include <boost/signal.hpp>
//#include <boost/bind.hpp>

#include "DImage.h"
#include "DImageArith.hpp"

#ifdef BUILD_GUI
#include <QApplication>
#endif // BUILD_GUI

#define bench(func, args) \
      t1 = clock(); \
      for (int i=0;i<nRuns;i++) \
	func args; \
        cout << #func << ": " << 1E3 * double(clock() - t1) / CLOCKS_PER_SEC / nRuns << " ms" << endl;


#ifdef __SSE__

#include <emmintrin.h>
	
// void testAdd(Image_UINT8 &im1, Image_UINT8 &im2, Image_UINT8 &im3)
// {
//     __m128i r0,r1;
//     int size = im1.getWidth();
//     int nlines = im1.getLineCount();
//     
// //     addLine<UINT8> al;
//         
//     for(int l=0;l<nlines;l++)
//     {
//     UINT8 *lineIn1 = im1.getLines()[l];
//     UINT8 *lineIn2 = im2.getLines()[l];
//     UINT8 *lineOut = im3.getLines()[l];
//     
// //     al._exec(lineIn1, lineIn2, size, lineOut);
// 	for(int i=0 ; i<size+16 ; i+=16)  
// 	{
// 	  r0 = _mm_load_si128((__m128i *) lineIn1);
// 	  r1 = _mm_load_si128((__m128i *) lineIn2);
// // 	  _mm_add_epi8(r0, r1);
// 	  _mm_adds_epi8(r0, r1);
// 	  _mm_store_si128((__m128i *) lineOut,r1);
// 
// 	  lineIn1 += 16;
// 	  lineIn2 += 16;
// 	  lineOut += 16;
// 	}
//     }
// };

#endif // __SSE__


int testAdd()
{
    int w = 20	;
    int h = 2;
    Image_UINT8 im1(w,h), im2(w,h), im3(w,h);
    
    for (int i=0;i<h;i++)
      cout << PTR_OFFSET(im1.getLines()[i]) << endl;
    return 0;
    
    for (int i=0;i<im1.getPixelCount();i++)
    {
	im1.getPixels()[i] = i;
	im2.getPixels()[i] = i;
    }
    
    UINT8 vec1[20] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };
    UINT8 vec2[20] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };
    UINT8 vec3[20] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };

//     im1 << vec1;
//     im2 << vec2;
    
/*    im1.setSize(100,100);
    im2.setSize(100,100);
    im3.setSize(100,100);*/
    
    inf(im1, im2, im3);
    
    im3.printSelf(1);
}

int main(int argc, char *argv[])
{
#ifdef BUILD_GUI
    QApplication qapp(argc, argv);
#endif // BUILD_GUI
    
    testAdd();
    return 0;
    
      Image_UINT8 im1(10,10);
      Image_UINT8 im2;
      Image_UINT8 im3;
      
      Image_UINT16 im4;
      
      int sx = 48; //24;
      int sy = 48; //24;
/*      sx = 40;
      sy = 20;*/
      
      im1.setSize(sx, sy);
      im2.setSize(sx, sy);
      im3.setSize(sx, sy);
      im4.setSize(sx, sy);
      
      sup(im1, im2, im3);
      
      return 0;
     fill(im1, UINT8(100));
     fill(im2, UINT8(5));
     
      int t1 = clock();

      int nRuns = (int)1E3;
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
      
      UINT8 v[2];
      
      im2 << "/home/faessel/DATA/BANQUE_IMAGES/IVP024-1/Bon/C0805_C22_3_20100326-105216/1.bmp";
      im2 << (UINT8)50;
      im2.setPixel(155, 25,25);
      rangeVal(im2, &v[0], &v[1]);
      cout << (int)minVal(im2) << ", " << (int)maxVal(im2) << endl;
      cout << (int)v[0] << ", " << (int)v[1] << endl;
      
      im5.show();
      
      qapp.exec();
      
//       fill(im1, UINT8(100));
//       fill(im3, UINT8(0));
      
  
}

