
#include <stdio.h>
#include <time.h>

#include <boost/signal.hpp>
#include <boost/bind.hpp>

#include "DImage.h"
#include "DImageArith.hpp"
#include "DImageMorph.hpp"
// #include "D_ImageIO_PNG.h"

#ifdef USE_QT
#include <QApplication>
#endif // USE_QT


#define bench(func, args) \
      t1 = clock(); \
      for (int i=0;i<nRuns;i++) \
	func args; \
        cout << #func << ": " << 1E3 * double(clock() - t1) / CLOCKS_PER_SEC / nRuns << " ms" << endl;


#ifdef __SSE__
   
void testAdd(Image_UINT8 &im1, Image_UINT8 &im2, Image_UINT8 &im3)
{
    __m128i r0,r1;
    int size = im1.getWidth();
    int nlines = im1.getLineCount();
    
//     addLine<UINT8> al;
        
    for(int l=0;l<nlines;l++)
    {
    UINT8 *lineIn1 = im1.getLines()[l];
    UINT8 *lineIn2 = im2.getLines()[l];
    UINT8 *lineOut = im3.getLines()[l];
    
//     al._exec(lineIn1, lineIn2, size, lineOut);
	for(int i=0 ; i<size+16 ; i+=16)  
	{
	  r0 = _mm_load_si128((__m128i *) lineIn1);
	  r1 = _mm_load_si128((__m128i *) lineIn2);
// 	  _mm_add_epi8(r0, r1);
	  _mm_adds_epi8(r0, r1);
	  _mm_store_si128((__m128i *) lineOut,r1);

	  lineIn1 += 16;
	  lineIn2 += 16;
	  lineOut += 16;
	}
    }
};

#endif // __SSE__


void func(StrElt *se)
{
    cout << "se base" << endl;
}

void func(hSE *se)
{
    cout << "hSE" << endl;
}



int main(int argc, char *argv[])
{
#ifdef USE_QT
    QApplication qapp(argc, argv);
#endif // USE_QT
    
      int c;
      Image_UINT8 im1(10,10);
      Image_UINT8 im2;
      Image_UINT8 im3;
      
      Image_UINT16 im4;
      
      int sx = 1000;
      int sy = 1000;
/*      sx = 40;
      sy = 20;*/
      
      im1.setSize(sx, sy);
      im2.setSize(sx, sy);
      im3.setSize(sx, sy);
      im4.setSize(sx, sy);
      
     fillIm(im1, UINT8(100));
     fillIm(im2, UINT8(5));
     
      int t1 = clock();

      int nRuns = 1E3;
      UINT8 val = 10;
      
//       bench(fillIm, (im3, val));
//       bench(copyIm, (im1, im3));
//       bench(copyIm, (im1, im4));
//       bench(invIm, (im1, im2));
//       bench(infIm, (im1, im2, im3));
//       bench(infIm, (im1, val, im3));
      bench(supIm, (im1, im2, im3));
//       bench(supIm, (im1, val, im3));
//       bench(addIm, (im1, im2, im3));
//       bench(addNoSatIm, (im1, im2, im3));
//       bench(addIm, (im1, val, im3));
//       bench(subIm, (im1, im2, im3));
//       bench(subIm, (im1, val, im3));
//       bench(grtIm, (im1, im2, im3));
//       bench(divIm, (im1, im2, im3));
//       bench(mulIm, (im1, im2, im3));
//       bench(mulIm, (im1, val, im3));
//       bench(mulNoSatIm, (im1, im2, im3));
//       bench(mulNoSatIm, (im1, val, im3));
      
// 	bench(testAdd, (im1, im2, im3));
//       bench(supIm, (im1, im2, im3));
      
      im3.printSelf(sx < 50);
      
/*      fill((UINT8)1, im1);
      fill((UINT8)2, im2);*/
      
      Image_UINT8 im5(10,10), im6(10,10);
//       fill(UINT8(5), im6);
//       im5 = im1 + im2;

      fillIm(im5, UINT8(100));
      StrElt se = hSE();
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
      
      supLine<UINT8> f;
//       unaryMorphImageFunction<UINT8, supLine<UINT8> > mf;
      bench(dilateIm, (im1, im3));
      bench(volIm, (im1));
//       im6.show();
      
//       addIm(im1, im2, im5);
//       im5.printSelf(sx < 50);

//       im5.show();
      
//       qapp.Exec();
      
      fillIm(im1, UINT8(100));
      fillIm(im3, UINT8(0));
      
      dilateIm(im1, im3, se);
      
//       im1.show();
//       im3.show();
//       qapp.exec();

//       baseImage *im = createImage(c);
//       copy(im, im);
      
//       maFunc<UINT8> fi;
      
//       fi.test((UINT8)5);
}

