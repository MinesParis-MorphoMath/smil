
#include <stdio.h>
#include <time.h>

//#include <boost/signal.hpp>
//#include <boost/bind.hpp>

#include "DImage.h"
#include "DImageArith.hpp"
#include "DImageIO.h"

#ifdef BUILD_GUI
#include <QApplication>
#endif // BUILD_GUI


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



int main(int argc, char *argv[])
{
#ifdef BUILD_GUI
    QApplication qapp(argc, argv);
#endif // BUILD_GUI
    
//      int c;
      Image_UINT8 im1(10,10);
      Image_UINT8 im2;
      Image_UINT8 im3;
      
     
//       readPNGFile("/home/faessel/src/morphee/trunk/utilities/Images/Gray/akiyo_y.png", &im1);
      im1 << "/home/faessel/src/morphee/trunk/utilities/Images/Gray/akiyo_y.png";
      cout << im1;
//       dilate(im1, im3, se);
      
//       im1.show();
//       im3.show();
//       qapp.exec();

//       baseImage *im = createImage(c);
//       copy(im, im);
      
//       maFunc<UINT8> fi;
      
//       fi.test((UINT8)5);
}

