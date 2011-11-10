
#include <stdio.h>
#include <time.h>

//#include <boost/signal.hpp>
//#include <boost/bind.hpp>

#include "DImage.h"
#include "DImageHistogram.hpp"

#ifdef BUILD_GUI
#include <QApplication>
// #include "DGui.h"
#endif // BUILD_GUI

#define bench(func, args) \
      t1 = clock(); \
      for (int i=0;i<nRuns;i++) \
	func args; \
        cout << #func << ": " << 1E3 * double(clock() - t1) / CLOCKS_PER_SEC / nRuns << " ms" << endl;


int testStretchHist()
{
      Image_UINT8 im1(4,4);
      Image_UINT8 im2(4,4);
      Image_UINT8 im3(4,4);

      UINT8 vec1[16] = { 50, 51, 52, 50, 
			 50, 55, 60, 45, 
			 98, 54, 65, 50, 
			 35, 59, 20, 48 };
			 
      UINT8 vec2[16] = { 98,   101,  104,  98,
			 98,   114,  130,  81, 
			 255,  111,  147,  98,  
			 49,   127,  0,    91};
			 
      im1 << vec1;
      im2 << vec2;
      
      stretchHist(im1, im3);
      
      return im2==im3;
}

int main(int argc, char *argv[])
{
#ifdef BUILD_GUI
    QApplication qapp(argc, argv);
#endif // BUILD_GUI
    
      Image_UINT8 im1(4,4);
      Image_UINT8 im2(4,4);
      Image_UINT8 im3(4,4);
      Image_UINT8 im4(4,4);

      UINT8 vec1[16] = { 50, 51, 52, 50, \
			 50, 55, 60, 45, \
			 98, 54, 65, 50, \
			 35, 59, 20, 48 };
			 
      UINT8 vec2[16] = { 10, 51, 20, 10, \
			 40, 15, 10, 15, \
			 58, 24, 25, 50, \
			 15, 29, 10, 48 };
			 
      UINT8 vec3[16] = { 50, 51, 52, 50, \
			 50, 55, 58, 45, \
			 58, 54, 58, 50, \
			 35, 58, 20, 48 }; 

      im1 << vec1;
//       im2 << vec2;
//       inf(im1, im2, im3);
      thresh(im1, UINT8(100), im4);
//       inf(im1, im2, im3);
      
      testStretchHist();
      
      im4.printSelf(1);
}

