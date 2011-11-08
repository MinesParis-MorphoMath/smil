
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



int main(int argc, char *argv[])
{
#ifdef BUILD_GUI
    QApplication qapp(argc, argv);
#endif // BUILD_GUI
    
      Image_UINT8 im1(10,10);
      Image_UINT8 im2;
      Image_UINT8 im3;
      
      thresh(im1, (UINT8)0, (UINT8)5, (UINT8)255, (UINT8)0, im2);
      im1.show();
      
      qapp.exec();
      
//       fill(im1, UINT8(100));
//       fill(im3, UINT8(0));
      
  
}

