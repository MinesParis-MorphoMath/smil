
#include <stdio.h>
#include <time.h>

//#include <boost/signal.hpp>
//#include <boost/bind.hpp>

#include "DImage.h"
#include "DImageArith.hpp"
#include "DMorpho.h"
#include "DImageIO.h"
#include "DGui.h"

#ifdef USE_QT
#include <QApplication>
#endif // USE_QT


#define bench(func, args) \
      t1 = clock(); \
      for (int i=0;i<nRuns;i++) \
	func args; \
        cout << #func << ": " << 1E3 * double(clock() - t1) / CLOCKS_PER_SEC / nRuns << " ms" << endl;




int main(int argc, char *argv[])
{
#ifdef USE_QT
    QApplication qapp(argc, argv);
#endif // USE_QT
    
//      int c;
      Image_UINT8 im1;
      im1 << "/home/mat/src/morphee/trunk/utilities/Images/Gray/akiyo_y.png";
      Image_UINT8 im2(im1);

//       hMinima(im1, 2, im2);
      add(im1, UINT8(10), im2);
      dualBuild(im1, im2, im2);
      im1.show();
      im2.show();
      
      im1.setSize(4, 4);
      UINT8 vec[16] = { 2, 3, 4, 1, \
			2, 5, 7, 1, \
			2, 5, 7, 1, \
			2, 5, 7, 1 };
      im1 << vec;

       qapp.exec();
}

