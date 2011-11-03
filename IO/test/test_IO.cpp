
#include <stdio.h>
#include <time.h>

//#include <boost/signal.hpp>
//#include <boost/bind.hpp>

#include "DImage.h"
#include "DImageIO.h"

#ifdef BUILD_GUI
#include "DGui.h"
#include <QApplication>
#endif // BUILD_GUI


int main(int argc, char *argv[])
{
#ifdef BUILD_GUI
    QApplication qapp(argc, argv);
#endif // BUILD_GUI
    
      Image_UINT8 im1;
      Image_UINT8 im2;
      Image_UINT8 im3;
      
     
//       im1 << "/home/faessel/src/morphee/trunk/utilities/Images/Gray/akiyo_y.png";
      im1 << "/home/faessel/src/ivp/faessel/DATA/BANQUE_IMAGES/IVP024-1/Bon/C0603_C1_1_20100326-105102/1.bmp";
      im1 >> "/home/faessel/tmp/tmp.bmp";
      cout << endl;
      im2 << "/home/faessel/tmp/tmp.bmp";
      im2.show();
//       im3.show();
      qapp.exec();

}

