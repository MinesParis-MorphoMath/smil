/*
 * Smil
 * Copyright (c) 2010 Matthieu Faessel
 *
 * This file is part of Smil.
 *
 * Smil is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * Smil is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Smil.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 */



#include <stdio.h>
#include <time.h>

//#include <boost/signal.hpp>
//#include <boost/bind.hpp>

#include "DImage.h"
#include "DImageArith.hpp"
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




int main(int argc, char *argv[])
{
#ifdef BUILD_GUI
    QApplication qapp(argc, argv);
#endif // BUILD_GUI

//      int c;
    Image_UINT8 im1(4,4);
    Image_UINT8 im2(4,4);
    Image_UINT8 im3(4,4);

    UINT8 vec1[16] = { 50, 51, 52, 50, \
                       50, 55, 60, 45, \
                       98, 54, 65, 50, \
                       35, 59, 20, 48
                     };

    UINT8 vec2[16] = { 10, 51, 20, 10, \
                       40, 15, 10, 15, \
                       58, 24, 25, 50, \
                       15, 29, 10, 48
                     };

    UINT8 vec3[16] = { 50, 51, 52, 50, \
                       50, 55, 58, 45, \
                       58, 54, 58, 50, \
                       35, 58, 20, 48
                     };

//       im1 << vec1;
//       im2 << vec2;
//       inf(im1, im2, im3);
    build(im2, im1, im3);
//       inf(im1, im2, im3);

    im3.printSelf(1);
    im3.show();
    qapp.exec();
}

