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
#include "DImageIO.h"
#include "DImageIO_PNG.h"

#ifdef BUILD_GUI
#include <QApplication>
#include "DGui.h"
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
    Image_UINT8 im1(10,10);
    Image_UINT8 im2;
    Image_UINT8 im3;

    im1 << "/home/faessel/src/morphee/trunk/utilities/Images/Gray/akiyo_y.png";

    im1.show();
    qapp.exec();
}

