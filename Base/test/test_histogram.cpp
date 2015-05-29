/*
 * Smil
 * Copyright (c) 2011-2015 Matthieu Faessel
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


#include "Core/include/DCore.h"
#include "DImageHistogram.hpp"

#include "Gui/include/DGui.h"

using namespace smil;

class Test_Histogram : public TestCase
{
  virtual void run()
  {
      Image_UINT8 im1(4,4);
      
      UINT8 vec1[16] = { 50, 51, 45, 50,
                        50, 35, 254, 45,
                        254, 254, 254, 50,
                        35, 45, 254, 48
                      };
      im1 << vec1;
      
      map<UINT8, UINT> hist = histogram(im1, true);
      map<UINT8, UINT> truth;
      
      for (int i=0;i<256;i++)
        truth.insert(pair<UINT8, UINT>(i, 0));
      
      truth[35] = 2;
      truth[45] = 3;
      truth[48] = 1;
      truth[50] = 4;
      truth[51] = 1;
      truth[254] = 5;
     
      TEST_ASSERT(hist==truth);
      
      if (retVal!=RES_OK)
      {
          map<UINT8, UINT>::iterator it1 = hist.begin();
          map<UINT8, UINT>::iterator it2 = truth.begin();
          for (;it1!=hist.end();it1++,it2++)
            cout << int(it1->first) << ": " << int(it1->second) << " " << int(it2->second) << endl;
      }
  }
};


class Test_Threshold : public TestCase
{
  virtual void run()
  {
      Image_UINT8 im1(4,4);
      Image_UINT8 im2(4,4);
      Image_UINT8 im3(4,4);

      UINT8 vec1[16] = 
      { 
        150, 21, 52, 50,
        50, 55, 60, 45,
        98, 54, 65, 50,
        35, 59, 20, 48
      };

      UINT8 vec2[16] = 
      { 
        255, 0, 0, 0,
        0, 0, 255, 0,
        255, 0, 255, 0,
        0, 0, 0, 0,
      };

      im1 << vec1;
      im2 << vec2;

      threshold(im1, UINT8(60), im3);

      TEST_ASSERT(im2==im3);
  }
};

class Test_Stretch_Histogram : public TestCase
{
  virtual void run()
  {
      Image_UINT8 im1(4,4);
      Image_UINT8 im2(4,4);
      Image_UINT8 im3(4,4);

      UINT8 vec1[16] = { 
	50, 51, 52, 50,
	50, 55, 60, 45,
	98, 54, 65, 50,
	35, 59, 20, 48
      };

      UINT8 vec2[16] = { 
	98, 101, 105,  98,
	98, 114, 131,  82,
	255, 111, 147,  98,
	49, 128,   0,  92,
      };

      im1 << vec1;
      im2 << vec2;

      stretchHist(im1, im3);

      TEST_ASSERT(im2==im3);

      if (retVal!=RES_OK)
      {
          im2.printSelf(1);
          im3.printSelf(1);
      }
  }
};

class Test_Otsu : public TestCase
{
  virtual void run()
  {
      Image_UINT8 im1(10,10);

      UINT8 vec1[100] = 
      { 
          113, 77, 84, 185, 20, 55, 150, 198, 99, 49, 
          130, 7, 186, 99, 233, 150, 209, 160, 203, 70, 
          95, 25, 79, 64, 241, 237, 145, 50, 16, 100, 
          97, 219, 238, 214, 69, 29, 188, 102, 183, 206, 
          221, 234, 124, 27, 6, 1, 53, 249, 107, 162, 
          254, 93, 203, 36, 176, 223, 181, 88, 11, 69, 
          80, 250, 36, 37, 99, 101, 91, 67, 224, 26, 
          10, 98, 89, 139, 118, 56, 252, 63, 123, 169, 
          241, 82, 223, 215, 118, 191, 241, 212, 47, 232, 
          118, 227, 128, 123, 78, 211, 95, 21, 121, 148
      };

      im1 << vec1;
      
      vector<UINT8> tvals = otsuThresholdValues(im1, 2);
//       for (vector<UINT8>::iterator it=tvals.begin();it!=tvals.end();it++)
//           cout << int(*it) << endl;
      
      TEST_ASSERT(tvals[0]==70);
      TEST_ASSERT(tvals[1]==162);
      
  }
};

int main()
{
    TestSuite ts;

    ADD_TEST(ts, Test_Histogram);
    ADD_TEST(ts, Test_Threshold);
    ADD_TEST(ts, Test_Stretch_Histogram);
    ADD_TEST(ts, Test_Otsu);

    return ts.run();
}

