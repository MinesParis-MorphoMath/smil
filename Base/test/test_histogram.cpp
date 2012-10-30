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
#include "DImageHistogram.hpp"

#include "DGui.h"


class Test_Histogram : public TestCase
{
  virtual void run()
  {
      Image_UINT8 im1(4,4);
      
      UINT8 vec1[16] = { 50, 51, 45, 50,
			50, 35, 255, 45,
			255, 255, 255, 50,
			35, 45, 255, 48
		      };
      im1 << vec1;
      
      map<UINT8, UINT> hist = histogram(im1);
      map<UINT8, UINT> truth;
      
      for (int i=0;i<256;i++)
	truth.insert(pair<UINT8, UINT>(i, 0));
      
      truth[(UINT8)35] = 2;
      truth[(UINT8)45] = 3;
      truth[(UINT8)48] = 1;
      truth[(UINT8)50] = 4;
      truth[(UINT8)51] = 1;
      truth[(UINT8)255] = 5;
     
      map<UINT8, UINT>::iterator it1 = hist.begin();
      map<UINT8, UINT>::iterator it2 = truth.begin();
      for (;it1!=hist.end();it1++,it2++)
      {
// 	cout << int((*it1).first) << ": " << int((*it1).second) << " " << int((*it2).second) << endl;
	TEST_ASSERT((*it1).second==(*it2).second);
      }
  }
};


class Test_Stretch_Histogram : public TestCase
{
  virtual void run()
  {
      Image_UINT8 im1(4,4);
      Image_UINT8 im2(4,4);
      Image_UINT8 im3(4,4);

      UINT8 vec1[16] = { 50, 51, 52, 50,
			50, 55, 60, 45,
			98, 54, 65, 50,
			35, 59, 20, 48
		      };

      UINT8 vec2[16] = { 98,   101,  104,  98,
			98,   114,  130,  81,
			255,  111,  147,  98,
			49,   127,  0,    91
		      };

      im1 << vec1;
      im2 << vec2;

      stretchHist(im1, im3);

      TEST_ASSERT(im2==im3);
  }
};

class Test_Otsu : public TestCase
{
  virtual void run()
  {
      Image_UINT8 im1(4,4);

      UINT8 vec1[16] = 
      { 
	161, 220, 87, 124, 
	208, 148, 13, 239, 
	151, 67, 12, 0, 
	134, 244, 101, 168, 
      };

      im1 << vec1;
      
      vector<UINT8> tvals = otsuThresholdValues(im1);
      for (vector<UINT8>::iterator it=tvals.begin();it!=tvals.end();it++)
	  cout << int(*it) << endl;
      
      TEST_ASSERT(tvals[0]==118);
      
  }
};

int main(int argc, char *argv[])
{
    TestSuite ts;

    ADD_TEST(ts, Test_Histogram);
    ADD_TEST(ts, Test_Stretch_Histogram);
    ADD_TEST(ts, Test_Otsu);

    return ts.run()==RES_OK;
}

