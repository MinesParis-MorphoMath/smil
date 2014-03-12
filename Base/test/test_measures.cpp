/*
 * Smil
 * Copyright (c) 2011 Matthieu Faessel
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


#include "DImage.h"
#include "DBase.h"
#include "DBlobMeasures.hpp"
#include "DBench.h"
#include "DTest.h"
#include "DMeasures.hpp"

using namespace smil;

class Test_MeasureVolAndArea : public TestCase
{
  virtual void run()
  {
      Image_UINT8 im(256,256);
      
      UINT8 **lines = im.getLines();
      
      fill(im, UINT8(0));
      for (UINT j=10;j<60;j++)
	for (UINT i=20;i<70;i+=2)
	  lines[j][i] = 255;

      double surf = area(im);
      double volume = vol(im);
      
      TEST_ASSERT(surf==1250);
      TEST_ASSERT(volume==318750);
  }
};

class Test_MeasureBarycenter : public TestCase
{
  virtual void run()
  {
      Image_UINT8 im(256,256);
      
      UINT8 **lines = im.getLines();
      
      fill(im, UINT8(0));
      for (UINT j=10;j<60;j++)
	for (UINT i=20;i<70;i++)
	  lines[j][i] = 255;

//       double xc = 0, yc = 0;
      double xcTruth = 44.5, ycTruth = 34.5;
      
      Vector_double bary = measBarycenter(im);
      
      TEST_ASSERT(bary[0]==xcTruth);
      TEST_ASSERT(bary[1]==ycTruth);
  }
};

class Test_MeasBoundingBox : public TestCase
{
  virtual void run()
  {
      UINT8 vec[125] = 
      {
	0, 0, 0, 0, 0,
	0, 0, 0, 0, 0,
	0, 0, 0, 1, 0,
	0, 1, 1, 0, 0,
	0, 0, 0, 0, 0,

	0, 0, 0, 0, 0,
	0, 0, 0, 0, 0,
	0, 0, 0, 0, 0,
	0, 0, 0, 0, 0,
	0, 0, 0, 0, 0,

	0, 0, 0, 0, 0,
	0, 0, 0, 0, 0,
	0, 0, 0, 0, 0,
	0, 0, 1, 0, 0,
	0, 0, 0, 0, 0,

	0, 0, 0, 0, 0,
	0, 0, 0, 0, 0,
	0, 0, 0, 0, 0,
	0, 0, 0, 0, 0,
	0, 0, 0, 0, 0,

	0, 0, 0, 0, 0,
	0, 0, 0, 0, 0,
	0, 0, 0, 0, 0,
	0, 0, 0, 0, 0,
	0, 1, 0, 0, 0,
	
      };
      
      
      Image_UINT8 im(5,5,5);
      im << vec;
      
      vector<UINT> bbox = measBoundBox(im);
      
      TEST_ASSERT(bbox[0]==1 && bbox[1]==2 && bbox[2]==0 && bbox[3]==3 && bbox[4]==4 && bbox[5]==4);
  }
};

class Test_MeasInertiaMatrix : public TestCase
{
  virtual void run()
  {
      UINT8 vec[25] = 
      {
	0, 0, 0, 0, 0,
	0, 0, 0, 1, 0,
	0, 0, 1, 1, 0,
	0, 1, 1, 0, 0,
	0, 0, 0, 0, 0,
      };
      
      
      Image_UINT8 im(5,5);
      im << vec;
      
      Vector_double mat = measInertiaMatrix(im, true);

      TEST_ASSERT(mat[0]==5 && mat[1]==11 && mat[2]==11 && mat[3]==22 && mat[4]==27 && mat[5]==27);
  }
};



class Test_MeasCovariance: public TestCase
{
  virtual void run()
  {
      UINT8 vec1[25] = 
      {
	  10, 207,  10, 182, 118,
	222,  96,  36,  14, 147,
	150, 104, 159,  47,  81,
	  15, 130,  98, 103, 182,
	158, 147,  37,  88, 129,
      };
      
      UINT8 vec2[25] = 
      {
	127,  80, 222, 166, 211,
	224,  12, 116, 187, 133,
	230,  13, 143,   1,  84,
	  46, 118,   6, 136,  96,
	240, 170, 200,  91,  31,
      };
      
      
      Image_UINT8 im1(5,5);
      Image_UINT8 im2(im1);
      im1 << vec1;
      im2 << vec2;
      
       Vector_double cov = measCovariance(im1, im2, 1, 0, 0, 5);

      TEST_ASSERT(cov[0]==13929.6);
      TEST_ASSERT(cov[1]==10913);
      TEST_ASSERT(cov[2]==13026.4);
      TEST_ASSERT(cov[3]==14196);
      TEST_ASSERT(cov[4]==10114.8);
      TEST_ASSERT(cov[5]==0);
      
      if (retVal!=RES_OK)
	for (vector<double>::iterator it=cov.begin();it!=cov.end();it++)
	  cout << *it << endl;
      
  }
};


class Test_MeanVal : public TestCase
{
  virtual void run()
  {
      Image_UINT16 im(10,1);
      
      fill(im, UINT16(10));
      im.setPixel(0, UINT16(65000));
      Vector_double res = meanVal(im);

      TEST_ASSERT(res[0]==6509);
      TEST_ASSERT(res[1]==19497);
  }
};

int main(int argc, char *argv[])
{
      TestSuite ts;
      

//       ADD_TEST(ts, Test_MeasureVolAndArea);
//       ADD_TEST(ts, Test_MeanVal);
//       ADD_TEST(ts, Test_MeasureBarycenter);
//       ADD_TEST(ts, Test_MeasBoundingBox);
      ADD_TEST(ts, Test_MeasCovariance);
//       ADD_TEST(ts, Test_MeasInertiaMatrix);
      
      Image_UINT8 im(512,512);
      measAreas(im);
//       UINT BENCH_NRUNS = 1E3;
//       BENCH(measBarycenter, im, &xc, &yc);
//       BENCH(measBarycenters, im);
      
      return ts.run();
}

