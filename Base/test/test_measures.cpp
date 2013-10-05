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


#include "DImage.h"
#include "DBase.h"
#include "DLabelMeasures.hpp"
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

      double xc = 0, yc = 0;
      double xcTruth = 44.5, ycTruth = 34.5;
      
      measBarycenter(im, &xc, &yc);
      
      TEST_ASSERT(xc==xcTruth);
      TEST_ASSERT(yc==ycTruth);
  }
};

class Test_MeasBoundingBox : public TestCase
{
  virtual void run()
  {
      UINT8 vec[125] = 
      {
	0, 0, 0, 0, 0,   0, 0, 0, 0, 0,   0, 0, 0, 0, 0,   0, 0, 0, 0, 0,   0, 0, 0, 0, 0,
	0, 0, 0, 0, 0,   0, 0, 0, 0, 0,   0, 0, 0, 0, 0,   0, 0, 0, 0, 0,   0, 0, 0, 0, 0,
	0, 0, 0, 1, 0,   0, 0, 0, 0, 0,   0, 0, 0, 0, 0,   0, 0, 0, 0, 0,   0, 0, 0, 0, 0,
	0, 1, 1, 0, 0,   0, 0, 0, 0, 0,   0, 0, 1, 0, 0,   0, 0, 0, 0, 0,   0, 0, 0, 0, 0,
	0, 0, 0, 0, 0,   0, 0, 0, 0, 0,   0, 0, 0, 0, 0,   0, 0, 0, 0, 0,   0, 1, 0, 0, 0,
      };
      
      
      Image_UINT8 im(5,5,5);
      size_t xMin, xMax, yMin, yMax, zMin, zMax;
      im << vec;
      
      measBoundBox(im, &xMin, &yMin, &zMin, &xMax, &yMax, &zMax);
      TEST_ASSERT(xMin==1 && yMin==0 && zMin==2 && xMax==2 && yMax==4 && zMax==4);
  }
};

class Test_LabelMeasures : public TestCase
{
  virtual void run()
  {
      Image_UINT8 im(256,256);
      
      UINT8 **lines = im.getLines();
      
      fill(im, UINT8(0));
      
      for (UINT j=10;j<60;j++)
	for (UINT i=20;i<70;i++)
	  lines[j][i] = 1;

      for (UINT j=100;j<170;j++)
	for (UINT i=80;i<150;i++)
	  lines[j][i] = 2;

      // Barycenters
	  
      double xc1Truth = 44.5, yc1Truth = 34.5;
      double xc2Truth = 114.5, yc2Truth = 134.5;
      
      map<UINT8, DoublePoint> bMap = measBarycenters(im);
      
      TEST_ASSERT(bMap[1].x==xc1Truth && bMap[1].y==yc1Truth);
      TEST_ASSERT(bMap[2].x==xc2Truth && bMap[2].y==yc2Truth);
      

      // Areas
      
      double area1Thruth = 2500, area2Thruth = 4900;
      map<UINT8, double> areas = measAreas(im);
      
      TEST_ASSERT(areas[1]==area1Thruth);
      TEST_ASSERT(areas[2]==area2Thruth);
  }
};

class Test_MeanVal : public TestCase
{
  virtual void run()
  {
      Image_UINT16 im(10,1);
      
      fill(im, UINT16(10));
      im.setPixel(0, UINT16(65000));
      double mv, stdd;
      DoubleVector res = meanVal(im);

      TEST_ASSERT(res[0]==6509);
      TEST_ASSERT(res[1]==19497);
  }
};

int main(int argc, char *argv[])
{
      TestSuite ts;

      ADD_TEST(ts, Test_MeasureVolAndArea);
      ADD_TEST(ts, Test_MeasureBarycenter);
      ADD_TEST(ts, Test_MeasBoundingBox);
      ADD_TEST(ts, Test_LabelMeasures);
      ADD_TEST(ts, Test_MeanVal);
      
      Image_UINT8 im(512,512);
      measAreas(im);
//       UINT BENCH_NRUNS = 1E3;
//       BENCH(measBarycenter, im, &xc, &yc);
//       BENCH(measBarycenters, im);
      
      return ts.run();
}

