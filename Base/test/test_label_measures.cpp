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

class Test_ComputeBlobs : public TestCase
{
  virtual void run()
  {
      Image_UINT8 im(10,10);
      Image_UINT8::lineType pixels = im.getPixels();
      
      fill(im, UINT8(0));
      for (int i=12;i<27;i++)
	pixels[i] = 127;
      for (int i=54;i<76;i++)
	pixels[i] = 255;
      for (int i=80;i<100;i++)
	pixels[i] = 255;
      
      map<UINT8, Blob> blobs = computeBlobs(im);
      
//       map<UINT8, Blob>::iterator bit = blobs.begin();
//       for (bit=blobs.begin();bit!=blobs.end();bit++)
//       {
// 	Blob::sequences_iterator sit = bit->second.sequences.begin();
// 	for (sit=bit->second.sequences.begin();sit!=bit->second.sequences.end();sit++)
// 	{
// 	  cout << (int)bit->first << " " << (*sit).offset << " " << (*sit).size << endl;
// 	}
//       }
      
      TEST_ASSERT(blobs[127].sequences[0].offset==12);
      TEST_ASSERT(blobs[127].sequences[0].size==15);
      TEST_ASSERT(blobs[255].sequences[1].offset==80);
      TEST_ASSERT(blobs[255].sequences[1].size==20);
  }
};

class Test_Areas : public TestCase
{
  virtual void run()
  {
      Image_UINT8 im(1024,1024);
      Image_UINT8::lineType pixels = im.getPixels();
      
      fill(im, UINT8(0));
      drawRectangle(im, 200,200,512,512,UINT8(127), 1);
      
      map<UINT8, double> areas = measAreas2(im);
      map<UINT8, double> areas2 = measAreas2(im);
      cout << areas[127] << endl;
      cout << areas2[127] << endl;
      
  }
};


int main(int argc, char *argv[])
{
      TestSuite ts;

      ADD_TEST(ts, Test_ComputeBlobs);
      ADD_TEST(ts, Test_Areas);
      
      return ts.run();
}

