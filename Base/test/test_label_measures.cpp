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

#include "Core/include/DCore.h"
#include "DBase.h"
#include "DBlobMeasures.hpp"
#include "DMeasures.hpp"

using namespace smil;

class Test_ComputeBlobs : public TestCase
{
  virtual void run()
  {
    Image_UINT8 im(10, 10);
    Image_UINT8::lineType pixels = im.getPixels();

    fill(im, UINT8(0));
    for (int i = 12; i < 27; i++)
      pixels[i] = 127;
    for (int i = 54; i < 76; i++)
      pixels[i] = 255;
    for (int i = 80; i < 100; i++)
      pixels[i] = 255;

    map<UINT8, Blob> blobs = computeBlobs(im, true);

    TEST_ASSERT(blobs[127].sequences[0].offset == 12);
    TEST_ASSERT(blobs[127].sequences[0].size == 8);
    TEST_ASSERT(blobs[127].sequences[1].offset == 20);
    TEST_ASSERT(blobs[127].sequences[1].size == 7);
    TEST_ASSERT(blobs[255].sequences[0].offset == 54);
    TEST_ASSERT(blobs[255].sequences[0].size == 6);
    TEST_ASSERT(blobs[255].sequences[1].offset == 60);
    TEST_ASSERT(blobs[255].sequences[1].size == 10);
    // ...

    //       map<UINT8, Blob>::iterator bit = blobs.begin();
    //       for (bit=blobs.begin();bit!=blobs.end();bit++)
    //       {
    //         Blob::sequences_iterator sit = bit->second.sequences.begin();
    //         for
    //         (sit=bit->second.sequences.begin();sit!=bit->second.sequences.end();sit++)
    //         {
    //           cout << (int)bit->first << " " << (*sit).offset << " " <<
    //           (*sit).size << endl;
    //         }
    //       }
  }
};

class Test_Areas : public TestCase
{
  virtual void run()
  {
    Image_UINT8 im(1024, 1024);
    // Image_UINT8::lineType pixels = im.getPixels();

    fill(im, UINT8(0));
    drawRectangle(im, 200, 200, 512, 512, UINT8(127), 1);

    map<UINT8, double> areas = measAreas(im);
    TEST_ASSERT(areas[127] == 262144);
  }
};

class Test_Barycenters : public TestCase
{
  virtual void run()
  {
    Image_UINT8 im(1024, 1024);
    // Image_UINT8::lineType pixels = im.getPixels();

    fill(im, UINT8(0));
    drawRectangle(im, 200, 200, 50, 50, UINT8(127), 1);
    drawRectangle(im, 600, 200, 70, 70, UINT8(255), 1);

    map<UINT8, Vector_double> barycenters = measBarycenters(im);
    TEST_ASSERT(barycenters[127][0] == 224.5);
    TEST_ASSERT(barycenters[127][1] == 224.5);
    TEST_ASSERT(barycenters[255][0] == 634.5);
    TEST_ASSERT(barycenters[255][1] == 234.5);
  }
};

class Test_MeasureVolumes : public TestCase
{
  virtual void run()
  {
    Image_UINT8 im(10, 10);
    Image_UINT8::lineType pixels = im.getPixels();

    fill(im, UINT8(0));
    for (int i = 12; i < 27; i++)
      pixels[i] = 127;
    for (int i = 54; i < 76; i++)
      pixels[i] = 255;
    for (int i = 80; i < 100; i++)
      pixels[i] = 255;

    map<UINT8, Blob> blobs = computeBlobs(im, true);

    Image_UINT8 im2(im);
    fill(im2, UINT8(100));

    map<UINT8, double> vols = measVolumes(im2, blobs);
    TEST_ASSERT(vols[127] == 1500 && vols[255] == 4200);
  }
};

int main()
{
  TestSuite ts;

  ADD_TEST(ts, Test_ComputeBlobs);
  ADD_TEST(ts, Test_Areas);
  ADD_TEST(ts, Test_Barycenters);
  ADD_TEST(ts, Test_MeasureVolumes);

  return ts.run();
}
