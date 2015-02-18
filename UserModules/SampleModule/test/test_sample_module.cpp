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


#include "SampleModule.hpp"

using namespace smil;


class Test_SampleModule : public TestCase
{
  virtual void run()
  {
      UINT8 vec1[16] = {
        1,   2,   3,   4,
        5,   6,   7,   8,
        9,  10,  11,  12,
        13,  14,  15,  16,
      };
      
      Image_UINT8 im1(4,4), im2(im1);
      im1 << vec1;
      
      UINT8 vecTruth[16] = {
        254, 253, 252, 251,
        250, 249, 248, 247,
        246, 245, 244, 243,
        242, 241, 240, 239,
      };
      
      Image_UINT8 imTruth(im1);
      imTruth << vecTruth;
      
      samplePixelFunction(im1, im2);
      
      TEST_ASSERT(im2==imTruth)
      
      if (retVal!=RES_OK)
      {
          im2.printSelf(1);
          imTruth.printSelf(1);
      }
      
  }
};

int main(int argc, char *argv[])
{
      TestSuite ts;

      ADD_TEST(ts, Test_SampleModule);
      
      return ts.run();
}

