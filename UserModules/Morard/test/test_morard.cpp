/*
 * Smil
 * Copyright (c) 2011-2014 Matthieu Faessel
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


#include "DMorard.hpp"
#include "Core/include/DCore.h"



using namespace smil;


class Test_FastBilateralFilter : public TestCase
{
  virtual void run()
  {
      Image_UINT8 im1(256, 256);
      Image_UINT8 im2(im1);
      
      fastBilateralFilter(im1, 1, 5, 1, 1, im2);
  }
};

int main(int argc, char *argv[])
{
      TestSuite ts;

      ADD_TEST(ts, Test_FastBilateralFilter);
      
      return ts.run();
}

