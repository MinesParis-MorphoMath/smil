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


#include "MagicWand.hpp"
//#include "DTest.h"



using namespace smil;


class Test_Marcoteg : public TestCase
{
  virtual void run()
  {
      Image_UINT8 im1(256, 256);
      Image_UINT8 im2(im1);
		Graph<UINT8,UINT8> g;
		//      Graph <UINT,UINT16> g;
      magicWandSeg(im1,im2,g);

  }
};

int main(int argc, char *argv[])
{
      TestSuite ts;

      ADD_TEST(ts, Test_Marcoteg);
      
      return ts.run();
}

