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
#include "DImageDraw.hpp"
#include "DTest.h"

using namespace smil;


class Test_DrawText : public TestCase
{
  virtual void run()
  {

#ifdef USE_FREETYPE

      Image_UINT8 im1(20,15);
      Image_UINT8 im2(20,15);
      fill(im1, UINT8(127));
      
      TEST_ASSERT(drawText(im1, 4, 10, "ok", "/usr/share/fonts/truetype/ttf-dejavu/DejaVuSans.ttf", 10)==RES_OK);
      
      UINT8 vec[20*15] = 
      {   
	127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 
	127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 
	127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 255, 127, 127, 127, 127, 127, 127, 127, 127, 
	127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 255, 127, 127, 127, 127, 127, 127, 127, 127, 
	127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 255, 127, 127, 127, 127, 127, 127, 127, 127, 
	127, 127, 127, 127, 127, 127, 255, 255, 255, 127, 127, 255, 127, 127, 255, 127, 127, 127, 127, 127, 
	127, 127, 127, 127, 127, 255, 127, 127, 127, 255, 127, 255, 127, 255, 127, 127, 127, 127, 127, 127, 
	127, 127, 127, 127, 127, 255, 127, 127, 127, 255, 127, 255, 255, 127, 127, 127, 127, 127, 127, 127, 
	127, 127, 127, 127, 127, 255, 127, 127, 127, 255, 127, 255, 127, 255, 127, 127, 127, 127, 127, 127, 
	127, 127, 127, 127, 127, 127, 255, 255, 255, 127, 127, 255, 127, 127, 255, 127, 127, 127, 127, 127, 
	127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 
	127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 
	127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 
	127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 
	127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 
      };
      
      im2 << vec;
//       im1.printSelf(1);
      TEST_ASSERT(im1==im2);
      
#endif // USE_FREETYPE
      
  }
};

int main(int argc, char *argv[])
{
      TestSuite ts;

      ADD_TEST(ts, Test_DrawText);
      
      return ts.run();
}

