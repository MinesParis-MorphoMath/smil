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


#include "Core/include/DImage.h"
#include "DQVtkViewer.hpp"
#include "Core/include/DTest.h"

using namespace smil;


class Test_Show : public TestCase
{
  virtual void run()
  {
//       Image_UINT8 im1("crop_300_OK_BIN.vtk");
//       Image_UINT8 im1("crop_300_OK_Labeled_UINT8.vtk");
      Image_UINT16 im1("crop_300_OK_Labeled.vtk");
      
      QVtkViewer<UINT16> viewer(im1);
      viewer.showLabel();
//       viewer.show();
//       im1 << UINT8(127);
      
      Gui::execLoop();
  }
};

int main()
{
      TestSuite ts;

      ADD_TEST(ts, Test_Show);
      
      return ts.run();
}

