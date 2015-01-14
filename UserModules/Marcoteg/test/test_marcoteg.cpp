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


#include "Core/include/DCore.h"
#include "Morpho/include/DMorpho.h"
#include "Morpho/include/private/DMorphoGraph.hpp"

#include "MagicWand.hpp"
//#include "DTest.h"



using namespace smil;


class Test_Marcoteg : public TestCase
{
  virtual void run()
  {
//      Image_UINT8 im1(256, 256);
//      Image_UINT8 im2(im1);
//                Graph<UINT8,UINT8> g;
                //      Graph <UINT,UINT16> g;
//      magicWandSeg(im1,im2,g);

      typedef UINT8 dataType;
      typedef Image<dataType> imType;
      
      imType im1(7,7);
      imType imgra(im1);
      imType imMark(im1,"UINT16");
      imType imMos(imMark);

      imType imtmp(im1);
      imType imMW(im1);
      
      // Mosaic
      dataType vec1[] = {
	100, 100, 100, 179, 179,179,179,
	100, 100, 100, 179, 179,179,179,
	100, 100, 44,44,44,179,179,
	100, 100, 44,44,44,179,179,
	148, 148, 44,44,44,235,235,
	148, 148, 148,148,235,235,235,
	148, 148, 148,148,235,235,235,
      };
      im1 << vec1;
      
      // Values (gradient)
      dataType vecGra[] = {
	0, 0, 0, 79, 79, 0, 0, 
	0, 0, 0, 10, 10, 10, 10, 
	0, 2, 0, 7, 20, 10, 10, 
	2, 2, 0, 10, 10, 10, 10, 
	2, 0, 3, 0, 10, 4, 10, 
	2, 0, 3, 0, 0, 4, 0, 
	0, 0, 3, 0, 4, 4, 0
      };
      imgra << vecGra;

      // Values (MW)
      dataType vecMW[] = {
	0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 
	10, 0, 0, 0, 0, 0, 0, 
	10, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0
      };
      imMW << vecMW;



//graph = watershedExtinctionGraph(imgra,imMark,imMos,"v");

      
      Graph<UINT8,UINT8> graph;
      graph.addEdge(1,2,1);
      graph.addEdge(2,5,2);
      graph.addEdge(1,4,3);
      graph.addEdge(1,3,4);

      //      Graph <UINT,UINT16> g;
      magicWandSeg(imMark,imMos,graph);

      vector<Edge<UINT8> > trueEdges;
      trueEdges.push_back(Edge<UINT8>(1,0,7));
      trueEdges.push_back(Edge<UINT8>(2,0,0));
      trueEdges.push_back(Edge<UINT8>(3,2,2));
      trueEdges.push_back(Edge<UINT8>(3,0,0));
      

      TEST_ASSERT(trueEdges==graph.getEdges());
  }
};

int main(int argc, char *argv[])
{
      TestSuite ts;

      ADD_TEST(ts, Test_Marcoteg);
      
      return ts.run();
}

