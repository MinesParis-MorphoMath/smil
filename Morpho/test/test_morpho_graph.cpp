/*
 * Copyright (c) 2011-2015, Matthieu FAESSEL and ARMINES
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of Matthieu FAESSEL, or ARMINES nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS AND CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#include "Core/include/DCore.h"
#include "DMorpho.h"
#include "DMorphoGraph.hpp"

using namespace smil;

class Test_MosaicToGraph : public TestCase
{
  virtual void run()
  {
      typedef UINT16 dataType1;
      typedef UINT8 dataType2;
      typedef Image<dataType1> imType1;
      typedef Image<dataType2> imType2;
      
      imType1 im1(7,7);
      imType2 im2(im1);
      imType2 im3(im1);
      imType1 im4(im1);
      imType1 im5(im1);
      
      // Mosaic
      dataType1 vec1[] = {
        0, 0, 0, 0, 0, 0, 1, 
        0, 0, 0, 0, 1, 1, 1, 
        0, 2, 0, 0, 1, 1, 1, 
        2, 2, 0, 0, 0, 1, 0, 
        2, 0, 3, 0, 0, 4, 0, 
        2, 0, 3, 0, 0, 4, 0, 
        0, 0, 3, 0, 4, 4, 0
      };
      im1 << vec1;
      
      // Edge Values (gradient)
      dataType2 vec2[] = {
        0, 0, 0, 10, 20, 20, 60, 
        0, 0, 0, 10, 10, 10, 10, 
        0, 2, 0, 7, 20, 10, 10, 
        2, 2, 0, 10, 10, 10, 10, 
        2, 0, 3, 0, 10, 4, 10, 
        2, 0, 3, 0, 0, 4, 0, 
        0, 0, 3, 0, 4, 4, 0
      };
      im2 << vec2;
      
      // Node Values
      dataType2 vec3[] = {
        0, 0, 0, 0, 0, 0, 10, 
        0, 0, 0, 0, 10, 10, 10, 
        0, 20, 0, 0, 10, 10, 10, 
        20, 20, 0, 0, 0, 10, 0, 
        20, 0, 30, 0, 0, 40, 0, 
        20, 0, 30, 0, 00, 40, 0, 
        0, 0, 30, 0, 40, 40, 0
      };
      im3 << vec3;
      
      Graph<> graph;
      mosaicToGraph(im1, im2, im3, graph);
      
      vector<Edge<> > trueEdges;
      trueEdges.push_back(Edge<>(1,0,7));
      trueEdges.push_back(Edge<>(2,0,0));
      trueEdges.push_back(Edge<>(3,2,2));
      trueEdges.push_back(Edge<>(3,0,0));
      trueEdges.push_back(Edge<>(4,0,0));
      trueEdges.push_back(Edge<>(4,1,4));
      
      TEST_ASSERT(trueEdges==graph.getEdges());
      if (retVal!=RES_OK)
          graph.printSelf();
      
      TEST_ASSERT(graph.getNodeValues()[1]==10);
      
//       for (vector<Edge>::const_iterator it=graph.getEdges().begin();it!=graph.getEdges().end();it++)
//         cout << (*it).source << "-" << (*it).target << " (" << (*it).weight << ")" << endl;

//       graph.printSelf();
      
      graph.removeEdge(3,2);
      graph.removeEdge(3,0);
      
      graphToMosaic(im1, graph, im4);
      
      dataType1 vec5[] = {
        0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,
        0,   0,   3,   0,   0,   0,   0,
        0,   0,   3,   0,   0,   0,   0,
        0,   0,   3,   0,   0,   0,   0,
      };
      im5 << vec5;
      
      TEST_ASSERT(im5==im4);
      if (retVal!=RES_OK)
      {
        im4.printSelf(1);
        im5.printSelf(1);
      }
  }
};


class Test_DrawGraph : public TestCase
{
  virtual void run()
  {
      typedef UINT8 dataType;
      typedef Image<dataType> imType;
      
      imType im1(7,7);
      imType im2(im1);
      imType im3(im1);
      imType im4(im1);
      
      // Mosaic
      dataType vec1[] = {
        1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 
        2, 2, 2, 2, 1, 1, 1, 
        2, 2, 2, 2, 4, 1, 4, 
        2, 3, 3, 3, 4, 4, 4, 
        2, 3, 3, 3, 4, 4, 4, 
        3, 3, 3, 3, 4, 4, 4
      };
      im1 << vec1;
      
      fill(im2, UINT8(0));
      
      Graph<> graph;
      mosaicToGraph(im1, im2, graph);
      
      drawGraph(im1, graph, im2);

      dataType vec2[] = {
        0,   0,   0, 255,   0,   0,   0,
        0,   0, 255, 255,   0,   0,   0,
        0, 255, 255,   0, 255,   0,   0,
        0, 255,   0, 255, 255,   0,   0,
        0, 255,   0,   0, 255, 255,   0,
        0, 255, 255, 255,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,
      };
      im1 << vec2;
      
      TEST_ASSERT(im1==im2);
      if (retVal!=RES_OK)
        im2.printSelf(1);
      
  }
};


int main()
{
      TestSuite ts;
      ADD_TEST(ts, Test_MosaicToGraph);
      ADD_TEST(ts, Test_DrawGraph);
      
      return ts.run();
  
}

