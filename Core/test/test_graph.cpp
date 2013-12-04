/*
 * Copyright (c) 2011, Matthieu FAESSEL and ARMINES
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


#include "DTest.h"

#include "DGraph.hpp"

#include <iostream>
#include <fstream>



using namespace smil;




class Test_MST : public TestCase
{
    virtual void run()
    {
	Graph graph;
	graph.addEdge(Edge(0,2, 1));
	graph.addEdge(Edge(1,3, 1));
	graph.addEdge(Edge(1,4, 2));
	graph.addEdge(Edge(2,1, 7));
	graph.addEdge(Edge(2,3, 3));
	graph.addEdge(Edge(3,4, 1));
	graph.addEdge(Edge(4,0, 1));
	graph.addEdge(Edge(4,1, 3));
	
	Graph mst = MST(graph);
	vector<Edge> mstTruth;
	mstTruth.push_back(Edge(4,0,1));
	mstTruth.push_back(Edge(3,4,1));
	mstTruth.push_back(Edge(1,3,1));
	mstTruth.push_back(Edge(0,2,1));
	
	TEST_ASSERT(mst.getEdges()==mstTruth);
	
	if (retVal!=RES_OK)
	{
	    for (vector<Edge>::const_iterator it=mst.getEdges().begin();it!=mst.getEdges().end();it++)
	      cout << (*it).source << "-" << (*it).target << " (" << (*it).weight << ")" << endl;
	}
    }
};



int main()
{
      TestSuite ts;

      ADD_TEST(ts, Test_MST);
      
      return ts.run();
}

