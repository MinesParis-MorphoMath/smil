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

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>
#include <iostream>
#include <fstream>



using namespace smil;


class Test_Graph : public TestCase
{
    virtual void run()
    {
    }
};


vector<Edge> MST(const Graph<UINT> &graph)
{
    std::set<size_t> visitedInd;
    std::priority_queue<Edge> pq;
    vector<Edge> mst;
    
    int u = (*graph.nodeEdges.begin()).first;
    visitedInd.insert(u);
    vector<UINT> const& neigh = graph.nodeEdges.at(u);
    for (vector<UINT>::const_iterator it=neigh.begin();it!=neigh.end();it++)
    {
	pq.push(graph.edges[*it]);
    }
    
    while(!pq.empty())
    {
	Edge edge = pq.top();
	pq.pop();
	
	if(visitedInd.find(edge.source) == visitedInd.end())
	  u = edge.source;
        else if (visitedInd.find(edge.target) == visitedInd.end())
	  u = edge.target;
	else u = -1;
	
	if (u>=0)
        {
            mst.push_back(edge);
	    visitedInd.insert(u);
	    vector<UINT> const& neigh = graph.nodeEdges.at(u);
	    for (vector<UINT>::const_iterator it=neigh.begin();it!=neigh.end();it++)
	    {
		pq.push(graph.edges[*it]);
	    }
        }
    }
    return mst;
}
  
int main()
{
  using namespace boost;
  typedef adjacency_list < vecS, vecS, undirectedS, no_property, property < edge_weight_t, int > > boost_Graph;
  typedef graph_traits < boost_Graph >::edge_descriptor boost_Edge;
  typedef graph_traits < boost_Graph >::vertex_descriptor Vertex;
  typedef std::pair<int, int> E;

  const int num_nodes = 5;
  E edge_array[] = { E(0, 2), E(1, 3), E(1, 4), E(2, 1), E(2, 3),
    E(3, 4), E(4, 0), E(4, 1)
  };
  int weights[] = { 1, 1, 2, 7, 3, 1, 1, 1 };
  std::size_t num_edges = sizeof(edge_array) / sizeof(E);
#if defined(BOOST_MSVC) && BOOST_MSVC <= 1300
  boost_Graph g(num_nodes);
  property_map<boost_Graph, edge_weight_t>::type weightmap = get(edge_weight, g);
  for (std::size_t j = 0; j < num_edges; ++j) {
    boost_Edge e; bool inserted;
    tie(e, inserted) = add_edge(edge_array[j].first, edge_array[j].second, g);
    weightmap[e] = weights[j];
  }
#else
  boost_Graph g(edge_array, edge_array + num_edges, weights, num_nodes);
#endif
  property_map < boost_Graph, edge_weight_t >::type weight = get(edge_weight, g);
  std::vector < boost_Edge > spanning_tree;

  kruskal_minimum_spanning_tree(g, std::back_inserter(spanning_tree));

  std::cout << "Print the edges in the MST:" << std::endl;
  for (std::vector < boost_Edge >::iterator ei = spanning_tree.begin();
       ei != spanning_tree.end(); ++ei) {
    std::cout << source(*ei, g) << " <--> " << target(*ei, g)
      << " with weight of " << weight[*ei]
      << std::endl;
  }

  std::ofstream fout("figs/kruskal-eg.dot");
  fout << "graph A {\n"
    << " rankdir=LR\n"
    << " size=\"3,3\"\n"
    << " ratio=\"filled\"\n"
    << " edge[style=\"bold\"]\n" << " node[shape=\"circle\"]\n";
  graph_traits<boost_Graph>::edge_iterator eiter, eiter_end;
  for (tie(eiter, eiter_end) = edges(g); eiter != eiter_end; ++eiter) {
    fout << source(*eiter, g) << " -- " << target(*eiter, g);
    if (std::find(spanning_tree.begin(), spanning_tree.end(), *eiter)
        != spanning_tree.end())
      fout << "[color=\"black\", label=\"" << get(edge_weight, g, *eiter)
           << "\"];\n";
    else
      fout << "[color=\"gray\", label=\"" << get(edge_weight, g, *eiter)
           << "\"];\n";
  }
  fout << "}\n";


  
    typedef std::pair<int, int> E;
    
    Graph<UINT> graph;
    graph.addEdge(Edge(0,2, 1));
    graph.addEdge(Edge(1,3, 1));
    graph.addEdge(Edge(1,4, 2));
    graph.addEdge(Edge(2,1, 7));
    graph.addEdge(Edge(2,3, 3));
    graph.addEdge(Edge(3,4, 1));
    graph.addEdge(Edge(4,0, 1));
    graph.addEdge(Edge(4,1, 1));
    
    
//     for (std::vector< Node<UINT> >::const_iterator it=graph.nodes.begin();it!=graph.nodes.end();it++)
//       cout << (*it).index << " (" << (*it).value << ")" << endl;
      
//     for (std::vector<Edge>::const_iterator it=graph.edges.begin();it!=graph.edges.end();it++)
//       cout << (*it).source << "-" << (*it).target << " weight: " << (*it).weight <<  endl;
    
    vector<Edge> mstG = MST(graph);
    for (std::vector<Edge>::const_iterator it=mstG.begin();it!=mstG.end();it++)
      cout << (*it).source << "-" << (*it).target << " weight: " << (*it).weight <<  endl;
    
    return EXIT_SUCCESS;
}

