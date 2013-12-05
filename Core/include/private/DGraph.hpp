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

#ifndef _D_GRAPH_HPP
#define _D_GRAPH_HPP

#include <list>
#include <algorithm>
#include <vector>
#include <utility>
#include <iostream>
#include <stdexcept>
#include <set>
#include <queue>

namespace smil
{
    /**
     * Non-oriented edge
     */
    class Edge
    {
    public:
	Edge(UINT a, UINT b, UINT w=1.)
	  : source(a), target(b), weight(w)
	{
	}
	Edge(const Edge &rhs)
	  : source(rhs.source), target(rhs.target), weight(rhs.weight)
	{
	}
	
	UINT source;
	UINT target;
	UINT weight;
	
	inline bool operator ==(const Edge &rhs) const
	{
	    if (rhs.source==source && rhs.target==target)
	      return true;
	    else if (rhs.source==target && rhs.target==source)
	      return true;
	    return false;
	}
	
	inline bool operator <(const Edge &rhs) const
	{
	    return weight>=rhs.weight;
	}
    };

    
    /**
     * Non-oriented graph
     */
    class Graph
    {
    public:
	
	Graph()
	  : edgeNbr(0)
	{
	}
	
	void addNode(const UINT &ind, const UINT &val=0)
	{
	    nodeValues[ind] = val;
	}
	/**
	 * Add an edge to the graph. 
	 * If checkIfExists is \b true:
	 * 	If the edge doen't exist, create a new one.
	 * 	If the edge already exists, the edge weight will be the minimum between the existing a the new weight.
	 */
	
	void addEdge(const Edge &e, bool checkIfExists=true)
	{
	    if (checkIfExists)
	    {
		vector<Edge>::iterator foundEdge = find(edges.begin(), edges.end(), e);
		if (foundEdge!=edges.end())
		{
		    (*foundEdge).weight = min((*foundEdge).weight, e.weight);
		    return;
		}
	    }
	    edges.push_back(e);
	    nodes.insert(e.source);
	    nodes.insert(e.target);
	    nodeEdges[e.source].push_back(edgeNbr);
	    nodeEdges[e.target].push_back(edgeNbr);
	    
	    edgeNbr++;
	}
	void addEdge(const UINT src, const UINT targ, UINT weight=1, bool checkIfExists=true)
	{
	    addEdge(Edge(src, targ, weight), checkIfExists);
	}
	void removeNode(const UINT ind)
	{
	    set<UINT>::iterator fNode = nodes.find(ind);
	    if (fNode!=nodes.end())
	      nodes.erase(fNode);
	    map< UINT, vector<UINT> >::iterator fNodeEdges = nodeEdges.find(ind);
	    if (fNodeEdges!=nodeEdges.end())
	      nodeEdges.erase(fNodeEdges);
	}
	void removeNodeEdge(const UINT nodeIndex, const UINT edgeIndex)
	{
	    map< UINT, std::vector<UINT> >::iterator nEdges = nodeEdges.find(nodeIndex);
	    if (nEdges==nodeEdges.end())
	      return;
	    
	    std::vector<UINT> &eList = nEdges->second;
	    
	    vector<UINT>::iterator ei = find(eList.begin(), eList.end(), edgeIndex);
	    if (ei!=eList.end())
	      eList.erase(ei);
	}
	void removeEdge(const UINT src, const UINT targ)
	{
	    vector<Edge>::iterator foundEdge = find(edges.begin(), edges.end(), Edge(src,targ));
	    if (foundEdge==edges.end())
	      return;
	    
	    UINT edge_index = foundEdge-edges.begin();
	    	    
 	    removeNodeEdge((*foundEdge).source, edge_index);
	    removeNodeEdge((*foundEdge).target, edge_index);
	    
	    (*foundEdge) = Edge(0,0,0);
	}
	
	const vector<Edge> &getEdges() const { return edges; }
	const map< UINT, std::vector<UINT> > &getNodeEdges() const { return nodeEdges; }
	
	
	map<UINT,UINT> labelizeNodes() const
	{
	    map<UINT,UINT> lookup;
	    set<UINT> nodeList(nodes);
	    
	    UINT curLabel = 1;
	    
	    while(!nodeList.empty())
	    {
		propagateLabel(*(nodeList.begin()), curLabel++, lookup, nodeList);
	    }
	    return lookup;
	}
	
	void printSelf(ostream &os = std::cout, string ="")
	{
	    for (vector<Edge>::const_iterator it=edges.begin();it!=edges.end();it++)
	      os << (*it).source << "-" << (*it).target << " (" << (*it).weight << ")" << endl;
	}
	
	std::map<UINT, UINT> nodeValues;
    protected:
	UINT edgeNbr;
	set<UINT> nodes;
	std::vector<Edge> edges;
	std::map< UINT, std::vector<UINT> > nodeEdges;
	
	
	void propagateLabel(const UINT &ind, const UINT &lbl, map<UINT,UINT> &lookup, set<UINT> &nList) const
	{
	    set<UINT>::iterator foundNode = nList.find(ind);
	    if (foundNode==nList.end())
	      return;
	    
	    lookup[ind] = lbl;
	    nList.erase(foundNode);
	    
	    const vector<UINT> &nEdges = nodeEdges.at(ind);
	    
	    for (vector<UINT>::const_iterator it=nEdges.begin();it!=nEdges.end();it++)
	    {
		const Edge &e = edges[*it];
		if (e.source!=ind)
		  propagateLabel(e.source, lbl, lookup, nList);
		else if (e.target!=ind)
		  propagateLabel(e.target, lbl, lookup, nList);
	    }
	}
	
    };
    
    Graph MST(const Graph &graph)
    {
	std::set<size_t> visitedInd;
	std::priority_queue<Edge> pq;
	Graph mst;
	
	const map< UINT, std::vector<UINT> > &nodeEdges = graph.getNodeEdges();
	const vector<Edge> &edges = graph.getEdges();
	
	int u = (*nodeEdges.begin()).first;
	visitedInd.insert(u);
	const vector<UINT> &neigh = nodeEdges.at(u);
	for (vector<UINT>::const_iterator it=neigh.begin();it!=neigh.end();it++)
	    pq.push(edges[*it]);
	
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
		mst.addEdge(edge, false);
		visitedInd.insert(u);
		vector<UINT> const& neigh = nodeEdges.at(u);
		for (vector<UINT>::const_iterator it=neigh.begin();it!=neigh.end();it++)
		    pq.push(edges[*it]);
	    }
	}
	return mst;
    }

} // namespace smil

#endif // _D_GRAPH_HPP

