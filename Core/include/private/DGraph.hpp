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
#include <assert.h>

namespace smil
{
    template <class T=UINT>
    class Node
    {
    public:
	Node(UINT ind, T val=0)
	  : index(ind), value(val)
	{
	}
	UINT index;
	T value;
	
	inline bool operator ==(const Node<T> &rhs)
	{
	    return rhs.index==index;
	}
    };
    
  
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
	    return weight<rhs.weight;
	}
    };

    
    template <class T=UINT>
    class Graph
    {
    public:
	
	Graph()
	  : edgeNbr(0)
	{
	}
	
	typedef Node<T> NodeType;
	
	std::vector<Edge> edges;
	std::vector<NodeType> nodes;
	std::map< UINT, std::vector<UINT> > nodeEdges;
	
	void addNode(const NodeType &n)
	{
	    if (find(nodes.begin(), nodes.end(), n)==nodes.end())
	      nodes.push_back(n);
	}
	void addNode(const UINT &ind, const T &val=0)
	{
	    addNode(NodeType(ind, val));
	}
	void addEdge(const Edge &e, bool createNodes=true)
	{
	    if (find(edges.begin(), edges.end(), e)!=edges.end())
	      return;
	    if (createNodes)
	    {
		addNode(e.source);
		addNode(e.target);
	    }
	    edges.push_back(e);
	    nodeEdges[e.source].push_back(edgeNbr);
	    nodeEdges[e.target].push_back(edgeNbr);
	    
	    edgeNbr++;
	}
	void addEdge(const UINT src, const UINT targ, UINT weight=1, bool createNodes=true)
	{
	    addEdge(Edge(src, targ, weight), createNodes);
	}
    protected:
	UINT edgeNbr;
	
    };

} // namespace smil

#endif // _D_GRAPH_HPP
