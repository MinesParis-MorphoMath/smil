/*
 * Copyright (c) 2011-2014, Matthieu FAESSEL and ARMINES
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

#include "Core/include/DBaseObject.h"

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
     * \see Graph
     */
    template <class T=size_t>
    class Edge
    {
    public:
	typedef T WeightType;
	//! Default constructor
	Edge()
	  : source(0), target(0), weight(1)
	{
	}
	//! Constructor using two nodes and an optional weight (default 1).
	Edge(size_t a, size_t b, T w=1)
	  : source(a), target(b), weight(w)
	{
	}
	//! Copy constructor
	Edge(const Edge &rhs)
	  : source(rhs.source), target(rhs.target), weight(rhs.weight)
	{
	}
	
	~Edge()
        {
        }
	
        Edge &operator =(const Edge &rhs)
        {
            source = rhs.source;
            target = rhs.target;
            weight = rhs.weight;
            return *this;
        }
        
	//! Source node
	size_t source;
	//! Target node
	size_t target;
	//! Edge weight/value
	T weight;
	
	inline bool isActive() const { return (source!=0 || target!=0);  }
	inline void desactivate()
	{
	    source = 0;
	    target = 0;
	}
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
	    return weight>rhs.weight;
	}
	
	virtual void printSelf(ostream &os = std::cout, string s="") const
	{
	  os << s << (int)source << "-" << (int)target << " (" << (int)weight << ")" << endl;
	}
    };

    /**
     * Non-oriented graph
     * \see Edge
     */
    template <class nodeT=size_t, class edgeWT=size_t>
    class Graph : public BaseObject
    {
    public:
	
	typedef nodeT NodeType;
	typedef Edge<edgeWT> EdgeType;
	typedef edgeWT EdgeWeightType;
	
    protected:
	size_t edgeNbr;
	set<size_t> nodes;
	std::map<size_t, size_t> nodeValues;
	std::vector< EdgeType > edges;
	std::map< size_t, std::vector<size_t> > nodeEdges;
	
    public:
	//! Default constructor
	Graph()
	  : BaseObject("Graph"),
	    edgeNbr(0)
	{
	}
	//! Copy constructor
	Graph(const Graph &rhs)
	  : BaseObject("Graph"),
	    edgeNbr(rhs.edgeNbr),
	    nodes(rhs.nodes),
	    nodeValues(rhs.nodeValues),
	    edges(rhs.edges),
	    nodeEdges(rhs.nodeEdges)
	{
	}
	
        virtual ~Graph() {}
	
	Graph &operator =(const Graph &rhs)
	{
	    nodes = rhs.nodes;
	    nodeValues = rhs.nodeValues;
	    edges = rhs.edges;
	    nodeEdges = rhs.nodeEdges;
	    edgeNbr = rhs.edgeNbr;
	    return *this;
	}
	
	//! Clear graph content
	void clear()
	{
	    nodes.clear();
	    nodeValues.clear();
	    edges.clear();
	    nodeEdges.clear();
	    edgeNbr = 0;
	}
	
	//! Add a node given its index and its optional value
	void addNode(const size_t &ind, const nodeT &val=0)
	{
            nodes.insert(ind);
            if (val!=0)
              nodeValues[ind] = val;
	}
	
	/**
	 * Add an edge to the graph. 
	 * If checkIfExists is \b true:
	 * 	If the edge doen't exist, create a new one.
	 * 	If the edge already exists, the edge weight will be the minimum between the existing a the new weight.
	 */
	void addEdge(const EdgeType &e, bool checkIfExists=true)
	{
	    if (checkIfExists)
	    {
		typename vector< EdgeType >::iterator foundEdge = find(edges.begin(), edges.end(), e);
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
	
	/**
	 * Add an edge to the graph given two nodes \b src and \b targ and an optional weight
	 * If checkIfExists is \b true:
	 * 	If the edge doen't exist, create a new one.
	 * 	If the edge already exists, the edge weight will be the minimum between the existing a the new weight.
	 */
	void addEdge(const size_t src, const size_t targ, edgeWT weight=1, bool checkIfExists=true)
	{
	    addEdge(EdgeType(src, targ, weight), checkIfExists);
	}
	
	void sortEdges()
	{
            vector< Edge<edgeWT> > sEdges = edges;
	    sort(sEdges.begin(), sEdges.end());
            
            nodes.clear();
            edges.clear();
            nodeEdges.clear();
            edgeNbr = 0;
            
            for (typename vector< Edge<edgeWT> >::const_iterator it=sEdges.begin();it!=sEdges.end();it++)
              addEdge(*it, false);
	}
	
	Graph<nodeT, edgeWT> clone()
        {
            return Graph<nodeT, edgeWT>(*this);
        }
        
        size_t getNodeNbr()
        {
            return nodes.size();
        }
        
        size_t getEdgeNbr()
        {
            return edges.size();
        }
        
    protected:
	void removeNode(const size_t ind)
	{
	    set<size_t>::iterator fNode = nodes.find(ind);
	    if (fNode!=nodes.end())
	      nodes.erase(fNode);
	    removeNodeEdges(ind);
	}
	void removeNodeEdge(const size_t nodeIndex, const size_t edgeIndex)
	{
	    map< size_t, std::vector<size_t> >::iterator nEdges = nodeEdges.find(nodeIndex);
	    if (nEdges==nodeEdges.end())
	      return;
	    
	    std::vector<size_t> &eList = nEdges->second;
	    
	    vector<size_t>::iterator ei = find(eList.begin(), eList.end(), edgeIndex);
	    if (ei!=eList.end())
	      eList.erase(ei);
	}
    public:
	//! Remove all edges linked to the node \b nodeIndex
	void removeNodeEdges(const size_t nodeIndex)
	{
	    map< size_t, vector<size_t> >::iterator fNodeEdges = nodeEdges.find(nodeIndex);
	    if (fNodeEdges==nodeEdges.end())
	      return;
	    
	    vector<size_t> &nedges = fNodeEdges->second;
	    for (vector<size_t>::iterator it=nedges.begin();it!=nedges.end();it++)
	    {
		EdgeType &e = edges[*it];
		if (e.source==nodeIndex)
		  removeNodeEdge(e.target, *it);
		else
		  removeNodeEdge(e.source, *it);
		e.desactivate();
	    }
	    nedges.clear();
	}
	//! Remove an edge
	void removeEdge(const size_t index)
	{
	    if (index>=edges.size())
	      return;
	    
	    EdgeType &edge = edges[index];
	    
 	    removeNodeEdge(edge.source, index);
	    removeNodeEdge(edge.target, index);
	    edge.desactivate();
	}
	//! Find and remove an edge linking \b src to \b targ
	void removeEdge(const size_t src, const size_t targ)
	{
	    typename vector< EdgeType >::iterator foundEdge = find(edges.begin(), edges.end(), EdgeType(src,targ));
	    if (foundEdge==edges.end())
	      return;
	    
	    return removeEdge(foundEdge-edges.begin());
	}
	// Remove a given edge
	void removeEdge(const EdgeType &edge)
	{
	    typename vector< EdgeType >::iterator foundEdge = find(edges.begin(), edges.end(), edge);
	    if (foundEdge==edges.end())
	      return;
	    
	    return removeEdge(foundEdge-edges.begin());
	}
      void removeHighEdges( EdgeWeightType EdgeThreshold)
      {
	size_t nb_edges = edges.size();
	for (int index= 0; index < nb_edges; index++){//vector<size_t>::iterator it=edges.begin();it!=edges.end();it++)
	 	    EdgeType &e = edges[index];
	    if (e.weight>EdgeThreshold)
	      removeEdge(index);

	}// for all edges
      }// END removeHighEdges
      void removeLowEdges( EdgeWeightType EdgeThreshold)
      {
	size_t nb_edges = edges.size();
	for (int index= 0; index < nb_edges; index++){//vector<size_t>::iterator it=edges.begin();it!=edges.end();it++)
	 	    EdgeType &e = edges[index];
	    if (e.weight<EdgeThreshold)
	      removeEdge(index);

	}// for all edges
      }// END removeHighEdges

#ifndef SWIG
	const vector< EdgeType > &getEdges() const { return edges; }  // lvalue
#endif // SWIG
	//! Get a vector containing the graph edges
	vector< EdgeType > &getEdges() { return edges; }  // rvalue
	const map< size_t, std::vector<size_t> > &getNodeEdges() const { return nodeEdges; } // lvalue
	//! Get a map containing the edges linked to a given node
        std::vector<size_t> getNodeEdges(const size_t &node) 
        { 
            map< size_t, std::vector<size_t> >::iterator it = nodeEdges.find(node);
            if (it!=nodeEdges.end())
              return it->second;
            else return vector<size_t>();
          
        }
#ifndef SWIG
	map< size_t, std::vector<size_t> > &getNodeEdges() { return nodeEdges; } // rvalue
#endif // SWIG
	
	//! Compute the Minimum Spanning Tree graph
	Graph<nodeT,edgeWT> computeMST()
	{
	    return graphMST(*this);
	}
	
	virtual void printSelf(ostream &os = std::cout, string s ="")
	{
	    os << s << "Number of nodes: " << nodes.size() << endl;
	    os << s << "Number of edges: " << edges.size() << endl;
	    os << s << "Edges: " << endl << "source-target (weight) " << endl;
	    
	    string s2 = s + "\t";
	    for (typename vector< EdgeType >::const_iterator it=edges.begin();it!=edges.end();it++)
	      if ((*it).isActive())
		(*it).printSelf(os, s2);
	}
	
	//! Labelize the nodes.
	//! Give a different label to each group of connected nodes.
	//! Return a map [ node_index, label_value ]
	map<size_t,size_t> labelizeNodes() const
	{
	    map<size_t,size_t> lookup;
	    set<size_t> nodeList(nodes);
	    
	    size_t curLabel = 1;
	    
	    while(!nodeList.empty())
	    {
		propagateLabel(*(nodeList.begin()), *(nodeList.begin()), lookup, nodeList);
	    }
	    return lookup;
	}
	
    protected:	
	void propagateLabel(const size_t &ind, const size_t &lbl, map<size_t,size_t> &lookup, set<size_t> &nList) const
	{
	    set<size_t>::iterator foundNode = nList.find(ind);
	    if (foundNode==nList.end())
	      return;
	    
	    lookup[ind] = lbl;
	    nList.erase(foundNode);
	    
	    const vector<size_t> &nEdges = nodeEdges.at(ind);
	    
	    for (vector<size_t>::const_iterator it=nEdges.begin();it!=nEdges.end();it++)
	    {
		const EdgeType &e = edges[*it];
		if (e.source!=ind)
		  propagateLabel(e.source, lbl, lookup, nList);
		else if (e.target!=ind)
		  propagateLabel(e.target, lbl, lookup, nList);
	    }
	}
	
    };
    
    template <class graphT>
    graphT graphMST(const graphT &graph)
    {
	typedef typename graphT::EdgeType EdgeType;
	std::set<size_t> visitedInd;
	std::priority_queue< EdgeType > pq;
	graphT mst;
	
	const map< size_t, std::vector<size_t> > &nodeEdges = graph.getNodeEdges();
	const vector< EdgeType > &edges = graph.getEdges();
	
	int u = (*nodeEdges.begin()).first;
	visitedInd.insert(u);
	const vector<size_t> &neigh = nodeEdges.at(u);
	for (vector<size_t>::const_iterator it=neigh.begin();it!=neigh.end();it++)
	    pq.push(edges[*it]);
	
	while(!pq.empty())
	{
	    EdgeType edge = pq.top();
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
		vector<size_t> const& neigh = nodeEdges.at(u);
		for (vector<size_t>::const_iterator it=neigh.begin();it!=neigh.end();it++)
		    pq.push(edges[*it]);
	    }
	}
	return mst;
    }
    
} // namespace smil

#endif // _D_GRAPH_HPP

