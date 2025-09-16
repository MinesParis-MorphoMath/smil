/*
 * Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS AND CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
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
   * @addtogroup GraphTypes
   * @{
   */

  //
  // ######  #####    ####   ######
  // #       #    #  #    #  #
  // #####   #    #  #       #####
  // #       #    #  #  ###  #
  // #       #    #  #    #  #
  // ######  #####    ####   ######
  //
  /**
   * Non-oriented edge
   * @see Graph
   */
  template <class NodeT = size_t, class WeightT = size_t>
  class Edge
  {
  public:
    typedef WeightT WeightType;

    /** Default constructor
     */
    Edge() : source(0), target(0), weight(1)
    {
    }

    /** Constructor using two nodes and an optional weight (default 1).
     */
    Edge(NodeT a, NodeT b, WeightT w = 1) : source(a), target(b), weight(w)
    {
    }

    /** Copy constructor
     */
    Edge(const Edge &rhs)
        : source(rhs.source), target(rhs.target), weight(rhs.weight)
    {
    }

    /** @cond */
    virtual ~Edge()
    {
    }
    /** @endcond */

    /**
     * Copy an edge
     */
    Edge &operator=(const Edge &rhs)
    {
      source = rhs.source;
      target = rhs.target;
      weight = rhs.weight;
      return *this;
    }

    //! Source node
    NodeT source;
    //! Target node
    NodeT target;
    //! Edge weight/value
    WeightT weight;

    /**
     * Check if the edge is active
     *
     * An @b Edge is considered @b active if both @b source and @b target node
     * indexes are @TB{non zero}.
     */
    inline bool isActive() const
    {
      return (source != 0 || target != 0);
    }

    /**
     * Deactivate the @b Edge
     */
    inline void desactivate()
    {
      source = 0;
      target = 0;
    }

    // Don't test the weight values (only to check if the edge exists)
    inline bool operator==(const Edge &rhs) const
    {
      if (rhs.source == source && rhs.target == target)
        return true;
      if (rhs.source == target && rhs.target == source)
        return true;
      return false;
    }

    inline bool operator!=(const Edge &rhs) const
    {
      return !this->operator==(rhs);
    }

    inline bool operator<(const Edge &rhs) const
    {
      return weight > rhs.weight;
    }

    virtual void printSelf(std::ostream &os = std::cout, std::string s = "") const
    {
      os << s << (int) source << "-" << (int) target << " (" << (int) weight
         << ")" << std::endl;
    }
  };

  // Compare two vectors of edges (test also the weight values)
  /**
   * Check if two vectors of edges are equal
   */
  template <class NodeT, class WeightT>
  bool operator==(const std::vector<Edge<NodeT, WeightT>> &e1,
                  const std::vector<Edge<NodeT, WeightT>> &e2)
  {
    if (e1.size() != e2.size())
      return false;

    typedef Edge<NodeT, WeightT>           EdgeT;
    typename std::vector<EdgeT>::const_iterator it1 = e1.begin(), it2 = e2.begin();

    for (; it1 != e1.end() && it2 != e2.end(); it1++, it2++) {
      if ((*it1) != (*it2))
        return false;
      if (it1->weight != it2->weight)
        return false;
    }

    return true;
  }

  //
  //  ####   #####     ##    #####   #    #
  // #    #  #    #   #  #   #    #  #    #
  // #       #    #  #    #  #    #  ######
  // #  ###  #####   ######  #####   #    #
  // #    #  #   #   #    #  #       #    #
  //  ####   #    #  #    #  #       #    #
  //
  /**
   * Non-oriented graph
   * @see Edge
   */
  template <class NodeT = size_t, class WeightT = size_t>
  class Graph : public BaseObject
  {
  public:
    typedef Graph<NodeT, WeightT> GraphType;

    typedef NodeT                    NodeType;
    typedef WeightT                  NodeWeightType;
    typedef std::map<NodeT, WeightT> NodeValuesType;
    typedef std::set<NodeT>               NodeListType;

    typedef Edge<NodeT, WeightT> EdgeType;
    typedef WeightT              EdgeWeightType;

    typedef std::vector<Edge<NodeT, WeightT>> EdgeListType;
    typedef std::vector<size_t>               NodeEdgesType;
    typedef std::map<NodeT, NodeEdgesType>    NodeEdgeListType;

  protected:
    size_t           edgeNbr;
    NodeListType     nodes;
    NodeValuesType   nodeValues;
    EdgeListType     edges;
    NodeEdgeListType nodeEdgeList;

  public:
    //! Default constructor
    Graph() : BaseObject("Graph"), edgeNbr(0)
    {
    }

    //! Copy constructor
    Graph(const Graph &rhs)
        : BaseObject("Graph"), edgeNbr(rhs.edgeNbr), nodes(rhs.nodes),
          nodeValues(rhs.nodeValues), edges(rhs.edges),
          nodeEdgeList(rhs.nodeEdgeList)
    {
    }

    /** @cond */
    virtual ~Graph()
    {
    }
    /** @endcond */

    Graph &operator=(const Graph &rhs)
    {
      nodes        = rhs.nodes;
      nodeValues   = rhs.nodeValues;
      edges        = rhs.edges;
      nodeEdgeList = rhs.nodeEdgeList;
      edgeNbr      = rhs.edgeNbr;
      return *this;
    }

    //! Clear graph content
    void clear()
    {
      nodes.clear();
      nodeValues.clear();
      edges.clear();
      nodeEdgeList.clear();
      edgeNbr = 0;
    }

    //! Add a node given its index
    void addNode(const NodeT &ind)
    {
      nodes.insert(ind);
    }

    //! Add a node given its index and its optional value
    void addNode(const NodeT &ind, const WeightT &val)
    {
      nodes.insert(ind);
      nodeValues[ind] = val;
    }

    /**
     * findEdge() - Find an edge by its content - return its index
     */
    int findEdge(const EdgeType &e)
    {
      typename EdgeListType::iterator foundEdge =
          find(edges.begin(), edges.end(), e);
      if (foundEdge != edges.end())
        return foundEdge - edges.begin();
      else
        return -1;
    }

    /**
     * findEdge() - Find an edge by its nodes - return its index
     */
    int findEdge(const NodeT &src, const NodeT &targ)
    {
      return findEdge(EdgeType(src, targ));
    }

    /**
     * Add an edge to the graph.
     * If checkIfExists is @b true:
     * - If the edge doen't exist, create a new one.
     * - If the edge already exists, the edge weight will be the minimum
     * between the existing a the new weight.
     */
    void addEdge(const EdgeType &e, bool checkIfExists = true)
    {
      if (checkIfExists)
        if (findEdge(e) != -1)
          return;

      edges.push_back(e);
      nodes.insert(e.source);
      nodes.insert(e.target);
      nodeEdgeList[e.source].push_back(edgeNbr);
      nodeEdgeList[e.target].push_back(edgeNbr);

      edgeNbr++;
    }

    /**
     * Add an edge to the graph given two nodes @b src and @b targ and an
     * optional weight
     *
     * If checkIfExists is @b true:
     * - If the edge doen't exist, create a new one.
     * - If the edge already exists, the edge weight will be the
     * minimum between the existing a the new weight.
     */
    void addEdge(const NodeT src, const NodeT targ, WeightT weight = 0,
                 bool checkIfExists = true)
    {
      addEdge(EdgeType(src, targ, weight), checkIfExists);
    }

    /**
     * Sort edges (by weight as defined by the operator @b < of class Edge)
     */
    void sortEdges(bool reverse = false)
    {
      EdgeListType sEdges = edges;
      if (!reverse)
        sort(sEdges.begin(), sEdges.end());
      else
        sort(sEdges.rbegin(), sEdges.rend());

      nodes.clear();
      edges.clear();
      nodeEdgeList.clear();
      edgeNbr = 0;

      for (typename EdgeListType::const_iterator it = sEdges.begin();
           it != sEdges.end(); it++)
        addEdge(*it, false);
    }

    /**
     * clone() - 
     */
    GraphType clone()
    {
      return GraphType(*this);
    }

    /**
     * getNodeNbr() - 
     */
    size_t getNodeNbr()
    {
      return nodes.size();
    }

    /**
     * getEdgeNbr() -
     */
    size_t getEdgeNbr()
    {
      return edges.size();
    }

  protected:
    void removeNode(const NodeT ind)
    {
      typename NodeListType::iterator fNode = nodes.find(ind);
      if (fNode != nodes.end())
        nodes.erase(fNode);
      removeNodeEdges(ind);
    }

    void removeNodeEdge(const NodeT node, const size_t edgeIndex)
    {
      typename NodeEdgeListType::iterator nEdges = nodeEdgeList.find(node);
      if (nEdges == nodeEdgeList.end())
        return;

      NodeEdgesType &eList = nEdges->second;

      typename NodeEdgesType::iterator ei =
          find(eList.begin(), eList.end(), edgeIndex);
      if (ei != eList.end())
        eList.erase(ei);
    }

  public:
    /**
     * Remove all edges linked to the node @b nodeIndex
     */
    void removeNodeEdges(const NodeT node)
    {
      typename NodeEdgeListType::iterator fNodeEdges = nodeEdgeList.find(node);
      if (fNodeEdges == nodeEdgeList.end())
        return;

      NodeEdgesType &nedges = fNodeEdges->second;
      for (NodeEdgesType::iterator it = nedges.begin(); it != nedges.end();
           it++) {
        EdgeType &e = edges[*it];
        if (e.source == node)
          removeNodeEdge(e.target, *it);
        else
          removeNodeEdge(e.source, *it);
        e.desactivate();
      }
      nedges.clear();
    }

    /**
     * Remove an edge
     */
    void removeEdge(const size_t index)
    {
      if (index >= edges.size())
        return;

      EdgeType &edge = edges[index];

      removeNodeEdge(edge.source, index);
      removeNodeEdge(edge.target, index);
      edge.desactivate();
    }

    /**
     * Find and remove an edge linking @b src to @b targ
     */
    void removeEdge(const NodeT src, const NodeT targ)
    {
      typename EdgeListType::iterator foundEdge =
          find(edges.begin(), edges.end(), EdgeType(src, targ));
      if (foundEdge == edges.end())
        return;

      return removeEdge(foundEdge - edges.begin());
    }

    /**
     *  Remove a given edge
     */
    void removeEdge(const EdgeType &edge)
    {
      typename EdgeListType::iterator foundEdge =
          find(edges.begin(), edges.end(), edge);
      if (foundEdge == edges.end())
        return;

      return removeEdge(foundEdge - edges.begin());
    }

    /**
     * removeHighEdges() - remove edges whose weight are greater then some
     * threshold
     *
     * @param[in] EdgeThreshold :
     */
    void removeHighEdges(EdgeWeightType EdgeThreshold)
    {
      size_t nb_edges = edges.size();
      for (size_t index = 0; index < nb_edges; index++) {
        EdgeType &e = edges[index];
        if (e.weight > EdgeThreshold)
          removeEdge(index);
      }
    }

    /**
     * removeHighEdges() - remove edges whose weight are lesser then some
     * threshold
     *
     * @param[in] EdgeThreshold :
     */
    void removeLowEdges(EdgeWeightType EdgeThreshold)
    {
      size_t nb_edges = edges.size();
      for (size_t index = 0; index < nb_edges; index++) {
        EdgeType &e = edges[index];
        if (e.weight < EdgeThreshold)
          removeEdge(index);
      }
    }

    /** @cond */
#ifndef SWIG
    const NodeListType &getNodes() const
    {
      return nodes;
    } // lvalue

    const EdgeListType &getEdges() const
    {
      return edges;
    } // lvalue

    const NodeValuesType &getNodeValues() const
    {
      return nodeValues;
    } // lvalue

    const NodeEdgeListType &getNodeEdges() const
    {
      return nodeEdgeList;
    }  // lvalue
#endif // SWIG
    /** @endcond */

    /**
     * getNodes() - get the list of nodes
     *
     * @returns a handle to the list of nodes in the graph as a @TB{set}
     */
    NodeListType &getNodes()
    {
      return nodes;
    } // rvalue

    /**
     * getEdges() - Get a vector containing the graph edges
     *
     * @returns a handle to the list of edges as a @TB{vector}
     */
    EdgeListType &getEdges()
    {
      return edges;
    } // rvalue

    /**
     * getNodeValues() -
     */
    NodeValuesType &getNodeValues()
    {
      return nodeValues;
    } // rvalue

    /**
     * getNodeEdges()-
     */
    NodeEdgeListType &getNodeEdges()
    {
      return nodeEdgeList;
    } // rvalue

    /**
     * getNodeEdges() - Get a map containing the edges linked to a given node
     */
    NodeEdgesType getNodeEdges(const size_t &node)
    {
      typename NodeEdgeListType::iterator it = nodeEdgeList.find(node);
      if (it != nodeEdgeList.end())
        return it->second;
      else
        return std::vector<size_t>();
    }

    /**
     * computeMST() - Compute the Minimum Spanning Tree graph
     */
    GraphType computeMST()
    {
      return graphMST(*this);
    }

    /**
     * printSelf() -
     */
    virtual void printSelf(std::ostream &os = std::cout, std::string s = "") const
    {
      os << s << "Number of nodes: " << nodes.size() << std::endl;
      os << s << "Number of edges: " << edges.size() << std::endl;
      os << s << "Edges: " << std::endl << "source-target (weight) " << std::endl;

      std::string s2 = s + "\t";
      for (typename EdgeListType::const_iterator it = edges.begin();
           it != edges.end(); it++)
        if ((*it).isActive())
          (*it).printSelf(os, s2);
    }

    /**
     * labelizeNodes() - Labelize the nodes.
     * 
     * Give a different label to each group of connected nodes.
     *
     * @returns a map [ node, label_value ]
     */
    std::map<NodeT, NodeT> labelizeNodes() const
    {
      std::map<NodeT, NodeT> lookup;
      std::set<NodeT>        nodeList(nodes);

      while (!nodeList.empty()) {
        propagateLabel(*(nodeList.begin()), *(nodeList.begin()), lookup,
                       nodeList);
      }
      return lookup;
    }

  protected:
    void propagateLabel(const NodeT ind, const NodeT lbl,
                        std::map<NodeT, NodeT> &lookup, std::set<NodeT> &nList) const
    {
      typename NodeListType::iterator foundNode = nList.find(ind);
      if (foundNode == nList.end())
        return;

      lookup[ind] = lbl;
      nList.erase(foundNode);

      const NodeEdgesType &nEdges = nodeEdgeList.at(ind);

      for (typename NodeEdgesType::const_iterator it = nEdges.begin();
           it != nEdges.end(); it++) {
        const EdgeType &e = edges[*it];
        if (e.source != ind)
          propagateLabel(e.source, lbl, lookup, nList);
        else if (e.target != ind)
          propagateLabel(e.target, lbl, lookup, nList);
      }
    }
  };

  /** graphMST() - create a Mininum Spanning Tree
   *
   * @param[in] graph : input graph
   * 
   * @returns Minimum Spanning Tree built from input graph
   */
  template <class graphT>
  graphT graphMST(const graphT &graph)
  {
    typedef typename graphT::NodeType         NodeType;
    typedef typename graphT::EdgeType         EdgeType;
    typedef typename graphT::EdgeListType     EdgeListType;
    typedef typename graphT::NodeEdgesType    NodeEdgesType;
    typedef typename graphT::NodeEdgeListType NodeEdgeListType;

    std::set<NodeType>            visitedNodes;
    std::priority_queue<EdgeType> pq;
    graphT                        mst;

    const EdgeListType &    edges        = graph.getEdges();
    const NodeEdgeListType &nodeEdgeList = graph.getNodeEdges();

    NodeType curNode = (*nodeEdgeList.begin()).first;
    visitedNodes.insert(curNode);

    const NodeEdgesType &nodeEdges = nodeEdgeList.at(curNode);
    for (typename NodeEdgesType::const_iterator it = nodeEdges.begin();
         it != nodeEdges.end(); it++)
      pq.push(edges[*it]);

    while (!pq.empty()) {
      EdgeType edge = pq.top();
      pq.pop();

      if (visitedNodes.find(edge.source) == visitedNodes.end())
        curNode = edge.source;
      else if (visitedNodes.find(edge.target) == visitedNodes.end())
        curNode = edge.target;
      else
        continue;

      mst.addEdge(edge, false);
      visitedNodes.insert(curNode);
      NodeEdgesType const &nodeEdges = nodeEdgeList.at(curNode);
      for (typename NodeEdgesType::const_iterator it = nodeEdges.begin();
           it != nodeEdges.end(); it++)
        pq.push(edges[*it]);
    }
    // Copy node values
    mst.getNodeValues() = graph.getNodeValues();

    return mst;
  }

  /** @} */

} // namespace smil

#endif // _D_GRAPH_HPP
