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

namespace Graph
{
  template <class T>
  class graph
  {
  public :
    explicit graph(const std::vector<std::pair<T, T> > &vertices);
    ~graph()
    {}
    void insert_vertex_pair_by_keys(T key1, T key2);

  // Private contained classes
  private:
   // Forward Definition of vertex
   class vertex;

    struct edge
    {
      edge(vertex *edge, T weight) :
        m_Edge(edge),
        m_Weight(weight)
      {}
      vertex *m_Edge;
      T m_Weight;
    }; // END EDGE

    class vertex
    {
    public:
      vertex(T key) :
        m_Key(key)
      {}
      void connect_edge(vertex *adjacent);
      const T key() const {return m_Key;}
      const std::list<edge> &edges() const {return m_Edges;}
    private:
      std::list<edge> m_Edges;
      T m_Key;
      bool contains_edge_to_vertex_with_key(const T key);
    }; // END VERTEX

   // Private methods and member variables
   private:
     std::list<vertex> m_Vertices;
     vertex *contains_vertex(const T key);
  };
}

/*!
 * Constructor of graph: Take a pair of vertices as connection, attempt 
 * to insert if not already in graph. Then connect them in edge list
 */
template <class T>
Graph::graph<T>::graph(const std::vector<std::pair<T, T> > &vertices_relation)
{
#ifndef NDEBUG
  std::cout << "Inserting pairs: " << std::endl;
#endif
  typename std::vector<std::pair<T, T> >::const_iterator insert_it = vertices_relation.begin();
  for(; insert_it != vertices_relation.end(); ++insert_it) {
#ifndef NDEBUG
    std::cout << insert_it->first << " -- > " << insert_it->second << 
std::endl;
#endif
    insert_vertex_pair_by_keys(insert_it->first, insert_it->second);
  }
#ifndef NDEBUG
  std::cout << "Printing results: " << std::endl;
  typename std::list<vertex>::iterator print_it = m_Vertices.begin();
  for(; print_it != m_Vertices.end(); ++print_it) {
    std::cout << print_it->key();
    typename std::list<edge>::const_iterator edge_it = print_it->edges().begin();
    for(; edge_it != print_it->edges().end(); ++edge_it) {
      std::cout << "-->" << edge_it->m_Edge->key();
    }
    std::cout << std::endl;
  }
#endif
}

/*!
 * Takes in a value of type T as a key and 
 * inserts it into graph data structure if 
 * key not already present
 */
template <typename T>
void Graph::graph<T>::insert_vertex_pair_by_keys(T key1, T key2)
{
  /*!
   * Check if vertices already in graph
   */
  Graph::graph<T>::vertex *insert1 = contains_vertex(key1);
  Graph::graph<T>::vertex *insert2 = contains_vertex(key2);
  /*!
   * If not in graph then insert it and get a pointer to it
   * to pass into edge. See () for information on how
   * to build graph
   */ 
  if (insert1 == NULL) {
    m_Vertices.push_back(vertex(key1));
    insert1 = contains_vertex(key1);
  }
  if (insert2 == NULL) {
    m_Vertices.push_back(vertex(key2));
    insert2 = contains_vertex(key2);
  }

#ifndef NDEBUG
    assert(insert1 != NULL && "Failed to insert first vertex");
    assert(insert2 != NULL && "Failed to insert second vertex");
#endif

  /*!
   * At this point we should have a vertex to insert an edge on
   * if not throw an error.
   */ 
  if (insert1 != NULL && insert2 != NULL) {
    insert1->connect_edge(insert2);
    insert2->connect_edge(insert1);
  } else {
    throw std::runtime_error("Unknown");
  }
}

/*!
 * Search the std::list of vertices for key
 * if present return the vertex to indicate
 * already in graph else return NULL to indicate
 * new node
 */
template <typename T>
typename Graph::graph<T>::vertex *Graph::graph<T>::contains_vertex(T key)
{
  typename std::list<vertex >::iterator find_it = m_Vertices.begin();
  for(; find_it != m_Vertices.end(); ++find_it) {
    if (find_it->key() == key) {
      return &(*find_it);
    }
  }
  return NULL;
}

/*!
 * Take the oposing vertex from input and insert it
 * into adjacent list, you can have multiple edges
 * between vertices
 */
template <class T>
void Graph::graph<T>::vertex::connect_edge(Graph::graph<T>::vertex *adjacent)
{
  if (adjacent == NULL)
    return;

  if (!contains_edge_to_vertex_with_key(adjacent->key())) {
    Graph::graph<T>::edge e(adjacent, 1);
    m_Edges.push_back(e);
  }
}

/*!
 * Private member function that check if there is already
 * an edge between the two vertices
 */
template <class T>
bool Graph::graph<T>::vertex::contains_edge_to_vertex_with_key(const T key)
{
  typename std::list<edge>::iterator find_it = m_Edges.begin();
  for(; find_it != m_Edges.end(); ++find_it) {
    if (find_it->m_Edge->key() == key) {
      return true;
    }   
  }
  return false;
}

#endif // _D_GRAPH_HPP
