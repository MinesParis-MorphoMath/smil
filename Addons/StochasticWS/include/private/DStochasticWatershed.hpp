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

#ifndef _D_STOCHASTIC_CHABARDES_HPP_
#define _D_STOCHASTIC_CHABARDES_HPP_

#include "Morpho/include/DMorpho.h"
#include "DUtils.h"
#include <math.h>
#include <random>
#include <chrono>

namespace smil
{
  /** @cond */
  /**
   * @ingroup Addons
   * @addtogroup AddonStochasticWatershed Stochastic Watershed
   * @{
   */

  template <class labelT, class T> class stochastic_edge
  {
  public:
    stochastic_edge(const labelT &s, const labelT &d, const double &w,
                    const T &a, const T &l, const double &m)
        : source(s), dest(d), weight(w), altitude(a), length(l), dist_max(m)
    {
    }
    stochastic_edge(const stochastic_edge &s)
        : source(s.source), dest(s.dest), weight(s.weight),
          altitude(s.altitude), length(s.length), dist_max(s.dist_max)
    {
    }

    labelT source;
    labelT dest;
    double weight;
    T altitude;
    size_t length;
    double dist_max;
    labelT getSource()
    {
      return source;
    }
    labelT getDest()
    {
      return dest;
    }
    double getWeight()
    {
      return weight;
    }
    T getAltitude()
    {
      return altitude;
    }
    size_t getLength()
    {
      return length;
    }
    double getDistMax()
    {
      return dist_max;
    }
  };

  template <class labelT, class T>
  bool compAltitude(const stochastic_edge<labelT, T> &e1,
                    const stochastic_edge<labelT, T> &e2)
  {
    return e1.altitude < e2.altitude;
  }

  template <class labelT, class T>
  bool compWeight(const stochastic_edge<labelT, T> &e1,
                  const stochastic_edge<labelT, T> &e2)
  {
    return e1.weight < e2.weight;
  }

  template <class labelT> class stochastic_node
  {
  public:
    stochastic_node(const size_t s, const size_t a, const double d)
        : area(a), surface(s), dist_max(d)
    {
    }
    map<labelT, size_t> edges;
    size_t area;
    size_t surface;
    double dist_max;
    map<labelT, size_t> getEdges()
    {
      return edges;
    }
    size_t getArea()
    {
      return area;
    }
    size_t getSurface()
    {
      return surface;
    }
    double getDistMax()
    {
      return dist_max;
    }
  };

  template <class labelT, class T> class stochastic_graph
  {
  public:
    stochastic_graph()
    {
    }
    vector<stochastic_edge<labelT, T>> edges;
    vector<stochastic_node<labelT>> nodes;
    vector<stochastic_edge<labelT, T>> getEdges()
    {
      return edges;
    }
    vector<stochastic_node<labelT>> getNodes()
    {
      return nodes;
    }
  };

  template <class labelT, class T>
  void stochasticGraphToPDF(const Image<labelT> &img,
                            stochastic_graph<labelT, T> &graph,
                            Image<labelT> &out, const StrElt &s)
  {
    fill<labelT>(out, ImDtTypes<labelT>::max());

    labelT *pixels = img.getPixels();
    labelT *outs   = out.getPixels();
    {
      index p, q;
      UINT pts;
      size_t S[3];
      img.getSize(S);
      size_t nbrPixelsInSlice = S[0] * S[1];
      size_t nbrPixels        = nbrPixelsInSlice * S[2];
      StrElt se               = s.noCenter();
      UINT sePtsNumber        = se.points.size();

      ForEachPixel(p)
      {
        if (pixels[p.o] != ImDtTypes<T>::max() && pixels[p.o] != 0) {
          std::map<labelT, bool> is_done;
          outs[p.o] = ImDtTypes<labelT>::max();
          ForEachNeighborOf(p, q)
          {
            is_done[pixels[q.o]] = false;
          }
          ENDForEachNeighborOf ForEachNeighborOf(p, q)
          {
            if (!is_done[pixels[q.o]]) {
              if (pixels[p.o] != pixels[q.o] && pixels[q.o] != 0 &&
                  pixels[q.o] != ImDtTypes<labelT>::max()) {
                outs[p.o] = labelT(
                    (1. -
                     graph.edges[graph.nodes[pixels[p.o]].edges[pixels[q.o]]]
                         .weight) *
                    double(ImDtTypes<labelT>::max()));
              }
            }
            is_done[pixels[q.o]] = true;
          }
          ENDForEachNeighborOf ForEachNeighborOf(p, q)
          {
            if (pixels[q.o] == 0) {
              outs[p.o] = 0;
            }
          }
          ENDForEachNeighborOf
        }
      }
      ENDForEachPixel
    }
  }

  template <class labelT, class T>
  void displayStochasticMarker(const Image<labelT> &img,
                               std::vector<int> &markers,
                               stochastic_graph<labelT, T> &graph,
                               std::vector<labelT> &originals,
                               Image<labelT> &out, const StrElt &s)
  {
    std::map<labelT, labelT> correspondances;
    uint32_t i = 0;
    for (typename std::vector<labelT>::iterator it = originals.begin();
         it != originals.end(); ++it) {
      correspondances[*it] = i;
      ++i;
    }

    labelT *pixels = img.getPixels();
    labelT *outs   = out.getPixels();
    {
      index p, q;
      UINT pts;
      size_t S[3];
      img.getSize(S);
      size_t nbrPixelsInSlice = S[0] * S[1];
      size_t nbrPixels        = nbrPixelsInSlice * S[2];
      StrElt se               = s.noCenter();
      UINT sePtsNumber        = se.points.size();
      typename std::map<labelT, labelT>::iterator it;

      ForEachPixel(p)
      {
        if (pixels[p.o] != ImDtTypes<T>::max() && pixels[p.o] != 0) {
          it = correspondances.find(pixels[p.o]);
          if (it != correspondances.end())
            outs[p.o] =
                (markers[it->second] > 0) ? 0 : ImDtTypes<labelT>::max();
        }
        if (pixels[p.o] != 0) {
          ForEachNeighborOf(p, q)
          {
            if (pixels[q.o] == 0) {
              outs[p.o] = 0;
            }
          }
          ENDForEachNeighborOf
        }
      }
      ENDForEachPixel
    }
  }

  template <class labelT, class T>
  void mosaicToStochasticGraph(const Image<labelT> &img, const Image<T> &gradI,
                               stochastic_graph<labelT, T> &graph,
                               const StrElt &s)
  {
    Image<labelT> tmp = Image<labelT>(img);

    distEuclidean(img, tmp);

    labelT nbr_tiles = maxVal(img);
    labelT *pixels   = img.getPixels();
    labelT *dists    = tmp.getPixels();

    T *grads = gradI.getPixels();

    // Creating the nodes
    // NULL node at 0
    graph.nodes.push_back(stochastic_node<labelT>(0, 0, 0));
    for (labelT i = 0; i < nbr_tiles; ++i)
      graph.nodes.push_back(stochastic_node<labelT>(0, 0, 0));

    {
      index p, q;
      bool has_zero;
      UINT pts;
      size_t S[3];
      img.getSize(S);
      size_t nbrPixelsInSlice = S[0] * S[1];
      size_t nbrPixels        = nbrPixelsInSlice * S[2];
      StrElt se               = s.noCenter();
      UINT sePtsNumber        = se.points.size();

      ForEachPixel(p)
      {
        if (pixels[p.o] != ImDtTypes<T>::max() && pixels[p.o] != 0) {
          graph.nodes[pixels[p.o]].dist_max =
              (sqrt(dists[p.o]) > graph.nodes[pixels[p.o]].dist_max)
                  ? sqrt(dists[p.o])
                  : graph.nodes[pixels[p.o]].dist_max;

          std::map<labelT, bool> is_done;
          has_zero = false;
          ForEachNeighborOf(p, q)
          {
            is_done[pixels[q.o]] = false;
            if (pixels[q.o] == 0 || pixels[q.o] == ImDtTypes<labelT>::max()) {
              has_zero = true;
            }
          }
          ENDForEachNeighborOf ForEachNeighborOf(p, q)
          {
            if (!is_done[q.o] && pixels[p.o] != pixels[q.o] &&
                pixels[q.o] != 0 && pixels[q.o] != ImDtTypes<labelT>::max()) {
              if (graph.nodes[pixels[p.o]].edges.find(pixels[q.o]) ==
                  graph.nodes[pixels[p.o]].edges.end()) {
                graph.edges.push_back(stochastic_edge<labelT, T>(
                    pixels[p.o], pixels[q.o], 1., ImDtTypes<T>::max(), 0, 0));
                graph.nodes[pixels[q.o]].edges[pixels[p.o]] =
                    graph.edges.size() - 1;
                graph.nodes[pixels[p.o]].edges[pixels[q.o]] =
                    graph.edges.size() - 1;
              }
              graph.edges[graph.nodes[pixels[p.o]].edges[pixels[q.o]]]
                  .altitude =
                  (graph.edges[graph.nodes[pixels[p.o]].edges[pixels[q.o]]]
                       .altitude > grads[p.o])
                      ? grads[p.o]
                      : graph.edges[graph.nodes[pixels[p.o]].edges[pixels[q.o]]]
                            .altitude;
              graph.edges[graph.nodes[pixels[p.o]].edges[pixels[q.o]]]
                  .dist_max =
                  (graph.edges[graph.nodes[pixels[p.o]].edges[pixels[q.o]]]
                       .dist_max < sqrt(dists[p.o]))
                      ? sqrt(dists[p.o])
                      : graph.edges[graph.nodes[pixels[p.o]].edges[pixels[q.o]]]
                            .dist_max;
              if (!has_zero)
                ++graph.edges[graph.nodes[pixels[p.o]].edges[pixels[q.o]]]
                      .length;
            }
            is_done[pixels[q.o]] = true;
          }
          ENDForEachNeighborOf
        }
      }
      ENDForEachPixel
    }

    typedef map<labelT, double> map2;
    map2 m2 = blobsArea(img);
    for (typename map2::iterator it = m2.begin(); it != m2.end(); ++it) {
      graph.nodes[it->first].area = it->second;
    }
    gradient(img, tmp, s);
    compare(img, "==", labelT(0), labelT(0), tmp, tmp);
    compare(tmp, ">", labelT(0), img, labelT(0), tmp);
    map2 m3 = blobsArea(tmp);
    for (typename map2::iterator it = m3.begin(); it != m3.end(); ++it) {
      graph.nodes[it->first].surface = it->second;
    }
  }

  struct subset {
    int parent;
    int rank;
  };
  int Find(struct subset subsets[], int i)
  {
    if (subsets[i].parent != i)
      subsets[i].parent = Find(subsets, subsets[i].parent);
    return subsets[i].parent;
  }

  void Union(struct subset subsets[], int x, int y)
  {
    int xroot = Find(subsets, x);
    int yroot = Find(subsets, y);
    if (subsets[xroot].rank < subsets[yroot].rank)
      subsets[xroot].parent = yroot;
    else if (subsets[xroot].rank > subsets[yroot].rank)
      subsets[yroot].parent = xroot;
    else {
      subsets[yroot].parent = xroot;
      subsets[xroot].rank++;
    }
  }

  template <class labelT, class T>
  void visit_stochastic_node(const labelT &l, stochastic_graph<labelT, T> &r,
                             vector<labelT> &visited, const size_t &i)
  {
    visited[i] = l;
    for (typename map<labelT, size_t>::iterator it = r.nodes[i].edges.begin();
         it != r.nodes[i].edges.end(); ++it) {
      if (visited[it->first] == 0 && r.edges[it->second].weight == 1.) {
        visit_stochastic_node(l, r, visited, it->first);
      }
    }
  }

  template <class labelT, class T>
  size_t CCL_stochasticGraph(stochastic_graph<labelT, T> &r,
                             vector<labelT> &labels)
  {
    size_t nbr_nodes  = r.nodes.size();
    size_t nbr_labels = 0;
    labelT cur_label  = 0;
    labels            = vector<labelT>(nbr_nodes + 1, 0);

    for (uint32_t i = 1; i < nbr_nodes; ++i) {
      if (labels[i] == 0) {
        nbr_labels++;
        if (nbr_labels == ImDtTypes<labelT>::max())
          cur_label = 1;
        else
          cur_label++;
        visit_stochastic_node(cur_label, r, labels, i);
      }
    }

    return nbr_labels;
  }

  template <class labelT, class T>
  size_t CCLUnionFind_stochasticGraph(stochastic_graph<labelT, T> &graph,
                                      vector<labelT> &labels)
  {
    uint32_t nbr_nodes  = graph.nodes.size() - 1;
    uint32_t nbr_labels = 0;
    labels              = vector<labelT>(nbr_nodes + 1, 0);

    struct subset subsets[nbr_nodes + 1];
    for (uint32_t i = 1; i < nbr_nodes + 1; ++i) {
      subsets[i].parent = i;
      subsets[i].rank   = 0;
    }

    for (typename std::vector<stochastic_edge<labelT, T>>::iterator it =
             graph.edges.begin();
         it != graph.edges.end(); ++it) {
      Union(subsets, it->source, it->dest);
    }

    for (uint32_t i = 1; i < nbr_nodes + 1; ++i) {
      uint32_t root = Find(subsets, i);
      labels[i]     = root;
      if (root == i)
        nbr_labels++;
    }
    return nbr_labels;
  }

  template <class labelT, class T>
  stochastic_graph<labelT, T>
  getSubStochasticGraph(stochastic_graph<labelT, T> &graph, const size_t &i,
                        const vector<labelT> &labels, vector<labelT> &originals)
  {
    stochastic_graph<labelT, T> out;

    out.nodes.push_back(stochastic_node<labelT>(0, 0, 0));
    originals.push_back(0);

    map<labelT, labelT> back;
    // The nodes
    for (uint32_t j = 1; j < labels.size(); ++j) {
      if (labels[j] == i) {
        out.nodes.push_back(stochastic_node<labelT>(graph.nodes[j].surface,
                                               graph.nodes[j].area,
                                               graph.nodes[j].dist_max));
        originals.push_back(j);
        back[j] = labelT(out.nodes.size() - 1);
      }
    }
    // The edges
    for (uint32_t j = 1; j < out.nodes.size(); ++j) {
      for (typename map<labelT, size_t>::iterator it =
               graph.nodes[originals[j]].edges.begin();
           it != graph.nodes[originals[j]].edges.end(); ++it) {
        if (out.nodes[j].edges.find(back[it->first]) ==
            out.nodes[j].edges.end()) {
          out.edges.push_back(stochastic_edge<labelT, T>(
              j, back[it->first], graph.edges[it->second].weight,
              graph.edges[it->second].altitude, graph.edges[it->second].length,
              graph.edges[it->second].dist_max));
          out.nodes[j].edges[back[it->first]] = out.edges.size() - 1;
          out.nodes[back[it->first]].edges[j] = out.edges.size() - 1;
        }
      }
    }

    return out;
  }

  template <class labelT, class T>
  stochastic_graph<labelT, T> KruskalMST(stochastic_graph<labelT, T> &graph)
  {
    stochastic_graph<labelT, T> out;

    // Copy the nodes.
    out.nodes.push_back(stochastic_node<labelT>(0, 0, 0));
    for (uint32_t i = 1; i < graph.nodes.size(); ++i) {
      out.nodes.push_back(stochastic_node<labelT>(graph.nodes[i].surface,
                                             graph.nodes[i].area,
                                             graph.nodes[i].dist_max));
    }

    vector<stochastic_edge<labelT, T>> edges = graph.edges;
    std::sort(edges.begin(), edges.end(), compAltitude<labelT, T>);
    vector<stochastic_edge<labelT, T>> mst_edges;

    struct subset subsets[graph.nodes.size()];
    // Initialise subsets for union-find.
    for (uint32_t i = 0; i < graph.nodes.size(); ++i) {
      subsets[i].parent = i;
      subsets[i].rank   = 0;
    }

    uint32_t nbr_of_edges = 0;
    uint32_t i            = 0;
    while (nbr_of_edges != graph.nodes.size() - 2) {
      stochastic_edge<labelT, T> e = edges[i];
      ++i;

      int x = Find(subsets, e.source);
      int y = Find(subsets, e.dest);

      if (x != y) {
        mst_edges.push_back(e);
        ++nbr_of_edges;
        Union(subsets, x, y);
      }
    }

    out.edges = mst_edges;

    for (uint32_t i = 0; i < mst_edges.size(); ++i) {
      out.nodes[mst_edges[i].source].edges[mst_edges[i].dest] = i;
      out.nodes[mst_edges[i].dest].edges[mst_edges[i].source] = i;
    }

    return out;
  }

  template <class labelT, class T>
  std::vector<double> areaDistribution(stochastic_graph<labelT, T> graph)
  {
    std::vector<double> out;
    double total_area;

    for (uint32_t i = 0; i < graph.nodes.size(); ++i) {
      out.push_back(double(graph.nodes[i].area));
      total_area += double(graph.nodes[i].area);
    }
    for (uint32_t i = 0; i < graph.nodes.size(); ++i) {
      out[i] /= total_area;
    }
    return out;
  }
  template <class labelT, class T>
  std::vector<double> uniformDistribution(stochastic_graph<labelT, T> graph)
  {
    std::vector<double> out;
    double total_nodes = graph.nodes.size() - 1;
    double prob        = 1. / total_nodes;

    out.push_back(0);
    for (uint32_t i = 1; i < graph.nodes.size(); ++i) {
      out.push_back(prob);
    }
    return out;
  }

  std::vector<int> generateMarkers(const std::vector<double> &prob_dist,
                                   std::default_random_engine &generator)
  {
    std::vector<int> markers = std::vector<int>(prob_dist.size(), 0);

    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    std::uniform_int_distribution<int> dist2(1, prob_dist.size());
    //        size_t nbr_markers = prob_dist.size();
    size_t nbr_markers = dist2(generator);

    for (uint32_t n = 0; n < nbr_markers; ++n) {
      double number   = distribution(generator);
      double cumulate = prob_dist[0];
      int i           = 0;
      while (number > cumulate) {
        i += 1;
        cumulate += prob_dist[i];
      }
      ++markers[i];
    }
    return markers;
  }

  template <class labelT, class T>
  std::vector<size_t> watershedGraph(stochastic_graph<labelT, T> &graph,
                                     std::vector<int> &markers)
  {
    std::vector<size_t> out;
    std::map<T, int> dist_altitude;

    for (typename std::vector<stochastic_edge<labelT, T>>::iterator it =
             graph.edges.begin();
         it != graph.edges.end(); ++it) {
      dist_altitude[it->altitude]++;
    }

    // Pre-allocation
    map<T, std::queue<size_t>> hq;
    for (typename std::map<T, int>::iterator it = dist_altitude.begin();
         it != dist_altitude.end(); ++it) {
      hq[it->first] = std::queue<size_t>();
      // Pre-allocation.
      // hq[it->first].reserve (it->second);
    }

    std::vector<labelT> basins = std::vector<labelT>(graph.nodes.size(), 0);
    std::vector<bool> already_pushed =
        std::vector<bool>(graph.edges.size(), false);

    labelT basin = 0;
    // Initialisation
    int i = 0;
    for (typename std::vector<int>::iterator it = markers.begin();
         it != markers.end(); ++it) {
      if (*it > 0) {
        ++basin;
        basins[i] = basin;
        // Push the edges out of it into the HQ.
        for (typename std::map<labelT, size_t>::iterator jt =
                 graph.nodes[i].edges.begin();
             jt != graph.nodes[i].edges.end(); ++jt) {
          if (!already_pushed[jt->second]) {
            already_pushed[jt->second] = true;
            hq[graph.edges[jt->second].altitude].push(jt->second);
          }
        }
      }
      i += 1;
    }

    typename std::map<T, std::queue<size_t>>::iterator it = hq.begin();
    while (it != hq.end()) {
      if (it->second.size() != 0) {
        stochastic_edge<labelT, T> p = graph.edges[it->second.front()];

        int l1 = basins[p.source];
        int l2 = basins[p.dest];
        if (l1 > 0 && l2 == 0) {
          basins[p.dest] = l1;
          for (typename std::map<labelT, size_t>::iterator jt =
                   graph.nodes[p.dest].edges.begin();
               jt != graph.nodes[p.dest].edges.end(); ++jt) {
            if (!already_pushed[jt->second]) {
              already_pushed[jt->second] = true;
              hq[graph.edges[jt->second].altitude].push(jt->second);
            }
          }
        } else if (l1 == 0 && l2 > 0) {
          basins[p.source] = l2;
          for (typename std::map<labelT, size_t>::iterator jt =
                   graph.nodes[p.source].edges.begin();
               jt != graph.nodes[p.source].edges.end(); ++jt) {
            if (!already_pushed[jt->second]) {
              already_pushed[jt->second] = true;
              hq[graph.edges[jt->second].altitude].push(jt->second);
            }
          }
        } else {
          out.push_back(it->second.front());
        }
        it->second.pop();
      } else {
        it = hq.begin();
        while (it != hq.end() && it->second.size() == 0) {
          it++;
        }
      }
    }

    return out;
  }

  class hierarchy
  {
  public:
    hierarchy(size_t l, size_t r) : left(l), right(r)
    {
    }
    hierarchy(const hierarchy &h) : left(h.left), right(h.right)
    {
    }
    size_t left;
    size_t right;
  };

  template <class labelT, class T>
  std::vector<hierarchy> getHierarchy(stochastic_graph<labelT, T> &graph)
  {
    size_t nbr_nodes = graph.nodes.size() - 1;

    std::sort(graph.edges.begin(), graph.edges.end(), compWeight<labelT, T>);

    uint32_t i = 0;
    // Applying the sort to the graph.
    for (typename std::vector<stochastic_edge<labelT, T>>::iterator it =
             graph.edges.begin();
         it != graph.edges.end(); ++it) {
      graph.nodes[it->source].edges[it->dest] = i;
      graph.nodes[it->dest].edges[it->source] = i;
      ++i;
    }

    size_t clusters[nbr_nodes + 1];
    struct subset subsets[nbr_nodes + 1];
    // Initialise subsets for union-find.
    for (uint32_t i = 1; i < nbr_nodes + 1; ++i) {
      subsets[i].parent = i;
      subsets[i].rank   = 0;
      clusters[i]       = i;
    }

    std::vector<hierarchy> out;

    i = nbr_nodes + 1;
    for (typename std::vector<stochastic_edge<labelT, T>>::iterator it =
             graph.edges.begin();
         it != graph.edges.end(); ++it) {
      int x = Find(subsets, it->source);
      int y = Find(subsets, it->dest);

      Union(subsets, x, y);

      out.push_back(hierarchy(clusters[x], clusters[y]));
      clusters[x] = i;
      clusters[y] = i;

      i += 1;
    }

    return out;
  }

  template <class labelT, class T>
  std::vector<double> weightHierarchy(std::vector<hierarchy> &h,
                                      stochastic_graph<labelT, T> &graph)
  {
    size_t nbr_weights = graph.nodes.size() + h.size();

    std::vector<double> area     = std::vector<double>(nbr_weights, 0.);
    std::vector<double> surface  = std::vector<double>(nbr_weights, 0.);
    std::vector<double> dist_max = std::vector<double>(nbr_weights, 0.);
    std::vector<double> out      = std::vector<double>(nbr_weights, 0.);

    uint32_t i = 0;
    for (typename std::vector<stochastic_node<labelT>>::iterator it =
             graph.nodes.begin();
         it != graph.nodes.end(); ++it) {
      area[i]     = double(it->area);
      surface[i]  = double(it->surface);
      dist_max[i] = double(it->dist_max);
      ++i;
    }
    uint32_t j = 0;
    for (typename std::vector<hierarchy>::iterator it = h.begin();
         it != h.end(); ++it) {
      area[i] = area[it->left] + area[it->right];
      surface[i] =
          surface[it->left] + surface[it->right] - graph.edges[j].length;
      dist_max[i] = (dist_max[it->left] > dist_max[it->right])
                        ? dist_max[it->left]
                        : dist_max[it->right];
      ++i;
      ++j;
    }

    for (i = 1; i < nbr_weights; ++i) {
      out[i] = dist_max[i];
      //                out[i] = area[i] / (9*PI*pow(mean_dist[i], 2));
      //                out[i] = (pow(surface[i],2))/(4*PI*area[i]);
      //                std::cerr << i << " " << surface[i] << " " << area[i] <<
      //                " " << dist_max[i] << " " << out[i] << std::endl;
    }

    return out;
  }

  template <class labelT, class T>
  void bottomUpHierarchy(std::vector<double> &weights,
                         std::vector<hierarchy> &h, const double &r0,
                         stochastic_graph<labelT, T> &graph)
  {
    // Set all edges to disconnected.
    for (typename std::vector<stochastic_edge<labelT, T>>::iterator it =
             graph.edges.begin();
         it != graph.edges.end(); ++it) {
      it->weight = 0.;
    }

    std::vector<bool> connected = std::vector<bool>(weights.size(), false);
    // A node is connected to itself.
    for (uint32_t i = 0; i < graph.nodes.size(); ++i) {
      connected[i] = true;
    }

    uint32_t i = 0, j = graph.nodes.size();
    for (typename std::vector<hierarchy>::iterator it = h.begin();
         it != h.end(); ++it) {
      double ratio = (weights[it->left] < weights[it->right])
                         ? weights[it->left]
                         : weights[it->right];
      ratio /= graph.edges[i].dist_max;
      if (connected[it->left] && connected[it->right] && ratio <= r0) {
        graph.edges[i].weight = 1.;
        connected[j]          = true;
      }
      ++i;
      ++j;
    }
  }

  template <class labelT, class T>
  void cut(std::vector<double> &weights, std::vector<hierarchy> &h,
           stochastic_graph<labelT, T> &graph, const size_t &node)
  {
    // A leaf is reached.
    if (node < graph.nodes.size())
      return;
    size_t edge   = node - graph.nodes.size();
    double weight = weights[node];
    if (int(weights[h[edge].left]) <= int(weight) ||
        int(weights[h[edge].right]) <= int(weight)) {
      graph.edges[edge].weight = 0.;
      cut(weights, h, graph, h[edge].left);
      cut(weights, h, graph, h[edge].right);
    }
  }

  template <class labelT, class T>
  void topDownHierarchy(std::vector<double> &weights, std::vector<hierarchy> &h,
                        stochastic_graph<labelT, T> &graph)
  {
    // Set all edge to connected.
    for (typename std::vector<stochastic_edge<labelT, T>>::iterator it =
             graph.edges.begin();
         it != graph.edges.end(); ++it) {
      it->weight = 1.;
    }

    cut(weights, h, graph, weights.size() - 1);
  }

  /**
   *  Over Segmentation Correction
   *
   */
  // Parallel.
  template <class labelT, class T>
  void stochasticWatershedParallel(const Image<labelT> &primary,
                                   const Image<T> &gradient, Image<labelT> &out,
                                   const size_t &n_seeds, const StrElt &se)
  {
    fill<labelT>(out, ImDtTypes<labelT>::max());

    stochastic_graph<labelT, T> graph;
    mosaicToStochasticGraph(primary, gradient, graph, se);

    std::vector<labelT> labels;
    size_t nbr_subgraphs = CCLUnionFind_stochasticGraph(graph, labels);

    // Cutting all edges...
    for (typename std::vector<stochastic_edge<labelT, T>>::iterator it =
             graph.edges.begin();
         it != graph.edges.end(); ++it) {
      it->weight = 0.;
    }

    std::default_random_engine generator;
    generator.seed(std::chrono::system_clock::now().time_since_epoch().count());

#pragma omp parallel for
    for (uint32_t i = 1; i < nbr_subgraphs + 1; ++i) {
      vector<labelT> originals;
      stochastic_graph<labelT, T> sub =
          getSubStochasticGraph(graph, i, labels, originals);
      stochastic_graph<labelT, T> mst = KruskalMST(sub);

      for (uint32_t j = 0; j < n_seeds; ++j) {
        std::vector<double> prob_dist = uniformDistribution(mst);
        std::vector<int> markers      = generateMarkers(prob_dist, generator);

        std::vector<size_t> ws = watershedGraph(mst, markers);

        for (typename std::vector<size_t>::iterator it = ws.begin();
             it != ws.end(); ++it) {
          mst.edges[*it].weight += 1.;
        }
      }

      for (typename std::vector<stochastic_edge<labelT, T>>::iterator it =
               mst.edges.begin();
           it != mst.edges.end(); ++it) {
        it->weight /= n_seeds;
      }

      // Copy the weights of the subgraph into the graph.
      for (typename std::vector<stochastic_edge<labelT, T>>::iterator it =
               mst.edges.begin();
           it != mst.edges.end(); ++it) {
        graph
            .edges[graph.nodes[originals[it->source]]
                       .edges[originals[it->dest]]]
            .weight = it->weight;
      }
    }

    stochasticGraphToPDF(primary, graph, out, se);
  }

  /**
   *  Over Segmentation Correction
   *
   */
  template <class labelT, class T>
  void stochasticWatershed(const Image<labelT> &primary,
                           const Image<T> &gradient, Image<labelT> &out,
                           const size_t &n_seeds, const StrElt &se)
  {
    fill<labelT>(out, ImDtTypes<labelT>::max());

    stochastic_graph<labelT, T> graph;
    mosaicToStochasticGraph(primary, gradient, graph, se);

    std::vector<labelT> labels;
    // size_t nbr_subgraphs = CCLUnionFind_stochasticGraph (graph, labels);

    // Cutting all edges...
    for (typename std::vector<stochastic_edge<labelT, T>>::iterator it =
             graph.edges.begin();
         it != graph.edges.end(); ++it) {
      it->weight = 0.;
    }

    std::default_random_engine generator;
    generator.seed(std::chrono::system_clock::now().time_since_epoch().count());

    for (uint32_t j = 0; j < n_seeds; ++j) {
      std::vector<double> prob_dist = uniformDistribution(graph);
      std::vector<int> markers      = generateMarkers(prob_dist, generator);

      std::vector<size_t> ws = watershedGraph(graph, markers);

      for (typename std::vector<size_t>::iterator it = ws.begin();
           it != ws.end(); ++it) {
        graph.edges[*it].weight += 1.;
      }
    }

    for (typename std::vector<stochastic_edge<labelT, T>>::iterator it =
             graph.edges.begin();
         it != graph.edges.end(); ++it) {
      it->weight /= n_seeds;
    }

    stochasticGraphToPDF(primary, graph, out, se);
  }

  /**
   *  Over Segmentation Correction
   *
   */
  // Parallel
  template <class labelT, class T>
  size_t stochasticFlatZonesParallel(const Image<labelT> &primary,
                                     const Image<T> &gradient,
                                     Image<labelT> &out, const size_t &n_seeds,
                                     const double &t0, const StrElt &se)
  {
    fill<labelT>(out, ImDtTypes<labelT>::max());

    stochastic_graph<labelT, T> graph;
    mosaicToStochasticGraph(primary, gradient, graph, se);

    std::vector<labelT> labels;
    size_t nbr_subgraphs = CCLUnionFind_stochasticGraph(graph, labels);

    // Cutting all edges...
    for (typename std::vector<stochastic_edge<labelT, T>>::iterator it =
             graph.edges.begin();
         it != graph.edges.end(); ++it) {
      it->weight = 0.;
    }

    std::default_random_engine generator;
    generator.seed(std::chrono::system_clock::now().time_since_epoch().count());

#pragma omp parallel for
    for (uint32_t i = 1; i < nbr_subgraphs + 1; ++i) {
      vector<labelT> originals;
      stochastic_graph<labelT, T> sub =
          getSubStochasticGraph(graph, i, labels, originals);
      stochastic_graph<labelT, T> mst = KruskalMST(sub);

      for (uint32_t j = 0; j < n_seeds; ++j) {
        std::vector<double> prob_dist = uniformDistribution(mst);
        std::vector<int> markers      = generateMarkers(prob_dist, generator);

        std::vector<size_t> ws = watershedGraph(mst, markers);

        for (typename std::vector<size_t>::iterator it = ws.begin();
             it != ws.end(); ++it) {
          mst.edges[*it].weight += 1.;
        }
      }

      for (typename std::vector<stochastic_edge<labelT, T>>::iterator it =
               mst.edges.begin();
           it != mst.edges.end(); ++it) {
        it->weight /= n_seeds;
        if (it->weight < t0) {
          it->weight = 1.;
        } else {
          it->weight = 0.;
        }
      }

      // Copy the weights of the subgraph into the graph.
      for (typename std::vector<stochastic_edge<labelT, T>>::iterator it =
               mst.edges.begin();
           it != mst.edges.end(); ++it) {
        graph
            .edges[graph.nodes[originals[it->source]]
                       .edges[originals[it->dest]]]
            .weight = it->weight;
      }
    }

    nbr_subgraphs = CCL_stochasticGraph(graph, labels);
    applyThreshold(primary, labels, out);
    return nbr_subgraphs;
  }

  /**
   *  Over Segmentation Correction
   *
   */
  template <class labelT, class T>
  size_t stochasticFlatZones(const Image<labelT> &primary,
                             const Image<T> &gradient, Image<labelT> &out,
                             const size_t &n_seeds, const double &t0,
                             const StrElt &se)
  {
    fill<labelT>(out, ImDtTypes<labelT>::max());

    stochastic_graph<labelT, T> graph;
    mosaicToStochasticGraph(primary, gradient, graph, se);

    std::vector<labelT> labels;
    size_t nbr_subgraphs = CCLUnionFind_stochasticGraph(graph, labels);

    // Cutting all edges...
    for (typename std::vector<stochastic_edge<labelT, T>>::iterator it =
             graph.edges.begin();
         it != graph.edges.end(); ++it) {
      it->weight = 0.;
    }

    std::default_random_engine generator;
    generator.seed(std::chrono::system_clock::now().time_since_epoch().count());

    for (uint32_t j = 0; j < n_seeds; ++j) {
      std::vector<double> prob_dist = uniformDistribution(graph);
      std::vector<int> markers      = generateMarkers(prob_dist, generator);

      std::vector<size_t> ws = watershedGraph(graph, markers);

      for (typename std::vector<size_t>::iterator it = ws.begin();
           it != ws.end(); ++it) {
        graph.edges[*it].weight += 1.;
      }
    }

    for (typename std::vector<stochastic_edge<labelT, T>>::iterator it =
             graph.edges.begin();
         it != graph.edges.end(); ++it) {
      it->weight /= n_seeds;
      if (it->weight < t0) {
        it->weight = 1.;
      } else {
        it->weight = 0.;
      }
    }

    nbr_subgraphs = CCL_stochasticGraph(graph, labels);
    applyThreshold(primary, labels, out);
    return nbr_subgraphs;
  }

  /**
   *  Over Segmentation Correction
   *
   */
  // Parallel
  template <class labelT, class T>
  size_t overSegmentationCorrection(const Image<labelT> &primary,
                                    const Image<T> &gradient,
                                    Image<labelT> &out, const size_t &n_seeds,
                                    const double &r0, const StrElt &se)
  {
    fill<labelT>(out, labelT(0));

    stochastic_graph<labelT, T> graph;
    mosaicToStochasticGraph(primary, gradient, graph, se);

    std::vector<labelT> labels;
    size_t nbr_subgraphs = CCLUnionFind_stochasticGraph(graph, labels);

    // Cutting all edges...
    for (typename std::vector<stochastic_edge<labelT, T>>::iterator it =
             graph.edges.begin();
         it != graph.edges.end(); ++it) {
      it->weight = 0.;
    }

    std::default_random_engine generator;
    generator.seed(std::chrono::system_clock::now().time_since_epoch().count());

    //        #pragma omp parallel for
    for (uint32_t i = 1; i < nbr_subgraphs + 1; ++i) {
      vector<labelT> originals;
      stochastic_graph<labelT, T> sub =
          getSubStochasticGraph(graph, i, labels, originals);

      if (sub.edges.size() != 0) {
        stochastic_graph<labelT, T> mst = KruskalMST(sub);

        for (uint32_t j = 0; j < n_seeds; ++j) {
          std::vector<double> prob_dist = uniformDistribution(mst);
          std::vector<int> markers      = generateMarkers(prob_dist, generator);

          std::vector<size_t> ws = watershedGraph(mst, markers);
          for (typename std::vector<size_t>::iterator it = ws.begin();
               it != ws.end(); ++it) {
            mst.edges[*it].weight += 1.0;
          }
        }

        for (typename std::vector<stochastic_edge<labelT, T>>::iterator it =
                 mst.edges.begin();
             it != mst.edges.end(); ++it) {
          it->weight /= n_seeds;
        }

        std::vector<hierarchy> d    = getHierarchy(mst);
        std::vector<double> weights = weightHierarchy(d, mst);

        bottomUpHierarchy(weights, d, r0, mst);

        // Copy the weights of the subgraph into the graph.
        for (typename std::vector<stochastic_edge<labelT, T>>::iterator it =
                 mst.edges.begin();
             it != mst.edges.end(); ++it) {
          graph
              .edges[graph.nodes[originals[it->source]]
                         .edges[originals[it->dest]]]
              .weight = it->weight;
        }
      }
    }

    nbr_subgraphs = CCL_stochasticGraph(graph, labels);
    applyThreshold(primary, labels, out);
    return nbr_subgraphs;
  }
  /** }@*/
  /** @endcond */
} // namespace smil

#endif //_D_STOCHASTIC_CHABARDES_HPP_
