/* __HEAD__
 * Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES
 * Copyright (c) 2017-2019, Centre de Morphologie Mathematique
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
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Description :
 *   Porting GeoCuts module from Morph-M - This is the Watershed submodule
 *
 * History :
 *   - 20/03/2019 - by Jose-Marcio Martins da Cruz
 *     Just created it...
 *   - XX/XX/XXXX -
 *
 *
 * __HEAD__ - Stop here !
 */

#include <cstddef>
#include <vector>

#include "Core/include/DCore.h"
#include "Morpho/include/DMorpho.h"

// #include "GeoCuts.h"
#include "GeoCuts_Watershed.h"
#include "src-geo-cuts-tools.hpp"

/*
 *
 */
#ifndef __BOOST_INCLUDED__
#define __BOOST_INCLUDED__

#include <boost/config.hpp>
#include <boost/version.hpp>

// for boost::tie
// #include <boost/type_traits.hpp>
#include <boost/utility.hpp>
// for boost::graph_traits
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>

#include <boost/version.hpp>
#if BOOST_VERSION >= 104700
#include <boost/graph/boykov_kolmogorov_max_flow.hpp>
#elif BOOST_VERSION >= 103500
#include <boost/graph/kolmogorov_max_flow.hpp>
#endif

#endif //  __BOOST_INCLUDED__


namespace smil
{
#if 0
  /*
   *
   *
   *
   */
  template <class T1, class T2>
  RES_T geoCutsMultiway_Watershed(const Image<T1> &imIn,
                                   const Image<T2> &imMarker,
                                   const double Power, const StrElt &nl,
                                   Image<T2> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imMarker, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imMarker, &imOut);

    typename Image<T1>::lineType bufIn     = imIn.getPixels();
    typename Image<T2>::lineType bufMarker = imMarker.getPixels();
    typename Image<T2>::lineType bufOut    = imOut.getPixels();

    double exposant = Power;

    // needed for max flow: capacit map, rev_capacity map, etc.
    typedef boost::adjacency_list_traits<boost::vecS, boost::vecS,
                                         boost::directedS>
        Traits_d;

    typedef boost::adjacency_list<
        boost::vecS, boost::vecS, boost::directedS,
        boost::property<boost::vertex_name_t, std::string>,
        boost::property<
            boost::edge_capacity_t, double,
            boost::property<boost::edge_residual_capacity_t, double,
                            boost::property<boost::edge_reverse_t,
                                            Traits_d::edge_descriptor>>>>
        Graph_d;

    Graph_d g;

    boost::property_map<Graph_d, boost::edge_capacity_t>::type capacity =
        boost::get(boost::edge_capacity, g);

    boost::property_map<Graph_d, boost::edge_reverse_t>::type rev =
        get(boost::edge_reverse, g);

    boost::property_map<Graph_d, boost::edge_residual_capacity_t>::type
        residual_capacity = get(boost::edge_residual_capacity, g);

    Graph_d::edge_descriptor e1, e2, e3, e4;
    Graph_d::vertex_descriptor vSource, vSink;
    T1 numVertex = maxVal(imIn);
    T2 numLabels = maxVal(imMarker);

    std::cout << "build graph vertices" << std::endl;

    size_t pixelCount = imIn.getPixelCount();
    for (off_t i = 0; i < (off_t) pixelCount; i++) {
      boost::add_vertex(g);
      bufOut[i] = (T2) 1;
    }

    std::cout << "number of Labels: " << numLabels << std::endl;
    std::cout << "number of vertices: " << numVertex << std::endl;

    vSource = boost::add_vertex(g);
    vSink   = boost::add_vertex(g);

    std::cout << "build graph edges" << std::endl;

    int width  = imIn.getWidth();
    int height = imIn.getHeight();
    int depth  = imIn.getDepth();

    off_t strideY = width;
    off_t strideZ = width * height;
    for (int z = 0; z < depth; z++) {
      off_t p0Z = z * strideZ;
      for (int y = 0; y < height; y++) {
        off_t p0Y = y * strideY;
        for (int x = 0; x < width; x++) {
          off_t o1 = p0Z + p0Y + x;

          // iterators on the Structuring Element
          vector<IntPoint> pts = filterStrElt(nl);
          vector<IntPoint>::iterator itBegin, itEnd, it;
          itBegin = pts.begin();
          itEnd   = pts.end();

          T1 val1 = bufIn[o1];

          for (it = itBegin; it != itEnd; it++) {
            if (x + it->x > width - 1 || x + it->x < 0)
              continue;
            if (y + it->y > height - 1 || y + it->y < 0)
              continue;
            if (z + it->z > depth - 1 || z + it->z < 0)
              continue;

            off_t o2 = o1 + it->z * strideZ + it->y * strideY + it->x;
            if (o2 <= o1)
              continue;

            T2 val2       = bufIn[o2];
            double valeur = (255.0 / (std::abs((double) (val1 - val2)) + 1));
            double cost   = std::pow(valeur, exposant);

            bool hasEdge            = false;
            boost::tie(e4, hasEdge) = boost::add_edge(o1, o2, g);
            boost::tie(e3, hasEdge) = boost::add_edge(o2, o1, g);
            capacity[e4]            = cost;
            capacity[e3]            = cost;
            rev[e4]                 = e3;
            rev[e3]                 = e4;
          }
        }
      }
    }

    for (T2 nbk = 2; nbk <= numLabels; nbk++) {
      // for all pixels in imIn create a vertex
      for (off_t o0 = 0; o0 < (off_t) pixelCount; o0++) {
        T2 val1 = bufMarker[o0];
        T1 val2 = bufIn[o0];

        double cost = std::numeric_limits<double>::max();
        bool hasEdge;
        if (val1 == nbk) {
          boost::tie(e4, hasEdge) = boost::edge(vSource, o0, g);
          if (!hasEdge) {
            boost::tie(e4, hasEdge) = boost::add_edge(vSource, o0, g);
            boost::tie(e3, hasEdge) = boost::add_edge(o0, vSource, g);
            capacity[e4]            = cost;
            capacity[e3]            = cost;
            rev[e4]                 = e3;
            rev[e3]                 = e4;
          }
        } else if (val1 > 1 && val1 != nbk) {
          boost::tie(e4, hasEdge) = boost::edge(val2, vSink, g);
          if (!hasEdge) {
            boost::tie(e4, hasEdge) = boost::add_edge(o0, vSink, g);
            boost::tie(e3, hasEdge) = boost::add_edge(vSink, o0, g);
            capacity[e4]            = cost;
            capacity[e3]            = cost;
            rev[e4]                 = e3;
            rev[e3]                 = e4;
          }
        }
      }

      std::cout << "Compute Max flow" << std::endl;
      boost::property_map<Graph_d, boost::vertex_index_t>::type indexmap =
          boost::get(boost::vertex_index, g);
      std::vector<boost::default_color_type> color(boost::num_vertices(g));
#if BOOST_VERSION >= 104700
      double flow =
          boykov_kolmogorov_max_flow(g, capacity, residual_capacity, rev,
                                     &color[0], indexmap, vSource, vSink);
#else
      double flow = kolmogorov_max_flow(g, capacity, residual_capacity, rev,
                                        &color[0], indexmap, vSource, vSink);
#endif
      std::cout << "c  The total flow:" << std::endl;
      std::cout << "s " << flow << std::endl << std::endl;

      // for all pixels in imIn create a vertex and an edge
      for (off_t o1 = 0; o1 < (off_t) pixelCount; o1++) {
        T1 val1 = (int) bufIn[o1];
        T2 val2 = (int) bufOut[o1];
        T2 val3 = (int) bufMarker[o1];

        if (val2 == 1) {
          if (color[val1] == color[vSource])
            bufOut[o1] = (T2) nbk;
        }

        bool hasEdge;
        if (val3 == nbk) {
          boost::tie(e4, hasEdge) = boost::edge(vSource, o1, g);
          if (hasEdge) {
            boost::remove_edge(vSource, o1, g);
            boost::remove_edge(o1, vSource, g);
          }
        } else if (val3 > 1) {
          boost::tie(e4, hasEdge) = boost::edge(o1, vSink, g);
          if (hasEdge) {
            boost::remove_edge(o1, vSink, g);
            boost::remove_edge(vSink, o1, g);
          }
        }
      }
    }
    return RES_OK;
  }
#endif
} // namespace smil



