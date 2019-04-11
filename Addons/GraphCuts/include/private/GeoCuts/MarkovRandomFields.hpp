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

#ifndef _D_GEOCUTS_MARKOV_HPP_
#define _D_GEOCUTS_MARKOV_HPP_

#include <vector>

namespace smil
{
  /*
   *
   *
   *
   */
  template <class T1, class T2>
  RES_T geoCutsMRF_Ising(const Image<T1> &imIn, const Image<T2> &imMarker,
                         double Beta, double Sigma, const StrElt &nl,
                         Image<T2> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imMarker, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imMarker, &imOut);

    typename Image<T1>::lineType bufIn     = imIn.getPixels();
    typename Image<T2>::lineType bufMarker = imMarker.getPixels();
    typename Image<T2>::lineType bufOut    = imOut.getPixels();

    // needed for max flow: capacit map, rev_capacity map, etc.
    typedef boost::adjacency_list_traits<boost::vecS, boost::vecS,
                                         boost::directedS>
        Traits_d;

    typedef boost::adjacency_list<
        boost::listS, boost::vecS, boost::directedS,
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
    // T2 numLabels = maxVal(imMarker);
    
    double uBound = std::numeric_limits<T1>::max();

    std::cout << "build graph vertices" << std::endl;
    std::cout << "number of vertices : " << numVertex << std::endl;

    double meanFGround = 0;
    double meanBGround = 0;
    double nbFGround   = 0;
    double nbBGround   = 0;

    double meanImage = 0;

    size_t pixelCount = imIn.getPixelCount();
    for (off_t i = 0; i < (off_t) pixelCount; i++) {
      T1 valIn   = bufIn[i];
      T2 valMark = bufMarker[i];

      boost::add_vertex(g);

      if (valMark == 2) {
        meanFGround = meanFGround + valIn;
        nbFGround++;
      }
      if (valMark == 3) {
        meanBGround = meanBGround + valIn;
        nbBGround++;
      }

      meanImage = meanImage + valIn;
    }
    meanImage   = meanImage / pixelCount;
    meanFGround = meanFGround / nbFGround;
    meanBGround = meanBGround / nbBGround;
    // JOE - 255 ? Or max de l'image ???
    meanFGround = meanFGround / uBound;
    meanBGround = meanBGround / uBound;

    std::cout << "number of vertices: " << numVertex << std::endl;
    std::cout << "Foreground Mean   : " << meanFGround << std::endl;
    std::cout << "Background Mean   : " << meanBGround << std::endl;

    vSource = boost::add_vertex(g);
    vSink   = boost::add_vertex(g);

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

          T1 val1   = bufIn[o1];
          T2 marker = bufMarker[o1];

          bool hasEdge = false;

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

            // T2 val2     = bufIn[o2];
            double cost = Beta;

            boost::tie(e4, hasEdge) = boost::add_edge(o1, o2, g);
            boost::tie(e3, hasEdge) = boost::add_edge(o2, o1, g);
            capacity[e4]            = cost;
            capacity[e3]            = cost;
            rev[e4]                 = e3;
            rev[e3]                 = e4;
          }

          if (marker == 2) {
            boost::tie(e4, hasEdge) = boost::add_edge(o1, vSink, g);
            boost::tie(e3, hasEdge) = boost::add_edge(vSink, o1, g);
            capacity[e4]            = (std::numeric_limits<double>::max)();
            capacity[e3]            = (std::numeric_limits<double>::max)();
            rev[e4]                 = e3;
            rev[e3]                 = e4;
          } else if (marker == 3) {
            boost::tie(e4, hasEdge) = boost::add_edge(vSource, o1, g);
            boost::tie(e3, hasEdge) = boost::add_edge(o1, vSource, g);
            capacity[e4]            = (std::numeric_limits<double>::max)();
            capacity[e3]            = (std::numeric_limits<double>::max)();
            rev[e4]                 = e3;
            rev[e3]                 = e4;
          } else {
            // JOE - 255 ???
            val1 = val1 / uBound;

            double sigma  = (double) Sigma;
            double sigmab = 0.2;
            double slink  = (val1 - meanFGround) * (val1 - meanFGround) /
                           (2 * sigma * sigma);
            double tlink = (val1 - meanBGround) * (val1 - meanBGround) /
                           (2 * sigmab * sigmab);

            boost::tie(e4, hasEdge) = boost::add_edge(vSource, o1, g);
            boost::tie(e3, hasEdge) = boost::add_edge(o1, vSource, g);
            capacity[e4]            = slink;
            capacity[e3]            = slink;
            rev[e4]                 = e3;
            rev[e3]                 = e4;

            boost::tie(e4, hasEdge) = boost::add_edge(o1, vSink, g);
            boost::tie(e3, hasEdge) = boost::add_edge(vSink, o1, g);
            capacity[e4]            = tlink;
            capacity[e3]            = tlink;
            rev[e4]                 = e3;
            rev[e3]                 = e4;
          }
        }
      }
    }

    boost::property_map<Graph_d, boost::vertex_index_t>::type indexmap =
        boost::get(boost::vertex_index, g);
    std::vector<boost::default_color_type> color(boost::num_vertices(g));

    std::cout << "Compute Max flow" << std::endl;
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

    std::cout << "Source Label:" << color[vSource] << std::endl;
    std::cout << "Sink  Label:" << color[vSink] << std::endl;

    for (off_t i = 0; i < (off_t) pixelCount; i++) {
      if (color[i] == color[vSource]) {
        bufOut[1] = 3;
        continue;
      }
      if (color[i] == color[vSink]) {
        bufOut[i] = 2;
        continue;
      }
      if (color[i] == 1) {
        bufOut[i] = 4;
        continue;
      }
      bufOut[i] = 0;
    }

    return RES_OK;
  }

  /*
   *
   *
   *
   */
  template <class T1, class T2>
  RES_T geoCutsMRF_EdgePreserving(const Image<T1> &imIn,
                                  const Image<T2> &imMarker, double Beta,
                                  double Sigma, const StrElt &nl,
                                  Image<T2> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imMarker, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imMarker, &imOut);

    typename Image<T1>::lineType bufIn     = imIn.getPixels();
    typename Image<T2>::lineType bufMarker = imMarker.getPixels();
    typename Image<T2>::lineType bufOut    = imOut.getPixels();

    // needed for max flow: capacit map, rev_capacity map, etc.
    typedef boost::adjacency_list_traits<boost::vecS, boost::vecS,
                                         boost::directedS>
        Traits_d;

    typedef boost::adjacency_list<
        boost::listS, boost::vecS, boost::directedS,
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
    // T2 numLabels = maxVal(imMarker);

    double uBound = std::numeric_limits<T1>::max();

    std::cout << "build graph vertices" << std::endl;
    std::cout << "number of vertices : " << numVertex << std::endl;

    double maxIn = (double) maxVal(imIn);

    double meanFGround = 0;
    double meanBGround = 0;
    double nbFGround   = 0;
    double nbBGround   = 0;

    double meanDifference = 0;
    double nbDifference   = 0;

    double meanImage = 0;

    size_t pixelCount = imIn.getPixelCount();

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

          T1 valIn   = bufIn[o1];
          T2 valMark = bufMarker[o1];

          boost::add_vertex(g);

          if (valMark == 2) {
            meanFGround = meanFGround + valIn;
            nbFGround++;
          } else if (valMark == 3) {
            meanBGround = meanBGround + valIn;
            nbBGround++;
          }

          meanImage = meanImage + valIn;

          double val1 = (double) valIn;

          // iterators on the Structuring Element
          vector<IntPoint> pts = filterStrElt(nl);
          vector<IntPoint>::iterator itBegin, itEnd, it;
          itBegin = pts.begin();
          itEnd   = pts.end();

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

            double val2    = (double) bufIn[o2];
            meanDifference = meanDifference + (val2 - val1) * (val2 - val1);
            nbDifference++;
          }
        }
      }
    }

    meanDifference = meanDifference / (nbDifference * maxIn * maxIn);
    meanImage      = meanImage / pixelCount;

    meanFGround = meanFGround / nbFGround;
    meanBGround = meanBGround / nbBGround;
    meanFGround = meanFGround / maxIn;
    meanBGround = meanBGround / maxIn;

    std::cout << "number of vertices : " << numVertex << std::endl;
    std::cout << "Foreground Mean    : " << meanFGround << std::endl;
    std::cout << "Background Mean    : " << meanBGround << std::endl;
    std::cout << "Mean difference    : " << meanDifference << std::endl;

    vSource = boost::add_vertex(g);
    vSink   = boost::add_vertex(g);

    strideY = width;
    strideZ = width * height;
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

          double val1 = ((double) bufIn[o1]) / uBound;
          T2 marker   = bufMarker[o1];

          bool hasEdge = false;

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

            double val2 = ((double) bufIn[o2]) / uBound;
            double cost = Beta * (1. - std::pow(std::abs(val1 - val2), 0.25));

            bool hasEdge            = false;
            boost::tie(e4, hasEdge) = boost::add_edge(o1, o2, g);
            boost::tie(e3, hasEdge) = boost::add_edge(o2, o1, g);
            capacity[e4]            = cost;
            capacity[e3]            = cost;
            rev[e4]                 = e3;
            rev[e3]                 = e4;
          }

          if (marker == 2) {
            boost::tie(e4, hasEdge) = boost::add_edge(o1, vSink, g);
            boost::tie(e3, hasEdge) = boost::add_edge(vSink, o1, g);
            capacity[e4]            = (std::numeric_limits<double>::max)();
            capacity[e3]            = (std::numeric_limits<double>::max)();
            rev[e4]                 = e3;
            rev[e3]                 = e4;

            continue;
          }
          if (marker == 3) {
            boost::tie(e4, hasEdge) = boost::add_edge(vSource, o1, g);
            boost::tie(e3, hasEdge) = boost::add_edge(o1, vSource, g);
            capacity[e4]            = (std::numeric_limits<double>::max)();
            capacity[e3]            = (std::numeric_limits<double>::max)();
            rev[e4]                 = e3;
            rev[e3]                 = e4;

            continue;
          }

          double sigma  = (double) Sigma;
          double sigmab = 0.2;

          double slink =
              (val1 - meanFGround) * (val1 - meanFGround) / (2 * sigma * sigma);
          double tlink = (val1 - meanBGround) * (val1 - meanBGround) /
                         (2 * sigmab * sigmab);

          boost::tie(e4, hasEdge) = boost::add_edge(vSource, o1, g);
          boost::tie(e3, hasEdge) = boost::add_edge(o1, vSource, g);
          capacity[e4]            = slink;
          capacity[e3]            = slink;
          rev[e4]                 = e3;
          rev[e3]                 = e4;

          boost::tie(e4, hasEdge) = boost::add_edge(o1, vSink, g);
          boost::tie(e3, hasEdge) = boost::add_edge(vSink, o1, g);
          capacity[e4]            = tlink;
          capacity[e3]            = tlink;
          rev[e4]                 = e3;
          rev[e3]                 = e4;
        }
      }
    }

    boost::property_map<Graph_d, boost::vertex_index_t>::type indexmap =
        boost::get(boost::vertex_index, g);
    std::vector<boost::default_color_type> color(boost::num_vertices(g));

    std::cout << "Compute Max flow" << std::endl;
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

    std::cout << "Source Label:" << color[vSource] << std::endl;
    std::cout << "Sink  Label:" << color[vSink] << std::endl;

    for (off_t i = 0; i < (off_t) pixelCount; i++) {
      if (color[i] == color[vSource]) {
        bufOut[1] = 3;
        continue;
      }
      if (color[i] == color[vSink]) {
        bufOut[i] = 2;
        continue;
      }
      if (color[i] == 1) {
        bufOut[i] = 4;
        continue;
      }
      bufOut[i] = 0;
    }

    return RES_OK;
  }

  /*
   *
   *
   *
   */
  template <class T1, class T2>
  RES_T geoCutsMRF_Potts(const Image<T1> &imIn, const Image<T2> &imMarker,
                        double Beta, double Sigma, const StrElt &nl,
                        Image<T2> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imMarker, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imMarker, &imOut);

    typename Image<T1>::lineType bufIn     = imIn.getPixels();
    typename Image<T2>::lineType bufMarker = imMarker.getPixels();
    typename Image<T2>::lineType bufOut    = imOut.getPixels();

    // needed for max flow: capacit map, rev_capacity map, etc.
    typedef boost::adjacency_list_traits<boost::vecS, boost::vecS,
                                         boost::directedS>
        Traits_d;

    typedef boost::adjacency_list<
        boost::listS, boost::vecS, boost::directedS,
        boost::property<boost::vertex_name_t, std::string>,
        boost::property<
            boost::edge_capacity_t, double,
            boost::property<boost::edge_residual_capacity_t, double,
                            boost::property<boost::edge_reverse_t,
                                            Traits_d::edge_descriptor>>>>
        Graph_d;

    T1 numVertex = maxVal(imIn);
    // T2 numLabels = maxVal(imMarker);

   double uBound = std::numeric_limits<T1>::max();

    std::cout << "build graph vertices" << std::endl;
    std::cout << "number of vertices : " << numVertex << std::endl;

    
    typedef struct {
      double sigma;
      double valeur;
      double label;
    } combi_t;

    int nbCombi = 4;
    combi_t combi[4] = {
      {Sigma, 1.0, 4},
      {Sigma, 0.75, 3},
      {Sigma, 0.5, 2},
      {Sigma, 0, 1},
    };

    size_t pixelCount = imIn.getPixelCount();

    int width  = imIn.getWidth();
    int height = imIn.getHeight();
    int depth  = imIn.getDepth();

    double maxIn = (double) maxVal(imIn);

    for (int nbk = 0; nbk < nbCombi; nbk++) {
      Graph_d g;

      boost::property_map<Graph_d, boost::edge_capacity_t>::type capacity =
          boost::get(boost::edge_capacity, g);  

      boost::property_map<Graph_d, boost::edge_reverse_t>::type rev =
          get(boost::edge_reverse, g);

      boost::property_map<Graph_d, boost::edge_residual_capacity_t>::type
          residual_capacity = get(boost::edge_residual_capacity, g);

      Graph_d::edge_descriptor e1, e2, e3, e4;
      Graph_d::vertex_descriptor vSource, vSink;
    
      for (int i = 0; i < pixelCount; i++) {
        boost::add_vertex(g);
        bufOut[i] = 0;
      }
      vSource = boost::add_vertex(g);
      vSink   = boost::add_vertex(g);
      
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
            
            bool hasEdge = false;
  
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

              double cost         = Beta;
              bool hasEdge = false;
              boost::tie(e4, hasEdge) = boost::add_edge(o1, o2, g);
              boost::tie(e3, hasEdge) = boost::add_edge(o2, o1, g);
              capacity[e4]        = cost;
              capacity[e3]        = cost;
              rev[e4]             = e3;
              rev[e3]             = e4;
            }

            double val1 = bufIn[o1] / uBound;
            val1 = (val1 - combi[nbk].valeur) * (val1 - combi[nbk].valeur) /
                   (2 * combi[nbk].sigma * combi[nbk].sigma);

            double val2 = 0;
            double val3 = 0;
            double val4 = 0;            
            //
          }
        }
      }
    }


#if 0

    boost::property_map<Graph_d, boost::vertex_index_t>::type indexmap =
        boost::get(boost::vertex_index, g);
    std::vector<boost::default_color_type> color(boost::num_vertices(g));

    std::cout << "Compute Max flow" << std::endl;
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

    std::cout << "Source Label:" << color[vSource] << std::endl;
    std::cout << "Sink  Label:" << color[vSink] << std::endl;

    for (off_t i = 0; i < (off_t) pixelCount; i++) {
      if (color[i] == color[vSource]) {
        bufOut[1] = 3;
        continue;
      }
      if (color[i] == color[vSink]) {
        bufOut[i] = 2;
        continue;
      }
      if (color[i] == 1) {
        bufOut[i] = 4;
        continue;
      }
      bufOut[i] = 0;
    }
#endif
    return RES_OK;
  
  }

  /*
   *
   *
   *
   */
#if 0
  template <class T1, class T2>
  RES_T geoCutsMRF_Potts((const Image<T1> &imIn, const Image<T2> &imMarker,
                        double Beta, double Sigma, const SE &nl,
                        Image<T2> &imOut)
  {
    MORPHEE_ENTER_FUNCTION("t_geoCutsMRF_Potts");
    std::cout << "Enter function t_geoCutsMRF_Potts" << std::endl;

    if ((!imOut.isAllocated())) {
      MORPHEE_REGISTER_ERROR("Not allocated");
      return RES_NOT_ALLOCATED;
    }

    if ((!imIn.isAllocated())) {
      MORPHEE_REGISTER_ERROR("Not allocated");
      return RES_NOT_ALLOCATED;
    }

    if ((!imMarker.isAllocated())) {
      MORPHEE_REGISTER_ERROR("Not allocated");
      return RES_NOT_ALLOCATED;
    }

    // common image iterator
    typename ImageIn::const_iterator it, iend;
    morphee::selement::Neighborhood<SE, ImageIn> neighb(imIn, nl);
    typename morphee::selement::Neighborhood<SE, ImageIn>::iterator nit, nend;
    offset_t o0;
    offset_t o1;

    // needed for max flow: capacit map, rev_capacity map, etc.
    typedef boost::adjacency_list_traits<boost::vecS, boost::vecS,
                                         boost::directedS>
        Traits;
    typedef boost::adjacency_list<
        boost::listS, boost::vecS, boost::directedS,
        boost::property<boost::vertex_name_t, std::string>,
        boost::property<
            boost::edge_capacity_t, double,
            boost::property<boost::edge_residual_capacity_t, double,
                            boost::property<boost::edge_reverse_t,
                                            Traits::edge_descriptor>>>>
        Graph_d;

    Graph_d::edge_descriptor e1, e2, e3, e4;
    Graph_d::vertex_descriptor vSource, vSink;
    int nb_combi = 4;

    bool in1;
    double sigma[4][2];
    sigma[0][0] = Sigma;
    sigma[0][1] = Sigma;

    sigma[1][0] = Sigma;
    sigma[1][1] = Sigma;

    sigma[2][0] = Sigma;
    sigma[2][1] = Sigma;

    sigma[3][0] = Sigma;
    sigma[3][1] = Sigma;

    double combi_valeur[4][2];
    combi_valeur[0][0] = 1.0;
    combi_valeur[0][1] = 0.0;

    combi_valeur[1][0] = 0.75;
    combi_valeur[1][1] = 0.0;

    combi_valeur[2][0] = 0.5;
    combi_valeur[2][1] = 0.0;

    combi_valeur[3][0] = 0.0;
    combi_valeur[3][1] = 0.0;

    double combi_label[4][2];
    combi_label[0][0] = 4;
    combi_label[0][1] = 0;

    combi_label[1][0] = 3;
    combi_label[1][1] = 0;

    combi_label[2][0] = 2;
    combi_label[2][1] = 0;

    combi_label[3][0] = 1;
    combi_label[3][1] = 0;

    for (int nbk = 0; nbk < nb_combi; nbk++) {
      Graph_d g;

      boost::property_map<Graph_d, boost::edge_capacity_t>::type capacity =
          boost::get(boost::edge_capacity, g);

      boost::property_map<Graph_d, boost::edge_reverse_t>::type rev =
          get(boost::edge_reverse, g);

      boost::property_map<Graph_d, boost::edge_residual_capacity_t>::type
          residual_capacity = get(boost::edge_residual_capacity, g);

      std::cout << "build graph vertices" << std::endl;
      int numVert = 0;

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o0      = it.getOffset();
        int val = imIn.pixelFromOffset(o0);
        boost::add_vertex(g);
        numVert++;
        if (nbk == 0) {
          imOut.setPixel(o0, 0);
        }
      }

      vSource = boost::add_vertex(g);
      vSink   = boost::add_vertex(g);

      std::cout << "build graph edges" << std::endl;
      boost::property_map<Graph_d, boost::vertex_index_t>::type indexmap =
          boost::get(boost::vertex_index, g);
      std::vector<boost::default_color_type> color(boost::num_vertices(g));
      std::cout << "build neighborhood edges and terminal links" << std::endl;

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1 = it.getOffset();
        neighb.setCenter(o1);
        double valCenter = imOut.pixelFromOffset(o1);
        double val1      = imIn.pixelFromOffset(o1);

        for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
          const offset_t o2 = nit.getOffset();

          if (o2 <= o1)
            continue;
          if (o2 > o1) {
            double cost         = Beta;
            boost::tie(e4, in1) = boost::add_edge(o1, o2, g);
            boost::tie(e3, in1) = boost::add_edge(o2, o1, g);
            capacity[e4]        = cost;
            capacity[e3]        = cost;
            rev[e4]             = e3;
            rev[e3]             = e4;
          }
        }

        val1 = val1 / 255;
        val1 = (val1 - combi_valeur[nbk][0]) * (val1 - combi_valeur[nbk][0]) /
               (2 * sigma[nbk][0] * sigma[nbk][0]);

        double val2 = 0;
        double val3 = 0;
        double val4 = 0;

        if (nbk == 0) {
          val2 = (val1 - 0.75) * (val1 - 0.75) /
                 (2 * sigma[nbk][1] * sigma[nbk][1]);
          val3 =
              (val1 - 0.5) * (val1 - 0.5) / (2 * sigma[nbk][1] * sigma[nbk][1]);
          val4 = (val1) * (val1) / (2 * sigma[nbk][1] * sigma[nbk][1]);
        } else if (nbk == 1) {
          val2 =
              (val1 - 1.0) * (val1 - 1.0) / (2 * sigma[nbk][1] * sigma[nbk][1]);
          val3 =
              (val1 - 0.5) * (val1 - 0.5) / (2 * sigma[nbk][1] * sigma[nbk][1]);
          val4 = (val1) * (val1) / (2 * sigma[nbk][1] * sigma[nbk][1]);
        } else if (nbk == 2) {
          val2 =
              (val1 - 1.0) * (val1 - 1.0) / (2 * sigma[nbk][1] * sigma[nbk][1]);
          val3 = (val1 - 0.75) * (val1 - 0.75) /
                 (2 * sigma[nbk][1] * sigma[nbk][1]);
          val4 = (val1) * (val1) / (2 * sigma[nbk][1] * sigma[nbk][1]);
        } else if (nbk == 3) {
          val2 =
              (val1 - 1.0) * (val1 - 1.0) / (2 * sigma[nbk][1] * sigma[nbk][1]);
          val3 = (val1 - 0.75) * (val1 - 0.75) /
                 (2 * sigma[nbk][1] * sigma[nbk][1]);
          val4 =
              (val1 - 0.5) * (val1 - 0.5) / (2 * sigma[nbk][1] * sigma[nbk][1]);
        }

        boost::tie(e4, in1) = boost::add_edge(vSource, o1, g);
        boost::tie(e3, in1) = boost::add_edge(o1, vSource, g);
        capacity[e4]        = val1;
        capacity[e3]        = val1;
        rev[e4]             = e3;
        rev[e3]             = e4;

        boost::tie(e4, in1) = boost::add_edge(vSink, o1, g);
        boost::tie(e3, in1) = boost::add_edge(o1, vSink, g);
        capacity[e4]        = std::min(std::min(val2, val3), val4);
        capacity[e3]        = std::min(std::min(val2, val3), val4);
        rev[e4]             = e3;
        rev[e3]             = e4;
      }

      std::cout << "Compute Max flow" << std::endl;
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

      std::cout << "Source Label:" << color[vSource] << std::endl;
      std::cout << "Sink  Label:" << color[vSink] << std::endl;

      for (int i = 0; i < numVert; i++) {
        int valimout = imOut.pixelFromOffset(i);

        if (valimout == 0) {
          if (color[i] == 4)
            imOut.setPixel(i, combi_label[nbk][0]);
        }
      }
    }

    return RES_OK;
  }
#endif

} // namespace smil

#endif // _D_GEOCUTS_MARKOV_HPP_
