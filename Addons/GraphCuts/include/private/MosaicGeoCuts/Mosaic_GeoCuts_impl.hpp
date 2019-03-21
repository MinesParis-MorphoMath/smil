#ifndef D_MOSAIC_GEOCUTS_IMPL_HPP
#define D_MOSAIC_GEOCUTS_IMPL_HPP

#include <time.h>

#include <boost/config.hpp>
// for boost::tie
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

#include <vector>

static int debugOn = false;

#define SMIL_ENTER_FUNCTION(a)                                                 \
  {                                                                            \
    if (debugOn)                                                               \
      cout << "Entering function " << __func__ << " " << (a) << endl;          \
  }

#define SMIL_REGISTER_ERROR(a)

namespace smil
{
  /*
   * Some useful tools
   */
  /* This function selects points from an Structuring element
   * generating a positive offset
   */
  inline vector<IntPoint> filterStrElt(StrElt se)
  {
    vector<IntPoint> pts(0);

    vector<IntPoint>::iterator it, itStart, itEnd;
    itStart = se.points.begin();
    itEnd   = se.points.end();

    for (it = itStart; it != itEnd; it++) {
      bool ok = (4 * it->z + 2 * it->y + it->x) > 0;
      if (ok)
        pts.push_back(*it);

      if (debugOn) {
        std::cout << (ok ? "GOOD " : "BAD  ") << std::right << " "
                  << std::setw(6) << it->x << " " << std::setw(6) << it->y
                  << " " << std::setw(6) << it->z << endl;
      }
    }
    return pts;
  }

  /*
   *
   */
  using namespace boost;

  // needed for max flow: capacit map, rev_capacity map, etc.
  typedef adjacency_list_traits<vecS, vecS, directedS> Traits_T;

  typedef adjacency_list<
      vecS, vecS, directedS, property<vertex_name_t, std::string>,
      property<edge_capacity_t, double,
               property<edge_residual_capacity_t, double,
                        property<edge_reverse_t, Traits_T::edge_descriptor>>>>
      GraphVV_T;

  // edge capacity
  typedef property_map<GraphVV_T, edge_capacity_t>::type EdgeCapVV_T;
  // edge reverse
  typedef property_map<GraphVV_T, edge_reverse_t>::type EdgeRevCapVV_T;
  // edge residual capacity
  typedef property_map<GraphVV_T, edge_residual_capacity_t>::type EdgeResCapVV_T;
  //
  typedef property_map<GraphVV_T, vertex_index_t>::type VertexIndexVV_T;
  
  typedef adjacency_list<
      listS, vecS, directedS, property<vertex_name_t, std::string>,
      property<edge_capacity_t, double,
               property<edge_residual_capacity_t, double,
                        property<edge_reverse_t, Traits_T::edge_descriptor>>>>
      GraphLV_T;

  // edge capacity
  typedef property_map<GraphLV_T, edge_capacity_t>::type EdgeCapLV_T;
  // edge reverse
  typedef property_map<GraphLV_T, edge_reverse_t>::type EdgeRevCapLV_T;
  // edge residual capacity
  typedef property_map<GraphLV_T, edge_residual_capacity_t>::type EdgeResCapLV_T;
  //
  typedef property_map<GraphLV_T, vertex_index_t>::type VertexIndexLV_T;

  /*
   *
   *
   *
   *
   */
  template <class T>
  RES_T GeoCuts_MinSurfaces(const Image<T> &imIn, const Image<T> &imGrad,
                            const Image<T> &imMarker, const StrElt &nl,
                            Image<T> &imOut)
  {
    SMIL_ENTER_FUNCTION("");

    ASSERT_ALLOCATED(&imIn, &imGrad, &imMarker, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imGrad, &imMarker, &imOut);

    typename Image<T>::lineType bufIn     = imIn.getPixels();
    typename Image<T>::lineType bufOut    = imOut.getPixels();
    typename Image<T>::lineType bufMarker = imMarker.getPixels();
    typename Image<T>::lineType bufGrad   = imGrad.getPixels();

    GraphVV_T g;
    EdgeCapVV_T capacity = boost::get(boost::edge_capacity, g);
    EdgeRevCapVV_T rev   = boost::get(boost::edge_reverse, g);
    EdgeResCapVV_T residual_capacity =
        boost::get(boost::edge_residual_capacity, g);

    GraphVV_T::edge_descriptor e1, e2, e3, e4, e5;
    GraphVV_T::vertex_descriptor vSource, vSink;

    int numVertex = maxVal(imIn);
    int numEdges  = 0;

    std::cout << "build Region Adjacency Graph" << std::endl;
    std::cout << "build Region Adjacency Graph Vertices" << std::endl;

    clock_t t1 = clock();

    std::cout << "number of Vertices : " << numVertex << std::endl;

    for (int i = 1; i <= numVertex; i++) {
      boost::add_vertex(g);
    }

    vSource = boost::add_vertex(g);
    vSink   = boost::add_vertex(g);

    clock_t tt_marker2 = 0, tt_marker3 = 0, tt_new_edge = 0, tt_old_edge = 0;
    clock_t t2 = clock();
    std::cout << "Nodes creation time : " << double(t2 - t1) / CLOCKS_PER_SEC
              << " seconds\n";

    std::cout << "Building Region Adjacency Graph Edges" << std::endl;
    t1 = clock();

    // iterators on the Structuring Element
    vector<IntPoint> pts = filterStrElt(nl);
    vector<IntPoint>::iterator itBegin, itEnd;
    itBegin = pts.begin();
    itEnd   = pts.end();

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

          int val1        = (int) bufIn[o1];
          int marker      = (int) bufMarker[o1];
          int val_prec    = 0;
          int marker_prec = 0;

          if (val1 <= 0)
            continue;

          // do the following only for val1 > 0
          if (marker == 2 && marker_prec != marker && val_prec != val1) {
            clock_t temps_marker2 = clock();

            bool hasEdge;
            boost::tie(e4, hasEdge) = boost::edge(vSource, val1, g);
            if (!hasEdge) {
              // std::cout<<"Add new edge marker 2"<<std::endl;
              boost::tie(e4, hasEdge) = boost::add_edge(vSource, val1, g);
              boost::tie(e3, hasEdge) = boost::add_edge(val1, vSource, g);
              capacity[e4]            = (std::numeric_limits<double>::max)();
              capacity[e3]            = (std::numeric_limits<double>::max)();
              rev[e4]                 = e3;
              rev[e3]                 = e4;
            }
            tt_marker2 += clock() - temps_marker2;
          } else {
            if (marker == 3 && marker_prec != marker && val_prec != val1) {
              clock_t temps_marker3 = clock();
              bool hasEdge;
              boost::tie(e3, hasEdge) = boost::edge(vSink, val1, g);
              if (!hasEdge) {
                // std::cout<<"Add new edge marker 3"<<std::endl;
                boost::tie(e4, hasEdge) = boost::add_edge(val1, vSink, g);
                boost::tie(e3, hasEdge) = boost::add_edge(vSink, val1, g);
                capacity[e4]            = (std::numeric_limits<double>::max)();
                capacity[e3]            = (std::numeric_limits<double>::max)();
                rev[e4]                 = e3;
                rev[e3]                 = e4;
              }
              tt_marker3 += clock() - temps_marker3;
            }
          }

          double val_grad_o1 = (double) bufGrad[o1];
          // val de val2 precedente
          int val2_prec = val1;

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

            int val2 = bufIn[o2];
            if (val1 == val2)
              continue;
              
            double val_grad_o2 = bufGrad[o2];
            double maxi        = std::max(val_grad_o1, val_grad_o2);
            double cost        = 10000.0 / (1.0 + std::pow(maxi, 4));

            if (val2_prec == val2) {
              // same val2 means same edge (thus, keep e3 and e4)
              capacity[e4] = capacity[e4] + cost;
              capacity[e3] = capacity[e3] + cost;
            } else {
              bool hasEdge;
              
              boost::tie(e5, hasEdge) = boost::edge(val1, val2, g);
              if (hasEdge == 0) {
                clock_t temps_new_edge = clock();
                // std::cout<<"Add new edge "<< val1<<" --
                // "<<val2<<std::endl;
                numEdges++;
                boost::tie(e4, hasEdge) = boost::add_edge(val1, val2, g);
                boost::tie(e3, hasEdge) = boost::add_edge(val2, val1, g);
                capacity[e4]            = cost;
                capacity[e3]            = cost;
                rev[e4]                 = e3;
                rev[e3]                 = e4;
                tt_new_edge += clock() - temps_new_edge;                
              } else {
                clock_t temps_old_edge = clock();
                // std::cout<<"existing edge"<<std::endl;
                boost::tie(e4, hasEdge) = boost::edge(val1, val2, g);
                boost::tie(e3, hasEdge) = boost::edge(val2, val1, g);
                capacity[e4]            = capacity[e4] + cost;
                capacity[e3]            = capacity[e3] + cost;
                tt_old_edge += clock() - temps_old_edge;
              }
              val2_prec = val2;
            }
          }
          val_prec    = val1;
          marker_prec = marker;
        }
      }
    }

    std::cout << "Number of edges : " << numEdges << std::endl;
    t2 = clock();
    std::cout << "Edges creation time : " << double(t2 - t1) / CLOCKS_PER_SEC
              << " seconds\n";
    std::cout << "Marker2   : " << double(tt_marker2) / CLOCKS_PER_SEC
              << " seconds\n";
    std::cout << "Marker3   : " << double(tt_marker3) / CLOCKS_PER_SEC
              << " seconds\n";
    std::cout << "New edges : " << double(tt_new_edge) / CLOCKS_PER_SEC
              << " seconds\n";
    std::cout << "Old edges : " << double(tt_old_edge) / CLOCKS_PER_SEC
              << " seconds\n";

    boost::property_map<GraphVV_T, boost::vertex_index_t>::type indexmap =
        boost::get(boost::vertex_index, g);
    std::vector<boost::default_color_type> color(boost::num_vertices(g));

    std::cout << "Compute Max flow" << std::endl;
    t1 = clock();
#if BOOST_VERSION >= 104700
      // XXX - JOE - Check against this function prototype :
      // boykov_kolmogorov_max_flow(
      //     Graph& g,
      //     CapacityEdgeMap cap,
      //     ResidualCapacityEdgeMap res_cap,
      //     ReverseEdgeMap rev_map,
      //     PredecessorMap pre_map,
      //     ColorMap color,
      //     DistanceMap dist,
      //     IndexMap idx,
      //     typename graph_traits<Graph>::vertex_descriptor src,
      //     typename graph_traits<Graph>::vertex_descriptor sink)
    double flow =
        boykov_kolmogorov_max_flow(g, capacity, residual_capacity, rev,
                                   &color[0], indexmap, vSource, vSink);
#else
    double flow = kolmogorov_max_flow(g, capacity, residual_capacity, rev,
                                      &color[0], indexmap, vSource, vSink);
#endif
    std::cout << "c  The total flow:" << std::endl;
    std::cout << "s " << flow << std::endl << std::endl;
    t2 = clock();
    std::cout << "Flow computation time : " << double(t2 - t1) / CLOCKS_PER_SEC
              << " seconds\n";

    t1                = clock();
    size_t pixelCount = imIn.getPixelCount();
    for (off_t o1 = 0; o1 < (off_t) pixelCount; o1++) {
      int val = (T) bufIn[o1];

      if (val == 0) {
        bufOut[o1] = 0;
      } else {
        if (color[val] == color[vSource])
          bufOut[o1] = 2;
        else if (color[val] == color[vSink])
          bufOut[o1] = 3;
        else
          bufOut[o1] = 4;
      }
    }

    t2 = clock();
    std::cout << "Computing imOut took : " << double(t2 - t1) / CLOCKS_PER_SEC
              << " seconds\n";

    return RES_OK;
  }

  /*
   *
   *
   *
   *
   */
#if 0
  template <class ImageIn, class ImageGrad, class ImageMarker, class SE,
            class ImageOut>
  RES_T GeoCuts_MinSurfaces_with_Line(
      const ImageIn &imIn, const ImageGrad &imGrad, const ImageMarker &imMarker,
      const SE &nl, ImageOut &imOut)

  }
#endif

  /*
   *
   *
   *
   *
   */
#if 0
  // ImageLabel and ImageMarker should be unsigned integers
  template <class ImageLabel, class ImageVal, class ImageMarker, class SE,
            class ImageOut>
  RES_T GeoCuts_MinSurfaces_with_steps(
      const ImageLabel &imLabel, const ImageVal &imVal,
      const ImageMarker &imMarker, const SE &nl, F_SIMPLE step_x,
      F_SIMPLE step_y, F_SIMPLE step_z, ImageOut &imOut)
#endif

  /*
   *
   *
   *
   *
   */
#if 0
  // ImageLabel and ImageMarker should be unsigned integers
  template <class ImageLabel, class ImageVal, class ImageMarker, class SE,
            class ImageOut>
  RES_T GeoCuts_MinSurfaces_with_steps_vGradient(
      const ImageLabel &imLabel, const ImageVal &imVal,
      const ImageMarker &imMarker, const SE &nl, F_SIMPLE step_x,
      F_SIMPLE step_y, F_SIMPLE step_z, ImageOut &imOut)
#endif

  /*
   *
   *
   *
   *
   */
#if 0
  template <class ImageIn, class ImageGrad, class ImageMarker, class SE,
            class ImageOut>
  RES_T GeoCuts_MinSurfaces_with_steps_old(
      const ImageIn &imIn, const ImageGrad &imGrad, const ImageMarker &imMarker,
      const SE &nl, F_SIMPLE step_x, F_SIMPLE step_y, F_SIMPLE step_z,
      ImageOut &imOut)
#endif

  /*
   *
   *
   *
   *
   */
  template <class T>
  RES_T GeoCuts_MultiWay_MinSurfaces(const Image<T> &imIn,
                                     const Image<T> &imGrad,
                                     const Image<T> &imMarker, const StrElt &nl,
                                     Image<T> &imOut)
  {
    SMIL_ENTER_FUNCTION("");

    ASSERT_ALLOCATED(&imIn, &imGrad, &imMarker, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imGrad, &imMarker, &imOut);

    typename Image<T>::lineType bufIn     = imIn.getPixels();
    typename Image<T>::lineType bufOut    = imOut.getPixels();
    typename Image<T>::lineType bufMarker = imMarker.getPixels();
    typename Image<T>::lineType bufGrad   = imGrad.getPixels();

    std::cout << "build Region Adjacency Graph" << std::endl;

    GraphLV_T g;
    EdgeCapLV_T capacity = boost::get(boost::edge_capacity, g);
    EdgeRevCapLV_T rev   = boost::get(boost::edge_reverse, g);
    EdgeResCapLV_T residual_capacity =
        boost::get(boost::edge_residual_capacity, g);

    GraphLV_T::edge_descriptor e1, e2, e3, e4;
    GraphLV_T::vertex_descriptor vSource, vSink;

    int numVertex = maxVal(imIn);
    int numLabels = maxVal(imMarker);

    size_t pixelCount = imIn.getPixelCount();

    std::cout << "number of labels :" << numLabels << std::endl;

    std::cout << "build Region Adjacency Graph Vertices" << std::endl;

    for (int i = 0; i <= numVertex; i++) {
      boost::add_vertex(g);
    }

    vSource = boost::add_vertex(g);
    vSink   = boost::add_vertex(g);

    std::vector<boost::default_color_type> color(boost::num_vertices(g));
    VertexIndexLV_T indexmap = boost::get(boost::vertex_index, g);

    std::cout << "build Region Adjacency Graph Edges" << std::endl;

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
          int val1 = (int) bufIn[o1];
          // int marker = (int) bufMarker[o1]; // XXX unused ???

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

            int val2 = bufIn[o2];
            if (val1 == val2)
              continue;

            bool hasEdge;
            boost::tie(e3, hasEdge) = boost::edge(val1, val2, g);
            // std::cout<<hasEdge<<std::endl;
            // std::cout<<"Compute Gradient"<<std::endl;
            double val3 = (double) bufGrad[o1];
            double val4 = (double) bufGrad[o2];
            double maxi = std::max(val3, val4);
            double cost = 10000.0 / (1.0 + std::pow(maxi, 4));

            if (!hasEdge) {
              // std::cout<<"Add new edge"<<std::endl;
              boost::tie(e4, hasEdge) = boost::add_edge(val1, val2, g);
              boost::tie(e3, hasEdge) = boost::add_edge(val2, val1, g);
              capacity[e4]            = cost;
              capacity[e3]            = cost;
              rev[e4]                 = e3;
              rev[e3]                 = e4;
            } else {
              // std::cout<<"existing edge"<<std::endl;
              boost::tie(e4, hasEdge) = boost::edge(val1, val2, g);
              boost::tie(e3, hasEdge) = boost::edge(val2, val1, g);
              capacity[e4]            = capacity[e4] + cost;
              capacity[e3]            = capacity[e3] + cost;
            }
          }
        }
      }
    }

    for (int nbk = 2; nbk <= numLabels; nbk++) {
      // for all pixels in imIn create a vertex
      for (off_t o0 = 0; o0 < (off_t) pixelCount; o0++) {
        int val1 = (int) bufMarker[o0];
        int val2 = (int) bufIn[o0];

        bool hasEdge;
        if (val1 == nbk) {
          boost::tie(e4, hasEdge) = boost::edge(vSource, val2, g);
          if (!hasEdge) {
            boost::tie(e4, hasEdge) = boost::add_edge(vSource, val2, g);
            boost::tie(e3, hasEdge) = boost::add_edge(val2, vSource, g);
            capacity[e4]            = (std::numeric_limits<double>::max)();
            capacity[e3]            = (std::numeric_limits<double>::max)();
            rev[e4]                 = e3;
            rev[e3]                 = e4;
          }
        } else if (val1 > 1 && val1 != nbk) {
          boost::tie(e4, hasEdge) = boost::edge(val2, vSink, g);
          if (!hasEdge) {
            boost::tie(e4, hasEdge) = boost::add_edge(val2, vSink, g);
            boost::tie(e3, hasEdge) = boost::add_edge(vSink, val2, g);
            capacity[e4]            = (std::numeric_limits<double>::max)();
            capacity[e3]            = (std::numeric_limits<double>::max)();
            rev[e4]                 = e3;
            rev[e3]                 = e4;
          }
        }
      }

      std::cout << "Compute Max flow" << nbk << std::endl;
#if BOOST_VERSION >= 104700
      // XXX - JOE - Check against this function prototype :
      // boykov_kolmogorov_max_flow(
      //     Graph& g,
      //     CapacityEdgeMap cap,
      //     ResidualCapacityEdgeMap res_cap,
      //     ReverseEdgeMap rev_map,
      //     PredecessorMap pre_map,
      //     ColorMap color,
      //     DistanceMap dist,
      //     IndexMap idx,
      //     typename graph_traits<Graph>::vertex_descriptor src,
      //     typename graph_traits<Graph>::vertex_descriptor sink)
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
        int val1 = (int) bufIn[o1];
        int val2 = (int) bufOut[o1];
        int val3 = (int) bufMarker[o1];

        if (val2 == 1) {
          if (color[val1] == color[vSource])
            bufOut[o1] = (T) nbk;
        }

        bool hasEdge;
        if (val3 == nbk) {
          boost::tie(e4, hasEdge) = boost::edge(vSource, val1, g);
          if (hasEdge) {
            boost::remove_edge(vSource, val1, g);
            boost::remove_edge(val1, vSource, g);
          }
        } else if (val3 > 1) {
          boost::tie(e4, hasEdge) = boost::edge(val1, vSink, g);
          if (hasEdge) {
            boost::remove_edge(val1, vSink, g);
            boost::remove_edge(vSink, val1, g);
          }
        }
      }
    }

    return RES_OK;
  }

  /*
   *
   *
   *
   *
   */
#if 0
  template <class ImageIn, class ImageGrad, class ImageMosaic,
            class ImageMarker, class SE, class ImageOut>
  RES_T GeoCuts_Optimize_Mosaic(
      const ImageIn &imIn, const ImageGrad &imGrad, const ImageMosaic &imMosaic,
      const ImageMarker &imMarker, const SE &nl, ImageOut &imOut)
#endif

  /*
   *
   *
   *
   *
   */
#if 0
  template <class ImageIn, class ImageGrad, class ImageCurvature,
            class ImageMarker, typename _Beta, class SE, class ImageOut>
  RES_T GeoCuts_Regularized_MinSurfaces(
      const ImageIn &imIn, const ImageGrad &imGrad,
      const ImageCurvature &imCurvature, const ImageMarker &imMarker,
      const _Beta Beta, const SE &nl, ImageOut &imOut)
#endif

  /*
   *
   *
   *
   *
   */
#if 0
  template <class ImageIn, class ImageMosaic, class ImageMarker, class SE,
            class ImageOut>
  RES_T GeoCuts_Segment_Graph(const ImageIn &imIn, const ImageMosaic &imMosaic,
                              const ImageMarker &imMarker, const SE &nl,
                              ImageOut &imOut)
#endif

  /*
   *
   *
   *
   *
   */
#if 0
  template <class ImageIn, class ImageMosaic, class ImageMarker, typename _Beta,
            typename _Sigma, class SE, class ImageOut>
  RES_T MAP_MRF_edge_preserving(
      const ImageIn &imIn, const ImageMosaic &imMosaic,
      const ImageMarker &imMarker, const _Beta Beta, const _Sigma Sigma,
      const SE &nl, ImageOut &imOut)
#endif

  /*
   *
   *
   *
   *
   */
#if 0
  template <class ImageIn, class ImageMosaic, class ImageMarker, typename _Beta,
            typename _Sigma, class SE, class ImageOut>
  RES_T MAP_MRF_Ising(const ImageIn &imIn, const ImageMosaic &imMosaic,
                      const ImageMarker &imMarker, const _Beta Beta,
                      const _Sigma Sigma, const SE &nl, ImageOut &imOut)
#endif

  /*
   *
   *
   *
   *
   */
#if 0
  template <class ImageIn, class ImageMosaic, class ImageMarker, typename _Beta,
            typename _Sigma, class SE, class ImageOut>
  RES_T MAP_MRF_Potts(const ImageIn &imIn, const ImageMosaic &imMosaic,
                      const ImageMarker &imMarker, const _Beta Beta,
                      const _Sigma Sigma, const SE &nl, ImageOut &imOut)
#endif

/*
 *
 *
 *
 *
 */
#define indent "   "

  inline void printSE(StrElt ss)
  {
    cout << indent << "Structuring Element" << endl;
    cout << indent << "Type: " << ss.seT << endl;
    cout << indent << "Size: " << ss.size << endl;
    size_t ptNbr = ss.points.size();
    cout << indent << "Point Nbr: " << ptNbr << endl;
    if (ptNbr == 0)
      return;

    vector<IntPoint>::iterator itStart = ss.points.begin();
    vector<IntPoint>::iterator it, itEnd;
    itEnd = ss.points.end();

    vector<IntPoint> z(0);
    z       = filterStrElt(ss);
    itStart = z.begin();
    itEnd   = z.end();
    for (it = itStart; it != itEnd; it++) {
      // cout << "z      " << "* " << it->x << " " << it->y << " " << it->z <<
      // endl;
    }
    cout << " =================== " << endl;
  }

  template <class T> void testHandleSE(const Image<T> &img, StrElt se)
  {
    StrElt sx = se;
    // sx = CubeSE(2);
    printSE(sx);
    sx = sx.noCenter();
    printSE(sx);
    int sz = sx.getSize();
    sx     = sx.homothety(sz);
    printSE(sx);

#if 1
    img.printSelf();
    cout << "* width " << img.getWidth() << endl;
    cout << "* height " << img.getHeight() << endl;
    cout << "* depth " << img.getDepth() << endl;
#endif
  }

} // namespace smil

#endif // D_MOSAIC_GEOCUTS_IMPL_HPP
