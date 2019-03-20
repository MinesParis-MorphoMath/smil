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

#define SMIL_ENTER_FUNCTION(a)                                              \
  {                                                                            \
    if (debugOn)                                                               \
      cout << "Entering function " << __func__ << " " << (a) << endl;          \
  }

#define SMIL_REGISTER_ERROR(a)

typedef off_t offset_t;

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
      Graph_T;

  // edge capacity
  typedef property_map<Graph_T, edge_capacity_t>::type EdgeCap_T;
  // edge reverse
  typedef property_map<Graph_T, edge_reverse_t>::type EdgeRevCap_T;
  // edge residual capacity
  typedef property_map<Graph_T, edge_residual_capacity_t>::type EdgeResCap_T;
  //
  typedef property_map<Graph_T, vertex_index_t>::type VertexIndex_T;

  /*
   *
   *
   *
   *
   */
#if 1
  template <class T>
  RES_T GeoCuts_MinSurfaces(const Image<T> &imIn, const Image<T> &imGrad,
                            const Image<T> &imMarker, const StrElt &nl,
                            Image<T> &imOut)
  {
    SMIL_ENTER_FUNCTION("");

    ASSERT_ALLOCATED(&imIn, &imGrad, &imMarker, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imGrad, &imMarker, &imOut);

    offset_t o0;
    offset_t o1;

    Graph_T g;
    EdgeCap_T capacity = boost::get(boost::edge_capacity, g);
    EdgeRevCap_T rev   = boost::get(boost::edge_reverse, g);
    EdgeResCap_T residual_capacity =
        boost::get(boost::edge_residual_capacity, g);

    bool in1;
    Graph_T::edge_descriptor e1, e2, e3, e4, e5;
    Graph_T::vertex_descriptor vSource, vSink;
    int numVert  = 0;
    int numEdges = 0;
    int max, not_used;

    std::cout << "build Region Adjacency Graph" << std::endl;
    std::cout << "build Region Adjacency Graph Vertices" << std::endl;

    clock_t t1 = clock();

    max = maxVal(imIn);

    numVert = max;
    std::cout << "number of Vertices : " << numVert << std::endl;

    for (int i = 1; i <= numVert; i++) {
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

    typename Image<T>::lineType bufIn     = imIn.getPixels();
    typename Image<T>::lineType bufOut    = imOut.getPixels();
    typename Image<T>::lineType bufMarker = imMarker.getPixels();
    typename Image<T>::lineType bufGrad   = imGrad.getPixels();

    int pixelCount = imIn.getPixelCount();
    for (o1 = 0; o1 < pixelCount; o1++) {
      int val      = (T) bufIn[o1];
      int marker   = (T) bufMarker[o1];
      int val_prec = 0, marker_prec = 0;

      if (val <= 0)
        continue;

      if (val > 0) {
        if (marker == 2 && marker_prec != marker && val_prec != val) {
          clock_t temps_marker2 = clock();
          boost::tie(e4, in1)   = boost::edge(vSource, val, g);

          if (in1 == 0) {
            // std::cout<<"Add new edge marker 2"<<std::endl;
            boost::tie(e4, in1) = boost::add_edge(vSource, val, g);
            boost::tie(e3, in1) = boost::add_edge(val, vSource, g);
            capacity[e4]        = (std::numeric_limits<double>::max)();
            capacity[e3]        = (std::numeric_limits<double>::max)();
            rev[e4]             = e3;
            rev[e3]             = e4;
          }
          tt_marker2 += clock() - temps_marker2;
        } else {
          if (marker == 3 && marker_prec != marker && val_prec != val) {
            clock_t temps_marker3 = clock();
            boost::tie(e3, in1)   = boost::edge(vSink, val, g);
            if (in1 == 0) {
              // std::cout<<"Add new edge marker 3"<<std::endl;
              boost::tie(e4, in1) = boost::add_edge(val, vSink, g);
              boost::tie(e3, in1) = boost::add_edge(vSink, val, g);
              capacity[e4]        = (std::numeric_limits<double>::max)();
              capacity[e3]        = (std::numeric_limits<double>::max)();
              rev[e4]             = e3;
              rev[e3]             = e4;
            }
            tt_marker3 += clock() - temps_marker3;
          }
        }

        double val_grad_o1 = imGrad.getPixel(o1);
        // val de val2 precedente
        int val2_prec = val;

#if 0
        morphee::selement::Neighborhood<SE, ImageIn> neighb(imIn, nl);
        typename morphee::selement::Neighborhood<SE, ImageIn>::iterator nit, nend;

        neighb.setCenter(o1);

        for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
          const offset_t o2 = nit.getOffset();

          if (o2 <= o1)
            continue;
          if (o2 > o1) {
            int val2 = imIn.pixelFromOffset(o2);
            if (val != val2) {
              double val_grad_o2 = imGrad.pixelFromOffset(o2);
              double maxi        = std::max(val_grad_o1, val_grad_o2);
              double cost        = 10000.0 / (1.0 + std::pow(maxi, 4));

              if (val2_prec == val2) {
                // same val2 means same edge (thus, keep e3 and e4)
                capacity[e4] = capacity[e4] + cost;
                capacity[e3] = capacity[e3] + cost;
              } else {
                boost::tie(e5, in1) = boost::edge(val, val2, g);

                if (in1 == 0) {
                  clock_t temps_new_edge = clock();
                  // std::cout<<"Add new edge "<< val<<" --
                  // "<<val2<<std::endl;
                  numEdges++;
                  boost::tie(e4, in1) = boost::add_edge(val, val2, g);
                  boost::tie(e3, in1) = boost::add_edge(val2, val, g);
                  capacity[e4]        = cost;
                  capacity[e3]        = cost;
                  rev[e4]             = e3;
                  rev[e3]             = e4;
                  tt_new_edge += clock() - temps_new_edge;

                } else {
                  clock_t temps_old_edge = clock();
                  // std::cout<<"existing edge"<<std::endl;
                  boost::tie(e4, in1) = boost::edge(val, val2, g);
                  boost::tie(e3, in1) = boost::edge(val2, val, g);
                  capacity[e4]        = capacity[e4] + cost;
                  capacity[e3]        = capacity[e3] + cost;
                  tt_old_edge += clock() - temps_old_edge;
                }
                val2_prec = val2;
              }
            }
          }
        }
        val_prec    = val;
        marker_prec = marker;
#endif
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

    boost::property_map<Graph_T, boost::vertex_index_t>::type indexmap =
        boost::get(boost::vertex_index, g);
    std::vector<boost::default_color_type> color(boost::num_vertices(g));

    std::cout << "Compute Max flow" << std::endl;
    t1 = clock();
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
    t2 = clock();
    std::cout << "Flow computation time : " << double(t2 - t1) / CLOCKS_PER_SEC
              << " seconds\n";

    t1   = clock();
    pixelCount = imIn.getPixelCount();
    for (o1 = 0; o1 < pixelCount; o1++) {
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
#endif

#if 0
  /*
   *
   *
   *
   *
   */
  template <class ImageIn, class ImageGrad, class ImageMarker, class SE,
            class ImageOut>
  RES_T GeoCuts_MinSurfaces_with_Line(
      const ImageIn &imIn, const ImageGrad &imGrad, const ImageMarker &imMarker,
      const SE &nl, ImageOut &imOut)
  {
    SMIL_ENTER_FUNCTION("");

    std::cout << "Enter function t_GeoCuts_MinSurfaces_With_Line" << std::endl;

    if ((!imOut.isAllocated())) {
      SMIL_REGISTER_ERROR("Not allocated");
      return RES_NOT_ALLOCATED;
    }

    if ((!imIn.isAllocated())) {
      SMIL_REGISTER_ERROR("Not allocated");
      return RES_NOT_ALLOCATED;
    }

    if ((!imGrad.isAllocated())) {
      SMIL_REGISTER_ERROR("Not allocated");
      return RES_NOT_ALLOCATED;
    }

    if ((!imMarker.isAllocated())) {
      SMIL_REGISTER_ERROR("Not allocated");
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

    Graph_d g;

    double sigma = 1.0;

    boost::property_map<Graph_d, boost::edge_capacity_t>::type capacity =
        boost::get(boost::edge_capacity, g);

    boost::property_map<Graph_d, boost::edge_reverse_t>::type rev =
        get(boost::edge_reverse, g);

    boost::property_map<Graph_d, boost::edge_residual_capacity_t>::type
        residual_capacity = get(boost::edge_residual_capacity, g);

    bool in1;
    Graph_d::edge_descriptor e1, e2, e3, e4;
    Graph_d::vertex_descriptor vSource, vSink;
    int numVert = 0;

    std::cout << "build Region Adjacency Graph" << std::endl;

    for (it = imIn.begin(), iend = imIn.end(); it != iend; ++it) {
      // for all pixels in imIn create a vertex
      o0      = it.getOffset();
      int val = imIn.pixelFromOffset(o0);

      if (val > numVert) {
        numVert = val;
      }
    }

    std::cout << "build Region Adjacency Graph Vertices :" << numVert
              << std::endl;

    for (int i = 0; i <= numVert; i++) {
      boost::add_vertex(g);Mosaic_GeoCuts.hpp
    }

    vSource = boost::add_vertex(g);
    vSink   = boost::add_vertex(g);

    std::cout << "build Region Adjacency Graph Edges" << std::endl;

    for (it = imIn.begin(), iend = imIn.end(); it != iend; ++it) {
      o1 = it.getOffset();
      // std::cout<<o1<<std::endl;

      int val    = imIn.pixelFromOffset(o1);
      int marker = imMarker.pixelFromOffset(o1);

      if (marker == 2) {
        boost::tie(e4, in1) = boost::edge(vSource, val, g);
        if (in1 == 0) {
          // std::cout<<"Add new edge marker 2"<<std::endl;
          boost::tie(e4, in1) = boost::add_edge(vSource, val, g);
          boost::tie(e3, in1) = boost::add_edge(val, vSource, g);
          capacity[e4]        = (std::numeric_limits<double>::max)();
          capacity[e3]        = (std::numeric_limits<double>::max)();
          rev[e4]             = e3;
          rev[e3]             = e4;
        }
      } else if (marker == 3) {
        boost::tie(e3, in1) = boost::edge(vSink, val, g);
        if (in1 == 0) {
          // std::cout<<"Add new edge marker 3"<<std::endl;
          boost::tie(e4, in1) = boost::add_edge(val, vSink, g);
          boost::tie(e3, in1) = boost::add_edge(vSink, val, g);
          capacity[e4]        = (std::numeric_limits<double>::max)();
          capacity[e3]        = (std::numeric_limits<double>::max)();
          rev[e4]             = e3;
          rev[e3]             = e4;
        }
      }

      neighb.setCenter(o1);

      for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
        const offset_t o2 = nit.getOffset();
        int val2          = imIn.pixelFromOffset(o2);

        if (o2 > o1) {
          if (val != val2) {
            boost::tie(e3, in1) = boost::edge(val, val2, g);
            // std::cout<<in1<<std::endl;
            // std::cout<<"Compute Gradient"<<std::endl;
            double val3 = imGrad.pixelFromOffset(o1);
            double val4 = imGrad.pixelFromOffset(o2);
            double maxi = std::max(val3, val4);
            double cost = 10000000.0 / (1.0 + std::pow(maxi, 5));

            if (in1 == 0) {
              // std::cout<<"Add new edge"<<std::endl;
              boost::tie(e4, in1) = boost::add_edge(val, val2, g);
              boost::tie(e3, in1) = boost::add_edge(val2, val, g);
              capacity[e4]        = cost;
              capacity[e3]        = cost;
              rev[e4]             = e3;
              rev[e3]             = e4;
            } else {
              // std::cout<<"existing edge"<<std::endl;
              boost::tie(e4, in1) = boost::edge(val, val2, g);
              boost::tie(e3, in1) = boost::edge(val2, val, g);
              capacity[e4]        = capacity[e4] + cost;
              capacity[e3]        = capacity[e3] + cost;
            }
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
    double flow   = kolmogorov_max_flow(g, capacity, residual_capacity, rev,
                                      &color[0], indexmap, vSource, vSink);
#endif
    std::cout << "c  The total flow:" << std::endl;
    std::cout << "s " << flow << std::endl << std::endl;

    for (it = imIn.begin(), iend = imIn.end(); it != iend; ++it) {
      // for all pixels in imIn create a vertex and an edge
      o1      = it.getOffset();
      int val = imIn.pixelFromOffset(o1);

      if (color[val] == color[vSource])
        imOut.setPixel(o1, 2);
      else if (color[val] == color[vSink])
        imOut.setPixel(o1, 3);
      else
        imOut.setPixel(o1, 4);
    }

    for (it = imIn.begin(), iend = imIn.end(); it != iend; ++it) {
      // for all pixels in imIn create a vertex and an edge
      o1           = it.getOffset();
      int valeur_c = imOut.pixelFromOffset(o1);
      neighb.setCenter(o1);

      for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
        const offset_t o2 = nit.getOffset();
        int valeur_n      = imOut.pixelFromOffset(o2);

        if (valeur_n != valeur_c && valeur_n != 0 && valeur_c != 0) {
          imOut.setPixel(o1, 0);
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
  // ImageLabel and ImageMarker should be unsigned integers
  template <class ImageLabel, class ImageVal, class ImageMarker, class SE,
            class ImageOut>
  RES_T GeoCuts_MinSurfaces_with_steps(
      const ImageLabel &imLabel, const ImageVal &imVal,
      const ImageMarker &imMarker, const SE &nl, F_SIMPLE step_x,
      F_SIMPLE step_y, F_SIMPLE step_z, ImageOut &imOut)
  {
    SMIL_ENTER_FUNCTION("(multi-valued version)");

    // std::cout << "Enter function t_GeoCuts_MinSurfaces_with_steps
    // (multi_valued version)" << std::endl;

    if (!imOut.isAllocated() || !imLabel.isAllocated() ||
        !imVal.isAllocated() || !imMarker.isAllocated()) {
      SMIL_REGISTER_ERROR("Image not allocated");
      return RES_NOT_ALLOCATED;
    }

    // common image iterator
    typename ImageLabel::const_iterator it, iend;
    morphee::selement::Neighborhood<SE, ImageLabel> neighb(imLabel, nl);
    typename morphee::selement::Neighborhood<SE, ImageLabel>::iterator nit,
        nend;
    offset_t o0;
    offset_t o1;

    // needed for max flow: capacit map, rev_capacity map, etc.
    typedef boost::adjacency_list_traits<boost::vecS, boost::vecS,
                                         boost::directedS>
        Traits;
    typedef boost::adjacency_list<
        boost::vecS, boost::vecS, boost::directedS,
        boost::property<boost::vertex_name_t, std::string>,
        boost::property<
            boost::edge_capacity_t, F_DOUBLE,
            boost::property<boost::edge_residual_capacity_t, F_DOUBLE,
                            boost::property<boost::edge_reverse_t,
                                            Traits::edge_descriptor>>>>
        Graph_d;

    // if we had computed the number of vertices before, we could directly
    // initialize g with it
    Graph_d g;

    boost::property_map<Graph_d, boost::edge_capacity_t>::type capacity =
        boost::get(boost::edge_capacity, g);
    boost::property_map<Graph_d, boost::edge_reverse_t>::type rev =
        get(boost::edge_reverse, g);
    boost::property_map<Graph_d, boost::edge_residual_capacity_t>::type
        residual_capacity = get(boost::edge_residual_capacity, g);

    bool in1;
    Graph_d::edge_descriptor e1, e2, e3, e4, e5;
    Graph_d::vertex_descriptor vSource, vSink;
    int numVert  = 0;
    int numEdges = 0;

    // Vertices creation
    // std::cout<<"build Region Adjacency Graph Vertices"<<std::endl;
    clock_t t1 = clock();
    int max, not_used;
    morphee::stats::t_measMinMax(imLabel, not_used, max);
    numVert = max;
    // std::cout<<"number of Vertices (without source and sink):
    // "<<numVert<<std::endl;
    // Warning : numVert+1 nodes created, but node 0 is not used (in order to
    // simplify correspondance between labels and nodes)
    for (int i = 0; i <= numVert; i++) {
      boost::add_vertex(g);
    }
    vSource = boost::add_vertex(g);
    vSink   = boost::add_vertex(g);

    clock_t tt_marker2 = 0, tt_marker3 = 0, tt_new_edge = 0, tt_old_edge = 0;
    clock_t t2 = clock();
    // std::cout << "Nodes creation time : " << F_DOUBLE(t2-t1) /
    // CLOCKS_PER_SEC << " seconds\n" ;

    // Edges creation
    // std::cout<<"Building Region Adjacency Graph Edges"<<std::endl;
    t1 = clock();
    for (it = imLabel.begin(), iend = imLabel.end(); it != iend; ++it) {
      o1                                         = it.getOffset();
      typename ImageLabel::value_type label      = imLabel.pixelFromOffset(o1),
                                      label_prec = 0;
      typename ImageMarker::value_type marker    = imMarker.pixelFromOffset(o1),
                                       marker_prec = 0;

      if (label > 0) {
        if (marker == 2 && marker_prec != marker &&
            label_prec != label) // add edge to Source
        {
          clock_t temps_marker2 = clock();
          boost::tie(e4, in1)   = boost::edge(vSource, label, g);
          if (in1 == 0) // if in1 == 0 : edge had not yet been added
          {
            boost::tie(e4, in1) = boost::add_edge(vSource, label, g);
            boost::tie(e3, in1) = boost::add_edge(label, vSource, g);
            capacity[e4]        = (std::numeric_limits<F_DOUBLE>::max)();
            capacity[e3]        = (std::numeric_limits<F_DOUBLE>::max)();
            rev[e4]             = e3;
            rev[e3]             = e4;
          }
          tt_marker2 += clock() - temps_marker2;
        } else if (marker == 3 && marker_prec != marker &&
                   label_prec != label) // add edge to Sink
        {
          clock_t temps_marker3 = clock();
          boost::tie(e3, in1)   = boost::edge(vSink, label, g);
          if (in1 == 0) // if in1 == 0 : edge had not yet been added
          {
            // std::cout<<"Add new edge marker 3"<<std::endl;
            boost::tie(e4, in1) = boost::add_edge(label, vSink, g);
            boost::tie(e3, in1) = boost::add_edge(vSink, label, g);
            capacity[e4]        = (std::numeric_limits<F_DOUBLE>::max)();
            capacity[e3]        = (std::numeric_limits<F_DOUBLE>::max)();
            rev[e4]             = e3;
            rev[e3]             = e4;
          }
          tt_marker3 += clock() - temps_marker3;
        }

        neighb.setCenter(o1);
        typename ImageVal::value_type val_o1 = imVal.pixelFromOffset(o1);
        typename ImageLabel::value_type label2_prec =
            label; // val de label2 precedente

        for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
          const offset_t o2 = nit.getOffset();

          if (o2 > o1) {
            typename ImageLabel::value_type label2 =
                imLabel.pixelFromOffset(o2);
            if (label != label2) {
              typename ImageVal::value_type val_o2 = imVal.pixelFromOffset(o2);
              // F_SIMPLE max_diff = 255;   // EDF TODO : use data traits
              F_SIMPLE diff = t_Distance_L1(val_o1, val_o2);
              // F_SIMPLE leak = std::sqrt(max_diff - diff);

              INT8 dx, dy, dz;
              dx = std::abs(it.getX() - nit.getX());
              dy = std::abs(it.getY() - nit.getY());
              dz = std::abs(it.getZ() - nit.getZ());

              // F_SIMPLE dist ;
              // dist = std::sqrt(std::pow(step_x*dx,2) +
              // std::pow(step_y*dy,2) + std::pow(step_z*dz,2));
              // if(dist == 0)
              // {
              // 	 std::cout <<
              //     "ERROR : Distance between pixels equal to zero! " <<
              //     " Setting it to 1.\n" ;
              //   dist = 1 ;
              // }

              F_SIMPLE surf = 1; // TODO : this only works with 4-connexity
                                 // (in 2d) or 6-connexity (in 3d)
              if (dx == 0)
                surf *= step_x;
              if (dy == 0)
                surf *= step_y;
              if (dz == 0)
                surf *= step_z;
              if (surf == 0) {
                // std::cout << "ERROR : Surface between pixels equal to zero!
                // Setting it to 1.\n" ;
                surf = 1;
              }

              // 	if (o2%1000 == 0)
              // 	{
              // 		std::cout << " surf : " << surf ;
              // 	}

              // 	F_SIMPLE grad = diff/dist ;
              //  F_SIMPLE weighted_surf = grad * surf ;

              // Cette fonction devrait être remplacée par une fonction
              // paramètre
              // F_DOUBLE cost = 10000.0/(1.0+std::pow(diff/dist,4));
              // F_DOUBLE cost = dist/(1+diff);
              // F_DOUBLE cost = leak * surf ;
              F_DOUBLE cost = surf / (1 + diff);
              // if (o2%1000 == 0)
              // {
              // 	 std::cout << " cost : " << cost << "\n";
              // }

              // std::cout <<  " dx: " << (double)dx <<
              //               " dy: " <<  (double)dy <<
              //               " dz: " <<  (double)dz <<
              //               " dist: " <<  (double)dist <<
              //               " surf: " <<  (double)surf <<
              //               " grad: " <<  (double)grad <<
              //               " w_s: " <<  (double)weighted_surf <<
              //               " cost: " <<  (double)cost << "\n";

              if (label2_prec == label2) // same label2 means same edge (thus,
                                         // keep e3 and e4)
              {
                capacity[e4] = capacity[e4] + cost;
                capacity[e3] = capacity[e3] + cost;
              } else {
                boost::tie(e5, in1) = boost::edge(label, label2, g);
                if (in1 == 0) {
                  clock_t temps_new_edge = clock();
                  // std::cout<<"Add new edge "<< label<<" --
                  // "<<label2<<std::endl;
                  numEdges++;
                  boost::tie(e4, in1) = boost::add_edge(label, label2, g);
                  boost::tie(e3, in1) = boost::add_edge(label2, label, g);
                  capacity[e4]        = cost;
                  capacity[e3]        = cost;
                  rev[e4]             = e3;
                  rev[e3]             = e4;
                  tt_new_edge += clock() - temps_new_edge;

                } else {
                  clock_t temps_old_edge = clock();
                  // std::cout<<"existing edge"<<std::endl;
                  boost::tie(e4, in1) = boost::edge(label, label2, g);
                  boost::tie(e3, in1) = boost::edge(label2, label, g);
                  capacity[e4]        = capacity[e4] + cost;
                  capacity[e3]        = capacity[e3] + cost;
                  tt_old_edge += clock() - temps_old_edge;
                }
                label2_prec = label2;
              }
            }
          }
        }
        label_prec  = label;
        marker_prec = marker;
      }
    }

    t2 = clock();
    // std::cout << "Number of initial edges : " << numEdges << std::endl;
    //     std::cout << "Edges creation time : " << F_DOUBLE(t2-t1) /
    //     CLOCKS_PER_SEC << " seconds\n" ; std::cout << "Marker2   : " <<
    //     F_DOUBLE(tt_marker2)  / CLOCKS_PER_SEC << " seconds\n"; std::cout
    //     << "Marker3   : " << F_DOUBLE(tt_marker3)  / CLOCKS_PER_SEC << "
    //     seconds\n"; std::cout << "New edges : " << F_DOUBLE(tt_new_edge) /
    //     CLOCKS_PER_SEC << " seconds\n"; std::cout << "Old edges : " <<
    //     F_DOUBLE(tt_old_edge) / CLOCKS_PER_SEC << " seconds\n";

    // We should test that the same region node is not connected
    // simultaneously to source and sink :
    // * iterate on sink neighbors ;
    // * if neigbhbor is linked to source, remove edges to source and sink (in
    // fact, set its capacity to 0, to avoid the modification of the graph
    // structure)
    //
    t1 = clock();
    Graph_d::vertex_descriptor sink_neighbour;
    typename boost::graph_traits<Graph_d>::adjacency_iterator ai, ai_end;
    UINT32 rem_edges = 0;
    for (boost::tie(ai, ai_end) = adjacent_vertices(vSink, g); ai != ai_end;
         ++ai) {
      sink_neighbour = *ai;
      tie(e1, in1)   = edge(sink_neighbour, vSource, g);
      if (in1) {
        // remove_edge(vSource, sink_neighbour, g);
        // remove_edge(sink_neighbour, vSource, g);
        capacity[e1]      = 0;
        capacity[rev[e1]] = 0;
        rem_edges++;
        tie(e2, in1)      = edge(vSink, sink_neighbour, g);
        capacity[e2]      = 0;
        capacity[rev[e2]] = 0;
        rem_edges++;
      }
    }
    t2 = clock();
    // std::cout << "Graph post-processing : Removal of " << rem_edges << "
    // edges in  : " << F_DOUBLE(t2-t1) / CLOCKS_PER_SEC << " seconds\n" ;

    // Prepare to run the max-flow algorithm
    boost::property_map<Graph_d, boost::vertex_index_t>::type indexmap =
        boost::get(boost::vertex_index, g);
    std::vector<boost::default_color_type> color(boost::num_vertices(g));
    // std::cout << "Compute Max flow" << std::endl;
    t1 = clock();

#if BOOST_VERSION >= 104700
    F_DOUBLE flow =
        boykov_kolmogorov_max_flow(g, capacity, residual_capacity, rev,
                                   &color[0], indexmap, vSource, vSink);
#else
    F_DOUBLE flow = kolmogorov_max_flow(g, capacity, residual_capacity, rev,
                                        &color[0], indexmap, vSource, vSink);
#endif
    // std::cout << "c  The total flow:" << std::endl;
    // std::cout << "s " << flow << std::endl;
    t2 = clock();
    // std::cout << "Flow computation time : " << F_DOUBLE(t2-t1) /
    // CLOCKS_PER_SEC << " seconds\n" ;

    t1 = clock();
    for (it = imLabel.begin(), iend = imLabel.end(); it != iend; ++it) {
      o1                                    = it.getOffset();
      typename ImageLabel::value_type label = imLabel.pixelFromOffset(o1);

      if (label == 0) {
        imOut.setPixel(o1, 0);
      } else // if color[label] == 0 : source node ; else, sink node (accord
             // to boost graph doc)
      {
        if (color[label] == color[vSource])
          imOut.setPixel(o1, 2);
        else
          imOut.setPixel(o1, 3);
      }
    }
    t2 = clock();
    // std::cout << "Computing imOut took : " << F_DOUBLE(t2-t1) /
    // CLOCKS_PER_SEC << " seconds\n" ;

    // std::cout << "\n";

    return RES_OK;
  }

  /*
   *
   *
   *
   *
   */
  // ImageLabel and ImageMarker should be unsigned integers
  template <class ImageLabel, class ImageVal, class ImageMarker, class SE,
            class ImageOut>
  RES_T GeoCuts_MinSurfaces_with_steps_vGradient(
      const ImageLabel &imLabel, const ImageVal &imVal,
      const ImageMarker &imMarker, const SE &nl, F_SIMPLE step_x,
      F_SIMPLE step_y, F_SIMPLE step_z, ImageOut &imOut)
  {
    SMIL_ENTER_FUNCTION("(multi-valued version)");

    // std::cout << "Enter function t_GeoCuts_MinSurfaces_with_steps
    // (multi_valued version)" << std::endl;

    if (!imOut.isAllocated() || !imLabel.isAllocated() ||
        !imVal.isAllocated() || !imMarker.isAllocated()) {
      SMIL_REGISTER_ERROR("Image not allocated");
      return RES_NOT_ALLOCATED;
    }

    // common image iterator
    typename ImageLabel::const_iterator it, iend;
    morphee::selement::Neighborhood<SE, ImageLabel> neighb(imLabel, nl);
    typename morphee::selement::Neighborhood<SE, ImageLabel>::iterator nit,
        nend;
    offset_t o0;
    offset_t o1;

    // needed for max flow: capacit map, rev_capacity map, etc.
    typedef boost::adjacency_list_traits<boost::vecS, boost::vecS,
                                         boost::directedS>
        Traits;
    typedef boost::adjacency_list<
        boost::vecS, boost::vecS, boost::directedS,
        boost::property<boost::vertex_name_t, std::string>,
        boost::property<
            boost::edge_capacity_t, F_DOUBLE,
            boost::property<boost::edge_residual_capacity_t, F_DOUBLE,
                            boost::property<boost::edge_reverse_t,
                                            Traits::edge_descriptor>>>>
        Graph_d;

    // if we had computed the number of vertices before, we could directly
    // initialize g with it
    Graph_d g;

    boost::property_map<Graph_d, boost::edge_capacity_t>::type capacity =
        boost::get(boost::edge_capacity, g);
    boost::property_map<Graph_d, boost::edge_reverse_t>::type rev =
        get(boost::edge_reverse, g);
    boost::property_map<Graph_d, boost::edge_residual_capacity_t>::type
        residual_capacity = get(boost::edge_residual_capacity, g);

    bool in1;
    Graph_d::edge_descriptor e1, e2, e3, e4, e5;
    Graph_d::vertex_descriptor vSource, vSink;
    int numVert  = 0;
    int numEdges = 0;

    // Vertices creation
    // std::cout<<"build Region Adjacency Graph Vertices"<<std::endl;
    clock_t t1 = clock();
    int max, not_used;
    morphee::stats::t_measMinMax(imLabel, not_used, max);
    numVert = max;
    // std::cout<<"number of Vertices (without source and sink):
    // "<<numVert<<std::endl;
    // Warning : numVert+1 nodes created, but node 0 is not used (in order to
    // simplify correspondance between labels and nodes)
    for (int i = 0; i <= numVert; i++) {
      boost::add_vertex(g);
    }
    vSource = boost::add_vertex(g);
    vSink   = boost::add_vertex(g);

    clock_t tt_marker2 = 0, tt_marker3 = 0, tt_new_edge = 0, tt_old_edge = 0;
    clock_t t2 = clock();
    // std::cout << "Nodes creation time : " << F_DOUBLE(t2-t1) /
    // CLOCKS_PER_SEC << " seconds\n" ;

    // Edges creation
    // std::cout<<"Building Region Adjacency Graph Edges"<<std::endl;
    t1 = clock();
    for (it = imLabel.begin(), iend = imLabel.end(); it != iend; ++it) {
      o1                                         = it.getOffset();
      typename ImageLabel::value_type label      = imLabel.pixelFromOffset(o1),
                                      label_prec = 0;
      typename ImageMarker::value_type marker    = imMarker.pixelFromOffset(o1),
                                       marker_prec = 0;

      if (label > 0) {
        if (marker == 2 && marker_prec != marker &&
            label_prec != label) // add edge to Source
        {
          clock_t temps_marker2 = clock();
          boost::tie(e4, in1)   = boost::edge(vSource, label, g);
          if (in1 == 0) // if in1 == 0 : edge had not yet been added
          {
            boost::tie(e4, in1) = boost::add_edge(vSource, label, g);
            boost::tie(e3, in1) = boost::add_edge(label, vSource, g);
            capacity[e4]        = (std::numeric_limits<F_DOUBLE>::max)();
            capacity[e3]        = (std::numeric_limits<F_DOUBLE>::max)();
            rev[e4]             = e3;
            rev[e3]             = e4;
          }
          tt_marker2 += clock() - temps_marker2;
        } else if (marker == 3 && marker_prec != marker &&
                   label_prec != label) // add edge to Sink
        {
          clock_t temps_marker3 = clock();
          boost::tie(e3, in1)   = boost::edge(vSink, label, g);
          if (in1 == 0) // if in1 == 0 : edge had not yet been added
          {
            // std::cout<<"Add new edge marker 3"<<std::endl;
            boost::tie(e4, in1) = boost::add_edge(label, vSink, g);
            boost::tie(e3, in1) = boost::add_edge(vSink, label, g);
            capacity[e4]        = (std::numeric_limits<F_DOUBLE>::max)();
            capacity[e3]        = (std::numeric_limits<F_DOUBLE>::max)();
            rev[e4]             = e3;
            rev[e3]             = e4;
          }
          tt_marker3 += clock() - temps_marker3;
        }

        neighb.setCenter(o1);
        typename ImageVal::value_type val_o1 = imVal.pixelFromOffset(o1);
        typename ImageLabel::value_type label2_prec =
            label; // val de label2 precedente

        for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
          const offset_t o2 = nit.getOffset();

          if (o2 > o1) {
            typename ImageLabel::value_type label2 =
                imLabel.pixelFromOffset(o2);
            if (label != label2) {
              typename ImageVal::value_type val_o2 = imVal.pixelFromOffset(o2);
              F_SIMPLE diff                        = std::max(val_o1, val_o2);

              INT8 dx, dy, dz;
              dx = std::abs(it.getX() - nit.getX());
              dy = std::abs(it.getY() - nit.getY());
              dz = std::abs(it.getZ() - nit.getZ());

              // 		      F_SIMPLE dist ;
              // 		      dist = std::sqrt(std::pow(step_x*dx,2) +
              // std::pow(step_y*dy,2) + std::pow(step_z*dz,2)); if(dist == 0)
              // 			{
              // 			  std::cout << "ERROR : Distance between pixels equal to
              // zero! Setting it to 1.\n" ; 			  dist = 1 ;
              // 			}

              F_SIMPLE surf = 1; // TODO : this only works with 4-connexity
                                 // (in 2d) or 6-connexity (in 3d)
              if (dx == 0)
                surf *= step_x;
              if (dy == 0)
                surf *= step_y;
              if (dz == 0)
                surf *= step_z;
              if (surf == 0) {
                // std::cout << "ERROR : Surface between pixels equal to zero!
                // Setting it to 1.\n" ;
                surf = 1;
              }

              // 		      if (o2%1000 == 0)
              // 			{
              // 			  std::cout << " surf : " << surf ;
              // 			}

              // 		      F_SIMPLE grad = diff/dist ;
              // 		      F_SIMPLE weighted_surf = grad * surf ;

              // Cette fonction devrait être remplacée par une fonction
              // paramètre
              // F_DOUBLE cost = 10000.0/(1.0+std::pow(diff/dist,4));
              // F_DOUBLE cost = dist/(1+diff);
              // F_DOUBLE cost = leak * surf ;
              F_DOUBLE cost = surf / (1 + diff);
              // 		      if (o2%1000 == 0)
              // 			{
              // 			  std::cout << " cost : " << cost << "\n";
              // 			}

              // std::cout <<  "dx: " << (double)dx << " dy: " <<  (double)dy
              // << " dz: " <<  (double)dz << " dist: " <<  (double)dist << "
              // surf: " <<  (double)surf << " grad: " <<  (double)grad << "
              // w_s: " <<  (double)weighted_surf << " cost: " << (double)cost
              // << "\n";

              if (label2_prec == label2) // same label2 means same edge (thus,
                                         // keep e3 and e4)
              {
                capacity[e4] = capacity[e4] + cost;
                capacity[e3] = capacity[e3] + cost;
              } else {
                boost::tie(e5, in1) = boost::edge(label, label2, g);
                if (in1 == 0) {
                  clock_t temps_new_edge = clock();
                  // std::cout<<"Add new edge "<< label<<" --
                  // "<<label2<<std::endl;
                  numEdges++;
                  boost::tie(e4, in1) = boost::add_edge(label, label2, g);
                  boost::tie(e3, in1) = boost::add_edge(label2, label, g);
                  capacity[e4]        = cost;
                  capacity[e3]        = cost;
                  rev[e4]             = e3;
                  rev[e3]             = e4;
                  tt_new_edge += clock() - temps_new_edge;

                } else {
                  clock_t temps_old_edge = clock();
                  // std::cout<<"existing edge"<<std::endl;
                  boost::tie(e4, in1) = boost::edge(label, label2, g);
                  boost::tie(e3, in1) = boost::edge(label2, label, g);
                  capacity[e4]        = capacity[e4] + cost;
                  capacity[e3]        = capacity[e3] + cost;
                  tt_old_edge += clock() - temps_old_edge;
                }
                label2_prec = label2;
              }
            }
          }
        }
        label_prec  = label;
        marker_prec = marker;
      }
    }

    t2 = clock();
    // std::cout << "Number of initial edges : " << numEdges << std::endl;
    //     std::cout << "Edges creation time : " << F_DOUBLE(t2-t1) /
    //     CLOCKS_PER_SEC << " seconds\n" ; std::cout << "Marker2   : " <<
    //     F_DOUBLE(tt_marker2)  / CLOCKS_PER_SEC << " seconds\n"; std::cout
    //     << "Marker3   : " << F_DOUBLE(tt_marker3)  / CLOCKS_PER_SEC << "
    //     seconds\n"; std::cout << "New edges : " << F_DOUBLE(tt_new_edge) /
    //     CLOCKS_PER_SEC << " seconds\n"; std::cout << "Old edges : " <<
    //     F_DOUBLE(tt_old_edge) / CLOCKS_PER_SEC << " seconds\n";

    // We should test that the same region node is not connected
    // simultaneously to source and sink :
    // * iterate on sink neighbors ;
    // * if neigbhbor is linked to source, remove edges to source and sink (in
    // fact, set its capacity to 0, to avoid the modification of the graph
    // structure)
    //
    t1 = clock();
    Graph_d::vertex_descriptor sink_neighbour;
    typename boost::graph_traits<Graph_d>::adjacency_iterator ai, ai_end;
    UINT32 rem_edges = 0;
    for (boost::tie(ai, ai_end) = adjacent_vertices(vSink, g); ai != ai_end;
         ++ai) {
      sink_neighbour = *ai;
      tie(e1, in1)   = edge(sink_neighbour, vSource, g);
      if (in1) {
        // remove_edge(vSource, sink_neighbour, g);
        // remove_edge(sink_neighbour, vSource, g);
        capacity[e1]      = 0;
        capacity[rev[e1]] = 0;
        rem_edges++;
        tie(e2, in1)      = edge(vSink, sink_neighbour, g);
        capacity[e2]      = 0;
        capacity[rev[e2]] = 0;
        rem_edges++;
      }
    }
    t2 = clock();
    // std::cout << "Graph post-processing : Removal of " << rem_edges << "
    // edges in  : " << F_DOUBLE(t2-t1) / CLOCKS_PER_SEC << " seconds\n" ;

    // Prepare to run the max-flow algorithm
    boost::property_map<Graph_d, boost::vertex_index_t>::type indexmap =
        boost::get(boost::vertex_index, g);
    std::vector<boost::default_color_type> color(boost::num_vertices(g));
    // std::cout << "Compute Max flow" << std::endl;
    t1 = clock();
#if BOOST_VERSION >= 104700
    F_DOUBLE flow =
        boykov_kolmogorov_max_flow(g, capacity, residual_capacity, rev,
                                   &color[0], indexmap, vSource, vSink);
#else
    F_DOUBLE flow = kolmogorov_max_flow(g, capacity, residual_capacity, rev,
                                        &color[0], indexmap, vSource, vSink);
#endif
    // std::cout << "c  The total flow:" << std::endl;
    // std::cout << "s " << flow << std::endl;
    t2 = clock();
    // std::cout << "Flow computation time : " << F_DOUBLE(t2-t1) /
    // CLOCKS_PER_SEC << " seconds\n" ;

    t1 = clock();
    for (it = imLabel.begin(), iend = imLabel.end(); it != iend; ++it) {
      o1                                    = it.getOffset();
      typename ImageLabel::value_type label = imLabel.pixelFromOffset(o1);

      if (label == 0) {
        imOut.setPixel(o1, 0);
      } else // if color[label] == 0 : source node ; else, sink node (accord
             // to boost graph doc)
      {
        if (color[label] == color[vSource])
          imOut.setPixel(o1, 2);
        else
          imOut.setPixel(o1, 3);
      }
    }
    t2 = clock();
    // std::cout << "Computing imOut took : " << F_DOUBLE(t2-t1) /
    // CLOCKS_PER_SEC << " seconds\n" ;

    // std::cout << "\n";

    return RES_OK;
  }

  /*
   *
   *
   *
   *
   */
  template <class ImageIn, class ImageGrad, class ImageMarker, class SE,
            class ImageOut>
  RES_T GeoCuts_MinSurfaces_with_steps_old(
      const ImageIn &imIn, const ImageGrad &imGrad, const ImageMarker &imMarker,
      const SE &nl, F_SIMPLE step_x, F_SIMPLE step_y, F_SIMPLE step_z,
      ImageOut &imOut)
  {
    SMIL_ENTER_FUNCTION("");

    std::cout << "Enter function t_GeoCuts_MinSurfaces" << std::endl;

    if ((!imOut.isAllocated())) {
      SMIL_REGISTER_ERROR("Not allocated");
      return RES_NOT_ALLOCATED;
    }

    if ((!imIn.isAllocated())) {
      SMIL_REGISTER_ERROR("Not allocated");
      return RES_NOT_ALLOCATED;
    }

    if ((!imGrad.isAllocated())) {
      SMIL_REGISTER_ERROR("Not allocated");
      return RES_NOT_ALLOCATED;
    }

    if ((!imMarker.isAllocated())) {
      SMIL_REGISTER_ERROR("Not allocated");
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
        boost::vecS, boost::vecS, boost::directedS,
        boost::property<boost::vertex_name_t, std::string>,
        boost::property<
            boost::edge_capacity_t, double,
            boost::property<boost::edge_residual_capacity_t, double,
                            boost::property<boost::edge_reverse_t,
                                            Traits::edge_descriptor>>>>
        Graph_d;

    Graph_d g;

    boost::property_map<Graph_d, boost::edge_capacity_t>::type capacity =
        boost::get(boost::edge_capacity, g);

    boost::property_map<Graph_d, boost::edge_reverse_t>::type rev =
        get(boost::edge_reverse, g);

    boost::property_map<Graph_d, boost::edge_residual_capacity_t>::type
        residual_capacity = get(boost::edge_residual_capacity, g);

    bool in1;
    Graph_d::edge_descriptor e1, e2, e3, e4, e5;
    Graph_d::vertex_descriptor vSource, vSink;
    int numVert  = 0;
    int numEdges = 0;
    int max, not_used;

    std::cout << "build Region Adjacency Graph" << std::endl;

    std::cout << "build Region Adjacency Graph Vertices" << std::endl;

    clock_t t1 = clock();

    morphee::stats::t_measMinMax(imIn, not_used, max);
    numVert = max;
    std::cout << "number of Vertices : " << numVert << std::endl;

    for (int i = 1; i <= numVert; i++) {
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

    for (it = imIn.begin(), iend = imIn.end(); it != iend; ++it) {
      o1 = it.getOffset();
      // pas de raison que ce soit int ; il faut rendre ceci générique
      int val      = imIn.pixelFromOffset(o1);
      int marker   = imMarker.pixelFromOffset(o1);
      int val_prec = 0, marker_prec = 0;

      if (val > 0) {
        if (marker == 2 && marker_prec != marker && val_prec != val) {
          clock_t temps_marker2 = clock();
          boost::tie(e4, in1)   = boost::edge(vSource, val, g);

          if (in1 == 0) {
            // std::cout<<"Add new edge marker 2"<<std::endl;
            boost::tie(e4, in1) = boost::add_edge(vSource, val, g);
            boost::tie(e3, in1) = boost::add_edge(val, vSource, g);
            capacity[e4]        = (std::numeric_limits<double>::max)();
            capacity[e3]        = (std::numeric_limits<double>::max)();
            rev[e4]             = e3;
            rev[e3]             = e4;
          }
          tt_marker2 += clock() - temps_marker2;
        } else if (marker == 3 && marker_prec != marker && val_prec != val) {
          clock_t temps_marker3 = clock();
          boost::tie(e3, in1)   = boost::edge(vSink, val, g);
          if (in1 == 0) {
            // std::cout<<"Add new edge marker 3"<<std::endl;
            boost::tie(e4, in1) = boost::add_edge(val, vSink, g);
            boost::tie(e3, in1) = boost::add_edge(vSink, val, g);
            capacity[e4]        = (std::numeric_limits<double>::max)();
            capacity[e3]        = (std::numeric_limits<double>::max)();
            rev[e4]             = e3;
            rev[e3]             = e4;
          }
          tt_marker3 += clock() - temps_marker3;
        }

        neighb.setCenter(o1);
        // Enlever double et int ; prendre types génériques
        double val_grad_o1 = imGrad.pixelFromOffset(o1);
        // typename mageGrad::value_type val_grad_o1 =
        // imGrad.pixelFromOffset(o1);
        int val2_prec = val; // val de val2 precedente

        for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
          const offset_t o2 = nit.getOffset();

          if (o2 > o1) {
            // enlever int ; prendre type générique
            int val2 = imIn.pixelFromOffset(o2);
            if (val != val2) {
              // enlever double ; prendre type générique
              double val_grad_o2 = imGrad.pixelFromOffset(o2);
              // prendre la distance L1, remplacer double et F_SIMPLE par
              // valeur générique
              double diff = t_Distance_L1(val_grad_o1, val_grad_o2);
              // double diff = std::abs(val_grad_o1 - val_grad_o2);

              F_SIMPLE dist =
                  std::sqrt(std::pow(step_x * (it.getX() - nit.getX()), 2) +
                            std::pow(step_y * (it.getY() - nit.getY()), 2) +
                            std::pow(step_z * (it.getZ() - nit.getZ()), 2));

              // Cette fonction devrait être remplacée par une fonction
              // paramètre
              // double cost = 10000.0/(1.0+std::pow(diff/dist,4));
              double cost = dist / (1 + diff);

              if (val2_prec ==
                  val2) // same val2 means same edge (thus, keep e3 and e4)
              {
                capacity[e4] = capacity[e4] + cost;
                capacity[e3] = capacity[e3] + cost;
              } else {
                boost::tie(e5, in1) = boost::edge(val, val2, g);

                if (in1 == 0) {
                  clock_t temps_new_edge = clock();
                  // std::cout<<"Add new edge "<< val<<" --
                  // "<<val2<<std::endl;
                  numEdges++;
                  boost::tie(e4, in1) = boost::add_edge(val, val2, g);
                  boost::tie(e3, in1) = boost::add_edge(val2, val, g);
                  capacity[e4]        = cost;
                  capacity[e3]        = cost;
                  rev[e4]             = e3;
                  rev[e3]             = e4;
                  tt_new_edge += clock() - temps_new_edge;

                } else {
                  clock_t temps_old_edge = clock();
                  // std::cout<<"existing edge"<<std::endl;
                  boost::tie(e4, in1) = boost::edge(val, val2, g);
                  boost::tie(e3, in1) = boost::edge(val2, val, g);
                  capacity[e4]        = capacity[e4] + cost;
                  capacity[e3]        = capacity[e3] + cost;
                  tt_old_edge += clock() - temps_old_edge;
                }
                val2_prec = val2;
              }
            }
          }
        }
        val_prec    = val;
        marker_prec = marker;
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

    boost::property_map<Graph_d, boost::vertex_index_t>::type indexmap =
        boost::get(boost::vertex_index, g);
    std::vector<boost::default_color_type> color(boost::num_vertices(g));

    std::cout << "Compute Max flow" << std::endl;
    t1 = clock();
#if BOOST_VERSION >= 104700
    double flow =
        boykov_kolmogorov_max_flow(g, capacity, residual_capacity, rev,
                                   &color[0], indexmap, vSource, vSink);
#else
    double flow   = kolmogorov_max_flow(g, capacity, residual_capacity, rev,
                                      &color[0], indexmap, vSource, vSink);
#endif
    std::cout << "c  The total flow:" << std::endl;
    std::cout << "s " << flow << std::endl;
    t2 = clock();
    std::cout << "Flow computation time : " << double(t2 - t1) / CLOCKS_PER_SEC
              << " seconds\n";

    t1       = clock();
    int miss = 0;
    for (it = imIn.begin(), iend = imIn.end(); it != iend; ++it) {
      o1 = it.getOffset();
      // enlever int ; prendre type générique
      int val = imIn.pixelFromOffset(o1);

      if (val == 0) {
        imOut.setPixel(o1, 0);
      } else {
        if (color[val] == color[vSource])
          imOut.setPixel(o1, 2);
        else if (color[val] == color[vSink])
          imOut.setPixel(o1, 3);
        else {
          imOut.setPixel(o1, 20);
          miss++;
        }
      }
    }
    t2 = clock();
    std::cout << "Computing imOut took : " << double(t2 - t1) / CLOCKS_PER_SEC
              << " seconds\n";

    if (miss > 0)
      std::cout << "WARNING : Missclassified nodes : " << miss << "\n";

    return RES_OK;
  }
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

    offset_t o0, o1;

    typename Image<T>::lineType bufIn     = imIn.getPixels();
    typename Image<T>::lineType bufOut    = imOut.getPixels();
    typename Image<T>::lineType bufMarker = imMarker.getPixels();
    typename Image<T>::lineType bufGrad   = imGrad.getPixels();

    std::cout << "build Region Adjacency Graph" << std::endl;

    Graph_T g;
    EdgeCap_T capacity = boost::get(boost::edge_capacity, g);
    EdgeRevCap_T rev   = boost::get(boost::edge_reverse, g);
    EdgeResCap_T residual_capacity =
        boost::get(boost::edge_residual_capacity, g);

    bool in1;
    Graph_T::edge_descriptor e1, e2, e3, e4;
    Graph_T::vertex_descriptor vSource, vSink;
    int numVert   = 0;
    int numLabels = 0;

    int pixelCount = imIn.getPixelCount();
    for (o0 = 0; o0 < pixelCount; o0++) {
      int val  = (int) bufIn[o0];
      int val2 = (int) bufMarker[o0];

      bufOut[o0] = 1;
      if (val2 > numLabels) {
        numLabels = val2;
      }

      if (val > numVert) {
        numVert = val;
      }
    }

    std::cout << "number of labels :" << numLabels << std::endl;

    std::cout << "build Region Adjacency Graph Vertices" << std::endl;

    for (int i = 0; i <= numVert; i++) {
      boost::add_vertex(g);
    }

    vSource = boost::add_vertex(g);
    vSink   = boost::add_vertex(g);

    std::vector<boost::default_color_type> color(boost::num_vertices(g));
    VertexIndex_T indexmap = boost::get(boost::vertex_index, g);

    std::cout << "build Region Adjacency Graph Edges" << std::endl;

    vector<IntPoint> pts = filterStrElt(nl);
    vector<IntPoint>::iterator itBegin, itEnd;
    itBegin = pts.begin();
    itEnd   = pts.end();

    int width  = imIn.getWidth();
    int height = imIn.getHeight();
    int depth  = imIn.getDepth();

    int strideY = width;
    int strideZ = width * height;
    for (int z = 0; z < depth; z++) {
      off_t p0 = z * strideZ;
      for (int y = 0; y < height; y++) {
        p0 += y * strideY;
        for (int x = 0; x < width; x++) {
          int val    = (int) bufIn[o1];
          // int marker = (int) bufMarker[o1]; // XXX unused ???

          o1 = p0 + x;

          vector<IntPoint>::iterator it;
          for (it = itBegin; it != itEnd; it++) {
            if (x + it->x > width - 1 || x + it->x < 0)
              continue;
            if (y + it->y > height - 1 || y + it->y < 0)
              continue;
            if (z + it->z > depth - 1 || z + it->z < 0)
              continue;
            offset_t o2 = o1 + it->z *strideZ + it->y *strideY + it->x;
            if (o2 <= o1)
              continue;

            int val2 = bufIn[o2];
            if (val == val2)
              continue;

            boost::tie(e3, in1) = boost::edge(val, val2, g);
            // std::cout<<in1<<std::endl;
            // std::cout<<"Compute Gradient"<<std::endl;
            double val3 = (double) bufGrad[o1];
            double val4 = (double) bufGrad[o2];
            double maxi = std::max(val3, val4);
            double cost = 10000.0 / (1.0 + std::pow(maxi, 4));

            if (in1 == 0) {
              // std::cout<<"Add new edge"<<std::endl;
              boost::tie(e4, in1) = boost::add_edge(val, val2, g);
              boost::tie(e3, in1) = boost::add_edge(val2, val, g);
              capacity[e4]        = cost;
              capacity[e3]        = cost;
              rev[e4]             = e3;
              rev[e3]             = e4;
            } else {
              // std::cout<<"existing edge"<<std::endl;
              boost::tie(e4, in1) = boost::edge(val, val2, g);
              boost::tie(e3, in1) = boost::edge(val2, val, g);
              capacity[e4]        = capacity[e4] + cost;
              capacity[e3]        = capacity[e3] + cost;
            }
          }
        }
      }
    }

    for (int nbk = 2; nbk <= numLabels; nbk++) {
      // for all pixels in imIn create a vertex
      for (int o0 = 0; o0 < pixelCount; o0++) {
        int val  = (int) bufMarker[o0];
        int val2 = (int) bufIn[o0];

        if (val == nbk) {
          boost::tie(e4, in1) = boost::edge(vSource, val2, g);
          if (in1 == 0) {
            boost::tie(e4, in1) = boost::add_edge(vSource, val2, g);
            boost::tie(e3, in1) = boost::add_edge(val2, vSource, g);
            capacity[e4]        = (std::numeric_limits<double>::max)();
            capacity[e3]        = (std::numeric_limits<double>::max)();
            rev[e4]             = e3;
            rev[e3]             = e4;
          }
        } else if (val > 1 && val != nbk) {
          boost::tie(e4, in1) = boost::edge(val2, vSink, g);
          if (in1 == 0) {
            boost::tie(e4, in1) = boost::add_edge(val2, vSink, g);
            boost::tie(e3, in1) = boost::add_edge(vSink, val2, g);
            capacity[e4]        = (std::numeric_limits<double>::max)();
            capacity[e3]        = (std::numeric_limits<double>::max)();
            rev[e4]             = e3;
            rev[e3]             = e4;
          }
        }
      }

      std::cout << "Compute Max flow" << nbk << std::endl;
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
      for (int o1 = 0; o1 < pixelCount; o1++) {
        int val  = (int) bufIn[o1];
        int val2 = (int) bufOut[o1];
        int val3 = (int) bufMarker[o1];

        if (val2 == 1) {
          if (color[val] == color[vSource])
            bufOut[o1] = (T) nbk;
        }

        if (val3 == nbk) {
          boost::tie(e4, in1) = boost::edge(vSource, val, g);
          if (in1 == 1) {
            boost::remove_edge(vSource, val, g);
            boost::remove_edge(val, vSource, g);
          }
        } else if (val3 > 1 && val3 != nbk) {
          boost::tie(e4, in1) = boost::edge(val, vSink, g);
          if (in1 == 1) {
            boost::remove_edge(val, vSink, g);
            boost::remove_edge(vSink, val, g);
          }
        }
      }
    }

    return RES_OK;
  }

#if 0
  /*
   *
   *
   *
   *
   */
  template <class ImageIn, class ImageGrad, class ImageMosaic,
            class ImageMarker, class SE, class ImageOut>
  RES_T GeoCuts_Optimize_Mosaic(
      const ImageIn &imIn, const ImageGrad &imGrad, const ImageMosaic &imMosaic,
      const ImageMarker &imMarker, const SE &nl, ImageOut &imOut)
  {
    SMIL_ENTER_FUNCTION("");

    std::cout << "Enter function t_GeoCuts_Optimize_Mosaic" << std::endl;

    if ((!imOut.isAllocated())) {
      SMIL_REGISTER_ERROR("Not allocated");
      return RES_NOT_ALLOCATED;
    }

    if ((!imIn.isAllocated())) {
      SMIL_REGISTER_ERROR("Not allocated");
      return RES_NOT_ALLOCATED;
    }

    if ((!imGrad.isAllocated())) {
      SMIL_REGISTER_ERROR("Not allocated");
      return RES_NOT_ALLOCATED;
    }

    if ((!imMosaic.isAllocated())) {
      SMIL_REGISTER_ERROR("Not allocated");
      return RES_NOT_ALLOCATED;
    }

    if ((!imMarker.isAllocated())) {
      SMIL_REGISTER_ERROR("Not allocated");
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

    Graph_d g;

    double sigma = 1.0;

    boost::property_map<Graph_d, boost::edge_capacity_t>::type capacity =
        boost::get(boost::edge_capacity, g);

    boost::property_map<Graph_d, boost::edge_reverse_t>::type rev =
        get(boost::edge_reverse, g);

    boost::property_map<Graph_d, boost::edge_residual_capacity_t>::type
        residual_capacity = get(boost::edge_residual_capacity, g);

    bool in1;
    Graph_d::edge_descriptor e1, e2, e3, e4;
    Graph_d::vertex_descriptor vSource, vSink;
    int numVert   = 0;
    int numLabels = 0;

    std::cout << "build Region Adjacency Graph" << std::endl;

    for (it = imIn.begin(), iend = imIn.end(); it != iend; ++it) {
      // for all pixels in imIn create a vertex
      o0       = it.getOffset();
      int val  = imIn.pixelFromOffset(o0);
      int val2 = imMarker.pixelFromOffset(o0);

      imOut.setPixel(o0, 1);

      if (val2 > numLabels) {
        numLabels = val2;
      }

      if (val > numVert) {
        numVert = val;
      }
    }
    std::cout << "numlabels = " << numLabels << std::endl;

    std::cout << "build Region Adjacency Graph Vertices" << std::endl;

    for (int i = 0; i <= numVert; i++) {
      boost::add_vertex(g);
    }

    vSource = boost::add_vertex(g);
    vSink   = boost::add_vertex(g);

    std::vector<boost::default_color_type> color(boost::num_vertices(g));
    boost::property_map<Graph_d, boost::vertex_index_t>::type indexmap =
        boost::get(boost::vertex_index, g);

    std::cout << "build Region Adjacency Graph Edges" << std::endl;

    for (it = imIn.begin(), iend = imIn.end(); it != iend; ++it) {
      // for all pixels in imIn create a vertex and an edge
      o1         = it.getOffset();
      int val    = imIn.pixelFromOffset(o1);
      int marker = imMarker.pixelFromOffset(o1);
      neighb.setCenter(o1);

      for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
        const offset_t o2 = nit.getOffset();
        int val2          = imIn.pixelFromOffset(o2);

        if (o2 > o1) {
          if (val != val2) {
            boost::tie(e3, in1) = boost::edge(val, val2, g);
            // std::cout<<in1<<std::endl;
            // std::cout<<"Compute Gradient"<<std::endl;
            double val3 = imGrad.pixelFromOffset(o1);
            double val4 = imGrad.pixelFromOffset(o2);
            double maxi = std::max(val3, val4);
            double cost = 1000 / (1 + 0.1 * (maxi) * (maxi));

            if (in1 == 0) {
              // std::cout<<"Add new edge"<<std::endl;
              boost::tie(e4, in1) = boost::add_edge(val, val2, g);
              boost::tie(e3, in1) = boost::add_edge(val2, val, g);
              capacity[e4]        = cost;
              capacity[e3]        = cost;
              rev[e4]             = e3;
              rev[e3]             = e4;
            } else {
              // std::cout<<"existing edge"<<std::endl;
              boost::tie(e4, in1) = boost::edge(val, val2, g);
              boost::tie(e3, in1) = boost::edge(val2, val, g);
              capacity[e4]        = capacity[e4] + cost;
              capacity[e3]        = capacity[e3] + cost;
            }
          }
        }
      }
    }

    for (int label1 = 1; label1 < numLabels; label1++) {
      for (int label2 = label1 + 1; label2 <= numLabels; label2++) {
        if (label1 != label2) {
          std::cout << "Optimize Pair of labels: " << label1 << "	" << label2
                    << std::endl;

          for (it = imIn.begin(), iend = imIn.end(); it != iend; ++it) {
            // for all pixels in imIn create a vertex
            o0       = it.getOffset();
            int val  = imMarker.pixelFromOffset(o0);
            int val2 = imIn.pixelFromOffset(o0);

            if (val == label1) {
              boost::tie(e4, in1) = boost::edge(vSource, val2, g);
              if (in1 == 0) {
                boost::tie(e4, in1) = boost::add_edge(vSource, val2, g);
                boost::tie(e3, in1) = boost::add_edge(val2, vSource, g);
                capacity[e4]        = (std::numeric_limits<double>::max)();
                capacity[e3]        = (std::numeric_limits<double>::max)();
                rev[e4]             = e3;
                rev[e3]             = e4;
              }
            } else if (val == label2) {
              boost::tie(e4, in1) = boost::edge(val2, vSink, g);
              if (in1 == 0) {
                boost::tie(e4, in1) = boost::add_edge(val2, vSink, g);
                boost::tie(e3, in1) = boost::add_edge(vSink, val2, g);
                capacity[e4]        = (std::numeric_limits<double>::max)();
                capacity[e3]        = (std::numeric_limits<double>::max)();
                rev[e4]             = e3;
                rev[e3]             = e4;
              }
            }
          }

          std::cout << "Compute Max flow" << std::endl;
#if BOOST_VERSION >= 104700
          double flow =
              boykov_kolmogorov_max_flow(g, capacity, residual_capacity, rev,
                                         &color[0], indexmap, vSource, vSink);
#else
          double flow =
              kolmogorov_max_flow(g, capacity, residual_capacity, rev,
                                  &color[0], indexmap, vSource, vSink);
#endif
          std::cout << "c  The total flow:" << std::endl;
          std::cout << "s " << flow << std::endl << std::endl;

          for (it = imIn.begin(), iend = imIn.end(); it != iend; ++it) {
            // for all pixels in imIn create a vertex and an edge
            o1       = it.getOffset();
            int val  = imIn.pixelFromOffset(o1);
            int val2 = imMosaic.pixelFromOffset(o1);
            int val3 = imMarker.pixelFromOffset(o1);

            if (val2 == label1 || val2 == label2) {
              if (color[val] == color[vSource])
                imOut.setPixel(o1, label1);
              else if (color[val] == color[vSink])
                imOut.setPixel(o1, label2);
              else
                imOut.setPixel(o1, label2);
            }

            if (val3 == label1) {
              boost::tie(e4, in1) = boost::edge(vSource, val, g);
              if (in1 == 1) {
                boost::remove_edge(vSource, val, g);
                boost::remove_edge(val, vSource, g);
              }
            } else if (val3 == label2) {
              boost::tie(e4, in1) = boost::edge(val, vSink, g);
              if (in1 == 1) {
                boost::remove_edge(val, vSink, g);
                boost::remove_edge(vSink, val, g);
              }
            }
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
  template <class ImageIn, class ImageGrad, class ImageCurvature,
            class ImageMarker, typename _Beta, class SE, class ImageOut>
  RES_T GeoCuts_Regularized_MinSurfaces(
      const ImageIn &imIn, const ImageGrad &imGrad,
      const ImageCurvature &imCurvature, const ImageMarker &imMarker,
      const _Beta Beta, const SE &nl, ImageOut &imOut)
  {
    SMIL_ENTER_FUNCTION("");

    std::cout << "Enter function t_GeoCuts_Regularized_MinSurfaces"
              << std::endl;

    if ((!imOut.isAllocated())) {
      SMIL_REGISTER_ERROR("Not allocated");
      return RES_NOT_ALLOCATED;
    }

    if ((!imIn.isAllocated())) {
      SMIL_REGISTER_ERROR("Not allocated");
      return RES_NOT_ALLOCATED;
    }

    if ((!imGrad.isAllocated())) {
      SMIL_REGISTER_ERROR("Not allocated");
      return RES_NOT_ALLOCATED;
    }

    if ((!imCurvature.isAllocated())) {
      SMIL_REGISTER_ERROR("Not allocated");
      return RES_NOT_ALLOCATED;
    }

    if ((!imMarker.isAllocated())) {
      SMIL_REGISTER_ERROR("Not allocated");
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

    Graph_d g;

    double beta = Beta;

    std::cout << "Reg. :" << beta << std::endl;

    boost::property_map<Graph_d, boost::edge_capacity_t>::type capacity =
        boost::get(boost::edge_capacity, g);

    boost::property_map<Graph_d, boost::edge_reverse_t>::type rev =
        get(boost::edge_reverse, g);

    boost::property_map<Graph_d, boost::edge_residual_capacity_t>::type
        residual_capacity = get(boost::edge_residual_capacity, g);

    bool in1;
    Graph_d::edge_descriptor e1, e2, e3, e4;
    Graph_d::vertex_descriptor vSource, vSink;
    int numVert = 0;

    std::cout << "build Region Adjacency Graph" << std::endl;

    for (it = imIn.begin(), iend = imIn.end(); it != iend; ++it) {
      // for all pixels in imIn create a vertex
      o0      = it.getOffset();
      int val = imIn.pixelFromOffset(o0);
      if (val > numVert) {
        numVert = val;
      }
    }

    std::cout << "build Region Adjacency Graph Vertices" << std::endl;

    for (int i = 1; i <= numVert; i++) {
      boost::add_vertex(g);
    }

    vSource = boost::add_vertex(g);
    vSink   = boost::add_vertex(g);

    std::cout << "build Region Adjacency Graph Edges" << std::endl;

    for (it = imIn.begin(), iend = imIn.end(); it != iend; ++it) {
      // for all pixels in imIn create a vertex and an edge
      o1         = it.getOffset();
      int val    = imIn.pixelFromOffset(o1);
      int marker = imMarker.pixelFromOffset(o1);

      if (val > 0) {
        if (marker == 2) {
          boost::tie(e4, in1) = boost::edge(vSource, val, g);
          if (in1 == 0) {
            boost::tie(e4, in1) = boost::add_edge(vSource, val, g);
            boost::tie(e3, in1) = boost::add_edge(val, vSource, g);
            capacity[e4]        = (std::numeric_limits<double>::max)();
            capacity[e3]        = (std::numeric_limits<double>::max)();
            rev[e4]             = e3;
            rev[e3]             = e4;
          }
        } else if (marker == 3) {
          boost::tie(e3, in1) = boost::edge(vSink, val, g);
          if (in1 == 0) {
            boost::tie(e4, in1) = boost::add_edge(val, vSink, g);
            boost::tie(e3, in1) = boost::add_edge(vSink, val, g);
            capacity[e4]        = (std::numeric_limits<double>::max)();
            capacity[e3]        = (std::numeric_limits<double>::max)();
            rev[e4]             = e3;
            rev[e3]             = e4;
          }
        }

        neighb.setCenter(o1);

        for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
          const offset_t o2 = nit.getOffset();
          int val2          = imIn.pixelFromOffset(o2);

          if (o2 > o1) {
            if (val != val2) {
              boost::tie(e3, in1) = boost::edge(val, val2, g);

              double val3 = imGrad.pixelFromOffset(o1);
              double val4 = imGrad.pixelFromOffset(o2);

              double val5 = imCurvature.pixelFromOffset(o1);
              double val6 = imCurvature.pixelFromOffset(o2);

              double maxigrad = std::max(val3, val4);

              double maxicurv = std::max(val5, val6);

              // if (maxicurv >0) std::cout<<maxicurv<<std::endl;

              double costcurvature = (beta) *maxicurv;

              double costgradient = 10000.0 / (1 + std::pow(maxigrad, 2));

              double cost = costgradient + costcurvature;

              if (in1 == 0) {
                boost::tie(e4, in1) = boost::add_edge(val, val2, g);
                boost::tie(e3, in1) = boost::add_edge(val2, val, g);
                capacity[e4]        = cost;
                capacity[e3]        = cost;
                rev[e4]             = e3;
                rev[e3]             = e4;
              } else {
                boost::tie(e4, in1) = boost::edge(val, val2, g);
                boost::tie(e3, in1) = boost::edge(val2, val, g);
                capacity[e4]        = capacity[e4] + cost;
                capacity[e3]        = capacity[e3] + cost;
              }
            }
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

    for (it = imIn.begin(), iend = imIn.end(); it != iend; ++it) {
      // for all pixels in imIn create a vertex and an edge
      o1      = it.getOffset();
      int val = imIn.pixelFromOffset(o1);

      if (val == 0) {
        imOut.setPixel(o1, 0);
      } else {
        if (color[val] == color[vSource])
          imOut.setPixel(o1, 3);
        if (color[val] == 1)
          imOut.setPixel(o1, 4);
        if (color[val] == color[vSink])
          imOut.setPixel(o1, 2);
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
  template <class ImageIn, class ImageMosaic, class ImageMarker, class SE,
            class ImageOut>
  RES_T GeoCuts_Segment_Graph(const ImageIn &imIn, const ImageMosaic &imMosaic,
                              const ImageMarker &imMarker, const SE &nl,
                              ImageOut &imOut)
  {
    SMIL_ENTER_FUNCTION("");

    std::cout << "Enter function optimize mosaic t_GeoCuts_Segment_Graph"
              << std::endl;

    if ((!imOut.isAllocated())) {
      SMIL_REGISTER_ERROR("Not allocated");
      return RES_NOT_ALLOCATED;
    }

    if ((!imIn.isAllocated())) {
      SMIL_REGISTER_ERROR("Not allocated");
      return RES_NOT_ALLOCATED;
    }

    if ((!imMosaic.isAllocated())) {
      SMIL_REGISTER_ERROR("Not allocated");
      return RES_NOT_ALLOCATED;
    }

    if ((!imMarker.isAllocated())) {
      SMIL_REGISTER_ERROR("Not allocated");
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

    Graph_d g;

    double sigma = 1.0;

    boost::property_map<Graph_d, boost::edge_capacity_t>::type capacity =
        boost::get(boost::edge_capacity, g);

    boost::property_map<Graph_d, boost::edge_reverse_t>::type rev =
        get(boost::edge_reverse, g);

    boost::property_map<Graph_d, boost::edge_residual_capacity_t>::type
        residual_capacity = get(boost::edge_residual_capacity, g);

    bool in1;
    Graph_d::edge_descriptor e1, e2, e3, e4;
    Graph_d::vertex_descriptor vSource, vSink;
    int numVert         = 0;
    double meanclasse1  = 0.0;
    double meanclasse12 = 0.0;

    double meanclasse2 = 0.5;

    double sigma1 = 0.25;
    double sigma2 = 0.5;

    double max_value    = 0.0;
    double max_longueur = 0.0;

    std::cout << "build Region Adjacency Graph" << std::endl;

    for (it = imIn.begin(), iend = imIn.end(); it != iend; ++it) {
      // for all pixels in imIn create a vertex
      o0       = it.getOffset();
      int val  = imMosaic.pixelFromOffset(o0);
      int val2 = imIn.pixelFromOffset(o0);
      int val3 = imMarker.pixelFromOffset(o0);

      if (val > numVert) {
        numVert = val;
      }
      if (val2 > max_value) {
        max_value = val2;
      }
      if (val3 > max_longueur) {
        max_longueur = val3;
      }
    }

    std::cout << "build Region Adjacency Graph Vertices" << std::endl;

    std::cout << "Number of Vertices : " << numVert << std::endl;

    std::cout << "Max value : " << max_value << std::endl;

    for (int i = 0; i <= numVert; i++) {
      boost::add_vertex(g);
    }

    vSource = boost::add_vertex(g);
    vSink   = boost::add_vertex(g);

    std::cout << "build Region Adjacency Graph Edges" << std::endl;

    for (it = imIn.begin(), iend = imIn.end(); it != iend; ++it) {
      // for all pixels in imIn create a vertex and an edge
      o1      = it.getOffset();
      int val = imMosaic.pixelFromOffset(o1);

      if (val > 0) {
        boost::tie(e4, in1) = boost::edge(vSource, val, g);

        if (in1 == 0) {
          // std::cout<<"Add new edge marker 2"<<std::endl;
          double valee = (double) imIn.pixelFromOffset(o1) / max_value;
          double longueur =
              (double) imMarker.pixelFromOffset(o1) / max_longueur;

          double cost1 = 4 * (1 - longueur) + (valee - meanclasse1) *
                                                  (valee - meanclasse1) /
                                                  (2 * sigma1 * sigma1);
          double cost12 = 4 * (1 - longueur) + (valee - meanclasse12) *
                                                   (valee - meanclasse12) /
                                                   (2 * sigma1 * sigma1);
          double cost2 = 4 * (1 - 0.17) + (valee - meanclasse2) *
                                              (valee - meanclasse2) /
                                              (2 * sigma2 * sigma2);

          /*
            double cost1 =
            (valee-meanclasse1)*(valee-meanclasse1)/(2*sigma1*sigma1); double
            cost12 =
            (valee-meanclasse12)*(valee-meanclasse12)/(2*sigma1*sigma1);
            double cost2 =
            (valee-meanclasse2)*(valee-meanclasse2)/(2*sigma2*sigma2);
          */

          boost::tie(e4, in1) = boost::add_edge(vSource, val, g);
          boost::tie(e3, in1) = boost::add_edge(val, vSource, g);
          capacity[e4]        = std::min(cost1, cost12);
          capacity[e3]        = std::min(cost1, cost12);
          rev[e4]             = e3;
          rev[e3]             = e4;

          boost::tie(e4, in1) = boost::add_edge(vSink, val, g);
          boost::tie(e3, in1) = boost::add_edge(val, vSink, g);
          capacity[e4]        = cost2;
          capacity[e3]        = cost2;
          rev[e4]             = e3;
          rev[e3]             = e4;
        }

        neighb.setCenter(o1);

        for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
          const offset_t o2 = nit.getOffset();
          int val2          = imMosaic.pixelFromOffset(o2);

          if (o2 > o1) {
            if (val != val2 && val2 > 0) {
              boost::tie(e3, in1) = boost::edge(val, val2, g);

              double valee1 = (double) imIn.pixelFromOffset(o1) / max_value;
              double valee2 = (double) imIn.pixelFromOffset(o2) / max_value;

              double longueur1 =
                  (double) imMarker.pixelFromOffset(o1) / max_longueur;
              double longueur2 =
                  (double) imMarker.pixelFromOffset(o2) / max_longueur;

              double cost_diff     = 0.01 * std::exp(-0.01 * (valee1 - valee2) *
                                                 (valee1 - valee2));
              double cost_longueur = 0.1;

              if (in1 == 0) {
                // std::cout<<"Add new edge"<<std::endl;
                boost::tie(e4, in1) = boost::add_edge(val, val2, g);
                boost::tie(e3, in1) = boost::add_edge(val2, val, g);
                capacity[e4]        = cost_longueur;
                capacity[e3]        = cost_longueur;
                rev[e4]             = e3;
                rev[e3]             = e4;
              }
            }
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

    for (it = imIn.begin(), iend = imIn.end(); it != iend; ++it) {
      // for all pixels in imIn create a vertex and an edge
      o1      = it.getOffset();
      int val = imMosaic.pixelFromOffset(o1);

      if (color[val] == color[vSource])
        imOut.setPixel(o1, 10);
      if (color[val] == 1)
        imOut.setPixel(o1, 0);
      if (color[val] == color[vSink])
        imOut.setPixel(o1, 30);
    }

    return RES_OK;
  }

  /*
   *
   *
   *
   *
   */
  template <class ImageIn, class ImageMosaic, class ImageMarker, typename _Beta,
            typename _Sigma, class SE, class ImageOut>
  RES_T MAP_MRF_edge_preserving(
      const ImageIn &imIn, const ImageMosaic &imMosaic,
      const ImageMarker &imMarker, const _Beta Beta, const _Sigma Sigma,
      const SE &nl, ImageOut &imOut)
  {
    SMIL_ENTER_FUNCTION("");

    std::cout << "Enter function t_MAP_MRF_edge_preserving" << std::endl;

    if ((!imOut.isAllocated())) {
      SMIL_REGISTER_ERROR("Not allocated");
      return RES_NOT_ALLOCATED;
    }

    if ((!imIn.isAllocated())) {
      SMIL_REGISTER_ERROR("Not allocated");
      return RES_NOT_ALLOCATED;
    }

    if ((!imMosaic.isAllocated())) {
      SMIL_REGISTER_ERROR("Not allocated");
      return RES_NOT_ALLOCATED;
    }

    if ((!imMarker.isAllocated())) {
      SMIL_REGISTER_ERROR("Not allocated");
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

    Graph_d g;

    double sigma = Sigma;

    boost::property_map<Graph_d, boost::edge_capacity_t>::type capacity =
        boost::get(boost::edge_capacity, g);

    boost::property_map<Graph_d, boost::edge_reverse_t>::type rev =
        get(boost::edge_reverse, g);

    boost::property_map<Graph_d, boost::edge_residual_capacity_t>::type
        residual_capacity = get(boost::edge_residual_capacity, g);

    bool in1;
    Graph_d::edge_descriptor e1, e2, e3, e4;
    Graph_d::vertex_descriptor vSource, vSink;
    int numVert = 1;

    std::cout << "build Region Adjacency Graph" << std::endl;

    for (it = imIn.begin(), iend = imIn.end(); it != iend; ++it) {
      // for all pixels in imIn create a vertex
      o0      = it.getOffset();
      int val = imMosaic.pixelFromOffset(o0);
      imOut.setPixel(o0, 0);

      if (val > numVert) {
        numVert = val;
      }
    }

    double *mean = new double[numVert + 1];
    int *nb_val  = new int[numVert + 1];
    int *marker  = new int[numVert + 1];

    for (int i = 0; i <= numVert; i++) {
      boost::add_vertex(g);
      mean[i]   = 0;
      nb_val[i] = 0;
      marker[i] = 0;
    }

    vSource = boost::add_vertex(g);
    vSink   = boost::add_vertex(g);

    double meanforeground = 0;
    double meanbackground = 0;
    double nb_foreground  = 0;
    double nb_background  = 0;

    std::cout << "Compute Mean Value in Regions" << std::endl;

    for (it = imIn.begin(), iend = imIn.end(); it != iend; ++it) {
      // for all pixels in imIn create a vertex
      o0            = it.getOffset();
      int val       = imIn.pixelFromOffset(o0);
      int val2      = imMosaic.pixelFromOffset(o0);
      int val3      = imMarker.pixelFromOffset(o0);
      double valeur = (double) val;

      mean[val2]   = mean[val2] + (valeur / 255.0);
      nb_val[val2] = nb_val[val2] + 1;

      if (val3 == 2) {
        meanforeground = meanforeground + (valeur / 255.0);
        nb_foreground++;
        marker[val2] = 2;
      } else if (val3 == 3) {
        meanbackground = meanbackground + (valeur / 255.0);
        nb_background++;
        marker[val2] = 3;
      }
    }

    meanforeground = meanforeground / nb_foreground;
    meanbackground = meanbackground / nb_background;

    std::cout << "Mean Foreground " << meanforeground << std::endl;
    std::cout << "Mean Background " << meanbackground << std::endl;

    std::cout << "Compute terminal links" << std::endl;

    double sigmab = 0.2;

    for (int i = 0; i <= numVert; i++) {
      mean[i] = mean[i] / (nb_val[i]);

      if (marker[i] == 2 && nb_val[i] > 0) {
        boost::tie(e4, in1) = boost::add_edge(vSource, i, g);
        boost::tie(e3, in1) = boost::add_edge(i, vSource, g);
        capacity[e4]        = (std::numeric_limits<double>::max)();
        capacity[e3]        = (std::numeric_limits<double>::max)();
        rev[e4]             = e3;
        rev[e3]             = e4;
      } else if (marker[i] == 3 && nb_val[i] > 0) {
        boost::tie(e4, in1) = boost::add_edge(vSink, i, g);
        boost::tie(e3, in1) = boost::add_edge(i, vSink, g);
        capacity[e4]        = (std::numeric_limits<double>::max)();
        capacity[e3]        = (std::numeric_limits<double>::max)();
        rev[e4]             = e3;
        rev[e3]             = e4;

      }

      else if (nb_val[i] > 0) {
        double valee = mean[i];
        double sigma = Sigma;
        double val2  = (valee - meanforeground) * (valee - meanforeground) /
                      (2 * sigma * sigma);
        double val1 = (valee - meanbackground) * (valee - meanbackground) /
                      (2 * sigmab * sigmab);

        boost::tie(e4, in1) = boost::add_edge(vSource, i, g);
        boost::tie(e3, in1) = boost::add_edge(i, vSource, g);
        capacity[e4]        = nb_val[i] * val1;
        capacity[e3]        = nb_val[i] * val1;
        rev[e4]             = e3;
        rev[e3]             = e4;

        boost::tie(e4, in1) = boost::add_edge(i, vSink, g);
        boost::tie(e3, in1) = boost::add_edge(vSink, i, g);
        capacity[e4]        = nb_val[i] * val2;
        capacity[e3]        = nb_val[i] * val2;
        rev[e4]             = e3;
        rev[e3]             = e4;
      }
    }

    std::cout << "build Region Adjacency Graph Edges" << std::endl;

    for (it = imIn.begin(), iend = imIn.end(); it != iend; ++it) {
      // for all pixels in imIn create a vertex and an edge
      o1      = it.getOffset();
      int val = imMosaic.pixelFromOffset(o1);
      neighb.setCenter(o1);

      for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
        const offset_t o2 = nit.getOffset();
        int val2          = imMosaic.pixelFromOffset(o2);

        if (o2 > o1) {
          if (val != val2) {
            boost::tie(e3, in1) = boost::edge(val, val2, g);
            double cost         = (double) Beta -
                          ((double) Beta) *
                              std::pow(std::abs(mean[val] - mean[val2]), 6.0);
            if (in1 == 0) {
              boost::tie(e4, in1) = boost::add_edge(val, val2, g);
              boost::tie(e3, in1) = boost::add_edge(val2, val, g);
              capacity[e4]        = cost;
              capacity[e3]        = cost;
              rev[e4]             = e3;
              rev[e3]             = e4;
            } else {
              boost::tie(e4, in1) = boost::edge(val, val2, g);
              boost::tie(e3, in1) = boost::edge(val2, val, g);
              capacity[e4]        = capacity[e4] + cost;
              capacity[e3]        = capacity[e3] + cost;
            }
          }
        }
      }
    }

    delete[] nb_val;
    delete[] mean;
    delete[] marker;

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

    for (it = imIn.begin(), iend = imIn.end(); it != iend; ++it) {
      // for all pixels in imIn create a vertex and an edge
      o1      = it.getOffset();
      int val = imMosaic.pixelFromOffset(o1);

      if (color[val] == color[vSource])
        imOut.setPixel(o1, 2);
      else if (color[val] == color[vSink])
        imOut.setPixel(o1, 3);
      else
        imOut.setPixel(o1, 4);
    }

    return RES_OK;
  }

  /*
   *
   *
   *
   *
   */
  template <class ImageIn, class ImageMosaic, class ImageMarker, typename _Beta,
            typename _Sigma, class SE, class ImageOut>
  RES_T MAP_MRF_Ising(const ImageIn &imIn, const ImageMosaic &imMosaic,
                      const ImageMarker &imMarker, const _Beta Beta,
                      const _Sigma Sigma, const SE &nl, ImageOut &imOut)
  {
    SMIL_ENTER_FUNCTION("");

    std::cout << "Enter function t_MAP_MRF_Ising" << std::endl;

    if ((!imOut.isAllocated())) {
      SMIL_REGISTER_ERROR("Not allocated");
      return RES_NOT_ALLOCATED;
    }

    if ((!imIn.isAllocated())) {
      SMIL_REGISTER_ERROR("Not allocated");
      return RES_NOT_ALLOCATED;
    }

    if ((!imMosaic.isAllocated())) {
      SMIL_REGISTER_ERROR("Not allocated");
      return RES_NOT_ALLOCATED;
    }

    if ((!imMarker.isAllocated())) {
      SMIL_REGISTER_ERROR("Not allocated");
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

    Graph_d g;

    double sigma = Sigma;

    boost::property_map<Graph_d, boost::edge_capacity_t>::type capacity =
        boost::get(boost::edge_capacity, g);

    boost::property_map<Graph_d, boost::edge_reverse_t>::type rev =
        get(boost::edge_reverse, g);

    boost::property_map<Graph_d, boost::edge_residual_capacity_t>::type
        residual_capacity = get(boost::edge_residual_capacity, g);

    bool in1;
    Graph_d::edge_descriptor e1, e2, e3, e4;
    Graph_d::vertex_descriptor vSource, vSink;
    int numVert = 1;

    std::cout << "build Region Adjacency Graph" << std::endl;

    for (it = imIn.begin(), iend = imIn.end(); it != iend; ++it) {
      // for all pixels in imIn create a vertex
      o0      = it.getOffset();
      int val = imMosaic.pixelFromOffset(o0);
      imOut.setPixel(o0, 0);

      if (val > numVert) {
        numVert = val;
      }
    }

    double *mean = new double[numVert + 1];
    int *nb_val  = new int[numVert + 1];
    int *marker  = new int[numVert + 1];

    for (int i = 0; i <= numVert; i++) {
      boost::add_vertex(g);
      mean[i]   = 0;
      nb_val[i] = 0;
      marker[i] = 0;
    }

    vSource = boost::add_vertex(g);
    vSink   = boost::add_vertex(g);

    double meanforeground = 0;
    double meanbackground = 0;
    double nb_foreground  = 0;
    double nb_background  = 0;

    std::cout << "Compute Mean Value in Regions" << std::endl;

    for (it = imIn.begin(), iend = imIn.end(); it != iend; ++it) {
      // for all pixels in imIn create a vertex
      o0            = it.getOffset();
      int val       = imIn.pixelFromOffset(o0);
      int val2      = imMosaic.pixelFromOffset(o0);
      int val3      = imMarker.pixelFromOffset(o0);
      double valeur = (double) val;

      mean[val2]   = mean[val2] + (valeur / 255.0);
      nb_val[val2] = nb_val[val2] + 1;

      if (val3 == 2) {
        meanforeground = meanforeground + (valeur / 255.0);
        nb_foreground++;
        marker[val2] = 2;
      } else if (val3 == 3) {
        meanbackground = meanbackground + (valeur / 255.0);
        nb_background++;
        marker[val2] = 3;
      }
    }

    meanforeground = meanforeground / nb_foreground;
    meanbackground = meanbackground / nb_background;

    std::cout << "Mean Foreground " << meanforeground << std::endl;
    std::cout << "Mean Background " << meanbackground << std::endl;

    std::cout << "Compute terminal links" << std::endl;

    double sigmab = 0.2;
    sigmab        = Sigma;

    for (int i = 0; i <= numVert; i++) {
      mean[i] = mean[i] / (nb_val[i]);

      if (marker[i] == 2 && nb_val[i] > 0) {
        boost::tie(e4, in1) = boost::add_edge(vSource, i, g);
        boost::tie(e3, in1) = boost::add_edge(i, vSource, g);
        capacity[e4]        = (std::numeric_limits<double>::max)();
        capacity[e3]        = (std::numeric_limits<double>::max)();
        rev[e4]             = e3;
        rev[e3]             = e4;
      } else if (marker[i] == 3 && nb_val[i] > 0) {
        boost::tie(e4, in1) = boost::add_edge(vSink, i, g);
        boost::tie(e3, in1) = boost::add_edge(i, vSink, g);
        capacity[e4]        = (std::numeric_limits<double>::max)();
        capacity[e3]        = (std::numeric_limits<double>::max)();
        rev[e4]             = e3;
        rev[e3]             = e4;

      } else if (nb_val[i] > 0) {
        double valee = mean[i];
        double sigma = Sigma;
        double val2  = (valee - meanforeground) * (valee - meanforeground) /
                      (2 * sigma * sigma);
        double val1 = (valee - meanbackground) * (valee - meanbackground) /
                      (2 * sigmab * sigmab);

        boost::tie(e4, in1) = boost::add_edge(vSource, i, g);
        boost::tie(e3, in1) = boost::add_edge(i, vSource, g);
        capacity[e4]        = nb_val[i] * val1;
        capacity[e3]        = nb_val[i] * val1;
        rev[e4]             = e3;
        rev[e3]             = e4;

        boost::tie(e4, in1) = boost::add_edge(i, vSink, g);
        boost::tie(e3, in1) = boost::add_edge(vSink, i, g);
        capacity[e4]        = nb_val[i] * val2;
        capacity[e3]        = nb_val[i] * val2;
        rev[e4]             = e3;
        rev[e3]             = e4;
      }
    }

    delete[] nb_val;
    delete[] mean;
    int numEdges = 0;

    std::cout << "build Region Adjacency Graph Edges" << std::endl;

    for (it = imIn.begin(), iend = imIn.end(); it != iend; ++it) {
      // for all pixels in imIn create a vertex and an edge
      o1      = it.getOffset();
      int val = imMosaic.pixelFromOffset(o1);
      neighb.setCenter(o1);

      for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
        const offset_t o2 = nit.getOffset();
        int val2          = imMosaic.pixelFromOffset(o2);

        if (o2 > o1) {
          if (val != val2) {
            boost::tie(e3, in1) = boost::edge(val, val2, g);
            double cost         = (double) Beta;
            if (in1 == 0) {
              numEdges++;
              boost::tie(e4, in1) = boost::add_edge(val, val2, g);
              boost::tie(e3, in1) = boost::add_edge(val2, val, g);
              capacity[e4]        = cost;
              capacity[e3]        = cost;
              rev[e4]             = e3;
              rev[e3]             = e4;
            } else {
              boost::tie(e4, in1) = boost::edge(val, val2, g);
              boost::tie(e3, in1) = boost::edge(val2, val, g);
              capacity[e4]        = capacity[e4] + cost;
              capacity[e3]        = capacity[e3] + cost;
            }
          }
        }
      }
    }

    std::cout << "Number of vertices " << numVert << std::endl;
    std::cout << "Number of Edges " << numEdges << std::endl;

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

    for (it = imIn.begin(), iend = imIn.end(); it != iend; ++it) {
      // for all pixels in imIn create a vertex and an edge
      o1      = it.getOffset();
      int val = imMosaic.pixelFromOffset(o1);

      if (color[val] == color[vSource])
        imOut.setPixel(o1, 2);
      else if (color[val] == color[vSink])
        imOut.setPixel(o1, 3);
      else
        imOut.setPixel(o1, 4);
    }

    return RES_OK;
  }

  /*
   *
   *
   *
   *
   */
  template <class ImageIn, class ImageMosaic, class ImageMarker, typename _Beta,
            typename _Sigma, class SE, class ImageOut>
  RES_T MAP_MRF_Potts(const ImageIn &imIn, const ImageMosaic &imMosaic,
                      const ImageMarker &imMarker, const _Beta Beta,
                      const _Sigma Sigma, const SE &nl, ImageOut &imOut)
  {
    SMIL_ENTER_FUNCTION("");

    std::cout << "Enter function t_MAP_MRF_Potts" << std::endl;

    if ((!imOut.isAllocated())) {
      SMIL_REGISTER_ERROR("Not allocated");
      return RES_NOT_ALLOCATED;
    }

    if ((!imIn.isAllocated())) {
      SMIL_REGISTER_ERROR("Not allocated");
      return RES_NOT_ALLOCATED;
    }

    if ((!imMosaic.isAllocated())) {
      SMIL_REGISTER_ERROR("Not allocated");
      return RES_NOT_ALLOCATED;
    }

    if ((!imMarker.isAllocated())) {
      SMIL_REGISTER_ERROR("Not allocated");
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

    Graph_d g;

    boost::property_map<Graph_d, boost::edge_capacity_t>::type capacity =
        boost::get(boost::edge_capacity, g);

    boost::property_map<Graph_d, boost::edge_reverse_t>::type rev =
        get(boost::edge_reverse, g);

    boost::property_map<Graph_d, boost::edge_residual_capacity_t>::type
        residual_capacity = get(boost::edge_residual_capacity, g);

    bool in1;
    Graph_d::edge_descriptor e1, e2, e3, e4;
    Graph_d::vertex_descriptor vSource, vSink, vSource2, vSink2;

    int numVert = 0;

    double meanlabel1 = 0;
    double meanlabel2 = 0;
    double meanlabel3 = 0;

    double nb_label1 = 0;
    double nb_label2 = 0;
    double nb_label3 = 0;

    double Sigmal = (double) Sigma;

    for (it = imIn.begin(), iend = imIn.end(); it != iend; ++it) {
      // for all pixels in imIn create a vertex
      o0      = it.getOffset();
      int val = imMosaic.pixelFromOffset(o0);

      imOut.setPixel(o0, 0);

      if (val > numVert) {
        numVert = val;
      }
    }

    double *mean = new double[numVert + 1];
    int *nb_val  = new int[numVert + 1];
    int *marker  = new int[numVert + 1];

    std::cout << "number of regions : " << numVert << std::endl;

    for (int i = 0; i <= numVert; i++) {
      mean[i]   = 0;
      nb_val[i] = 0;
      marker[i] = 0;
    }

    std::cout << "build Region Adjacency Graph" << std::endl;

    for (it = imIn.begin(), iend = imIn.end(); it != iend; ++it) {
      // for all pixels in imIn create a vertex
      o0 = it.getOffset();
      imOut.setPixel(o0, 0);

      int val  = imIn.pixelFromOffset(o0);
      int val2 = imMosaic.pixelFromOffset(o0);
      int val3 = imMarker.pixelFromOffset(o0);

      double valeur = (double) val;

      mean[val2]   = mean[val2] + (valeur / 255.0);
      nb_val[val2] = nb_val[val2]++;

      if (val3 == 2) { /////////    LIVER LABEL
        meanlabel1 = meanlabel1 + (valeur / 255.0);
        nb_label1++;
        marker[val2] = 2;
      } else if (val3 == 3) { /////////    TUMOR LABEL
        meanlabel2 = meanlabel2 + (valeur / 255.0);
        nb_label2++;
        marker[val2] = 3;
      } else if (val3 == 4) { /////////    BACKGROUND LABEL
        meanlabel3 = meanlabel3 + (valeur / 255.0);
        nb_label3++;
        marker[val2] = 4;
      }
    }

    meanlabel1 = meanlabel1 / nb_label1;
    meanlabel2 = meanlabel2 / nb_label2;
    meanlabel3 = meanlabel3 / nb_label3;

    std::cout << "Mean Label 1 " << meanlabel1 << std::endl;
    std::cout << "Mean Label 2 " << meanlabel2 << std::endl;
    std::cout << "Mean Label 3 " << meanlabel3 << std::endl;

    std::cout << "Compute terminal links" << std::endl;

    for (int i = 0; i <= numVert; i++) {
      boost::add_vertex(g);
      mean[i] = mean[i] / (nb_val[i]);
    }

    vSource = boost::add_vertex(g);
    vSink   = boost::add_vertex(g);

    for (int i = 0; i <= numVert; i++) {
      if (marker[i] == 4 && nb_val[i] > 0) {
        boost::tie(e4, in1) = boost::add_edge(vSink, i, g);
        boost::tie(e3, in1) = boost::add_edge(i, vSink, g);
        capacity[e4]        = (std::numeric_limits<double>::max)();
        capacity[e3]        = (std::numeric_limits<double>::max)();
        rev[e4]             = e3;
        rev[e3]             = e4;
      } else if (marker[i] > 0 && nb_val[i] > 0) {
        boost::tie(e4, in1) = boost::add_edge(vSource, i, g);
        boost::tie(e3, in1) = boost::add_edge(i, vSource, g);
        capacity[e4]        = (std::numeric_limits<double>::max)();
        capacity[e3]        = (std::numeric_limits<double>::max)();
        rev[e4]             = e3;
        rev[e3]             = e4;
      } else if (nb_val[i] > 0) {
        double valee = mean[i];

        double val1 =
            (valee - meanlabel3) * (valee - meanlabel3) / (2 * 0.05 * 0.05);

        double val2 = 0;
        double val3 = 0;

        val2 =
            (valee - meanlabel1) * (valee - meanlabel1) / (2 * Sigmal * Sigmal);
        val3 = (valee - meanlabel2) * (valee - meanlabel2) / (2 * 0.2 * 0.2);

        boost::tie(e4, in1) = boost::add_edge(vSource, i, g);
        boost::tie(e3, in1) = boost::add_edge(i, vSource, g);
        capacity[e4]        = val1;
        capacity[e3]        = val1;
        rev[e4]             = e3;
        rev[e3]             = e4;

        boost::tie(e4, in1) = boost::add_edge(i, vSink, g);
        boost::tie(e3, in1) = boost::add_edge(vSink, i, g);
        capacity[e4]        = std::min(val2, val3);
        capacity[e3]        = std::min(val2, val3);
        rev[e4]             = e3;
        rev[e3]             = e4;
      }
    }

    std::cout << "build Region Adjacency Graph Edges" << std::endl;

    for (it = imIn.begin(), iend = imIn.end(); it != iend; ++it) {
      // for all pixels in imIn create a vertex and an edge
      o1      = it.getOffset();
      int val = imMosaic.pixelFromOffset(o1);
      neighb.setCenter(o1);

      for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
        const offset_t o2 = nit.getOffset();
        int val2          = imMosaic.pixelFromOffset(o2);

        if (o2 > o1) {
          if (val != val2) {
            boost::tie(e3, in1) = boost::edge(val, val2, g);
            double cost         = (double) Beta;
            if (in1 == 0) {
              boost::tie(e4, in1) = boost::add_edge(val, val2, g);
              boost::tie(e3, in1) = boost::add_edge(val2, val, g);
              capacity[e4]        = cost;
              capacity[e3]        = cost;
              rev[e4]             = e3;
              rev[e3]             = e4;
            } else {
              boost::tie(e4, in1) = boost::edge(val, val2, g);
              boost::tie(e3, in1) = boost::edge(val2, val, g);
              capacity[e4]        = capacity[e4] + cost;
              capacity[e3]        = capacity[e3] + cost;
            }
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

    for (it = imIn.begin(), iend = imIn.end(); it != iend; ++it) {
      // for all pixels in imIn create a vertex and an edge
      o1           = it.getOffset();
      int valimout = imOut.pixelFromOffset(o1);
      int val      = imMosaic.pixelFromOffset(o1);

      if (nb_val[val] > 0) {
        if (valimout == 0) {
          if (color[val] == 4)
            imOut.setPixel(o1, 4);
        }
      }
    }

    Graph_d g2;

    boost::property_map<Graph_d, boost::edge_capacity_t>::type capacity2 =
        boost::get(boost::edge_capacity, g2);

    boost::property_map<Graph_d, boost::edge_reverse_t>::type rev2 =
        get(boost::edge_reverse, g2);

    boost::property_map<Graph_d, boost::edge_residual_capacity_t>::type
        residual_capacity2 = get(boost::edge_residual_capacity, g2);

    for (int i = 0; i <= numVert; i++) {
      boost::add_vertex(g2);
    }

    vSource2 = boost::add_vertex(g2);
    vSink2   = boost::add_vertex(g2);

    for (int i = 0; i <= numVert; i++) {
      if (marker[i] == 3 && nb_val[i] > 0) {
        boost::tie(e4, in1) = boost::add_edge(vSink2, i, g2);
        boost::tie(e3, in1) = boost::add_edge(i, vSink2, g2);
        capacity2[e4]       = (std::numeric_limits<double>::max)();
        capacity2[e3]       = (std::numeric_limits<double>::max)();
        rev2[e4]            = e3;
        rev2[e3]            = e4;
      } else if (marker[i] == 2 && nb_val[i] > 0) {
        boost::tie(e4, in1) = boost::add_edge(vSource2, i, g2);
        boost::tie(e3, in1) = boost::add_edge(i, vSource2, g2);
        capacity2[e4]       = (std::numeric_limits<double>::max)();
        capacity2[e3]       = (std::numeric_limits<double>::max)();
        rev2[e4]            = e3;
        rev2[e3]            = e4;
      } else if (nb_val[i] > 0) {
        double valee = mean[i];
        double val1 =
            (valee - meanlabel2) * (valee - meanlabel2) / (2 * 0.2 * 0.2);
        double val2 =
            (valee - meanlabel1) * (valee - meanlabel1) / (2 * Sigmal * Sigmal);

        boost::tie(e4, in1) = boost::add_edge(vSource2, i, g2);
        boost::tie(e3, in1) = boost::add_edge(i, vSource2, g2);
        capacity2[e4]       = val1;
        capacity2[e3]       = val1;
        rev2[e4]            = e3;
        rev2[e3]            = e4;

        boost::tie(e4, in1) = boost::add_edge(i, vSink2, g2);
        boost::tie(e3, in1) = boost::add_edge(vSink2, i, g2);
        capacity2[e4]       = val2;
        capacity2[e3]       = val2;
        rev2[e4]            = e3;
        rev2[e3]            = e4;
      }
    }

    std::cout << "build Region Adjacency Graph Edges" << std::endl;

    for (it = imIn.begin(), iend = imIn.end(); it != iend; ++it) {
      // for all pixels in imIn create a vertex and an edge
      o1      = it.getOffset();
      int val = imMosaic.pixelFromOffset(o1);
      neighb.setCenter(o1);

      for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
        const offset_t o2 = nit.getOffset();
        int val2          = imMosaic.pixelFromOffset(o2);

        if (o2 > o1) {
          if (val != val2) {
            boost::tie(e3, in1) = boost::edge(val, val2, g2);
            double cost         = (double) Beta;
            if (in1 == 0) {
              boost::tie(e4, in1) = boost::add_edge(val, val2, g2);
              boost::tie(e3, in1) = boost::add_edge(val2, val, g2);
              capacity2[e4]       = cost;
              capacity2[e3]       = cost;
              rev2[e4]            = e3;
              rev2[e3]            = e4;
            } else {
              boost::tie(e4, in1) = boost::edge(val, val2, g2);
              boost::tie(e3, in1) = boost::edge(val2, val, g2);
              capacity2[e4]       = capacity2[e4] + cost;
              capacity2[e3]       = capacity2[e3] + cost;
            }
          }
        }
      }
    }

    boost::property_map<Graph_d, boost::vertex_index_t>::type indexmap2 =
        boost::get(boost::vertex_index, g2);
    std::vector<boost::default_color_type> color2(boost::num_vertices(g2));

    std::cout << "Compute Max flow" << std::endl;
#if BOOST_VERSION >= 104700
    flow = boykov_kolmogorov_max_flow(g2, capacity2, residual_capacity2, rev2,
                                      &color2[0], indexmap2, vSource2, vSink2);
#else
    flow        = kolmogorov_max_flow(g2, capacity2, residual_capacity2, rev2,
                               &color2[0], indexmap2, vSource2, vSink2);
#endif
    std::cout << "c  The total flow:" << std::endl;
    std::cout << "s " << flow << std::endl << std::endl;

    std::cout << "Source Label:" << color[vSource2] << std::endl;
    std::cout << "Sink  Label:" << color[vSink2] << std::endl;

    for (it = imIn.begin(), iend = imIn.end(); it != iend; ++it) {
      // for all pixels in imIn create a vertex and an edge
      o1           = it.getOffset();
      int valimout = imOut.pixelFromOffset(o1);
      int val      = imMosaic.pixelFromOffset(o1);

      if (nb_val[val] > 0) {
        if (valimout == 0) {
          if (color2[val] == 4)
            imOut.setPixel(o1, 3);
        }
      }
    }

    delete[] nb_val;
    delete[] mean;

    return RES_OK;
  }
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

  void testHandleSE(StrElt &se)
  {
    se = CubeSE(2);
    printSE(se);
    se = se.noCenter();
    printSE(se);
    int sz = se.getSize();
    se     = se.homothety(sz);
    printSE(se);
#if 0
    imIn.printSelf();
    cout << "* width "  << imIn.getWidth() << endl;
    cout << "* height " << imIn.getHeight() << endl;
    cout << "* depth "  << imIn.getDepth() << endl;
#endif
  }

} // namespace smil

#endif // D_MOSAIC_GEOCUTS_IMPL_HPP
