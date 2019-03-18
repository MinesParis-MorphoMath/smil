#ifndef MOSAIC_GEOCUTSALGO_IMPL_T_HPP
#define MOSAIC_GEOCUTSALGO_IMPL_T_HPP

#include <time.h>

#include <boost/config.hpp>
// for boost::tie
#include <boost/utility.hpp>
// for boost::graph_traits
#include <boost/graph/graph_traits.hpp> 
#include <boost/graph/adjacency_list.hpp>
//#include <boost/graph/graphviz.hpp>

#include <boost/version.hpp>
#if BOOST_VERSION >= 104700
#include <boost/graph/boykov_kolmogorov_max_flow.hpp>
#elif BOOST_VERSION >= 103500
#include <boost/graph/kolmogorov_max_flow.hpp>
#endif


//#include <morphee/selement/include/selementNeighborList.hpp>
//#include <morphee/selement/include/private/selementNeighborhood_T.hpp>
//#include <morphee/stats/include/private/statsMeasure_T.hpp>
//#include <morphee/image/include/private/imagePixelwise_T.hpp>

#include <vector>

#define MORPHEE_ENTER_FUNCTION(a) 
#define MORPHEE_REGISTER_ERROR(a) 

typedef off_t offset_t;

namespace smil
{
  using namespace boost;
#if 0

  {
    /*
     *
     *
     */
    template <class ImageIn, class ImageGrad, class ImageMarker, class SE,
              class ImageOut>
    RES_T t_GeoCuts_MinSurfaces_with_steps_old(const ImageIn &imIn,
                                               const ImageGrad &imGrad,
                                               const ImageMarker &imMarker,
                                               const SE &nl, F_SIMPLE step_x,
                                               F_SIMPLE step_y, F_SIMPLE step_z,
                                               ImageOut &imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_GeoCuts_MinSurfaces_with_steps");

      std::cout << "Enter function t_GeoCuts_MinSurfaces" << std::endl;

      if ((!imOut.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if ((!imIn.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if ((!imGrad.isAllocated())) {
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
      double flow = kolmogorov_max_flow(g, capacity, residual_capacity, rev,
                                        &color[0], indexmap, vSource, vSink);
#endif
      std::cout << "c  The total flow:" << std::endl;
      std::cout << "s " << flow << std::endl;
      t2 = clock();
      std::cout << "Flow computation time : "
                << double(t2 - t1) / CLOCKS_PER_SEC << " seconds\n";

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
     */
    template <class T>
    RES_T t_GeoCuts_MinSurfaces(const Image<T> &imIn, const Image<T> &imGrad,
                                const Image<T> &imMarker, const StrElt &nl,
                                Image<T> &imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_GeoCuts_MinSurfaces");

      ASSERT_ALLOCATED(&imIn, &imGrad, &imMarker, &imOut);
      ASSERT_SAME_SIZE(&imIn, &imGrad, &imMarker, &imOut);

#if 0
      morphee::selement::Neighborhood<SE, ImageIn> neighb(imIn, nl);
      typename morphee::selement::Neighborhood<SE, ImageIn>::iterator nit, nend;
#endif

      offset_t o0;
      offset_t o1;

      // needed for max flow: capacit map, rev_capacity map, etc.
      typedef boost::adjacency_list_traits<boost::vecS, boost::vecS,
                                           boost::directedS>
          Traits;

      typedef boost::adjacency_list<boost::vecS,
                                    boost::vecS,
                                    boost::directedS,
                                    boost::property<boost::vertex_name_t,
                                    std::string>,
          boost::property<boost::edge_capacity_t, double,
                          boost::property<boost::edge_residual_capacity_t,
                                          double,
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

      int oMax = imIn.getPixelCount();
      for (o1 = 0; o1 < oMax; o1++) {
        int val      = imIn.getPixel(o1);
        int marker   = imMarker.getPixel(o1);
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
          int val2_prec      = val; 

#if 0
          neighb.setCenter(o1);

          for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
            const offset_t o2 = nit.getOffset();

            if (o2 > o1) {
              int val2 = imIn.pixelFromOffset(o2);
              if (val != val2) {
                double val_grad_o2 = imGrad.pixelFromOffset(o2);
                double maxi        = std::max(val_grad_o1, val_grad_o2);
                double cost        = 10000.0 / (1.0 + std::pow(maxi, 4));

                if (val2_prec == val2) 
                {
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
      double flow = kolmogorov_max_flow(g, capacity, residual_capacity, rev,
                                        &color[0], indexmap, vSource, vSink);
#endif
      std::cout << "c  The total flow:" << std::endl;
      std::cout << "s " << flow << std::endl << std::endl;
      t2 = clock();
      std::cout << "Flow computation time : "
                << double(t2 - t1) / CLOCKS_PER_SEC << " seconds\n";

      t1 = clock();
#if 0
      for (it = imIn.begin(), iend = imIn.end(); it != iend; ++it) {
        o1      = it.getOffset();
        int val = imIn.pixelFromOffset(o1);

        if (val == 0) {
          imOut.setPixel(o1, 0);
        } else {
          if (color[val] == color[vSource])
            imOut.setPixel(o1, 2);
          else if (color[val] == color[vSink])
            imOut.setPixel(o1, 3);
          else
            imOut.setPixel(o1, 4);
        }
      }

      t2 = clock();
      std::cout << "Computing imOut took : " << double(t2 - t1) / CLOCKS_PER_SEC
                << " seconds\n";
#endif
      return RES_OK;
    }
#endif

#if 0
    /*
     *
     *
     */
    template <class ImageIn, class ImageGrad, class ImageMarker, class SE,
              class ImageOut>
    RES_T t_GeoCuts_MinSurfaces_With_Line(const ImageIn &imIn,
                                          const ImageGrad &imGrad,
                                          const ImageMarker &imMarker,
                                          const SE &nl, ImageOut &imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_GeoCuts_MinSurfaces_With_Line");

      std::cout << "Enter function t_GeoCuts_MinSurfaces_With_Line"
                << std::endl;

      if ((!imOut.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if ((!imIn.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if ((!imGrad.isAllocated())) {
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

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o0      = it.getOffset();
        int val = imIn.pixelFromOffset(o0);

        if (val > numVert) {
          numVert = val;
        }
      }

      std::cout << "build Region Adjacency Graph Vertices :" << numVert
                << std::endl;

      for (int i = 0; i <= numVert; i++) {
        boost::add_vertex(g);
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
      double flow = kolmogorov_max_flow(g, capacity, residual_capacity, rev,
                                        &color[0], indexmap, vSource, vSink);
#endif
      std::cout << "c  The total flow:" << std::endl;
      std::cout << "s " << flow << std::endl << std::endl;

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1      = it.getOffset();
        int val = imIn.pixelFromOffset(o1);

        if (color[val] == color[vSource])
          imOut.setPixel(o1, 2);
        else if (color[val] == color[vSink])
          imOut.setPixel(o1, 3);
        else
          imOut.setPixel(o1, 4);
      }

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
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
     */
    template <class ImageIn, class ImageGrad, class ImageCurvature,
              class ImageMarker, typename _Beta, class SE, class ImageOut>
    RES_T t_GeoCuts_Regularized_MinSurfaces(const ImageIn &imIn,
                                            const ImageGrad &imGrad,
                                            const ImageCurvature &imCurvature,
                                            const ImageMarker &imMarker,
                                            const _Beta Beta, const SE &nl,
                                            ImageOut &imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_GeoCuts_Regularized_MinSurfaces");

      std::cout << "Enter function t_GeoCuts_Regularized_MinSurfaces"
                << std::endl;

      if ((!imOut.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if ((!imIn.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if ((!imGrad.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if ((!imCurvature.isAllocated())) {
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

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
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

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
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

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
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
     */
    template <class ImageIn, class ImageGrad, class ImageMarker, class SE,
              class ImageOut>
    RES_T t_GeoCuts_MultiWay_MinSurfaces(const ImageIn &imIn,
                                         const ImageGrad &imGrad,
                                         const ImageMarker &imMarker,
                                         const SE &nl, ImageOut &imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_GeoCuts_MultiWay_MinSurfaces");

      std::cout << "Enter function t_GeoCuts_MultiWay_MinSurfaces" << std::endl;

      if ((!imOut.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if ((!imIn.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if ((!imGrad.isAllocated())) {
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

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
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

      std::cout << "number of labels :" << numLabels << std::endl;

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

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
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
        for (it = imIn.begin(), iend = imIn.end(); it != iend;
             ++it) // for all pixels in imIn create a vertex
        {
          o0       = it.getOffset();
          int val  = imMarker.pixelFromOffset(o0);
          int val2 = imIn.pixelFromOffset(o0);

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

        for (it = imIn.begin(), iend = imIn.end(); it != iend;
             ++it) // for all pixels in imIn create a vertex and an edge
        {
          o1       = it.getOffset();
          int val  = imIn.pixelFromOffset(o1);
          int val2 = imOut.pixelFromOffset(o1);
          int val3 = imMarker.pixelFromOffset(o1);

          if (val2 == 1) {
            if (color[val] == color[vSource])
              imOut.setPixel(o1, nbk);
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

    /*
     *
     *
     */
    template <class ImageIn, class ImageGrad, class ImageMosaic,
              class ImageMarker, class SE, class ImageOut>
    RES_T t_GeoCuts_Optimize_Mosaic(const ImageIn &imIn,
                                    const ImageGrad &imGrad,
                                    const ImageMosaic &imMosaic,
                                    const ImageMarker &imMarker, const SE &nl,
                                    ImageOut &imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_GeoCuts_Optimize_Mosaic");

      std::cout << "Enter function t_GeoCuts_Optimize_Mosaic" << std::endl;

      if ((!imOut.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if ((!imIn.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if ((!imGrad.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if ((!imMosaic.isAllocated())) {
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

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
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

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
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

            for (it = imIn.begin(), iend = imIn.end(); it != iend;
                 ++it) // for all pixels in imIn create a vertex
            {
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

            for (it = imIn.begin(), iend = imIn.end(); it != iend;
                 ++it) // for all pixels in imIn create a vertex and an edge
            {
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
     */
    template <class ImageIn, class ImageMosaic, class ImageMarker, class SE,
              class ImageOut>
    RES_T t_GeoCuts_Segment_Graph(const ImageIn &imIn,
                                  const ImageMosaic &imMosaic,
                                  const ImageMarker &imMarker, const SE &nl,
                                  ImageOut &imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_GeoCuts_Segment_Graph");

      std::cout << "Enter function optimize mosaic t_GeoCuts_Segment_Graph"
                << std::endl;

      if ((!imOut.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if ((!imIn.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if ((!imMosaic.isAllocated())) {
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

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
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

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
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

                double cost_diff = 0.01 * std::exp(-0.01 * (valee1 - valee2) *
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

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
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
     */
    template <class ImageIn, class ImageMosaic, class ImageMarker,
              typename _Beta, typename _Sigma, class SE, class ImageOut>
    RES_T t_MAP_MRF_Ising(const ImageIn &imIn, const ImageMosaic &imMosaic,
                          const ImageMarker &imMarker, const _Beta Beta,
                          const _Sigma Sigma, const SE &nl, ImageOut &imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_MAP_MRF_Ising");

      std::cout << "Enter function t_MAP_MRF_Ising" << std::endl;

      if ((!imOut.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if ((!imIn.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if ((!imMosaic.isAllocated())) {
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

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
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

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
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

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
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

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
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
     */
    template <class ImageIn, class ImageMosaic, class ImageMarker,
              typename _Beta, typename _Sigma, class SE, class ImageOut>
    RES_T t_MAP_MRF_edge_preserving(const ImageIn &imIn, const ImageMosaic &imMosaic,
                              const ImageMarker &imMarker, const _Beta Beta,
                              const _Sigma Sigma, const SE &nl, ImageOut &imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_MAP_MRF_edge_preserving");

      std::cout << "Enter function t_MAP_MRF_edge_preserving" << std::endl;

      if ((!imOut.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if ((!imIn.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if ((!imMosaic.isAllocated())) {
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

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
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

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
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

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
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

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
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
     */
    template <class ImageIn, class ImageMosaic, class ImageMarker,
              typename _Beta, typename _Sigma, class SE, class ImageOut>
    RES_T t_MAP_MRF_Potts(const ImageIn &imIn, const ImageMosaic &imMosaic,
                          const ImageMarker &imMarker, const _Beta Beta,
                          const _Sigma Sigma, const SE &nl, ImageOut &imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_MAP_MRF_Potts");

      std::cout << "Enter function t_MAP_MRF_Potts" << std::endl;

      if ((!imOut.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if ((!imIn.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if ((!imMosaic.isAllocated())) {
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

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
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

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
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

          val2 = (valee - meanlabel1) * (valee - meanlabel1) /
                 (2 * Sigmal * Sigmal);
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

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
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

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
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
          double val2 = (valee - meanlabel1) * (valee - meanlabel1) /
                        (2 * Sigmal * Sigmal);

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

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
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
      flow =
          boykov_kolmogorov_max_flow(g2, capacity2, residual_capacity2, rev2,
                                     &color2[0], indexmap2, vSource2, vSink2);
#else
      flow        = kolmogorov_max_flow(g2, capacity2, residual_capacity2, rev2,
                                 &color2[0], indexmap2, vSource2, vSink2);
#endif
      std::cout << "c  The total flow:" << std::endl;
      std::cout << "s " << flow << std::endl << std::endl;

      std::cout << "Source Label:" << color[vSource2] << std::endl;
      std::cout << "Sink  Label:" << color[vSink2] << std::endl;

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
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
} // namespace smil

#endif // GRAPHALGO_IMPL_T_HPP
