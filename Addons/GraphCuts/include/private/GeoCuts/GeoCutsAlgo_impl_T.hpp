#ifndef GEOCUTSALGO_IMPL_T_HPP
#define GEOCUTSALGO_IMPL_T_HPP
#include <morphee/selement/include/selementNeighborList.hpp>
#include <morphee/selement/include/private/selementNeighborhood_T.hpp>
#include <morphee/image/include/private/image_T.hpp>
#include <morphee/image/include/imageUtils.hpp>

#include <boost/config.hpp>
#include <boost/utility.hpp>            // for boost::tie
#include <boost/graph/graph_traits.hpp> // for boost::graph_traits
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/prim_minimum_spanning_tree.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/johnson_all_pairs_shortest.hpp>

#include <boost/version.hpp>
#if BOOST_VERSION >= 104700
#include <boost/graph/boykov_kolmogorov_max_flow.hpp>
#elif BOOST_VERSION >= 103500
#include <boost/graph/kolmogorov_max_flow.hpp>
#else
#include "../boost_ext/kolmogorov_max_flow.hpp"
#endif

#include "../boost_ext/kolmogorov_max_flow_min_cost.hpp" // FROM STAWIASKI JAN 2012
//#include "../boost_ext/maximum_spanning_tree.hpp"
//#include "../boost_ext/boost_compare.hpp"  //STAWIASKI JAN2012 commented, why?
#include <boost/graph/connected_components.hpp>

#include <morphee/image/include/private/image_T.hpp>
#include <morphee/image/include/private/imageManipulation_T.hpp>
#include <morphee/selement/include/selementNeighborList.hpp>
#include <morphee/selement/include/private/selementNeighborhood_T.hpp>
#include <morphee/morphoBase/include/private/morphoLabel_T.hpp>
#include <morphee/morphoBase/include/private/morphoLabel2_T.hpp>
#include <morphee/morphoBase/include/private/morphoGraphs_T.hpp>
#include <morphee/morphoBase/include/private/morphoHierarch_T.hpp>
#include <morphee/graph/include/private/graphProjection_T.hpp>
#include <morphee/graph/include/graph.hpp>
//#include <graphs/MorphoGraph/include/Morpho_Graph.hpp>// Required for
//t_Order_Edges_Weights
#include <graphs/MorphoGraph/include/Morpho_Graph_T.hpp> // Required for t_Order_Edges_Weights

#include <vector>

// ##################################################
// BEGIN FROM STAWIASKI JAN 2012
// ##################################################

#include <math.h>
#define M_PI 3.14159265358979323846
#define INFINI_POSITIF std::numeric_limits<double>::max)()
#define _SECURE_SCL 0
#include <stdio.h>

typedef struct {
  float x;
  float y;
  float p;
} morceau;

typedef std::list<morceau> affine_par_morceaux;

// ##################################################
// END FROM STAWIASKI JAN 2012
// ##################################################
//#include <morphee/common/include/commonTypesOperator.hpp>

namespace morphee
{
  namespace graphalgo
  {
    template <class BoostGraph>
    const BoostGraph t_CopyGraph(const BoostGraph &graphIn);
    template <class Graph>
    RES_C t_LabelConnectedComponent(const Graph &GIn, Graph &Gout);
    template <class Graph>
    RES_C t_LabelConnectedComponent(const Graph &GIn, Graph &Gout, int *num);
    // ##################################################
    // BEGIN FROM STAWIASKI JAN 2012
    // ##################################################

    //

    template <class ImageWs, class ImageIn, class ImageGrad, typename _alpha1,
              typename _alpha2, typename _alpha3, class SE, class BoostGraph>
    RES_C t_TreeReweighting_old(const ImageWs &imWs, const ImageIn &imIn,
                                const ImageGrad &imGrad, BoostGraph &Treein,
                                const _alpha1 alpha1, const _alpha2 alpha2,
                                const _alpha3 alpha3, const SE &nl,
                                BoostGraph &Tree_out)
    {
      MORPHEE_ENTER_FUNCTION("t_TreeReweighting");

      if ((!imWs.isAllocated())) {
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

      // common image iterator
      typename ImageWs::const_iterator it, iend;
      morphee::selement::Neighborhood<SE, ImageIn> neighb(imIn, nl);
      typename morphee::selement::Neighborhood<SE, ImageIn>::iterator nit, nend;
      offset_t o0;
      offset_t o1;

      typename BoostGraph::EdgeIterator ed_it, ed_end;
      typename BoostGraph::EdgeIterator ed_it2, ed_end2;
      typename BoostGraph::EdgeIterator ed_it3, ed_end3;
      typename BoostGraph::EdgeIterator ed_it4, ed_end4;

      typename BoostGraph::VertexIterator v_it, v_end;

      typename BoostGraph::VertexProperty vdata1, vdata2, vdata11, vdata22;
      typename BoostGraph::VertexProperty label1, label2;

      bool in1;
      int numVert  = 0;
      int numEdges = 0;
      typename BoostGraph::EdgeDescriptor e1, e2;
      typename BoostGraph::VertexDescriptor vs, vt;
      typename BoostGraph::EdgeProperty tmp, tmp2, tmp3;

      std::vector<double> val_edges;
      std::vector<double> val_edges2;
      val_edges.push_back(0.0);

      std::cout << "build Region Adjacency Graph" << std::endl;

      for (it = imWs.begin(), iend = imWs.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o0      = it.getOffset();
        int val = imWs.pixelFromOffset(o0);

        if (val > numVert) {
          numVert = val;
        }
      }

      std::cout << "Compute mean and quaderror" << std::endl;
      std::vector<double> mean_values;
      std::vector<double> number_of_values;
      std::vector<double> quad_error;
      std::vector<double> min_gradient_values;
      std::vector<double> max_gradient_values;
      std::vector<double> area_values;

      int **histogram;
      histogram = new int *[numVert];

      float *histogram_region1;
      histogram_region1 = new float[255];

      float *histogram_region2;
      histogram_region2 = new float[255];

      float *cumul_histogram_region1;
      cumul_histogram_region1 = new float[255];

      float *cumul_histogram_region2;
      cumul_histogram_region2 = new float[255];

      for (int dim_allouee0 = 0; dim_allouee0 < numVert; ++dim_allouee0) {
        histogram[dim_allouee0] = new int[255];
      }

      for (int dim_allouee0 = 0; dim_allouee0 < numVert; ++dim_allouee0) {
        for (int dim_allouee1 = 0; dim_allouee1 < 255; ++dim_allouee1) {
          histogram[dim_allouee0][dim_allouee1] = 0;
        }
      }

      //// INIT VALUES
      for (int i = 0; i < numVert; i++) {
        mean_values.push_back(0);
        number_of_values.push_back(0);
        quad_error.push_back(0);
        min_gradient_values.push_back(1000000000000);
        max_gradient_values.push_back(0);
        area_values.push_back(0);
      }

      //// COMPUTE MEAN VALUES, AREA, MIN AND MAX
      for (it = imWs.begin(), iend = imWs.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o1                  = it.getOffset();
        int val             = imWs.pixelFromOffset(o1);
        double val_image    = imIn.pixelFromOffset(o1);
        double val_gradient = imGrad.pixelFromOffset(o1);

        mean_values[val - 1]      = mean_values[val - 1] + val_image;
        number_of_values[val - 1] = number_of_values[val - 1] + 1.0;

        if (val_gradient < min_gradient_values[val - 1])
          min_gradient_values[val - 1] = val_gradient;
        if (val_gradient > max_gradient_values[val - 1])
          max_gradient_values[val - 1] = val_gradient;

        area_values[val - 1]     = area_values[val - 1] + 1;
        int vald                 = (int) (val_image);
        histogram[val - 1][vald] = histogram[val - 1][vald] + 1;
      }

      //// SET MEAN VALUES
      for (int i = 0; i < numVert; i++) {
        mean_values[i] = mean_values[i] / number_of_values[i];
      }

      //// COMPUTE QUADRATIC ERROR IN EACH REGION
      float max_quad_val = 0.0f;
      for (it = imWs.begin(), iend = imWs.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1               = it.getOffset();
        int val          = imWs.pixelFromOffset(o1);
        double val_image = imIn.pixelFromOffset(o1);

        quad_error[val - 1] =
            (double) quad_error[val - 1] +
            std::pow(std::abs(val_image - (double) mean_values[val - 1]), 2);

        if (quad_error[val - 1] > max_quad_val)
          max_quad_val = quad_error[val - 1];
      }

      std::cout << "build Region Adjacency Graph Vertices" << std::endl;
      std::cout << "number of Vertices:" << numVert << std::endl;

      // create some temp graphs
      Tree_out = morphee::graph::CommonGraph32(numVert);

      BoostGraph Gout   = morphee::graph::CommonGraph32(numVert);
      BoostGraph Gout_t = morphee::graph::CommonGraph32(numVert);
      BoostGraph Gtemp  = morphee::graph::CommonGraph32(numVert);

      BoostGraph Tree_temp  = morphee::graph::CommonGraph32(numVert);
      BoostGraph Tree_temp2 = morphee::graph::CommonGraph32(numVert);

      BoostGraph Gclassic = morphee::graph::CommonGraph32(numVert);

      morphee::morphoBase::t_NeighborhoodGraphFromMosaic(imWs, nl, Gclassic);

      // project area of regions on the graph nodes
      ImageWs ImTempSurfaces = imWs.getSame();
      morphee::morphoBase::t_ImLabelFlatZonesWithArea(imWs, nl, ImTempSurfaces);
      morphee::graph::t_ProjectMarkersOnGraph(ImTempSurfaces, imWs, Gclassic);

      morphee::morphoBase::t_ImLabelFlatZonesWithVolume(imWs, imGrad, nl,
                                                        ImTempSurfaces);
      morphee::graph::t_ProjectMarkersOnGraph(ImTempSurfaces, imWs, Gout);

      std::cout << "build Region Adjacency Graph Edges" << std::endl;

      double volume       = 0.0;
      double local_volume = 0.0;
      double area_total   = 0.0;

      //// SET VERTEX PROPERTIES TO AREA AND VOLUME
      for (boost::tie(v_it, v_end) = boost::vertices(Gout.getBoostGraph());
           v_it != v_end; ++v_it) {
        local_volume =
            area_values[(int) *v_it] * (max_gradient_values[(int) *v_it] -
                                        min_gradient_values[(int) *v_it]);

        Gout.setVertexData(*v_it, (float) local_volume);
        volume = volume + (double) local_volume;

        // Gout.vertexData(*v_it, &vdata1);
        // volume = volume + (double) vdata1;

        Gclassic.vertexData(*v_it, &vdata1);
        area_total = area_total + (double) vdata1;
      }

      //// COMPUTE MEAN GRADIENT VALUES ALONG THE REGIONS
      for (it = imWs.begin(), iend = imWs.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1      = it.getOffset();
        int val = imWs.pixelFromOffset(o1);

        if (val > 0) {
          neighb.setCenter(o1);

          for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
            const offset_t o2 = nit.getOffset();
            int val2          = imWs.pixelFromOffset(o2);

            if (o2 > o1 || o1 < o2) {
              if (val != val2) {
                boost::tie(e1, in1) =
                    boost::edge(val - 1, val2 - 1, Gout.getBoostGraph());
                boost::tie(e2, in1) =
                    boost::edge(val - 1, val2 - 1, Gtemp.getBoostGraph());

                double val3 = imGrad.pixelFromOffset(o1);
                double val4 = imGrad.pixelFromOffset(o2);
                double maxi = std::max(val3, val4);

                if (in1 == 0) {
                  numEdges++;
                  Gout.addEdge(val - 1, val2 - 1, maxi);
                  Gtemp.addEdge(val - 1, val2 - 1, 1);
                } else {
                  Gout.edgeWeight(e1, &tmp);
                  Gout.setEdgeWeight(e1, tmp + maxi);

                  Gtemp.edgeWeight(e2, &tmp2);
                  Gtemp.setEdgeWeight(e2, tmp2 + 1);
                }
              }
            }
          }
        }
      }

      std::cout << "number of Edges : " << numEdges << std::endl;

      Gout_t                      = t_CopyGraph(Gout);
      boost::tie(ed_it2, ed_end2) = boost::edges(Gtemp.getBoostGraph());
      boost::tie(ed_it3, ed_end3) = boost::edges(Gclassic.getBoostGraph());
      boost::tie(ed_it4, ed_end4) = boost::edges(Gout_t.getBoostGraph());
      float current_max_value     = 0.0f;

      // Weights the graph Gout with mean gradient value along boundary
      for (boost::tie(ed_it, ed_end) = boost::edges(Gout.getBoostGraph());
           ed_it != ed_end, ed_it2 != ed_end2, ed_it3 != ed_end3,
                             ed_it4 != ed_end4;
           ++ed_it, ++ed_it2, ++ed_it3, ++ed_it4) {
        Gout.edgeWeight(*ed_it, &tmp);    // GRADIENT SUM
        Gtemp.edgeWeight(*ed_it2, &tmp2); // REGION BOUNDARY LENGTH
        Gout_t.setEdgeWeight(*ed_it4, ((double) tmp2)); // length
      }

      // Gtemp is the min spanning tree of Gclassic weighted with pass-values
      // Gtemp = morphee::MinimumSpanningTreeFromGraph(Gclassic);
      // t_AverageLinkageTree(imWs, imGrad, nl, Gtemp) ;

      Gtemp = t_CopyGraph(Treein);

      for (boost::tie(ed_it, ed_end) = boost::edges(Gtemp.getBoostGraph());
           ed_it != ed_end; ++ed_it) {
        Gtemp.edgeWeight(*ed_it, &tmp); // PASS VALUE
        val_edges.push_back(tmp);
      }

      std::vector<typename BoostGraph::EdgeDescriptor> removed_edges;

      Tree_out   = t_CopyGraph(Gtemp);
      Tree_temp  = t_CopyGraph(Gtemp);
      Tree_temp2 = t_CopyGraph(Gtemp);

      // sort edges weights to explore the hierarchy
      std::cout << "sort" << std::endl;
      std::sort(val_edges.begin(), val_edges.end(), std::less<double>());

      double last_edge_value = val_edges.back();
      double last_analyzed   = last_edge_value;

      while (val_edges.size() > 1) {
        std::cout << last_edge_value << std::endl;

        // remove edge of maximal weight
        for (boost::tie(ed_it, ed_end) = boost::edges(Gtemp.getBoostGraph());
             ed_it != ed_end; ++ed_it) {
          Gtemp.edgeWeight(*ed_it, &tmp);

          if (tmp == last_edge_value) { // check is  current max weight
            vs                  = Gtemp.edgeSource(*ed_it);
            vt                  = Gtemp.edgeTarget(*ed_it);
            boost::tie(e1, in1) = boost::edge(vs, vt, Gtemp.getBoostGraph());
            removed_edges.push_back(e1);
            Tree_temp.removeEdge(vs, vt);
          }
        }

        // label trees
        t_LabelConnectedComponent(Tree_temp, Tree_temp2);

        // local variable for mean values and areas of regions
        float mean_value       = 0.0f;
        float number_of_points = 0.0f;
        float volume1          = 0.0f;
        float volume2          = 0.0f;

        float volume11 = 0.0f;
        float volume22 = 0.0f;

        float area1             = 0.0f;
        float area2             = 0.0f;
        float quad1             = 0.0f;
        float quad2             = 0.0f;
        float mean_val_1        = 0.0f;
        float mean_val_2        = 0.0f;
        float nb_val_1          = 0.0f;
        float nb_val_2          = 0.0f;
        float mean_value_1      = 0.0f;
        float mean_value_2      = 0.0f;
        float number_of_points1 = 0.0f;
        float number_of_points2 = 0.0f;
        float dist_histo        = 0.0f;

        // go through removed edges and look caracteristics of regions connected
        // to it
        while (removed_edges.size() > 0) {
          volume1          = 0.0f;
          volume2          = 0.0f;
          area1            = 0.0f;
          area2            = 0.0f;
          quad1            = 0.0f;
          quad2            = 0.0f;
          number_of_points = 0.0f;

          mean_value_1      = 0.0f;
          mean_value_2      = 0.0f;
          number_of_points1 = 0.0f;
          number_of_points2 = 0.0f;

          mean_value = 0.0f;
          mean_val_1 = 0.0f;
          mean_val_2 = 0.0f;
          nb_val_1   = 0.0f;
          nb_val_2   = 0.0f;

          volume11 = 0.0f;
          volume22 = 0.0f;

          dist_histo = 0.0f;

          for (int dim_allouee1 = 0; dim_allouee1 < 255; ++dim_allouee1) {
            histogram_region1[dim_allouee1]       = 0;
            histogram_region2[dim_allouee1]       = 0;
            cumul_histogram_region1[dim_allouee1] = 0;
            cumul_histogram_region2[dim_allouee1] = 0;
          }

          e1 = removed_edges.back();
          removed_edges.pop_back();

          // Get label of the regions
          Tree_temp2.vertexData(Tree_temp2.edgeSource(e1), &label1);
          Tree_temp2.vertexData(Tree_temp2.edgeTarget(e1), &label2);

          boost::tie(ed_it4, ed_end4) = boost::edges(Gout_t.getBoostGraph());

          for (boost::tie(ed_it, ed_end) = boost::edges(Gout.getBoostGraph());
               ed_it != ed_end, ed_it4 != ed_end4; ++ed_it, ++ed_it4) {
            vs = Gout.edgeSource(*ed_it);
            vt = Gout.edgeTarget(*ed_it);

            // labels of the regions
            Tree_temp2.vertexData(vs, &vdata1);
            Tree_temp2.vertexData(vt, &vdata2);

            // compute avergage mean gradient along regions
            if ((vdata1 == label1 && vdata2 == label2) ||
                (vdata1 == label2 && vdata2 == label1)) {
              Gout.edgeWeight(*ed_it, &tmp);
              Gout_t.edgeWeight(*ed_it4, &tmp2);

              mean_value       = mean_value + (float) tmp;
              number_of_points = number_of_points + (float) tmp2;
            }

            // compute avergage mean gradient along regions 1
            if ((vdata1 == label1 && vdata2 != label1) ||
                (vdata2 == label1 && vdata1 != label1)) {
              Gout.edgeWeight(*ed_it, &tmp);
              Gout_t.edgeWeight(*ed_it4, &tmp2);

              mean_value_1      = mean_value_1 + (float) tmp;
              number_of_points1 = number_of_points1 + (float) tmp2;
            }

            // compute avergage mean gradient along regions 2
            if ((vdata1 == label2 && vdata2 != label2) ||
                (vdata2 == label2 && vdata1 != label2)) {
              Gout.edgeWeight(*ed_it, &tmp);
              Gout_t.edgeWeight(*ed_it4, &tmp2);

              mean_value_2      = mean_value_2 + (float) tmp;
              number_of_points2 = number_of_points2 + (float) tmp2;
            }
          }

          // copying the properties of each vertex
          for (boost::tie(v_it, v_end) = boost::vertices(Gout.getBoostGraph());
               v_it != v_end; ++v_it) {
            // Get label of the regions
            Tree_temp2.vertexData(*v_it, &vdata1);
            Gout.vertexData(*v_it, &vdata11);

            Tree_out.setVertexData(*v_it, (float) alpha1 * vdata11 +
                                              (float) alpha3 *
                                                  quad_error[(int) *v_it]);

            // compute area of each regions
            if (vdata1 == label1) {
              volume1 =
                  volume1 +
                  (float) vdata11; // NODES OF GOUT ARE WEIGHTED WITH VOLUME
              Gclassic.vertexData(
                  *v_it, &vdata11); // NODES OF GCLASSIC ARE WEIGHTED WITH AREA
              area1      = area1 + (float) vdata11;
              mean_val_1 = mean_val_1 + mean_values[(int) *v_it];
              nb_val_1   = nb_val_1 + 1.0f;

              for (int dim_allouee1 = 0; dim_allouee1 < 255; ++dim_allouee1) {
                histogram_region1[dim_allouee1] =
                    histogram_region1[dim_allouee1] +
                    histogram[(int) *v_it][dim_allouee1];
              }
            }
            if (vdata1 == label2) {
              volume2 =
                  volume2 +
                  (float) vdata11; // NODES OF GOUT ARE WEIGHTED WITH VOLUME
              Gclassic.vertexData(
                  *v_it, &vdata11); // NODES OF GCLASSIC ARE WEIGHTED WITH AREA
              area2      = area2 + (float) vdata11;
              mean_val_2 = mean_val_2 + mean_values[(int) *v_it];
              nb_val_2   = nb_val_2 + 1.0f;

              for (int dim_allouee1 = 0; dim_allouee1 < 255; ++dim_allouee1) {
                histogram_region2[dim_allouee1] =
                    histogram_region2[dim_allouee1] +
                    histogram[(int) *v_it][dim_allouee1];
              }
            }
          }

          for (int dim_allouee1 = 0; dim_allouee1 < 255; ++dim_allouee1) {
            for (int dim_allouee0 = 0; dim_allouee0 <= dim_allouee1;
                 ++dim_allouee0) {
              cumul_histogram_region1[dim_allouee1] =
                  cumul_histogram_region1[dim_allouee1] +
                  (float) histogram_region1[dim_allouee0] / (float) area1;
              cumul_histogram_region2[dim_allouee1] =
                  cumul_histogram_region2[dim_allouee1] +
                  (float) histogram_region2[dim_allouee0] / (float) area2;
            }
          }

          for (int dim_allouee1 = 0; dim_allouee1 < 255; ++dim_allouee1) {
            dist_histo =
                dist_histo + std::pow(cumul_histogram_region1[dim_allouee1] -
                                          cumul_histogram_region2[dim_allouee1],
                                      2);
          }

          // copying the properties of each vertex
          for (boost::tie(v_it, v_end) = boost::vertices(Gout.getBoostGraph());
               v_it != v_end; ++v_it) {
            // Get label of the regions
            Tree_temp2.vertexData(*v_it, &vdata1);

            // compute area of each regions
            if (vdata1 == label1) {
              quad1 = quad1 +
                      (mean_values[(int) *v_it] - mean_val_1 / nb_val_1) *
                          (mean_values[(int) *v_it] - mean_val_1 / nb_val_1);
            }
            if (vdata1 == label2) {
              quad2 = quad2 +
                      (mean_values[(int) *v_it] - mean_val_2 / nb_val_2) *
                          (mean_values[(int) *v_it] - mean_val_2 / nb_val_2);
            }
          }

          quad1 = quad1 / nb_val_1;
          quad2 = quad2 / nb_val_2;

          volume11 = area1 * mean_value_1 / number_of_points1;
          volume22 = area2 * mean_value_2 / number_of_points2;

          float probability_volume =
              (1.0 - std::pow(1.0 - volume11 / ((double) volume), 50) -
               std::pow(1.0 - volume22 / ((double) volume), 50) +
               std::pow(1.0 - (volume11 + volume22) / ((double) volume), 50));

          // float probability_volume = ( 1.0 - std::pow( 1.0 -
          // volume1/((double) volume)  , 20 ) - std::pow( 1.0 -
          // volume2/((double) volume)  , 20 ) + std::pow( 1.0 -
          // (volume1+volume2)/((double) volume)  , 20 ) );
          float probability_area =
              (1.0 - std::pow(1.0 - area1 / ((double) area_total), 20) -
               std::pow(1.0 - area2 / ((double) area_total), 20) +
               std::pow(1.0 - (area1 + area2) / ((double) area_total), 20));
          float diff_mean =
              std::abs((mean_val_2 / nb_val_2) - (mean_val_1 / nb_val_1));

          probability_volume = std::min(probability_volume, 0.05f);
          // probability_area = std::min( probability_area , 0.5f ) ;

          // GET EDGE IN TRee_OUt
          boost::tie(e2, in1) =
              boost::edge(Tree_out.edgeSource(e1), Tree_out.edgeTarget(e1),
                          Tree_out.getBoostGraph());
          float val_gradient_mean_t =
              std::min(((float) last_edge_value) / (65535.0f), 0.7f);

          if (number_of_points > 0.0f) {
            // float value_test = 100000.0f * std::pow(dist_histo,(float)
            // alpha2) * std::pow( ((float) mean_value/ (float) number_of_points
            // )/( 65535.0f ) , (float) alpha1 ) * std::pow( probability_volume ,
            // (float) alpha3 ) ;
            float value_test = 2000000.0f *
                               std::pow(val_gradient_mean_t, (float) alpha1) *
                               std::pow(probability_volume, (float) alpha3);

            Tree_out.setEdgeWeight(e2, value_test);

            if (value_test > current_max_value)
              current_max_value = value_test;

          } else {
            Tree_out.setEdgeWeight(e2, 0.0f);
          }
        }

        while (last_edge_value == last_analyzed) {
          last_edge_value = val_edges.back();
          val_edges.pop_back();
        }
        last_analyzed = last_edge_value;
      }

      std::cout << "current_max_value" << current_max_value << std::endl;

      for (boost::tie(ed_it, ed_end) = boost::edges(Tree_out.getBoostGraph());
           ed_it != ed_end; ++ed_it) {
        Tree_out.edgeWeight(*ed_it, &tmp);
        float value_test = 65535.0f * std::pow((float) tmp / current_max_value,
                                               (float) alpha2);
        Tree_out.setEdgeWeight(*ed_it, value_test);
      }

      std::cout << "alpha2" << alpha2 << std::endl;

      // morphee::Morpho_Graph::t_Order_Edges_Weights(Tree_out, Tree_temp);
      // boost::tie(ed_it2, ed_end2)=boost::edges(Tree_temp.getBoostGraph());

      // for (boost::tie(ed_it, ed_end)=boost::edges(Tree_out.getBoostGraph()) ;
      // ed_it != ed_end,  ed_it2 != ed_end2 ; ++ed_it, ++ed_it2)
      //{
      //	Tree_temp.edgeWeight(*ed_it2,&tmp);
      //	Tree_out.setEdgeWeight(*ed_it, (float) tmp );
      //}

      return RES_OK;
    }

    template <class ImageWs, class ImageIn, class ImageGrad, typename _alpha1,
              typename _alpha2, typename _alpha3, class SE, class BoostGraph>
    RES_C t_TreeReweighting2(const ImageWs &imWs, const ImageIn &imIn,
                             const ImageGrad &imGrad, BoostGraph &Treein,
                             const _alpha1 alpha1, const _alpha2 alpha2,
                             const _alpha3 alpha3, const SE &nl,
                             BoostGraph &Tree_out)
    {
      MORPHEE_ENTER_FUNCTION("t_TreeReweighting");

      if ((!imWs.isAllocated())) {
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

      // common image iterator
      typename ImageWs::const_iterator it, iend;
      morphee::selement::Neighborhood<SE, ImageIn> neighb(imIn, nl);
      typename morphee::selement::Neighborhood<SE, ImageIn>::iterator nit, nend;
      offset_t o0;
      offset_t o1;

      typename BoostGraph::EdgeIterator ed_it, ed_end;
      typename BoostGraph::EdgeIterator ed_it2, ed_end2;
      typename BoostGraph::EdgeIterator ed_it3, ed_end3;
      typename BoostGraph::EdgeIterator ed_it4, ed_end4;

      typename BoostGraph::VertexIterator v_it, v_end;

      typename BoostGraph::VertexProperty vdata1, vdata2, vdata11, vdata22;
      typename BoostGraph::VertexProperty label1, label2;

      bool in1;
      int numVert  = 0;
      int numEdges = 0;
      typename BoostGraph::EdgeDescriptor e1, e2;
      typename BoostGraph::VertexDescriptor vs, vt;
      typename BoostGraph::EdgeProperty tmp, tmp2, tmp3;

      std::vector<double> val_edges;
      std::vector<double> val_edges2;
      val_edges.push_back(0.0);

      std::cout << "build Region Adjacency Graph" << std::endl;

      for (it = imWs.begin(), iend = imWs.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o0      = it.getOffset();
        int val = imWs.pixelFromOffset(o0);

        if (val > numVert) {
          numVert = val;
        }
      }

      std::cout << "Compute mean and quaderror" << std::endl;
      std::vector<double> mean_values;
      std::vector<double> number_of_values;
      std::vector<double> quad_error;
      std::vector<double> min_gradient_values;
      std::vector<double> max_gradient_values;
      std::vector<double> area_values;

      int **histogram; // image histogram
      histogram = new int *[numVert];

      float *histogram_region1; // region 1 histogram
      histogram_region1 = new float[255];

      float *histogram_region2; // region 2 histogram
      histogram_region2 = new float[255];

      float *cumul_histogram_region1; // cummulated region 1 histogram
      cumul_histogram_region1 = new float[255];

      float *cumul_histogram_region2; // cummulated region 2 histogram
      cumul_histogram_region2 = new float[255];

      for (int dim_allouee0 = 0; dim_allouee0 < numVert; ++dim_allouee0) {
        histogram[dim_allouee0] = new int[255];
      }

      for (int dim_allouee0 = 0; dim_allouee0 < numVert; ++dim_allouee0) {
        for (int dim_allouee1 = 0; dim_allouee1 < 255; ++dim_allouee1) {
          histogram[dim_allouee0][dim_allouee1] = 0;
        }
      }

      //// INIT VALUES TO ZERO
      for (int i = 0; i < numVert; i++) {
        mean_values.push_back(0);
        number_of_values.push_back(0);
        quad_error.push_back(0);
        min_gradient_values.push_back(1000000000000);
        max_gradient_values.push_back(0);
        area_values.push_back(0);
      }

      //// COMPUTE MEAN VALUES, AREA, HISTOGRAM, MIN AND MAX
      for (it = imWs.begin(), iend = imWs.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o1                  = it.getOffset();
        int val             = imWs.pixelFromOffset(o1);
        double val_image    = imIn.pixelFromOffset(o1);
        double val_gradient = imGrad.pixelFromOffset(o1);

        mean_values[val - 1]      = mean_values[val - 1] + val_image;
        number_of_values[val - 1] = number_of_values[val - 1] + 1.0;

        if (val_gradient < min_gradient_values[val - 1])
          min_gradient_values[val - 1] = val_gradient;
        if (val_gradient > max_gradient_values[val - 1])
          max_gradient_values[val - 1] = val_gradient;

        area_values[val - 1]     = area_values[val - 1] + 1;
        int vald                 = (int) (val_image);
        histogram[val - 1][vald] = histogram[val - 1][vald] + 1;
      }

      //// SET MEAN VALUES
      for (int i = 0; i < numVert; i++) {
        mean_values[i] = mean_values[i] / number_of_values[i];
      }

      //// COMPUTE QUADRATIC ERROR (COMPARED TO THE MEAN VALUE) IN EACH REGION
      float max_quad_val = 0.0f;
      for (it = imWs.begin(), iend = imWs.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1               = it.getOffset();
        int val          = imWs.pixelFromOffset(o1);
        double val_image = imIn.pixelFromOffset(o1);

        quad_error[val - 1] =
            (double) quad_error[val - 1] +
            std::pow(std::abs(val_image - (double) mean_values[val - 1]), 2);

        if (quad_error[val - 1] > max_quad_val)
          max_quad_val = quad_error[val - 1];
      }

      std::cout << "build Region Adjacency Graph Vertices" << std::endl;
      std::cout << "number of Vertices:" << numVert << std::endl;

      // create some temp graphs
      Tree_out = morphee::graph::CommonGraph32(numVert);

      BoostGraph Gout   = morphee::graph::CommonGraph32(numVert);
      BoostGraph Gout_t = morphee::graph::CommonGraph32(numVert);
      BoostGraph Gtemp  = morphee::graph::CommonGraph32(numVert);

      BoostGraph Tree_temp  = morphee::graph::CommonGraph32(numVert);
      BoostGraph Tree_temp2 = morphee::graph::CommonGraph32(numVert);

      // GRAPH WHOSE NODES ARE WEIGHTED WITH AREA
      BoostGraph G_area = morphee::graph::CommonGraph32(numVert);
      morphee::morphoBase::t_NeighborhoodGraphFromMosaic(imWs, nl, G_area);

      // project area of regions on the graph nodes
      ImageWs ImTempSurfaces = imWs.getSame();
      morphee::morphoBase::t_ImLabelFlatZonesWithArea(imWs, nl, ImTempSurfaces);
      morphee::graph::t_ProjectMarkersOnGraph(ImTempSurfaces, imWs, G_area);

      // GRAPH WHOSE NODES ARE WEIGHTED WITH VOLUME
      morphee::morphoBase::t_ImLabelFlatZonesWithVolume(imWs, imGrad, nl,
                                                        ImTempSurfaces);
      morphee::graph::t_ProjectMarkersOnGraph(ImTempSurfaces, imWs, Gout);

      std::cout << "build Region Adjacency Graph Edges" << std::endl;

      double volume       = 0.0;
      double local_volume = 0.0;
      double area_total   = 0.0;

      //// SET VERTEX PROPERTIES TO AREA AND VOLUME
      for (boost::tie(v_it, v_end) = boost::vertices(Gout.getBoostGraph());
           v_it != v_end; ++v_it) {
        local_volume =
            area_values[(int) *v_it] * (max_gradient_values[(int) *v_it] -
                                        min_gradient_values[(int) *v_it]);

        Gout.vertexData(*v_it, &vdata1);
        volume = volume + (double) vdata1;

        G_area.vertexData(*v_it, &vdata1);
        area_total = area_total + (double) vdata1;
      }

      //// COMPUTE MEAN GRADIENT VALUES ALONG THE REGIONS
      for (it = imWs.begin(), iend = imWs.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1      = it.getOffset();
        int val = imWs.pixelFromOffset(o1);

        if (val > 0) {
          neighb.setCenter(o1);

          for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
            const offset_t o2 = nit.getOffset();
            int val2          = imWs.pixelFromOffset(o2);

            if (o2 > o1) {
              if (val != val2) {
                boost::tie(e1, in1) =
                    boost::edge(val - 1, val2 - 1, Gout.getBoostGraph());
                boost::tie(e2, in1) =
                    boost::edge(val - 1, val2 - 1, Gtemp.getBoostGraph());

                double val3 = imGrad.pixelFromOffset(o1);
                double val4 = imGrad.pixelFromOffset(o2);
                double maxi = std::max(val3, val4);

                if (in1 == 0) {
                  numEdges++;
                  Gout.addEdge(val - 1, val2 - 1, maxi);
                  Gtemp.addEdge(val - 1, val2 - 1, 1);
                } else {
                  Gout.edgeWeight(e1, &tmp);
                  Gout.setEdgeWeight(e1, tmp + maxi);

                  Gtemp.edgeWeight(e2, &tmp2);
                  Gtemp.setEdgeWeight(e2, tmp2 + 1);
                }
              }
            }
          }
        }
      }

      std::cout << "number of Edges : " << numEdges << std::endl;

      Gout_t                      = t_CopyGraph(Gout);
      boost::tie(ed_it2, ed_end2) = boost::edges(Gtemp.getBoostGraph());
      boost::tie(ed_it3, ed_end3) = boost::edges(G_area.getBoostGraph());
      boost::tie(ed_it4, ed_end4) = boost::edges(Gout_t.getBoostGraph());
      float current_max_value     = 0.0f;

      // Weights the graph Gout with mean gradient value along boundary
      for (boost::tie(ed_it, ed_end) = boost::edges(Gout.getBoostGraph());
           ed_it != ed_end, ed_it2 != ed_end2, ed_it3 != ed_end3,
                             ed_it4 != ed_end4;
           ++ed_it, ++ed_it2, ++ed_it3, ++ed_it4) {
        Gout.edgeWeight(*ed_it, &tmp);    // GRADIENT SUM
        Gtemp.edgeWeight(*ed_it2, &tmp2); // REGION BOUNDARY LENGTH
        Gout_t.setEdgeWeight(*ed_it4, ((double) tmp2)); // length
      }

      // Gtemp is the min spanning tree of G_area weighted with pass-values
      // Gtemp = morphee::MinimumSpanningTreeFromGraph(G_area);
      // t_AverageLinkageTree(imWs, imGrad, nl, Gtemp) ;

      Gtemp = t_CopyGraph(Treein);

      for (boost::tie(ed_it, ed_end) = boost::edges(Gtemp.getBoostGraph());
           ed_it != ed_end; ++ed_it) {
        Gtemp.edgeWeight(*ed_it, &tmp); // PASS VALUE
        val_edges.push_back(tmp);
      }

      std::vector<typename BoostGraph::EdgeDescriptor> removed_edges;

      Tree_out   = t_CopyGraph(Gtemp);
      Tree_temp  = t_CopyGraph(Gtemp);
      Tree_temp2 = t_CopyGraph(Gtemp);

      // sort edges weights to explore the hierarchy
      std::cout << "sort" << std::endl;
      std::sort(val_edges.begin(), val_edges.end(), std::less<double>());

      double last_edge_value = val_edges.back();
      double last_analyzed   = last_edge_value;

      while (val_edges.size() > 1) {
        std::cout << last_edge_value << std::endl;

        // remove edge of maximal weight
        for (boost::tie(ed_it, ed_end) = boost::edges(Gtemp.getBoostGraph());
             ed_it != ed_end; ++ed_it) {
          Gtemp.edgeWeight(*ed_it, &tmp);

          if (tmp == last_edge_value) { // check is  current max weight
            vs                  = Gtemp.edgeSource(*ed_it);
            vt                  = Gtemp.edgeTarget(*ed_it);
            boost::tie(e1, in1) = boost::edge(vs, vt, Gtemp.getBoostGraph());
            removed_edges.push_back(e1);
            Tree_temp.removeEdge(vs, vt);
          }
        }

        // label trees
        int nb_of_connected = t_LabelConnectedComponent(Tree_temp, Tree_temp2);

        // std::cout<<" number of connected components =
        // "<<nb_of_connected<<std::endl;

        // local variable for mean values and areas of regions
        float mean_value       = 0.0f;
        float number_of_points = 0.0f;
        float volume1          = 0.0f;
        float volume2          = 0.0f;

        float volume11 = 0.0f;
        float volume22 = 0.0f;

        float area1             = 0.0f;
        float area2             = 0.0f;
        float quad1             = 0.0f;
        float quad2             = 0.0f;
        float mean_val_1        = 0.0f;
        float mean_val_2        = 0.0f;
        float nb_val_1          = 0.0f;
        float nb_val_2          = 0.0f;
        float mean_value_1      = 0.0f;
        float mean_value_2      = 0.0f;
        float number_of_points1 = 0.0f;
        float number_of_points2 = 0.0f;
        float dist_histo        = 0.0f;

        // go through removed edges and look caracteristics of regions connected
        // to it
        while (removed_edges.size() > 0) {
          volume1          = 0.0f;
          volume2          = 0.0f;
          area1            = 0.0f;
          area2            = 0.0f;
          quad1            = 0.0f;
          quad2            = 0.0f;
          number_of_points = 0.0f;

          mean_value_1      = 0.0f;
          mean_value_2      = 0.0f;
          number_of_points1 = 0.0f;
          number_of_points2 = 0.0f;

          mean_value = 0.0f;
          mean_val_1 = 0.0f;
          mean_val_2 = 0.0f;
          nb_val_1   = 0.0f;
          nb_val_2   = 0.0f;

          volume11 = 0.0f;
          volume22 = 0.0f;

          dist_histo = 0.0f;

          for (int dim_allouee1 = 0; dim_allouee1 < 255; ++dim_allouee1) {
            histogram_region1[dim_allouee1]       = 0;
            histogram_region2[dim_allouee1]       = 0;
            cumul_histogram_region1[dim_allouee1] = 0;
            cumul_histogram_region2[dim_allouee1] = 0;
          }

          e1 = removed_edges.back();
          removed_edges.pop_back();

          // Get label of the regions
          Tree_temp2.vertexData(Tree_temp2.edgeSource(e1), &label1);
          Tree_temp2.vertexData(Tree_temp2.edgeTarget(e1), &label2);

          boost::tie(ed_it4, ed_end4) = boost::edges(Gout_t.getBoostGraph());

          for (boost::tie(ed_it, ed_end) = boost::edges(Gout.getBoostGraph());
               ed_it != ed_end, ed_it4 != ed_end4; ++ed_it, ++ed_it4) {
            vs = Gout.edgeSource(*ed_it);
            vt = Gout.edgeTarget(*ed_it);

            // labels of the regions
            Tree_temp2.vertexData(vs, &vdata1);
            Tree_temp2.vertexData(vt, &vdata2);

            // compute avergage mean gradient along regions
            if ((vdata1 == label1 && vdata2 == label2) ||
                (vdata1 == label2 && vdata2 == label1)) {
              Gout.edgeWeight(*ed_it, &tmp);
              Gout_t.edgeWeight(*ed_it4, &tmp2);

              mean_value       = mean_value + (float) tmp;
              number_of_points = number_of_points + (float) tmp2;
            }

            // compute avergage mean gradient along regions 1
            if ((vdata1 == label1 && vdata2 != label1) ||
                (vdata2 == label1 && vdata1 != label1)) {
              Gout.edgeWeight(*ed_it, &tmp);
              Gout_t.edgeWeight(*ed_it4, &tmp2);

              mean_value_1      = mean_value_1 + (float) tmp;
              number_of_points1 = number_of_points1 + (float) tmp2;
            }

            // compute avergage mean gradient along regions 2
            if ((vdata1 == label2 && vdata2 != label2) ||
                (vdata2 == label2 && vdata1 != label2)) {
              Gout.edgeWeight(*ed_it, &tmp);
              Gout_t.edgeWeight(*ed_it4, &tmp2);

              mean_value_2      = mean_value_2 + (float) tmp;
              number_of_points2 = number_of_points2 + (float) tmp2;
            }
          }

          // copying the properties of each vertex
          for (boost::tie(v_it, v_end) = boost::vertices(Gout.getBoostGraph());
               v_it != v_end; ++v_it) {
            // Get label of the regions
            Tree_temp2.vertexData(*v_it, &vdata1);
            Gout.vertexData(*v_it, &vdata11);

            Tree_out.setVertexData(*v_it, (float) alpha1 * vdata11 +
                                              (float) alpha3 *
                                                  quad_error[(int) *v_it]);

            // compute area of each regions
            if (vdata1 == label1) {
              volume1 =
                  volume1 +
                  (float) vdata11; // NODES OF GOUT ARE WEIGHTED WITH VOLUME
              G_area.vertexData(
                  *v_it, &vdata11); // NODES OF GCLASSIC ARE WEIGHTED WITH AREA
              area1      = area1 + (float) vdata11;
              mean_val_1 = mean_val_1 + mean_values[(int) *v_it];
              nb_val_1   = nb_val_1 + 1.0f;

              for (int dim_allouee1 = 0; dim_allouee1 < 255; ++dim_allouee1) {
                histogram_region1[dim_allouee1] =
                    histogram_region1[dim_allouee1] +
                    histogram[(int) *v_it][dim_allouee1];
              }
            }
            if (vdata1 == label2) {
              volume2 =
                  volume2 +
                  (float) vdata11; // NODES OF GOUT ARE WEIGHTED WITH VOLUME
              G_area.vertexData(
                  *v_it, &vdata11); // NODES OF GCLASSIC ARE WEIGHTED WITH AREA
              area2      = area2 + (float) vdata11;
              mean_val_2 = mean_val_2 + mean_values[(int) *v_it];
              nb_val_2   = nb_val_2 + 1.0f;

              for (int dim_allouee1 = 0; dim_allouee1 < 255; ++dim_allouee1) {
                histogram_region2[dim_allouee1] =
                    histogram_region2[dim_allouee1] +
                    histogram[(int) *v_it][dim_allouee1];
              }
            }
          }

          for (int dim_allouee1 = 0; dim_allouee1 < 255; ++dim_allouee1) {
            for (int dim_allouee0 = 0; dim_allouee0 <= dim_allouee1;
                 ++dim_allouee0) {
              cumul_histogram_region1[dim_allouee1] =
                  cumul_histogram_region1[dim_allouee1] +
                  (float) histogram_region1[dim_allouee0] / (float) area1;
              cumul_histogram_region2[dim_allouee1] =
                  cumul_histogram_region2[dim_allouee1] +
                  (float) histogram_region2[dim_allouee0] / (float) area2;
            }
          }

          for (int dim_allouee1 = 0; dim_allouee1 < 255; ++dim_allouee1) {
            dist_histo =
                dist_histo + std::pow(cumul_histogram_region1[dim_allouee1] -
                                          cumul_histogram_region2[dim_allouee1],
                                      2);
          }

          // copying the properties of each vertex
          for (boost::tie(v_it, v_end) = boost::vertices(Gout.getBoostGraph());
               v_it != v_end; ++v_it) {
            // Get label of the regions
            Tree_temp2.vertexData(*v_it, &vdata1);

            // compute area of each regions
            if (vdata1 == label1) {
              quad1 = quad1 +
                      (mean_values[(int) *v_it] - mean_val_1 / nb_val_1) *
                          (mean_values[(int) *v_it] - mean_val_1 / nb_val_1);
            }
            if (vdata1 == label2) {
              quad2 = quad2 +
                      (mean_values[(int) *v_it] - mean_val_2 / nb_val_2) *
                          (mean_values[(int) *v_it] - mean_val_2 / nb_val_2);
            }
          }

          quad1 = quad1 / nb_val_1;
          quad2 = quad2 / nb_val_2;

          /*volume11 = area1 * mean_value_1/number_of_points1;
          volume22 = area2 * mean_value_2/number_of_points2;*/

          volume11 = area1 * last_edge_value / number_of_points1;
          volume22 = area2 * last_edge_value / number_of_points2;

          /// ALL KIND OF MEASURES THAT CAN BE DONE ON THE REGIONS
          // float probability_volume = ( 1.0 - std::pow( 1.0 -
          // volume11/((double) volume)  , 50 ) - std::pow( 1.0 -
          // volume22/((double) volume)  , 50 ) + std::pow( 1.0 -
          // (volume11+volume22)/((double) volume)  , 50 ) );
          float probability_volume =
              (1.0 - std::pow(1.0 - volume1 / ((double) volume), (int) alpha1) -
               std::pow(1.0 - volume2 / ((double) volume), (int) alpha1) +
               std::pow(1.0 - (volume1 + volume2) / ((double) volume),
                        (int) alpha1));
          float probability_area =
              (1.0 -
               std::pow(1.0 - area1 / ((double) area_total), (int) alpha1) -
               std::pow(1.0 - area2 / ((double) area_total), (int) alpha1) +
               std::pow(1.0 - (area1 + area2) / ((double) area_total),
                        (int) alpha1));

          float deterministic_area =
              std::min(area1, area2) / ((double) area_total);
          float deterministic_volume =
              std::min(volume11, volume22) / ((double) volume);

          float diff_mean =
              std::abs((mean_val_2 / nb_val_2) - (mean_val_1 / nb_val_1));
          float mean_gradient = mean_value / number_of_points;

          float mean_gradient_1 = mean_value_1 / number_of_points1;
          float mean_gradient_2 = mean_value_2 / number_of_points2;

          float shape_factor_1 =
              4.0f * M_PI * area1 / (number_of_points1 * number_of_points1);
          float shape_factor_2 =
              4.0f * M_PI * area2 / (number_of_points2 * number_of_points2);

          shape_factor_1 = std::min(shape_factor_1, (float) 1.0);
          shape_factor_2 = std::min(shape_factor_2, (float) 1.0);

          // std::cout<<"shape_factor_1"<<shape_factor_1<<std::endl;
          // std::cout<<"shape_factor_2"<<shape_factor_2<<std::endl;

          // probability_volume = std::min( probability_volume , 0.05f ) ;
          // probability_area = std::min( probability_area , 0.5f ) ;

          //// GET EDGE IN TRee_OUt
          boost::tie(e2, in1) =
              boost::edge(Tree_out.edgeSource(e1), Tree_out.edgeTarget(e1),
                          Tree_out.getBoostGraph());
          // float val_gradient_mean_t = std::min( ((float) last_edge_value)/(
          // 65535.0f ) , 0.7f );

          // std::cout<<"mean_gradient"<<mean_gradient/65535.0f<<std::endl;
          // std::cout<<"probability_volume"<<probability_volume<<std::endl;

          if (number_of_points > 0.0f) {
            // float value_test = 100000.0f * std::pow(dist_histo,(float)
            // alpha2) * std::pow( ((float) mean_value/ (float) number_of_points
            // )/( 65535.0f ) , (float) alpha1 ) * std::pow( probability_volume ,
            // (float) alpha3 ) ; float value_test = 2000000.0f * std::pow(
            // mean_gradient/65535.0f , (float) alpha1 ) * std::pow( std::min(
            // mean_gradient_1/65535.0f, mean_gradient_2/65535.0f ) , (float)
            // alpha2 ) * std::pow( probability_volume , (float) alpha3 ) ; float
            // value_test = 65535.0f * std::pow( (float) last_edge_value/(
            // 65535.0f ) , (float) alpha1 ) * std::pow( std::max( shape_factor_1
            // , shape_factor_2 ) , (float) alpha2 ) * std::pow(
            // probability_volume , (float) alpha3 ) ; float value_test =
            // 65535.0f * std::pow( mean_gradient/65535.0f , (float) alpha1 ) *
            // std::pow( (float) diff_mean/( 65535.0f ) , (float) alpha2 )  *
            // std::pow( probability_volume , (float) alpha3 ) ; float value_test
            // = 65535.0f * std::pow( mean_gradient/65535.0f , (float) alpha1 ) *
            // std::pow( probability_area , (float) alpha2 )  * std::pow(
            // probability_volume , (float) alpha3 ) ;
            float value_test = 65535.0f * (mean_gradient / 65535.0f) *
                               std::pow(probability_area, (float) alpha2) *
                               std::pow(probability_volume, (float) alpha3);
            // float value_test = 65535.0f * std::pow( mean_gradient/65535.0f ,
            // (float) alpha1 ) * std::pow( std::max( shape_factor_1 ,
            // shape_factor_2 ) , (float) alpha2 ) * std::pow( probability_volume
            // , (float) alpha3 ) ;
            Tree_out.setEdgeWeight(e2, value_test);

            if (value_test > current_max_value)
              current_max_value = value_test;

          } else {
            Tree_out.setEdgeWeight(e2, 0.0f);
          }
        }

        while (last_edge_value == last_analyzed) {
          last_edge_value = val_edges.back();
          val_edges.pop_back();
        }
        last_analyzed = last_edge_value;
      }

      std::cout << "current_max_value" << current_max_value << std::endl;

      return RES_OK;
    }

    template <class ImageWs, class ImageIn, class ImageGrad, typename _alpha1,
              typename _alpha2, typename _alpha3, class SE, class BoostGraph>
    RES_C t_TreeReweighting(const ImageWs &imWs, const ImageIn &imIn,
                            const ImageGrad &imGrad, BoostGraph &Treein,
                            const _alpha1 alpha1, const _alpha2 alpha2,
                            const _alpha3 alpha3, const SE &nl,
                            BoostGraph &Tree_out)
    {
      MORPHEE_ENTER_FUNCTION("t_TreeReweighting");

      if ((!imWs.isAllocated())) {
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

      // common image iterator
      typename ImageWs::const_iterator it, iend;
      morphee::selement::Neighborhood<SE, ImageIn> neighb(imIn, nl);
      typename morphee::selement::Neighborhood<SE, ImageIn>::iterator nit, nend;
      offset_t o0;
      offset_t o1;

      typename BoostGraph::EdgeIterator ed_it, ed_end;
      typename BoostGraph::EdgeIterator ed_it2, ed_end2;
      typename BoostGraph::EdgeIterator ed_it3, ed_end3;
      typename BoostGraph::EdgeIterator ed_it4, ed_end4;

      typename BoostGraph::VertexIterator v_it, v_end;

      typename BoostGraph::VertexProperty vdata1, vdata2, vdata11, vdata22;
      typename BoostGraph::VertexProperty label1, label2;

      bool in1;
      int numVert  = 0;
      int numEdges = 0;
      typename BoostGraph::EdgeDescriptor e1, e2;
      typename BoostGraph::VertexDescriptor vs, vt;
      typename BoostGraph::EdgeProperty tmp, tmp2, tmp3;

      std::vector<double> val_edges;
      std::vector<double> val_edges2;
      val_edges.push_back(0.0);

      std::cout << "build Region Adjacency Graph" << std::endl;

      for (it = imWs.begin(), iend = imWs.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o0      = it.getOffset();
        int val = imWs.pixelFromOffset(o0);

        if (val > numVert) {
          numVert = val;
        }
      }

      std::cout << "Compute mean and quaderror" << std::endl;
      std::vector<double> mean_values;
      std::vector<double> number_of_values;
      std::vector<double> quad_error;
      std::vector<double> min_gradient_values;
      std::vector<double> max_gradient_values;
      std::vector<double> area_values;

      int **histogram; // image histogram
      histogram = new int *[numVert];

      float *histogram_region1; // region 1 histogram
      histogram_region1 = new float[255];

      float *histogram_region2; // region 2 histogram
      histogram_region2 = new float[255];

      float *cumul_histogram_region1; // cummulated region 1 histogram
      cumul_histogram_region1 = new float[255];

      float *cumul_histogram_region2; // cummulated region 2 histogram
      cumul_histogram_region2 = new float[255];

      for (int dim_allouee0 = 0; dim_allouee0 < numVert; ++dim_allouee0) {
        histogram[dim_allouee0] = new int[255];
      }

      for (int dim_allouee0 = 0; dim_allouee0 < numVert; ++dim_allouee0) {
        for (int dim_allouee1 = 0; dim_allouee1 < 255; ++dim_allouee1) {
          histogram[dim_allouee0][dim_allouee1] = 0;
        }
      }

      //// INIT VALUES TO ZERO
      for (int i = 0; i < numVert; i++) {
        mean_values.push_back(0);
        number_of_values.push_back(0);
        quad_error.push_back(0);
        min_gradient_values.push_back(1000000000000);
        max_gradient_values.push_back(0);
        area_values.push_back(0);
      }

      //// COMPUTE MEAN VALUES, AREA, HISTOGRAM, MIN AND MAX
      for (it = imWs.begin(), iend = imWs.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o1                  = it.getOffset();
        int val             = imWs.pixelFromOffset(o1);
        double val_image    = imIn.pixelFromOffset(o1);
        double val_gradient = imGrad.pixelFromOffset(o1);

        mean_values[val - 1]      = mean_values[val - 1] + val_image;
        number_of_values[val - 1] = number_of_values[val - 1] + 1.0;

        if (val_gradient < min_gradient_values[val - 1])
          min_gradient_values[val - 1] = val_gradient;
        if (val_gradient > max_gradient_values[val - 1])
          max_gradient_values[val - 1] = val_gradient;

        area_values[val - 1]     = area_values[val - 1] + 1;
        int vald                 = (int) (val_image);
        histogram[val - 1][vald] = histogram[val - 1][vald] + 1;
      }

      //// SET MEAN VALUES
      for (int i = 0; i < numVert; i++) {
        mean_values[i] = mean_values[i] / number_of_values[i];
      }

      //// COMPUTE QUADRATIC ERROR (COMPARED TO THE MEAN VALUE) IN EACH REGION
      float max_quad_val = 0.0f;
      for (it = imWs.begin(), iend = imWs.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1               = it.getOffset();
        int val          = imWs.pixelFromOffset(o1);
        double val_image = imIn.pixelFromOffset(o1);

        quad_error[val - 1] =
            (double) quad_error[val - 1] +
            std::pow(std::abs(val_image - (double) mean_values[val - 1]), 2);

        if (quad_error[val - 1] > max_quad_val)
          max_quad_val = quad_error[val - 1];
      }

      std::cout << "build Region Adjacency Graph Vertices" << std::endl;
      std::cout << "number of Vertices:" << numVert << std::endl;

      // create some temp graphs
      Tree_out = morphee::graph::CommonGraph32(numVert);

      BoostGraph Gout   = morphee::graph::CommonGraph32(numVert);
      BoostGraph Gout_t = morphee::graph::CommonGraph32(numVert);
      BoostGraph Gtemp  = morphee::graph::CommonGraph32(numVert);

      BoostGraph Tree_temp  = morphee::graph::CommonGraph32(numVert);
      BoostGraph Tree_temp2 = morphee::graph::CommonGraph32(numVert);

      // GRAPH WHOSE NODES ARE WEIGHTED WITH AREA
      BoostGraph G_area = morphee::graph::CommonGraph32(numVert);
      morphee::morphoBase::t_NeighborhoodGraphFromMosaic(imWs, nl, G_area);

      // project area of regions on the graph nodes
      ImageWs ImTempSurfaces = imWs.getSame();
      morphee::morphoBase::t_ImLabelFlatZonesWithArea(imWs, nl, ImTempSurfaces);
      morphee::graph::t_ProjectMarkersOnGraph(ImTempSurfaces, imWs, G_area);

      // GRAPH WHOSE NODES ARE WEIGHTED WITH VOLUME
      morphee::morphoBase::t_ImLabelFlatZonesWithVolume(imWs, imGrad, nl,
                                                        ImTempSurfaces);
      morphee::graph::t_ProjectMarkersOnGraph(ImTempSurfaces, imWs, Gout);

      std::cout << "build Region Adjacency Graph Edges" << std::endl;

      double volume     = 0.0;
      double area_total = 0.0;

      //// SET VERTEX PROPERTIES TO AREA AND VOLUME
      for (boost::tie(v_it, v_end) = boost::vertices(Gout.getBoostGraph());
           v_it != v_end; ++v_it) {
        Gout.vertexData(*v_it, &vdata1);
        volume = volume + (double) vdata1;

        G_area.vertexData(*v_it, &vdata1);
        area_total = area_total + (double) vdata1;
      }

      //// COMPUTE MEAN GRADIENT VALUES ALONG THE REGIONS
      for (it = imWs.begin(), iend = imWs.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1      = it.getOffset();
        int val = imWs.pixelFromOffset(o1);

        if (val > 0) {
          neighb.setCenter(o1);

          for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
            const offset_t o2 = nit.getOffset();
            int val2          = imWs.pixelFromOffset(o2);

            if (o2 > o1) {
              if (val != val2) {
                boost::tie(e1, in1) =
                    boost::edge(val - 1, val2 - 1, Gout.getBoostGraph());
                boost::tie(e2, in1) =
                    boost::edge(val - 1, val2 - 1, Gtemp.getBoostGraph());

                double val3 = imGrad.pixelFromOffset(o1);
                double val4 = imGrad.pixelFromOffset(o2);
                double maxi = std::max(val3, val4);

                if (in1 == 0) {
                  numEdges++;
                  Gout.addEdge(val - 1, val2 - 1, maxi);
                  Gtemp.addEdge(val - 1, val2 - 1, 1);
                } else {
                  Gout.edgeWeight(e1, &tmp);
                  Gout.setEdgeWeight(e1, tmp + maxi);

                  Gtemp.edgeWeight(e2, &tmp2);
                  Gtemp.setEdgeWeight(e2, tmp2 + 1);
                }
              }
            }
          }
        }
      }

      std::cout << "number of Edges : " << numEdges << std::endl;

      Gout_t                      = t_CopyGraph(Gout);
      boost::tie(ed_it2, ed_end2) = boost::edges(Gtemp.getBoostGraph());
      boost::tie(ed_it3, ed_end3) = boost::edges(G_area.getBoostGraph());
      boost::tie(ed_it4, ed_end4) = boost::edges(Gout_t.getBoostGraph());
      float current_max_value     = 0.0f;

      // Weights the graph
      for (boost::tie(ed_it, ed_end) = boost::edges(Gout.getBoostGraph());
           ed_it != ed_end, ed_it2 != ed_end2, ed_it3 != ed_end3,
                             ed_it4 != ed_end4;
           ++ed_it, ++ed_it2, ++ed_it3, ++ed_it4) {
        Gout.edgeWeight(*ed_it, &tmp);    // GRADIENT SUM
        Gtemp.edgeWeight(*ed_it2, &tmp2); // REGION BOUNDARY LENGTH
        Gout_t.setEdgeWeight(*ed_it4, ((double) tmp2)); // length
      }

      Gtemp = t_CopyGraph(Treein);

      for (boost::tie(ed_it, ed_end) = boost::edges(Gtemp.getBoostGraph());
           ed_it != ed_end; ++ed_it) {
        Gtemp.edgeWeight(*ed_it, &tmp); // MEAN GRADIENT VALUE
        val_edges.push_back(tmp);
      }

      std::vector<typename BoostGraph::EdgeDescriptor> removed_edges;

      Tree_out   = t_CopyGraph(Gtemp);
      Tree_temp  = t_CopyGraph(Gtemp);
      Tree_temp2 = t_CopyGraph(Gtemp);

      // sort edges weights to explore the hierarchy
      std::cout << "sort" << std::endl;
      std::sort(val_edges.begin(), val_edges.end(), std::less<double>());
      double last_edge_value = val_edges.back();
      double last_analyzed   = last_edge_value;

      /*int min_nb_of_edges = (int) alpha1-1 ;
      int current_nb_of_edges = 0 ;

      for( int i = val_edges.size()-1 ; i>=0; i--){
        

        if( current_nb_of_edges < min_nb_of_edges ){
          last_edge_value = val_edges[i];
          last_analyzed = last_edge_value;
          val_edges.pop_back();
          current_nb_of_edges++;
        }

      }*/

      int iteration = 1;

      while (val_edges.size() > 1) {
        std::cout << last_edge_value << std::endl;

        // remove edge of maximal weight
        for (boost::tie(ed_it, ed_end) = boost::edges(Gtemp.getBoostGraph());
             ed_it != ed_end; ++ed_it) {
          Gtemp.edgeWeight(*ed_it, &tmp);
          vs                  = Gtemp.edgeSource(*ed_it);
          vt                  = Gtemp.edgeTarget(*ed_it);
          boost::tie(e1, in1) = boost::edge(vs, vt, Tree_temp.getBoostGraph());

          if (tmp >= last_edge_value && in1) { // check is  current max weight
            removed_edges.push_back(e1);
            Tree_temp.removeEdge(vs, vt);
          }
        }

        // label trees
        int nb_of_connected;
        t_LabelConnectedComponent(Tree_temp, Tree_temp2, &nb_of_connected);

        std::cout << " number of connected components = " << nb_of_connected
                  << std::endl;

        // local variable for mean values and areas of regions
        float mean_value       = 0.0f;
        float min_value_b      = 1000000000000.0f;
        float number_of_points = 0.0f;
        float volume1          = 0.0f;
        float volume2          = 0.0f;

        float volume11 = 0.0f;
        float volume22 = 0.0f;

        float area1 = 0.0f;
        float area2 = 0.0f;

        float quad1 = 0.0f;
        float quad2 = 0.0f;

        float mean_val_1 = 0.0f;
        float mean_val_2 = 0.0f;

        float nb_val_1 = 0.0f;
        float nb_val_2 = 0.0f;

        float mean_value_1 = 0.0f;
        float mean_value_2 = 0.0f;

        float number_of_points1 = 0.0f;
        float number_of_points2 = 0.0f;

        float dist_histo = 0.0f;

        // go through removed edges and look caracteristics of regions connected
        // to it
        while (removed_edges.size() > 0) {
          volume1          = 0.0f;
          volume2          = 0.0f;
          area1            = 0.0f;
          area2            = 0.0f;
          quad1            = 0.0f;
          quad2            = 0.0f;
          number_of_points = 0.0f;

          mean_value_1      = 0.0f;
          mean_value_2      = 0.0f;
          number_of_points1 = 0.0f;
          number_of_points2 = 0.0f;

          mean_value = 0.0f;
          mean_val_1 = 0.0f;
          mean_val_2 = 0.0f;
          nb_val_1   = 0.0f;
          nb_val_2   = 0.0f;

          volume11 = 0.0f;
          volume22 = 0.0f;

          dist_histo = 0.0f;

          for (int dim_allouee1 = 0; dim_allouee1 < 255; ++dim_allouee1) {
            histogram_region1[dim_allouee1]       = 0;
            histogram_region2[dim_allouee1]       = 0;
            cumul_histogram_region1[dim_allouee1] = 0;
            cumul_histogram_region2[dim_allouee1] = 0;
          }

          e1 = removed_edges.back();
          removed_edges.pop_back();

          // Get label of the regions
          Tree_temp2.vertexData(Tree_temp2.edgeSource(e1), &label1);
          Tree_temp2.vertexData(Tree_temp2.edgeTarget(e1), &label2);

          boost::tie(ed_it4, ed_end4) = boost::edges(Gout_t.getBoostGraph());

          for (boost::tie(ed_it, ed_end) = boost::edges(Gout.getBoostGraph());
               ed_it != ed_end, ed_it4 != ed_end4; ++ed_it, ++ed_it4) {
            vs = Gout.edgeSource(*ed_it);
            vt = Gout.edgeTarget(*ed_it);

            // labels of the regions
            Tree_temp2.vertexData(vs, &vdata1);
            Tree_temp2.vertexData(vt, &vdata2);

            // compute avergage mean gradient along regions
            if ((vdata1 == label1 && vdata2 == label2) ||
                (vdata1 == label2 && vdata2 == label1)) {
              Gout.edgeWeight(*ed_it, &tmp);
              Gout_t.edgeWeight(*ed_it4, &tmp2);

              mean_value       = mean_value + (float) tmp;
              number_of_points = number_of_points + (float) tmp2;

              if (((float) tmp / (float) tmp2) < min_value_b)
                min_value_b = ((float) tmp / (float) tmp2);
            }

            // compute avergage mean gradient along regions 1
            if ((vdata1 == label1 && vdata2 != label1) ||
                (vdata2 == label1 && vdata1 != label1)) {
              Gout.edgeWeight(*ed_it, &tmp);
              Gout_t.edgeWeight(*ed_it4, &tmp2);

              mean_value_1      = mean_value_1 + (float) tmp;
              number_of_points1 = number_of_points1 + (float) tmp2;
            }

            // compute avergage mean gradient along regions 2
            if ((vdata1 == label2 && vdata2 != label2) ||
                (vdata2 == label2 && vdata1 != label2)) {
              Gout.edgeWeight(*ed_it, &tmp);
              Gout_t.edgeWeight(*ed_it4, &tmp2);

              mean_value_2      = mean_value_2 + (float) tmp;
              number_of_points2 = number_of_points2 + (float) tmp2;
            }
          }

          // copying the properties of each vertex
          for (boost::tie(v_it, v_end) = boost::vertices(Gout.getBoostGraph());
               v_it != v_end; ++v_it) {
            // Get label of the regions
            Tree_temp2.vertexData(*v_it, &vdata1);
            Gout.vertexData(*v_it, &vdata11);

            Tree_out.setVertexData(*v_it, (float) alpha1 * vdata11 +
                                              (float) alpha3 *
                                                  quad_error[(int) *v_it]);

            // compute area of each regions
            if (vdata1 == label1) {
              volume1 =
                  volume1 +
                  (float) vdata11; // NODES OF GOUT ARE WEIGHTED WITH VOLUME
              G_area.vertexData(
                  *v_it, &vdata11); // NODES OF GCLASSIC ARE WEIGHTED WITH AREA
              area1      = area1 + (float) vdata11;
              mean_val_1 = mean_val_1 + mean_values[(int) *v_it];
              nb_val_1   = nb_val_1 + 1.0f;

              for (int dim_allouee1 = 0; dim_allouee1 < 255; ++dim_allouee1) {
                histogram_region1[dim_allouee1] =
                    histogram_region1[dim_allouee1] +
                    histogram[(int) *v_it][dim_allouee1];
              }
            }
            if (vdata1 == label2) {
              volume2 =
                  volume2 +
                  (float) vdata11; // NODES OF GOUT ARE WEIGHTED WITH VOLUME
              G_area.vertexData(
                  *v_it, &vdata11); // NODES OF GCLASSIC ARE WEIGHTED WITH AREA
              area2      = area2 + (float) vdata11;
              mean_val_2 = mean_val_2 + mean_values[(int) *v_it];
              nb_val_2   = nb_val_2 + 1.0f;

              for (int dim_allouee1 = 0; dim_allouee1 < 255; ++dim_allouee1) {
                histogram_region2[dim_allouee1] =
                    histogram_region2[dim_allouee1] +
                    histogram[(int) *v_it][dim_allouee1];
              }
            }
          }

          for (int dim_allouee1 = 0; dim_allouee1 < 255; ++dim_allouee1) {
            for (int dim_allouee0 = 0; dim_allouee0 <= dim_allouee1;
                 ++dim_allouee0) {
              cumul_histogram_region1[dim_allouee1] =
                  cumul_histogram_region1[dim_allouee1] +
                  (float) histogram_region1[dim_allouee0] / (float) area1;
              cumul_histogram_region2[dim_allouee1] =
                  cumul_histogram_region2[dim_allouee1] +
                  (float) histogram_region2[dim_allouee0] / (float) area2;
            }
          }

          for (int dim_allouee1 = 0; dim_allouee1 < 255; ++dim_allouee1) {
            dist_histo =
                dist_histo + std::pow(cumul_histogram_region1[dim_allouee1] -
                                          cumul_histogram_region2[dim_allouee1],
                                      2);
          }

          // copying the properties of each vertex
          for (boost::tie(v_it, v_end) = boost::vertices(Gout.getBoostGraph());
               v_it != v_end; ++v_it) {
            // Get label of the regions
            Tree_temp2.vertexData(*v_it, &vdata1);

            // compute area of each regions
            if (vdata1 == label1) {
              quad1 = quad1 +
                      (mean_values[(int) *v_it] - mean_val_1 / nb_val_1) *
                          (mean_values[(int) *v_it] - mean_val_1 / nb_val_1);
            }
            if (vdata1 == label2) {
              quad2 = quad2 +
                      (mean_values[(int) *v_it] - mean_val_2 / nb_val_2) *
                          (mean_values[(int) *v_it] - mean_val_2 / nb_val_2);
            }
          }

          quad1 = quad1 / nb_val_1;
          quad2 = quad2 / nb_val_2;

          /*volume11 = area1 * mean_value_1/number_of_points1;
          volume22 = area2 * mean_value_2/number_of_points2;*/

          volume11 = area1 * mean_value / number_of_points;
          volume22 = area2 * mean_value / number_of_points;

          // std::cout<<"dist_histo ="<<dist_histo<<std::endl;

          /// ALL KIND OF MEASURES THAT CAN BE DONE ON THE REGIONS
          float probability_volume =
              (1.0 - std::pow(1.0 - volume1 / ((double) volume), (int) alpha1) -
               std::pow(1.0 - volume2 / ((double) volume), (int) alpha1) +
               std::pow(1.0 - (volume1 + volume2) / ((double) volume),
                        (int) alpha1));
          float probability_area =
              (1.0 -
               std::pow(1.0 - area1 / ((double) area_total), (int) alpha1) -
               std::pow(1.0 - area2 / ((double) area_total), (int) alpha1) +
               std::pow(1.0 - (area1 + area2) / ((double) area_total),
                        (int) alpha1));

          float deterministic_area =
              std::min(area1, area2) / ((double) area_total);
          float deterministic_volume =
              std::min(volume11, volume22) / ((double) volume);

          float diff_mean =
              std::abs((mean_val_2 / nb_val_2) - (mean_val_1 / nb_val_1));
          float mean_gradient = mean_value / number_of_points;

          float mean_gradient_1 = mean_value_1 / number_of_points1;
          float mean_gradient_2 = mean_value_2 / number_of_points2;

          dist_histo = deterministic_area * dist_histo / 255.0f;

          //// GET EDGE IN TRee_OUt
          boost::tie(e2, in1) =
              boost::edge(Tree_out.edgeSource(e1), Tree_out.edgeTarget(e1),
                          Tree_out.getBoostGraph());

          if (number_of_points > 0.0f) {
            // float value_test = 65535.0f * std::pow( (
            // (float)last_edge_value/65535.0f) ,  (float) alpha2 ) * std::pow(
            // dist_histo , (float) alpha2 ) * std::pow( probability_volume ,
            // (float) alpha3 ) ;
            float value_test =
                65535.0f *
                std::pow(((float) mean_gradient / 65535.0f), (float) alpha2) *
                std::pow(probability_volume, (float) alpha3);
            // float value_test = 65535.0f * std::min( std::pow( (
            // (float)last_edge_value/65535.0f) ,  (float) alpha2 ) , std::pow(
            // probability_volume , (float) alpha3 ) ) ; value_test = std::min(
            // 65535.0f * log( 1.0f+value_test )/(log(65536.0f)) , 65535.0f ) ;
            value_test = std::min(value_test, 65535.0f);
            Tree_out.setEdgeWeight(e2, value_test);

            if (value_test > current_max_value)
              current_max_value = value_test;

          } else {
            Tree_out.setEdgeWeight(e2, 0.0f);
          }
        }

        while (last_edge_value == last_analyzed) {
          last_edge_value = val_edges.back();
          val_edges.pop_back();
        }
        last_analyzed = last_edge_value;
        iteration++;
      }

      std::cout << "current_max_value" << current_max_value << std::endl;

      return RES_OK;
    }

    void Addition(const affine_par_morceaux &A, const affine_par_morceaux &B,
                  affine_par_morceaux &S, int Pmax)
    {
      affine_par_morceaux::const_reverse_iterator ia  = A.rbegin(),
                                                  eia = A.rend();
      affine_par_morceaux::const_reverse_iterator ib  = B.rbegin(),
                                                  eib = B.rend();
      // Double parcours des morceaux de A et B, en sens inverse
      while ((ia != eia) && (ib != eib) && (S.size() < Pmax)) {
        // Si l'origine du morceau courant de A est >= a celle du morceau
        // courant de B Un morceau doit etre cree avec cette origine
        if ((*ia).x >= (*ib).x) {
          morceau nouveau = *ia;
          // On ajoute a l'ordonne du nouveau l'ordonnee de (*ib) au point
          // (*ia).x
          nouveau.y += (*ib).y + (*ib).p * ((*ia).x - (*ib).x);
          // Ajout des pentes
          nouveau.p += (*ib).p;
          // Empilement au debut de la somme
          S.push_front(nouveau);
          // Si les debuts des morceaux sont egaux : il faut aussi passer
          // au morceau precedent de B
          if ((*ia).x == (*ib).x)
            ++ib;
          ++ia;
        }
        // Memes operations mais en inversant les roles de (*ia) et (*ib)
        else {
          morceau nouveau = *ib;
          nouveau.y += (*ia).y + (*ia).p * ((*ib).x - (*ia).x);
          nouveau.p += (*ia).p;
          S.push_front(nouveau);
          ++ib;
        }
      }
      // Remise en 0 de l'origine du premier morceau :
      // si l'addition a ete stoppee par Pmax, le dernier morceau ajoute peut
      // avoir x>0
      if (S.front().x > 0) {
        S.front().y -= S.front().p * S.front().x;
        S.front().x = 0;
      }
      // FIN
    }

    float Inf_affine(affine_par_morceaux &A, const morceau &m)
    {
      // utilisation d'un iterateur bidirectionnel comme un reverse_iterator
      affine_par_morceaux::iterator i = A.end(), ei = A.begin();
      --i;
      --ei;
      // cas particulier de fonction de cout additive :
      // les pentes de m et du dernier morceau de A sont alors egales
      if (m.p == (*i).p) {
        // ordonnee de m en (*i).x
        float y = m.y + m.p * ((*i).x - m.x);
        // Si m est au dessus de (*i) : intersection en +infini
        if (y > (*i).y) {
          return 0;
        }
        // Si m est confondu a (*i) : apparition au debut de (*i)
        else if (y == (*i).y) {
          return (*i).x;
        }
        // Sinon, l'intersection se situe sur un morceau precedent,
        // on passe le dernier morceau et on continue
        else {
          --i;
        }
      }
      // Recherche du morceau sur lequel se situe l'intersection
      float xi;
      for (; i != ei; --i) {
        xi = (m.x * m.p - (*i).x * (*i).p - (m.y - (*i).y)) / (m.p - (*i).p);
        if (xi > (*i).x)
          break;
      }
      // Suppression des morceaux suivant l'intersection
      A.erase(++i, A.end());
      // Insertion nouveau morceau final
      morceau m1;
      m1.x = xi;
      m1.y = m.y + m.p * (xi - m.x);
      m1.p = m.p;
      A.push_back(m1);
      // FIN
      return xi;
    }

    template <class ImageWs, class ImageIn, class ImageGrad, class SE,
              class BoostGraph>
    RES_C t_ScaleSetHierarchyReweighting(const ImageWs &imWs,
                                         const ImageIn &imIn,
                                         const ImageGrad &imGrad,
                                         const BoostGraph &Treein, const SE &nl,
                                         BoostGraph &Tree_out)
    {
      MORPHEE_ENTER_FUNCTION("t_ScaleSetHierarchyReweighting");

      // SCALE SET HIERARCHY FUNCTION
      if ((!imWs.isAllocated())) {
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

      // common image iterator
      typename ImageWs::const_iterator it, iend;
      morphee::selement::Neighborhood<SE, ImageIn> neighb(imIn, nl);
      typename morphee::selement::Neighborhood<SE, ImageIn>::iterator nit, nend;
      offset_t o0;
      offset_t o1;

      typename BoostGraph::EdgeIterator ed_it, ed_end;
      typename BoostGraph::EdgeIterator ed_it2, ed_end2;
      typename BoostGraph::EdgeIterator ed_it3, ed_end3;
      typename BoostGraph::EdgeIterator ed_it4, ed_end4;

      typename BoostGraph::VertexIterator v_it, v_end;

      typename BoostGraph::VertexProperty vdata1, vdata2, vdata11, vdata22;
      typename BoostGraph::VertexProperty label1, label2;

      bool in1;
      int numVert  = 0;
      int numEdges = 0;
      typename BoostGraph::EdgeDescriptor e1, e2, ef;
      typename BoostGraph::VertexDescriptor vs, vt;
      typename BoostGraph::EdgeProperty tmp, tmp2, tmp3;

      std::vector<double> val_edges;
      val_edges.push_back(0.0);

      float lambda = 1.0;

      std::cout << "Compute number of regions" << std::endl;

      for (it = imWs.begin(), iend = imWs.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o0      = it.getOffset();
        int val = imWs.pixelFromOffset(o0);

        if (val > numVert) {
          numVert = val;
        }
      }

      // create some temp graphs
      Tree_out = morphee::graph::CommonGraph32(numVert);
      Tree_out = t_CopyGraph(Treein);
      // copy of input tree
      BoostGraph Tree_temp      = morphee::graph::CommonGraph32(numVert);
      BoostGraph Tree_ordered   = morphee::graph::CommonGraph32(numVert);
      BoostGraph Tree_temp2     = morphee::graph::CommonGraph32(numVert);
      BoostGraph Tree_old_label = morphee::graph::CommonGraph32(numVert);

      // G_gradient : contain integral of gradient along boundaries
      BoostGraph G_gradient = morphee::graph::CommonGraph32(numVert);
      // G_blength : contain length of boundaries
      BoostGraph G_blength = morphee::graph::CommonGraph32(numVert);

      std::cout << "1) Compute statistics of each regions : ... " << std::endl;
      std::vector<double> mean_values;
      std::vector<double> number_of_values;
      std::vector<double> quad_error;

      float x_star            = 0.0f;
      float sum_gradient      = 0.0f;
      float current_max_value = 0.0f;

      int **histogram; // image histogram
      histogram = new int *[numVert];

      float *histogram_region1; // region 1 histogram
      histogram_region1 = new float[256];

      float *histogram_region_merged_1; // region 1 histogram
      histogram_region_merged_1 = new float[256];

      float *histogram_region_merged_2; // region 1 histogram
      histogram_region_merged_2 = new float[256];

      for (int dim_allouee1 = 0; dim_allouee1 < 256; ++dim_allouee1) {
        histogram_region1[dim_allouee1]         = 0;
        histogram_region_merged_1[dim_allouee1] = 0;
        histogram_region_merged_2[dim_allouee1] = 0;
      }

      for (int dim_allouee0 = 0; dim_allouee0 < numVert; ++dim_allouee0) {
        histogram[dim_allouee0] = new int[256];
      }

      for (int dim_allouee0 = 0; dim_allouee0 < numVert; ++dim_allouee0) {
        for (int dim_allouee1 = 0; dim_allouee1 < 256; ++dim_allouee1) {
          histogram[dim_allouee0][dim_allouee1] = 0;
        }
      }

      //// INIT VALUES TO ZERO
      for (int i = 0; i < numVert; i++) {
        mean_values.push_back(0);
        number_of_values.push_back(0);
        quad_error.push_back(0);
      }

      //// COMPUTE MEAN VALUES, AREA, HISTOGRAM, MIN AND MAX
      for (it = imWs.begin(), iend = imWs.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o1               = it.getOffset();
        int val          = imWs.pixelFromOffset(o1);
        double val_image = imIn.pixelFromOffset(o1);

        mean_values[val - 1]      = mean_values[val - 1] + val_image;
        number_of_values[val - 1] = number_of_values[val - 1] + 1.0;

        int vald                 = (int) (val_image);
        histogram[val - 1][vald] = histogram[val - 1][vald] + 1;
      }

      //// MEAN VALUES INSIDE REGIONS
      for (int i = 0; i < numVert; i++) {
        mean_values[i] = mean_values[i] / number_of_values[i];
      }

      //// COMPUTE QUADRATIC ERROR (COMPARED TO THE MEAN VALUE) IN EACH REGION
      float max_quad_val = 0.0f;
      for (it = imWs.begin(), iend = imWs.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1               = it.getOffset();
        int val          = imWs.pixelFromOffset(o1);
        double val_image = imIn.pixelFromOffset(o1);

        // Label regions
        Tree_out.setVertexData(val - 1, 0);

        // quadratic error
        quad_error[val - 1] =
            (double) quad_error[val - 1] +
            std::pow(std::abs(val_image - (double) mean_values[val - 1]), 2);

        // Gradient integral and regions boundaries length
        if (val > 0) {
          neighb.setCenter(o1);

          for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
            const offset_t o2 = nit.getOffset();
            int val2          = imWs.pixelFromOffset(o2);

            if (o2 > o1) {
              if (val != val2) {
                boost::tie(e1, in1) =
                    boost::edge(val - 1, val2 - 1, G_gradient.getBoostGraph());
                boost::tie(e2, in1) =
                    boost::edge(val - 1, val2 - 1, G_blength.getBoostGraph());

                double val3 = imGrad.pixelFromOffset(o1);
                double val4 = imGrad.pixelFromOffset(o2);
                double maxi = std::max(val3, val4);
                sum_gradient =
                    sum_gradient + 1.0f / (1.0f + 100.0f * (maxi / 65535.0f) *
                                                      (maxi / 65535.0f));

                if (in1 == 0) {
                  numEdges++;
                  G_gradient.addEdge(val - 1, val2 - 1,
                                     1.0f / (1.0f + 100.0f * (maxi / 65535.0f) *
                                                        (maxi / 65535.0f)));
                  G_blength.addEdge(val - 1, val2 - 1, 1);
                } else {
                  G_gradient.edgeWeight(e1, &tmp);
                  G_gradient.setEdgeWeight(
                      e1, tmp + 1.0f / (1.0f + 100.0f * (maxi / 65535.0f) *
                                                   (maxi / 65535.0f)));

                  G_blength.edgeWeight(e2, &tmp2);
                  G_blength.setEdgeWeight(e2, tmp2 + 1);
                }
              }
            }
          }
        }
      }

      std::cout << "1) Compute statistics of each regions : done !"
                << std::endl;

      std::cout << "Number of of Tree Edges : " << numEdges << std::endl;

      std::cout << "2) Go through hierarchy in ascendant order ... "
                << std::endl;

      // ENSURE THAT ONLY TWO REGIONS CAN BE MERGED AT EACH STEP
      // morphee::Morpho_Graph::t_Order_Edges_Weights(Treein, Tree_ordered);

      Tree_ordered = t_CopyGraph(Tree_out);

      boost::tie(ed_it2, ed_end2) = boost::edges(Tree_out.getBoostGraph());

      for (boost::tie(ed_it, ed_end) =
               boost::edges(Tree_ordered.getBoostGraph());
           ed_it != ed_end, ed_it2 != ed_end2; ++ed_it, ++ed_it2) {
        Tree_ordered.edgeWeight(*ed_it, &tmp);
        val_edges.push_back(tmp);
        Tree_out.setEdgeWeight(*ed_it2, 0);
      }

      BoostGraph Tree_out2 = t_CopyGraph(Tree_out);

      // INIT TREETEMP
      Tree_temp = morphee::graph::CommonGraph32(numVert);

      std::vector<typename BoostGraph::EdgeDescriptor> added_edges;
      std::vector<typename BoostGraph::EdgeDescriptor> frontiers_edges;
      std::vector<typename BoostGraph::EdgeDescriptor> inside_edges;
      std::vector<typename BoostGraph::VertexDescriptor> merged_nodes1;
      std::vector<typename BoostGraph::VertexDescriptor> merged_nodes2;

      // sort edges weights to explore the hierarchy
      std::cout << "sort edges of tree" << std::endl;
      std::sort(val_edges.begin(), val_edges.end(), std::greater<double>());

      double last_edge_value             = val_edges.back();
      double last_analyzed               = last_edge_value;
      int current_label                  = numVert - 1;
      int number_of_connected_components = numVert;

      while (val_edges.size() > 0 || number_of_connected_components >= 2) {
        std::cout << last_edge_value << std::endl;
        // BEFORE MERGING
        int number_of_old_connected_components;
        t_LabelConnectedComponent(Tree_temp, Tree_old_label,
                                  &number_of_old_connected_components);

        // add edge of minimal weight
        for (boost::tie(ed_it, ed_end) =
                 boost::edges(Tree_ordered.getBoostGraph());
             ed_it != ed_end; ++ed_it) {
          Tree_ordered.edgeWeight(*ed_it, &tmp);

          if (tmp == last_edge_value) { // check if current min

            vs = Tree_ordered.edgeSource(*ed_it);
            vt = Tree_ordered.edgeTarget(*ed_it);
            boost::tie(e1, in1) =
                boost::add_edge(vs, vt, Tree_temp.getBoostGraph());
            added_edges.push_back(e1);
          }
        }

        // AFTER MERGING
        t_LabelConnectedComponent(Tree_temp, Tree_temp2,
                                  &number_of_connected_components);
        std::cout << "number_of_connected_components = "
                  << number_of_connected_components << std::endl;

        // go through removed edges and look caracteristics of regions connected
        // to it
        while (added_edges.size() > 0) {
          // local variable for mean values and areas of the merged region
          float int_value_1        = 0.0f;
          float number_of_bpoints1 = 0.0f;
          float mean_value_in1     = 0.0f;
          float quad_error_in1     = 0.0f;
          float nb_val_in1         = 0.0f;

          // local variable for mean values and areas of the regions before
          // merging
          int merged_region_1                     = 0;
          float int_value_merged_region_1         = 0.0f;
          float number_of_bpoints_merged_region_1 = 0.0f;
          float mean_value_in_merged_region_1     = 0.0f;
          float quad_error_in_merged_region_1     = 0.0f;
          float nb_val_in_merged_region_1         = 0.0f;

          int merged_region_2                     = 0;
          float int_value_merged_region_2         = 0.0f;
          float number_of_bpoints_merged_region_2 = 0.0f;
          float mean_value_in_merged_region_2     = 0.0f;
          float quad_error_in_merged_region_2     = 0.0f;
          float nb_val_in_merged_region_2         = 0.0f;

          // last added edges
          e1 = added_edges.back();
          added_edges.pop_back();

          // Get label of the regions : should be the same !!!
          Tree_temp2.vertexData(Tree_ordered.edgeSource(e1), &label1);
          Tree_temp2.vertexData(Tree_ordered.edgeTarget(e1), &label2);

          // Get old label of the regions : only two regions are merged at each
          // step !!
          Tree_old_label.vertexData(Tree_ordered.edgeSource(e1), &vdata1);
          merged_region_1 = (int) vdata1;

          Tree_old_label.vertexData(Tree_ordered.edgeTarget(e1), &vdata2);
          merged_region_2 = (int) vdata2;

          if (label1 != 0) {
            boost::tie(ed_it2, ed_end2) =
                boost::edges(G_gradient.getBoostGraph());

            for (boost::tie(ed_it, ed_end) =
                     boost::edges(G_blength.getBoostGraph());
                 ed_it != ed_end, ed_it2 != ed_end2; ++ed_it, ++ed_it2) {
              vs = G_blength.edgeSource(*ed_it);
              vt = G_blength.edgeTarget(*ed_it);

              // new label of the regions
              Tree_temp2.vertexData(vs, &vdata1);
              Tree_temp2.vertexData(vt, &vdata2);

              // old labels of the regions
              Tree_old_label.vertexData(vs, &vdata11);
              Tree_old_label.vertexData(vt, &vdata22);

              if ((vdata1 == label1 && vdata2 == label1))
                inside_edges.push_back(*ed_it);

              if ((vdata11 == merged_region_1 && vdata22 != merged_region_1 &&
                   vdata22 != merged_region_2) ||
                  (vdata22 == merged_region_1 && vdata11 != merged_region_1 &&
                   vdata11 != merged_region_2)) {
                frontiers_edges.push_back(*ed_it);
              }

              // compute integral of gradient and length of the merged region
              if ((vdata1 == label1 && vdata2 != label1) ||
                  (vdata2 == label1 && vdata1 != label1)) {
                G_gradient.edgeWeight(*ed_it2, &tmp);
                G_blength.edgeWeight(*ed_it, &tmp2);
                int_value_1        = int_value_1 + (float) tmp;
                number_of_bpoints1 = number_of_bpoints1 + (float) tmp2;
                // frontiers_edges.push_back( *ed_it );
              }

              // compute integral of gradient and length of the merged region
              if ((vdata11 == merged_region_1 && vdata22 != merged_region_1) ||
                  (vdata22 == merged_region_1 && vdata11 != merged_region_1)) {
                G_gradient.edgeWeight(*ed_it2, &tmp);
                G_blength.edgeWeight(*ed_it, &tmp2);
                int_value_merged_region_1 =
                    int_value_merged_region_1 + (float) tmp;
                number_of_bpoints_merged_region_1 =
                    number_of_bpoints_merged_region_1 + (float) tmp2;
              }

              // compute integral of gradient and length of the merged region
              if ((vdata11 == merged_region_2 && vdata22 != merged_region_2) ||
                  (vdata22 == merged_region_2 && vdata11 != merged_region_2)) {
                G_gradient.edgeWeight(*ed_it2, &tmp);
                G_blength.edgeWeight(*ed_it, &tmp2);
                int_value_merged_region_2 =
                    int_value_merged_region_2 + (float) tmp;
                number_of_bpoints_merged_region_2 =
                    number_of_bpoints_merged_region_2 + (float) tmp2;
              }
            }

            // LOOK INSIDE REGIONS, COMPUTE MEAN AND QUAD ERROR
            for (boost::tie(v_it, v_end) =
                     boost::vertices(G_blength.getBoostGraph());
                 v_it != v_end; ++v_it) {
              // Get label of the regions
              Tree_temp2.vertexData(*v_it, &vdata1);
              Tree_old_label.vertexData(*v_it, &vdata11);

              // compute mean and quad error
              if (vdata1 == label1) {
                // GET STATISTICS OF REGIONS AFTER MERGING
                for (int dim_allouee1 = 0; dim_allouee1 < 256; ++dim_allouee1) {
                  histogram_region1[dim_allouee1] =
                      histogram_region1[dim_allouee1] +
                      histogram[(int) *v_it][dim_allouee1];
                }

                if (vdata11 == merged_region_1) {
                  merged_nodes1.push_back(*v_it);

                  // GET STATISTICS OF REGIONS BEFORE MERGING
                  for (int dim_allouee1 = 0; dim_allouee1 < 256;
                       ++dim_allouee1) {
                    histogram_region_merged_1[dim_allouee1] =
                        histogram_region_merged_1[dim_allouee1] +
                        histogram[(int) *v_it][dim_allouee1];
                  }
                }

                if (vdata11 == merged_region_2) {
                  merged_nodes2.push_back(*v_it);

                  // GET STATISTICS OF REGIONS BEFORE MERGING
                  for (int dim_allouee1 = 0; dim_allouee1 < 256;
                       ++dim_allouee1) {
                    histogram_region_merged_2[dim_allouee1] =
                        histogram_region_merged_2[dim_allouee1] +
                        histogram[(int) *v_it][dim_allouee1];
                  }
                }
              }
            }

            for (int dim_allouee1 = 0; dim_allouee1 < 256; ++dim_allouee1) {
              mean_value_in1 =
                  mean_value_in1 + (float) dim_allouee1 *
                                       (float) histogram_region1[dim_allouee1];
              nb_val_in1 = nb_val_in1 + (float) histogram_region1[dim_allouee1];

              mean_value_in_merged_region_1 =
                  mean_value_in_merged_region_1 +
                  (float) dim_allouee1 *
                      (float) histogram_region_merged_1[dim_allouee1];
              nb_val_in_merged_region_1 =
                  nb_val_in_merged_region_1 +
                  (float) histogram_region_merged_1[dim_allouee1];

              mean_value_in_merged_region_2 =
                  mean_value_in_merged_region_2 +
                  (float) dim_allouee1 *
                      (float) histogram_region_merged_2[dim_allouee1];
              nb_val_in_merged_region_2 =
                  nb_val_in_merged_region_2 +
                  (float) histogram_region_merged_2[dim_allouee1];
            }

            // mean value of merged region
            mean_value_in1 = mean_value_in1 / nb_val_in1;
            mean_value_in_merged_region_1 =
                mean_value_in_merged_region_1 / nb_val_in_merged_region_1;
            mean_value_in_merged_region_2 =
                mean_value_in_merged_region_2 / nb_val_in_merged_region_2;

            for (int dim_allouee1 = 0; dim_allouee1 < 256; ++dim_allouee1) {
              quad_error_in1 =
                  quad_error_in1 + (float) histogram_region1[dim_allouee1] *
                                       ((float) dim_allouee1 - mean_value_in1) *
                                       ((float) dim_allouee1 - mean_value_in1);
              histogram_region1[dim_allouee1] = 0;

              quad_error_in_merged_region_1 =
                  quad_error_in_merged_region_1 +
                  (float) histogram_region_merged_1[dim_allouee1] *
                      ((float) dim_allouee1 - mean_value_in_merged_region_1) *
                      ((float) dim_allouee1 - mean_value_in_merged_region_1);
              histogram_region_merged_1[dim_allouee1] = 0;

              quad_error_in_merged_region_2 =
                  quad_error_in_merged_region_2 +
                  (float) histogram_region_merged_2[dim_allouee1] *
                      ((float) dim_allouee1 - mean_value_in_merged_region_2) *
                      ((float) dim_allouee1 - mean_value_in_merged_region_2);
              histogram_region_merged_2[dim_allouee1] = 0;
            }

            // float v_1 = (1.0f/(nb_val_in1-1) ) * quad_error_in1 ;
            // float v_m1 = (1.0f/(nb_val_in_merged_region_1-1) ) *
            // quad_error_in_merged_region_1 ; float v_m2 =
            // (1.0f/(nb_val_in_merged_region_2-1) ) *
            // quad_error_in_merged_region_2 ;

            // std::cout<<"mean_value_in1 =  "<<mean_value_in1<<std::endl;
            // std::cout<<"v_1 =  "<<v_1<<std::endl;
            //
            // std::cout<<"mean_value_in_merged_region_1 =
            // "<<mean_value_in_merged_region_1<<std::endl; std::cout<<"v_m1 =
            // "<<v_m1<<std::endl;
            //
            // std::cout<<"mean_value_in_merged_region_2 =
            // "<<mean_value_in_merged_region_2<<std::endl; std::cout<<"v_m2 =
            // "<<v_m2<<std::endl;

            quad_error_in1 = quad_error_in1 / 100.0f;
            quad_error_in_merged_region_1 =
                quad_error_in_merged_region_1 / 100.0f;
            quad_error_in_merged_region_2 =
                quad_error_in_merged_region_2 / 100.0f;

            // COMPUTE AFFINE ENERGY FUNCTION OF THE NEW MERGED REGION
            float D1 = quad_error_in1; // quad error ;
            float C1 =
                int_value_1 +
                number_of_bpoints1; // int_value_1 ; /// number_of_bpoints1 ; //
                                    // mean gradient value;

            // COMPUTE AFFINE ENERGY FUNCTION OF REGIONS BEFORE MERGING
            float DS1 = quad_error_in_merged_region_1; // quad error ;
            float CS1 =
                int_value_merged_region_1 +
                number_of_bpoints_merged_region_1; // int_value_merged_region_1
                                                   // ; ///
                                                   // number_of_bpoints_merged_region_1
                                                   // ;  // mean gradient value;

            float DS2 = quad_error_in_merged_region_2; // quad error ;
            float CS2 =
                int_value_merged_region_2 +
                number_of_bpoints_merged_region_2; // int_value_merged_region_2
                                                   // ; ///
                                                   // number_of_bpoints_merged_region_2
                                                   // ;  // mean gradient value;

            // Compute intersection between the two affine enrgies E1 = D1 +
            // lambda * C1 and ES1 + ES2 = DS1 + lambda * CS2 + DS2 + lambda *
            // CS2 intersection gives the scale of appearance lambda of the
            // frontier of the merged region
            float valk =
                std::max(1000.0f * -(DS1 + DS2 - D1) / (CS1 + CS2 - C1), 0.0f);
            // float lambda_star = 10000.0f * sqrt( valk ) ;
            float E_star =
                100.0f * (D1 + C1 * (-(DS1 + DS2 - D1) / (CS1 + CS2 - C1)));
            float lambda_star = 10000.0f * valk;

            // Tree_out.setVertexData(val-1,0);

            /*while( merged_nodes1.size() > 0 ){

                vs = merged_nodes1.back();
                merged_nodes1.pop_back();
                

                Tree_out2.vertexData(vs,&vdata2);

                if( vdata2 == 0 ){
                  Tree_out2.setVertexData(vs, E_star ) ;
                  Tree_out.setVertexData(vs, lambda_star ) ;
                }
                else{
                  if( vdata2 > E_star ){
                    Tree_out2.setVertexData(vs, E_star ) ;
                    Tree_out.setVertexData(vs, lambda_star ) ;
                  }
                }
            }
            

            while( merged_nodes2.size() > 0 ){
                vs = merged_nodes2.back();
                merged_nodes2.pop_back();
                

                Tree_out2.vertexData(vs,&vdata2);
                if( vdata2 == 0 ){
                  Tree_out2.setVertexData(vs, E_star ) ;
                  Tree_out.setVertexData(vs, lambda_star ) ;
                }
                else{
                  if( vdata2 > E_star ){
                    Tree_out2.setVertexData(vs, E_star ) ;
                    Tree_out.setVertexData(vs, lambda_star ) ;
                  }
                }
            }*/

            std::cout << lambda_star << std::endl;

            /*boost::tie(e1, in1) = boost::edge( Tree_ordered.edgeSource(e1) ,
            Tree_ordered.edgeTarget(e1) , Tree_out.getBoostGraph());
            Tree_out.setEdgeWeight(e1 ,  std::max( lambda_star , 0.0f   ) );

            boost::tie(e1, in1) = boost::edge( Tree_ordered.edgeSource(e1) ,
            Tree_ordered.edgeTarget(e1) , Tree_out2.getBoostGraph());
            Tree_out2.setEdgeWeight(e1 ,  std::max( E_star , 0.0f   ) );*/

            while (frontiers_edges.size() > 0) {
              ef = frontiers_edges.back();
              frontiers_edges.pop_back();

              boost::tie(e1, in1) = boost::edge(G_blength.edgeSource(ef),
                                                G_blength.edgeTarget(ef),
                                                Tree_out2.getBoostGraph());

              if (in1) {
                Tree_out2.edgeWeight(e1, &tmp);
                if (tmp == 0)
                  Tree_out2.setEdgeWeight(e1, lambda_star);
                else
                  Tree_out2.setEdgeWeight(
                      e1, std::max((float) lambda_star, (float) tmp));
              }
            }

            // Go trough merged regions chack if there is higher lambda values,
            // in wich case we have to decrease them
            while (inside_edges.size() > 0) {
              ef = inside_edges.back();
              inside_edges.pop_back();

              boost::tie(e1, in1) = boost::edge(G_blength.edgeSource(ef),
                                                G_blength.edgeTarget(ef),
                                                Tree_out2.getBoostGraph());

              if (in1) {
                Tree_out2.edgeWeight(e1, &tmp);
                if (tmp > lambda_star)
                  Tree_out2.setEdgeWeight(e1, lambda_star);
              }
            }

            inside_edges.clear();
            frontiers_edges.clear();
            merged_nodes1.clear();
            merged_nodes2.clear();

            for (int dim_allouee1 = 0; dim_allouee1 < 256; ++dim_allouee1) {
              histogram_region1[dim_allouee1]         = 0;
              histogram_region_merged_1[dim_allouee1] = 0;
              histogram_region_merged_2[dim_allouee1] = 0;
            }

          } // label1 != 0   // do not compute the same region multiple
            // times....
        }   // while( added_edges.size() > 0 ){

        while (last_edge_value == last_analyzed) {
          last_edge_value = val_edges.back();
          val_edges.pop_back();
        }
        last_analyzed = last_edge_value;
      } // while( val_edges.size() > 0 || number_of_connected_components >= 2 ){

      std::cout << "current_max_value = " << current_max_value << std::endl;

      morphee::Morpho_Graph::t_Order_Edges_Weights(Tree_out2, Tree_out);

      for (boost::tie(ed_it, ed_end) = boost::edges(Tree_out.getBoostGraph());
           ed_it != ed_end; ++ed_it) {
        Tree_out.edgeWeight(*ed_it, &tmp);
        if (tmp > current_max_value)
          current_max_value = tmp;
      }

      for (boost::tie(ed_it, ed_end) = boost::edges(Tree_out.getBoostGraph());
           ed_it != ed_end; ++ed_it) {
        Tree_out.edgeWeight(*ed_it, &tmp);
        float value_test = 65535.0f * ((float) tmp / current_max_value);
        Tree_out.setEdgeWeight(*ed_it, value_test);
      }

      // for (boost::tie(v_it, v_end)=boost::vertices(Tree_out.getBoostGraph())
      // ; v_it != v_end ; ++v_it)
      //{
      //	Tree_out.vertexData( *v_it, &vdata1);
      //	if ( vdata1 > current_max_value) current_max_value = vdata1 ;
      //}

      // for (boost::tie(v_it, v_end)=boost::vertices(Tree_out.getBoostGraph())
      // ; v_it != v_end ; ++v_it)
      //{
      //	Tree_out.vertexData( *v_it, &vdata1);
      //	float value_test = 65535.0f *  ( (float) vdata1 / current_max_value );
      //	Tree_out.setVertexData(*v_it, value_test );
      //}

      return RES_OK;
    }

    template <class ImageWs, class ImageIn, class ImageGrad, typename _alpha1,
              class SE, class BoostGraph>
    RES_C t_MSMinCutInHierarchy(const ImageWs &imWs, const ImageIn &imIn,
                                const ImageGrad &imGrad, const _alpha1 alpha1,
                                const BoostGraph &Treein, const SE &nl,
                                BoostGraph &Tree_out)
    {
      MORPHEE_ENTER_FUNCTION("t_MSMinCutInHierarchy");

      // SCALE SET HIERARCHY FUNCTION
      if ((!imWs.isAllocated())) {
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

      // common image iterator
      typename ImageWs::const_iterator it, iend;
      morphee::selement::Neighborhood<SE, ImageIn> neighb(imIn, nl);
      typename morphee::selement::Neighborhood<SE, ImageIn>::iterator nit, nend;
      offset_t o0;
      offset_t o1;

      typename BoostGraph::EdgeIterator ed_it, ed_end;
      typename BoostGraph::EdgeIterator ed_it2, ed_end2;
      typename BoostGraph::EdgeIterator ed_it3, ed_end3;
      typename BoostGraph::EdgeIterator ed_it4, ed_end4;

      typename BoostGraph::VertexIterator v_it, v_end;

      typename BoostGraph::VertexProperty vdata1, vdata2, vdata11, vdata22;
      typename BoostGraph::VertexProperty label1, label2;

      bool in1;
      int numVert  = 0;
      int numEdges = 0;
      typename BoostGraph::EdgeDescriptor e1, e2;
      typename BoostGraph::VertexDescriptor vs, vt;
      typename BoostGraph::EdgeProperty tmp, tmp2, tmp3;

      std::vector<double> val_edges;
      val_edges.push_back(0.0);

      float lambda = (float) alpha1;

      std::cout << "Compute number of regions" << std::endl;

      for (it = imWs.begin(), iend = imWs.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o0      = it.getOffset();
        int val = imWs.pixelFromOffset(o0);

        if (val > numVert) {
          numVert = val;
        }
      }
      // BMI
      INT32 imsize = imWs.getXSize() * imWs.getYSize() * imWs.getZSize();
      F_DOUBLE var_norm_factor =
          1. / (static_cast<F_DOUBLE>(imsize) * 256. * 256.);
      // END BMI
      // create some temp graphs
      Tree_out = morphee::graph::CommonGraph32(numVert);
      Tree_out = t_CopyGraph(Treein);
      // copy of input tree
      BoostGraph Tree_temp      = morphee::graph::CommonGraph32(numVert);
      BoostGraph Tree_ordered   = morphee::graph::CommonGraph32(numVert);
      BoostGraph Tree_temp2     = morphee::graph::CommonGraph32(numVert);
      BoostGraph Tree_old_label = morphee::graph::CommonGraph32(numVert);

      // G_gradient : contain integral of gradient along boundaries
      BoostGraph G_gradient = morphee::graph::CommonGraph32(numVert);
      // G_blength : contain length of boundaries
      BoostGraph G_blength = morphee::graph::CommonGraph32(numVert);

      std::cout << "1) Compute statistics of each regions : ... " << std::endl;
      std::vector<double> mean_values;
      std::vector<double> number_of_values;
      std::vector<double> quad_error;

      float x_star       = 0.0f;
      float sum_gradient = 0.0f;

      int **histogram; // image histogram
      histogram = new int *[numVert];

      float *histogram_region1; // region 1 histogram
      histogram_region1 = new float[256];

      float *histogram_region_merged_1; // region 1 histogram
      histogram_region_merged_1 = new float[256];

      float *histogram_region_merged_2; // region 1 histogram
      histogram_region_merged_2 = new float[256];

      for (int dim_allouee1 = 0; dim_allouee1 < 256; ++dim_allouee1) {
        histogram_region1[dim_allouee1]         = 0;
        histogram_region_merged_1[dim_allouee1] = 0;
        histogram_region_merged_2[dim_allouee1] = 0;
      }

      for (int dim_allouee0 = 0; dim_allouee0 < numVert; ++dim_allouee0) {
        histogram[dim_allouee0] = new int[256];
      }

      for (int dim_allouee0 = 0; dim_allouee0 < numVert; ++dim_allouee0) {
        for (int dim_allouee1 = 0; dim_allouee1 < 256; ++dim_allouee1) {
          histogram[dim_allouee0][dim_allouee1] = 0;
        }
      }

      //// INIT VALUES TO ZERO
      for (int i = 0; i < numVert; i++) {
        mean_values.push_back(0);
        number_of_values.push_back(0);
        quad_error.push_back(0);
      }

      //// COMPUTE MEAN VALUES, AREA, HISTOGRAM, MIN AND MAX
      for (it = imWs.begin(), iend = imWs.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o1               = it.getOffset();
        int val          = imWs.pixelFromOffset(o1);
        double val_image = imIn.pixelFromOffset(o1);

        mean_values[val - 1]      = mean_values[val - 1] + val_image;
        number_of_values[val - 1] = number_of_values[val - 1] + 1.0;

        int vald                 = (int) (val_image);
        histogram[val - 1][vald] = histogram[val - 1][vald] + 1;
      }

      //// MEAN VALUES INSIDE REGIONS
      for (int i = 0; i < numVert; i++) {
        mean_values[i] = mean_values[i] / number_of_values[i];
      }

      //// COMPUTE QUADRATIC ERROR (COMPARED TO THE MEAN VALUE) IN EACH REGION
      float max_quad_val = 0.0f;
      int total_length   = 0;
      for (it = imWs.begin(), iend = imWs.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1               = it.getOffset();
        int val          = imWs.pixelFromOffset(o1);
        double val_image = imIn.pixelFromOffset(o1);

        // Label regions
        Tree_out.setVertexData(val - 1, val);

        // quadratic error
        quad_error[val - 1] =
            (double) quad_error[val - 1] +
            std::pow(std::abs(val_image - (double) mean_values[val - 1]), 2);

        // Gradient integral and regions boundaries length
        if (val > 0) {
          neighb.setCenter(o1);

          for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
            const offset_t o2 = nit.getOffset();
            int val2          = imWs.pixelFromOffset(o2);

            if (o2 > o1) {
              if (val != val2) {
                boost::tie(e1, in1) =
                    boost::edge(val - 1, val2 - 1, G_gradient.getBoostGraph());
                boost::tie(e2, in1) =
                    boost::edge(val - 1, val2 - 1, G_blength.getBoostGraph());

                double val3 = imGrad.pixelFromOffset(o1);
                double val4 = imGrad.pixelFromOffset(o2);
                double maxi = std::max(val3, val4);
                sum_gradient =
                    sum_gradient + 1.0f / (1.0f + 100.0f * (maxi / 65535.0f) *
                                                      (maxi / 65535.0f));

                if (in1 == 0) {
                  numEdges++;
                  G_gradient.addEdge(val - 1, val2 - 1,
                                     1.0f / (1.0f + 100.0f * (maxi / 65535.0f) *
                                                        (maxi / 65535.0f)));
                  G_blength.addEdge(val - 1, val2 - 1, 1);
                  total_length += 1;
                } else {
                  G_gradient.edgeWeight(e1, &tmp);
                  G_gradient.setEdgeWeight(
                      e1, tmp + 1.0f / (1.0f + 100.0f * (maxi / 65535.0f) *
                                                   (maxi / 65535.0f)));

                  G_blength.edgeWeight(e2, &tmp2);
                  G_blength.setEdgeWeight(e2, tmp2 + 1);
                  total_length += 1;
                }
              }
            }
          }
        }
      }
      F_DOUBLE length_norm_factor = 1. / total_length;
      std::cout << "length_factor = " << length_norm_factor << "\n";

      std::cout << "1) Compute statistics of each regions : done !"
                << std::endl;

      std::cout << "Number of of Tree Edges : " << numEdges << std::endl;

      std::cout << "2) Go through hierarchy in ascendant order ... "
                << std::endl;

      // ENSURE THAT ONLY TWO REGIONS CAN BE MERGED AT EACH STEP
      morphee::Morpho_Graph::t_Order_Edges_Weights(Treein, Tree_ordered);

      boost::tie(ed_it2, ed_end2) = boost::edges(Tree_out.getBoostGraph());

      for (boost::tie(ed_it, ed_end) =
               boost::edges(Tree_ordered.getBoostGraph());
           ed_it != ed_end, ed_it2 != ed_end2; ++ed_it, ++ed_it2) {
        Tree_ordered.edgeWeight(*ed_it, &tmp);
        val_edges.push_back(tmp);
        Tree_out.setEdgeWeight(*ed_it2, 0);
      }

      // INIT TREETEMP
      Tree_temp = morphee::graph::CommonGraph32(numVert);

      std::vector<typename BoostGraph::EdgeDescriptor> added_edges;
      std::vector<typename BoostGraph::VertexDescriptor> merged_nodes1;
      std::vector<typename BoostGraph::VertexDescriptor> merged_nodes2;

      // sort edges weights to explore the hierarchy
      std::cout << "sort edges of tree" << std::endl;
      std::sort(val_edges.begin(), val_edges.end(), std::greater<double>());

      double last_edge_value             = val_edges.back();
      double last_analyzed               = last_edge_value;
      int current_label                  = numVert - 1;
      int number_of_connected_components = numVert;

      while (val_edges.size() > 0 || number_of_connected_components >= 2) {
        //				std::cout<<last_edge_value<<std::endl;
        // BEFORE MERGING
        int number_of_old_connected_components;
        t_LabelConnectedComponent(Tree_temp, Tree_old_label,
                                  &number_of_old_connected_components);

        // add edge of minimal weight
        for (boost::tie(ed_it, ed_end) =
                 boost::edges(Tree_ordered.getBoostGraph());
             ed_it != ed_end; ++ed_it) {
          Tree_ordered.edgeWeight(*ed_it, &tmp);

          if (tmp == last_edge_value) { // check if current min

            vs = Tree_ordered.edgeSource(*ed_it);
            vt = Tree_ordered.edgeTarget(*ed_it);
            boost::tie(e1, in1) =
                boost::add_edge(vs, vt, Tree_temp.getBoostGraph());
            added_edges.push_back(e1);
          }
        }

        // AFTER MERGING
        t_LabelConnectedComponent(Tree_temp, Tree_temp2,
                                  &number_of_connected_components);
        //				std::cout<<"number_of_connected_components =
        //"<<number_of_connected_components<<std::endl;

        // go through removed edges and look caracteristics of regions connected
        // to it
        while (added_edges.size() > 0) {
          // local variable for mean values and areas of the merged region
          float int_value_1        = 0.0f;
          float number_of_bpoints1 = 0.0f;
          float mean_value_in1     = 0.0f;
          float quad_error_in1     = 0.0f;
          float nb_val_in1         = 0.0f;

          // local variable for mean values and areas of the regions before
          // merging
          int merged_region_1                     = 0;
          float int_value_merged_region_1         = 0.0f;
          float number_of_bpoints_merged_region_1 = 0.0f;
          float mean_value_in_merged_region_1     = 0.0f;
          float quad_error_in_merged_region_1     = 0.0f;
          float nb_val_in_merged_region_1         = 0.0f;

          int merged_region_2                     = 0;
          float int_value_merged_region_2         = 0.0f;
          float number_of_bpoints_merged_region_2 = 0.0f;
          float mean_value_in_merged_region_2     = 0.0f;
          float quad_error_in_merged_region_2     = 0.0f;
          float nb_val_in_merged_region_2         = 0.0f;

          // last added edges
          e1 = added_edges.back();
          added_edges.pop_back();

          // Get label of the regions : should be the same !!!
          Tree_temp2.vertexData(Tree_ordered.edgeSource(e1), &label1);
          Tree_temp2.vertexData(Tree_ordered.edgeTarget(e1), &label2);

          // Get old label of the regions : only two regions are merged at each
          // step !!
          Tree_old_label.vertexData(Tree_ordered.edgeSource(e1), &vdata1);
          merged_region_1 = (int) vdata1;

          Tree_old_label.vertexData(Tree_ordered.edgeTarget(e1), &vdata2);
          merged_region_2 = (int) vdata2;

          if (label1 != 0) {
            boost::tie(ed_it2, ed_end2) =
                boost::edges(G_gradient.getBoostGraph());

            for (boost::tie(ed_it, ed_end) =
                     boost::edges(G_blength.getBoostGraph());
                 ed_it != ed_end, ed_it2 != ed_end2; ++ed_it, ++ed_it2) {
              vs = G_blength.edgeSource(*ed_it);
              vt = G_blength.edgeTarget(*ed_it);

              // new label of the regions
              Tree_temp2.vertexData(vs, &vdata1);
              Tree_temp2.vertexData(vt, &vdata2);

              // old labels of the regions
              Tree_old_label.vertexData(vs, &vdata11);
              Tree_old_label.vertexData(vt, &vdata22);

              // compute integral of gradient and length of the merged region
              if ((vdata1 == label1 && vdata2 != label1) ||
                  (vdata2 == label1 && vdata1 != label1)) {
                G_gradient.edgeWeight(*ed_it2, &tmp);
                G_blength.edgeWeight(*ed_it, &tmp2);
                int_value_1        = int_value_1 + (float) tmp;
                number_of_bpoints1 = number_of_bpoints1 + (float) tmp2;
              }

              // compute integral of gradient and length of the merged region
              if ((vdata11 == merged_region_1 && vdata22 != merged_region_1) ||
                  (vdata22 == merged_region_1 && vdata11 != merged_region_1)) {
                G_gradient.edgeWeight(*ed_it2, &tmp);
                G_blength.edgeWeight(*ed_it, &tmp2);
                int_value_merged_region_1 =
                    int_value_merged_region_1 + (float) tmp;
                number_of_bpoints_merged_region_1 =
                    number_of_bpoints_merged_region_1 + (float) tmp2;
              }

              // compute integral of gradient and length of the merged region
              if ((vdata11 == merged_region_2 && vdata22 != merged_region_2) ||
                  (vdata22 == merged_region_2 && vdata11 != merged_region_2)) {
                G_gradient.edgeWeight(*ed_it2, &tmp);
                G_blength.edgeWeight(*ed_it, &tmp2);
                int_value_merged_region_2 =
                    int_value_merged_region_2 + (float) tmp;
                number_of_bpoints_merged_region_2 =
                    number_of_bpoints_merged_region_2 + (float) tmp2;
              }
            }

            // LOOK INSIDE REGIONS, COMPUTE MEAN AND QUAD ERROR
            for (boost::tie(v_it, v_end) =
                     boost::vertices(G_blength.getBoostGraph());
                 v_it != v_end; ++v_it) {
              // Get label of the regions
              Tree_temp2.vertexData(*v_it, &vdata1);
              Tree_old_label.vertexData(*v_it, &vdata11);

              // compute mean and quad error
              if (vdata1 == label1) {
                // GET STATISTICS OF REGIONS AFTER MERGING
                for (int dim_allouee1 = 0; dim_allouee1 < 256; ++dim_allouee1) {
                  histogram_region1[dim_allouee1] =
                      histogram_region1[dim_allouee1] +
                      histogram[(int) *v_it][dim_allouee1];
                }

                if (vdata11 == merged_region_1) {
                  merged_nodes1.push_back(*v_it);

                  // GET STATISTICS OF REGIONS BEFORE MERGING
                  for (int dim_allouee1 = 0; dim_allouee1 < 256;
                       ++dim_allouee1) {
                    histogram_region_merged_1[dim_allouee1] =
                        histogram_region_merged_1[dim_allouee1] +
                        histogram[(int) *v_it][dim_allouee1];
                  }
                }

                if (vdata11 == merged_region_2) {
                  merged_nodes2.push_back(*v_it);

                  // GET STATISTICS OF REGIONS BEFORE MERGING
                  for (int dim_allouee1 = 0; dim_allouee1 < 256;
                       ++dim_allouee1) {
                    histogram_region_merged_2[dim_allouee1] =
                        histogram_region_merged_2[dim_allouee1] +
                        histogram[(int) *v_it][dim_allouee1];
                  }
                }
              }
            }

            for (int dim_allouee1 = 0; dim_allouee1 < 256; ++dim_allouee1) {
              mean_value_in1 =
                  mean_value_in1 + (float) dim_allouee1 *
                                       (float) histogram_region1[dim_allouee1];
              nb_val_in1 = nb_val_in1 + (float) histogram_region1[dim_allouee1];

              mean_value_in_merged_region_1 =
                  mean_value_in_merged_region_1 +
                  (float) dim_allouee1 *
                      (float) histogram_region_merged_1[dim_allouee1];
              nb_val_in_merged_region_1 =
                  nb_val_in_merged_region_1 +
                  (float) histogram_region_merged_1[dim_allouee1];

              mean_value_in_merged_region_2 =
                  mean_value_in_merged_region_2 +
                  (float) dim_allouee1 *
                      (float) histogram_region_merged_2[dim_allouee1];
              nb_val_in_merged_region_2 =
                  nb_val_in_merged_region_2 +
                  (float) histogram_region_merged_2[dim_allouee1];
            }

            // mean value of merged region
            mean_value_in1 = mean_value_in1 / nb_val_in1;
            mean_value_in_merged_region_1 =
                mean_value_in_merged_region_1 / nb_val_in_merged_region_1;
            mean_value_in_merged_region_2 =
                mean_value_in_merged_region_2 / nb_val_in_merged_region_2;

            for (int dim_allouee1 = 0; dim_allouee1 < 256; ++dim_allouee1) {
              quad_error_in1 =
                  quad_error_in1 + (float) histogram_region1[dim_allouee1] *
                                       ((float) dim_allouee1 - mean_value_in1) *
                                       ((float) dim_allouee1 - mean_value_in1);
              histogram_region1[dim_allouee1] = 0;

              quad_error_in_merged_region_1 =
                  quad_error_in_merged_region_1 +
                  (float) histogram_region_merged_1[dim_allouee1] *
                      ((float) dim_allouee1 - mean_value_in_merged_region_1) *
                      ((float) dim_allouee1 - mean_value_in_merged_region_1);
              histogram_region_merged_1[dim_allouee1] = 0;

              quad_error_in_merged_region_2 =
                  quad_error_in_merged_region_2 +
                  (float) histogram_region_merged_2[dim_allouee1] *
                      ((float) dim_allouee1 - mean_value_in_merged_region_2) *
                      ((float) dim_allouee1 - mean_value_in_merged_region_2);
              histogram_region_merged_2[dim_allouee1] = 0;
            }

            quad_error_in1 = quad_error_in1 / 100.0f;
            quad_error_in_merged_region_1 =
                quad_error_in_merged_region_1 / 100.0f;
            quad_error_in_merged_region_2 =
                quad_error_in_merged_region_2 / 100.0f;

            // COMPUTE AFFINE ENERGY FUNCTION OF THE NEW MERGED REGION
            float D1 = quad_error_in1; // quad error ;
            float C1 = number_of_bpoints1 +
                       int_value_1; // int_value_1 ; /// number_of_bpoints1 ; //
                                    // mean gradient value;
            float E1 = D1 + lambda * C1;

            // COMPUTE AFFINE ENERGY FUNCTION OF REGIONS BEFORE MERGING
            float DS1 = quad_error_in_merged_region_1; // quad error ;
            float CS1 =
                number_of_bpoints_merged_region_1 +
                int_value_merged_region_1; // int_value_merged_region_1 ; ///
                                           // number_of_bpoints_merged_region_1
                                           // ;  // mean gradient value;
            float ES1 = DS1 + lambda * CS1;

            float DS2 = quad_error_in_merged_region_2; // quad error ;
            float CS2 =
                number_of_bpoints_merged_region_2 +
                int_value_merged_region_2; // int_value_merged_region_2 ; ///
                                           // number_of_bpoints_merged_region_2
                                           // ;  // mean gradient value;
            float ES2 = DS2 + lambda * CS2;

            // BMI Try to normalize
            float beta = 0.0005;

            D1 = quad_error_in1 * var_norm_factor; // quad error ;
            C1 = (number_of_bpoints1 + int_value_1) *
                 length_norm_factor; // int_value_1 ; /// number_of_bpoints1 ;
                                     // // mean gradient value;
            E1 = D1 - lambda;
            E1 = D1 + lambda * C1;

            DS1 =
                quad_error_in_merged_region_1 * var_norm_factor; // quad error ;
            CS1 = (number_of_bpoints_merged_region_1 +
                   int_value_merged_region_1) *
                  length_norm_factor; // int_value_merged_region_1 ; ///
                                      // number_of_bpoints_merged_region_1 ;  //
                                      // mean gradient value;
            ES1 = DS1 + lambda * CS1;

            DS2 =
                quad_error_in_merged_region_2 * var_norm_factor; // quad error ;
            CS2 = (number_of_bpoints_merged_region_2 +
                   int_value_merged_region_2) *
                  length_norm_factor; // int_value_merged_region_2 ; ///
                                      // number_of_bpoints_merged_region_2 ;  //
                                      // mean gradient value;
            ES2 = DS2 + lambda * CS2;
            // BMI						std::cout<<"D1 = "<<D1<<"; DS1 = "<<DS1<<"; DS2 =
            // "<<DS2<<"; sum = "<<DS1+DS2<<std::endl;

            // END BMI

            // Cut nodes of merged regions if merging increases the energy
            if (E1 < ES1 + ES2) {
              // std::cout<<"D1 = "<<D1<<std::endl;
              // std::cout<<"C1 = "<<C1<<std::endl;
              // std::cout<<"mean_value_in1 = " << mean_value_in1 <<std::endl;

              // std::cout<<"DS1 = "<<DS1<<std::endl;
              // std::cout<<"CS1 = "<<CS1<<std::endl;
              // std::cout<<"mean_value_in_merged_region_1 = " <<
              // mean_value_in_merged_region_1 <<std::endl;
              //
              // std::cout<<"DS2 = "<<DS2<<std::endl;
              // std::cout<<"CS2 = "<<CS2<<std::endl;
              // std::cout<<"mean_value_in_merged_region_2 = " <<
              // mean_value_in_merged_region_2 <<std::endl;

              // std::cout<< " merged_nodes1.size() = "  << merged_nodes1.size()
              // <<std::endl; std::cout<< " merged_nodes2.size() = "  <<
              // merged_nodes2.size() <<std::endl;

              while (merged_nodes1.size() > 0) {
                vs = merged_nodes1.back();
                merged_nodes1.pop_back();
                Tree_out.vertexData(vs, &vdata1);
                Tree_out.setVertexData(vs, current_label);
              }

              while (merged_nodes2.size() > 0) {
                vs = merged_nodes2.back();
                merged_nodes2.pop_back();
                Tree_out.vertexData(vs, &vdata1);
                Tree_out.setVertexData(vs, current_label);
              }

              current_label++;

              boost::tie(e1, in1) = boost::edge(Tree_ordered.edgeSource(e1),
                                                Tree_ordered.edgeTarget(e1),
                                                Tree_out.getBoostGraph());
              Tree_out.setEdgeWeight(e1, E1 - (ES1 + ES2));
            }

            merged_nodes1.clear();
            merged_nodes2.clear();

            for (int dim_allouee1 = 0; dim_allouee1 < 256; ++dim_allouee1) {
              histogram_region1[dim_allouee1]         = 0;
              histogram_region_merged_1[dim_allouee1] = 0;
              histogram_region_merged_2[dim_allouee1] = 0;
            }

          } // label1 != 0   // do not compute the same region multiple
            // times....
        }   // WHILE( added_edges.size() > 0 ){

        while (last_edge_value == last_analyzed) {
          last_edge_value = val_edges.back();
          val_edges.pop_back();
        }
        last_analyzed = last_edge_value;
      } // while( val_edges.size() > 0 || number_of_connected_components >= 2 ){

      return RES_OK;
    } // END t_MSMinCutInHierarchy

    template <class ImageIn, class ImageGrad, class SE, class BoostGraph>
    RES_C t_AverageLinkageTree_minimean(const ImageIn &imIn,
                                        const ImageGrad &imGrad, const SE &nl,
                                        BoostGraph &Tout)
    {
      MORPHEE_ENTER_FUNCTION("t_AverageLinkageTree");

      if ((!imIn.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if ((!imGrad.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      // common image iterator
      typename ImageIn::const_iterator it, iend;
      morphee::selement::Neighborhood<SE, ImageIn> neighb(imIn, nl);
      typename morphee::selement::Neighborhood<SE, ImageIn>::iterator nit, nend;
      offset_t o0;
      offset_t o1;

      typename BoostGraph::EdgeIterator ed_it, ed_end;
      typename BoostGraph::EdgeIterator ed_it2, ed_end2;
      typename BoostGraph::EdgeIterator ed_it3, ed_end3;

      typename BoostGraph::VertexProperty vdata1, vdata2;
      typename BoostGraph::VertexDescriptor v1_remove, v2_remove;
      typename BoostGraph::VertexProperty label1, label2;

      bool in1;
      int numVert  = 0;
      int numEdges = 0;
      typename BoostGraph::EdgeDescriptor e1, e2, e3;
      typename BoostGraph::EdgeProperty tmp, tmp2, tmp3, tmp4;

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
      std::cout << "number of Vertices:" << numVert << std::endl;

      Tout                         = morphee::graph::CommonGraph32(numVert);
      BoostGraph Tout_label        = morphee::graph::CommonGraph32(numVert);
      BoostGraph Tout_current      = morphee::graph::CommonGraph32(numVert);
      BoostGraph Tout_current_temp = morphee::graph::CommonGraph32(numVert);
      BoostGraph Tout_current_mini = morphee::graph::CommonGraph32(numVert);

      BoostGraph Ttemp   = morphee::graph::CommonGraph32(numVert);
      BoostGraph Ttemp_2 = morphee::graph::CommonGraph32(numVert);
      BoostGraph Ttemp_3 = morphee::graph::CommonGraph32(numVert);

      int number_of_edges = 0;

      std::cout << "build Region Adjacency Graph Edges" << std::endl;

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1      = it.getOffset();
        int val = imIn.pixelFromOffset(o1);

        if (val > 0) {
          neighb.setCenter(o1);

          for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
            const offset_t o2 = nit.getOffset();
            int val2          = imIn.pixelFromOffset(o2);

            if (o2 > o1) {
              if (val != val2) {
                boost::tie(e1, in1) =
                    boost::edge(val - 1, val2 - 1, Ttemp.getBoostGraph());
                boost::tie(e2, in1) =
                    boost::edge(val - 1, val2 - 1, Ttemp_2.getBoostGraph());
                boost::tie(e3, in1) =
                    boost::edge(val - 1, val2 - 1, Ttemp_3.getBoostGraph());

                double val3 = imGrad.pixelFromOffset(o1);
                double val4 = imGrad.pixelFromOffset(o2);
                double mini = std::max(val3, val4);

                if (in1 == 0) {
                  Ttemp.addEdge(val - 1, val2 - 1, mini);
                  Ttemp_2.addEdge(val - 1, val2 - 1, 1);
                  Ttemp_3.addEdge(val - 1, val2 - 1, mini);

                  number_of_edges = number_of_edges + 1;
                } else {
                  Ttemp.edgeWeight(e1, &tmp);
                  Ttemp.setEdgeWeight(e1, tmp + mini);

                  Ttemp_2.edgeWeight(e2, &tmp2);
                  Ttemp_2.setEdgeWeight(e2, tmp2 + 1);

                  Ttemp_3.edgeWeight(e3, &tmp3);
                  if (mini < tmp3)
                    Ttemp_3.setEdgeWeight(e3, mini);
                }
              }
            }
          }
        }
      }

      std::cout << "Done" << std::endl;

      std::vector<double> val_edges;
      int current_index = 1;

      int num_connected_component;
      t_LabelConnectedComponent(Tout, Tout_label, &num_connected_component);

      Tout_current = morphee::graph::CommonGraph32(num_connected_component);
      Tout_current_temp =
          morphee::graph::CommonGraph32(num_connected_component);
      Tout_current_mini =
          morphee::graph::CommonGraph32(num_connected_component);

      std::cout << "num_connected_component : " << num_connected_component
                << std::endl;
      std::cout << "Tout.numEdges() :" << Tout.numEdges() << std::endl;

      std::cout << "construct region adjacency graph of connected components"
                << std::endl;

      boost::tie(ed_it2, ed_end2) = boost::edges(Ttemp_2.getBoostGraph());
      boost::tie(ed_it3, ed_end3) = boost::edges(Ttemp_3.getBoostGraph());

      for (boost::tie(ed_it, ed_end) = boost::edges(Ttemp.getBoostGraph());
           ed_it != ed_end, ed_it2 != ed_end2, ed_it3 != ed_end3;
           ++ed_it, ++ed_it2, ++ed_it3) {
        Tout_label.vertexData(Ttemp.edgeSource(*ed_it), &vdata1);
        Tout_label.vertexData(Ttemp.edgeTarget(*ed_it), &vdata2);

        if (vdata1 != vdata2) {
          boost::tie(e1, in1) = boost::edge((int) vdata1 - 1, (int) vdata2 - 1,
                                            Tout_current.getBoostGraph());
          boost::tie(e2, in1) = boost::edge((int) vdata1 - 1, (int) vdata2 - 1,
                                            Tout_current_temp.getBoostGraph());
          boost::tie(e3, in1) = boost::edge((int) vdata1 - 1, (int) vdata2 - 1,
                                            Tout_current_mini.getBoostGraph());

          Ttemp.edgeWeight(*ed_it, &tmp);
          Ttemp_2.edgeWeight(*ed_it2, &tmp2);
          Ttemp_3.edgeWeight(*ed_it3, &tmp4);

          if (in1 == 0) {
            Tout_current.addEdge((int) vdata1 - 1, (int) vdata2 - 1,
                                 ((double) tmp));
            Tout_current_temp.addEdge((int) vdata1 - 1, (int) vdata2 - 1,
                                      ((double) tmp2));
            Tout_current_mini.addEdge((int) vdata1 - 1, (int) vdata2 - 1,
                                      ((double) tmp3));
          } else {
            Tout_current.edgeWeight(e1, &tmp3);
            Tout_current.setEdgeWeight(e1, (double) tmp3 + ((double) tmp));

            Tout_current_temp.edgeWeight(e2, &tmp3);
            Tout_current_temp.setEdgeWeight(e2,
                                            (double) tmp3 + ((double) tmp2));

            Tout_current_mini.edgeWeight(e3, &tmp3);
            if (tmp4 < tmp3)
              Tout_current_temp.setEdgeWeight(e3, tmp4);
          }
        }
      }

      std::cout << "get edges values and sort them" << std::endl;
      // get edges values and sort them

      double min_edge_value = 10000000000000.0f;
      double minimum_value  = 10000000000000.0f;

      boost::tie(ed_it2, ed_end2) =
          boost::edges(Tout_current_temp.getBoostGraph());
      boost::tie(ed_it3, ed_end3) =
          boost::edges(Tout_current_mini.getBoostGraph());

      for (boost::tie(ed_it, ed_end) =
               boost::edges(Tout_current.getBoostGraph());
           ed_it != ed_end, ed_it2 != ed_end2, ed_it3 != ed_end3;
           ++ed_it, ++ed_it2, ++ed_it3) {
        Tout_current.edgeWeight(*ed_it, &tmp);
        Tout_current_temp.edgeWeight(*ed_it2, &tmp2);
        Tout_current_mini.edgeWeight(*ed_it3, &tmp3);

        if (sqrt(((double) tmp3) * ((double) tmp / (double) tmp2)) <
            min_edge_value) {
          min_edge_value =
              sqrt(((double) tmp3) * ((double) tmp / (double) tmp2));
          v1_remove = Tout_current.edgeSource(*ed_it);
          v2_remove = Tout_current.edgeTarget(*ed_it);
        }
      }

      std::cout << "label trees" << std::endl;
      // label trees
      while (num_connected_component > 1) {
        std::cout << num_connected_component << std::endl;

        bool added_edge = false;

        while (added_edge == false) {
          boost::tie(ed_it2, ed_end2) = boost::edges(Ttemp_2.getBoostGraph());

          for (boost::tie(ed_it, ed_end) = boost::edges(Ttemp.getBoostGraph());
               ed_it != ed_end, ed_it2 != ed_end2; ++ed_it, ++ed_it2) {
            Tout_label.vertexData(Ttemp.edgeSource(*ed_it), &vdata1);
            Tout_label.vertexData(Ttemp.edgeTarget(*ed_it), &vdata2);

            if (((vdata1 - 1) == v1_remove && (vdata2 - 1) == v2_remove) ||
                ((vdata2 - 1) == v1_remove && (vdata1 - 1) == v2_remove)) {
              added_edge = true;
              Tout.addEdge(Ttemp.edgeSource(*ed_it), Ttemp.edgeTarget(*ed_it),
                           min_edge_value);
              current_index = current_index + 1;
            }

            if (added_edge == true)
              break;
          }
        }

        // clear edges and update edges values
        t_LabelConnectedComponent(Tout, Tout_label, &num_connected_component);

        Tout_current = morphee::graph::CommonGraph32(num_connected_component);
        Tout_current_temp =
            morphee::graph::CommonGraph32(num_connected_component);
        Tout_current_mini =
            morphee::graph::CommonGraph32(num_connected_component);

        // construct region adjacency graph of connected components
        boost::tie(ed_it2, ed_end2) = boost::edges(Ttemp_2.getBoostGraph());
        boost::tie(ed_it3, ed_end3) = boost::edges(Ttemp_3.getBoostGraph());

        for (boost::tie(ed_it, ed_end) = boost::edges(Ttemp.getBoostGraph());
             ed_it != ed_end, ed_it2 != ed_end2, ed_it3 != ed_end3;
             ++ed_it, ++ed_it2, ++ed_it3) {
          Tout_label.vertexData(Ttemp.edgeSource(*ed_it), &vdata1);
          Tout_label.vertexData(Ttemp.edgeTarget(*ed_it), &vdata2);

          if (vdata1 != vdata2) {
            boost::tie(e1, in1) =
                boost::edge((int) vdata1 - 1, (int) vdata2 - 1,
                            Tout_current.getBoostGraph());
            boost::tie(e2, in1) =
                boost::edge((int) vdata1 - 1, (int) vdata2 - 1,
                            Tout_current_temp.getBoostGraph());
            boost::tie(e3, in1) =
                boost::edge((int) vdata1 - 1, (int) vdata2 - 1,
                            Tout_current_mini.getBoostGraph());

            Ttemp.edgeWeight(*ed_it, &tmp);
            Ttemp_2.edgeWeight(*ed_it2, &tmp2);
            Ttemp_3.edgeWeight(*ed_it3, &tmp4);

            if (in1 == 0) {
              Tout_current.addEdge((int) vdata1 - 1, (int) vdata2 - 1,
                                   ((double) tmp));
              Tout_current_temp.addEdge((int) vdata1 - 1, (int) vdata2 - 1,
                                        ((double) tmp2));
              Tout_current_mini.addEdge((int) vdata1 - 1, (int) vdata2 - 1,
                                        ((double) tmp3));
            } else {
              Tout_current.edgeWeight(e1, &tmp3);
              Tout_current.setEdgeWeight(e1, (double) tmp3 + ((double) tmp));

              Tout_current_temp.edgeWeight(e2, &tmp3);
              Tout_current_temp.setEdgeWeight(e2,
                                              (double) tmp3 + ((double) tmp2));

              Tout_current_mini.edgeWeight(e3, &tmp3);
              if (tmp4 < tmp3)
                Tout_current_temp.setEdgeWeight(e3, tmp4);
            }
          }
        }

        min_edge_value = 10000000000000.0f;
        boost::tie(ed_it2, ed_end2) =
            boost::edges(Tout_current_temp.getBoostGraph());
        boost::tie(ed_it3, ed_end3) =
            boost::edges(Tout_current_mini.getBoostGraph());

        for (boost::tie(ed_it, ed_end) =
                 boost::edges(Tout_current.getBoostGraph());
             ed_it != ed_end, ed_it2 != ed_end2, ed_it3 != ed_end3;
             ++ed_it, ++ed_it2, ++ed_it3) {
          Tout_current.edgeWeight(*ed_it, &tmp);
          Tout_current_temp.edgeWeight(*ed_it2, &tmp2);
          Tout_current_mini.edgeWeight(*ed_it3, &tmp3);

          if (sqrt(((double) tmp3) * ((double) tmp / (double) tmp2)) <
              min_edge_value) {
            min_edge_value =
                sqrt(((double) tmp3) * ((double) tmp / (double) tmp2));
            v1_remove = Tout_current.edgeSource(*ed_it);
            v2_remove = Tout_current.edgeTarget(*ed_it);
          }
        }
      }

      return RES_OK;
    }

    template <class ImageIn, class ImageGrad, class SE, class BoostGraph>
    RES_C t_AverageLinkageTree(const ImageIn &imIn, const ImageGrad &imGrad,
                               const SE &nl, BoostGraph &Tout)
    {
      MORPHEE_ENTER_FUNCTION("t_AverageLinkageTree");

      if ((!imIn.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if ((!imGrad.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      // common image iterator
      typename ImageIn::const_iterator it, iend;
      morphee::selement::Neighborhood<SE, ImageIn> neighb(imIn, nl);
      typename morphee::selement::Neighborhood<SE, ImageIn>::iterator nit, nend;
      offset_t o0;
      offset_t o1;

      typename BoostGraph::EdgeIterator ed_it, ed_end;
      typename BoostGraph::EdgeIterator ed_it2, ed_end2;

      typename BoostGraph::VertexProperty vdata1, vdata2;
      typename BoostGraph::VertexDescriptor v1_remove, v2_remove;
      typename BoostGraph::VertexProperty label1, label2;

      bool in1;
      int numVert  = 0;
      int numEdges = 0;
      typename BoostGraph::EdgeDescriptor e1, e2;
      typename BoostGraph::EdgeProperty tmp, tmp2, tmp3;

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
      std::cout << "number of Vertices:" << numVert << std::endl;

      Tout                         = morphee::graph::CommonGraph32(numVert);
      BoostGraph Tout_label        = morphee::graph::CommonGraph32(numVert);
      BoostGraph Tout_current      = morphee::graph::CommonGraph32(numVert);
      BoostGraph Tout_current_temp = morphee::graph::CommonGraph32(numVert);

      BoostGraph Ttemp   = morphee::graph::CommonGraph32(numVert);
      BoostGraph Ttemp_2 = morphee::graph::CommonGraph32(numVert);

      int number_of_edges = 0;

      std::cout << "build Region Adjacency Graph Edges" << std::endl;

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1      = it.getOffset();
        int val = imIn.pixelFromOffset(o1);

        if (val > 0) {
          neighb.setCenter(o1);

          for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
            const offset_t o2 = nit.getOffset();
            int val2          = imIn.pixelFromOffset(o2);

            if (o2 > o1) {
              if (val != val2) {
                boost::tie(e1, in1) =
                    boost::edge(val - 1, val2 - 1, Ttemp.getBoostGraph());
                boost::tie(e2, in1) =
                    boost::edge(val - 1, val2 - 1, Ttemp_2.getBoostGraph());

                double val3 = imGrad.pixelFromOffset(o1);
                double val4 = imGrad.pixelFromOffset(o2);
                double mini = std::max(val3, val4);

                if (in1 == 0) {
                  Ttemp.addEdge(val - 1, val2 - 1, mini);
                  Ttemp_2.addEdge(val - 1, val2 - 1, 1);
                  number_of_edges = number_of_edges + 1;
                } else {
                  Ttemp.edgeWeight(e1, &tmp);
                  Ttemp.setEdgeWeight(e1, tmp + mini);

                  Ttemp_2.edgeWeight(e2, &tmp2);
                  Ttemp_2.setEdgeWeight(e2, tmp2 + 1);
                }
              }
            }
          }
        }
      }

      std::cout << "Done" << std::endl;

      std::vector<double> val_edges;
      int current_index = 1;
      int num_connected_component;
      t_LabelConnectedComponent(Tout, Tout_label, &num_connected_component);

      Tout_current = morphee::graph::CommonGraph32(num_connected_component);
      Tout_current_temp =
          morphee::graph::CommonGraph32(num_connected_component);

      std::cout << "num_connected_component : " << num_connected_component
                << std::endl;
      std::cout << "Tout.numEdges() :" << Tout.numEdges() << std::endl;

      std::cout << "construct region adjacency graph of connected components"
                << std::endl;

      boost::tie(ed_it2, ed_end2) = boost::edges(Ttemp_2.getBoostGraph());

      for (boost::tie(ed_it, ed_end) = boost::edges(Ttemp.getBoostGraph());
           ed_it != ed_end, ed_it2 != ed_end2; ++ed_it, ++ed_it2) {
        Tout_label.vertexData(Ttemp.edgeSource(*ed_it), &vdata1);
        Tout_label.vertexData(Ttemp.edgeTarget(*ed_it), &vdata2);

        if (vdata1 != vdata2) {
          boost::tie(e1, in1) = boost::edge((int) vdata1 - 1, (int) vdata2 - 1,
                                            Tout_current.getBoostGraph());
          boost::tie(e2, in1) = boost::edge((int) vdata1 - 1, (int) vdata2 - 1,
                                            Tout_current_temp.getBoostGraph());

          Ttemp.edgeWeight(*ed_it, &tmp);
          Ttemp_2.edgeWeight(*ed_it2, &tmp2);

          if (in1 == 0) {
            Tout_current.addEdge((int) vdata1 - 1, (int) vdata2 - 1,
                                 ((double) tmp));
            Tout_current_temp.addEdge((int) vdata1 - 1, (int) vdata2 - 1,
                                      ((double) tmp2));
          } else {
            Tout_current.edgeWeight(e1, &tmp3);
            Tout_current.setEdgeWeight(e1, (double) tmp3 + ((double) tmp));

            Tout_current_temp.edgeWeight(e2, &tmp3);
            Tout_current_temp.setEdgeWeight(e2,
                                            (double) tmp3 + ((double) tmp2));
          }
        }
      }

      std::cout << "get edges values and sort them" << std::endl;
      // get edges values and sort them

      double min_edge_value = 10000000000000.0f;
      double minimum_value  = 10000000000000.0f;

      boost::tie(ed_it2, ed_end2) =
          boost::edges(Tout_current_temp.getBoostGraph());
      for (boost::tie(ed_it, ed_end) =
               boost::edges(Tout_current.getBoostGraph());
           ed_it != ed_end, ed_it2 != ed_end2; ++ed_it, ++ed_it2) {
        Tout_current.edgeWeight(*ed_it, &tmp);
        Tout_current_temp.edgeWeight(*ed_it2, &tmp2);

        if (((double) tmp / (double) tmp2) < min_edge_value) {
          min_edge_value = (double) tmp / (double) tmp2;
          v1_remove      = Tout_current.edgeSource(*ed_it);
          v2_remove      = Tout_current.edgeTarget(*ed_it);
        }
      }

      std::cout << "label trees" << std::endl;
      // label trees
      while (num_connected_component > 1) { // TOTO

        std::cout << num_connected_component << std::endl;

        bool added_edge = false;

        while (added_edge == false) {
          boost::tie(ed_it2, ed_end2) = boost::edges(Ttemp_2.getBoostGraph());

          for (boost::tie(ed_it, ed_end) = boost::edges(Ttemp.getBoostGraph());
               ed_it != ed_end, ed_it2 != ed_end2; ++ed_it, ++ed_it2) {
            Tout_label.vertexData(Ttemp.edgeSource(*ed_it), &vdata1);
            Tout_label.vertexData(Ttemp.edgeTarget(*ed_it), &vdata2);

            if (((vdata1 - 1) == v1_remove && (vdata2 - 1) == v2_remove) ||
                ((vdata2 - 1) == v1_remove && (vdata1 - 1) == v2_remove)) {
              added_edge = true;
              std::cout << "ADDED_EDGE" << vdata1 << "," << vdata2 << std::endl;

              Tout.addEdge(Ttemp.edgeSource(*ed_it), Ttemp.edgeTarget(*ed_it),
                           min_edge_value);
              current_index = current_index + 1;
            }

            if (added_edge == true)
              break;
          }
        }

        // clear edges and update edges values
        t_LabelConnectedComponent(Tout, Tout_label, &num_connected_component);
        std::cout << "after clear edges" << num_connected_component
                  << std::endl;

        Tout_current = morphee::graph::CommonGraph32(num_connected_component);
        Tout_current_temp =
            morphee::graph::CommonGraph32(num_connected_component);

        // construct region adjacency graph of connected components
        boost::tie(ed_it2, ed_end2) = boost::edges(Ttemp_2.getBoostGraph());
        for (boost::tie(ed_it, ed_end) = boost::edges(Ttemp.getBoostGraph());
             ed_it != ed_end, ed_it2 != ed_end2; ++ed_it, ++ed_it2) {
          Tout_label.vertexData(Ttemp.edgeSource(*ed_it), &vdata1);
          Tout_label.vertexData(Ttemp.edgeTarget(*ed_it), &vdata2);

          if (vdata1 != vdata2) {
            boost::tie(e1, in1) =
                boost::edge((int) vdata1 - 1, (int) vdata2 - 1,
                            Tout_current.getBoostGraph());
            boost::tie(e2, in1) =
                boost::edge((int) vdata1 - 1, (int) vdata2 - 1,
                            Tout_current_temp.getBoostGraph());

            Ttemp.edgeWeight(*ed_it, &tmp);
            Ttemp_2.edgeWeight(*ed_it2, &tmp2);

            if (in1 == 0) {
              Tout_current.addEdge((int) vdata1 - 1, (int) vdata2 - 1,
                                   ((double) tmp));
              Tout_current_temp.addEdge((int) vdata1 - 1, (int) vdata2 - 1,
                                        ((double) tmp2));
            } else {
              Tout_current.edgeWeight(e1, &tmp3);
              Tout_current.setEdgeWeight(e1, (double) tmp3 + ((double) tmp));

              Tout_current_temp.edgeWeight(e2, &tmp3);
              Tout_current_temp.setEdgeWeight(e2,
                                              (double) tmp3 + ((double) tmp2));
            }
          }
        }

        min_edge_value = 10000000000000.0f;
        boost::tie(ed_it2, ed_end2) =
            boost::edges(Tout_current_temp.getBoostGraph());

        for (boost::tie(ed_it, ed_end) =
                 boost::edges(Tout_current.getBoostGraph());
             ed_it != ed_end, ed_it2 != ed_end2; ++ed_it, ++ed_it2) {
          Tout_current.edgeWeight(*ed_it, &tmp);
          Tout_current_temp.edgeWeight(*ed_it2, &tmp2);

          if (((double) tmp / (double) tmp2) < min_edge_value) {
            min_edge_value = (double) tmp / (double) tmp2;
            v1_remove      = Tout_current.edgeSource(*ed_it);
            v2_remove      = Tout_current.edgeTarget(*ed_it);
          }
        }

      } // END	while ( num_connected_component > 1)

      return RES_OK;
    }

    template <class ImageIn, class BoostGraph>
    RES_C t_Centrality_Edges_Weighting(const ImageIn &imIn, std::vector<int> p,
                                       int vRoot, BoostGraph &Tout)
    {
      MORPHEE_ENTER_FUNCTION("t_Centrality_Edges_Weighting");

      typename BoostGraph::VertexIterator v_it, v_end;
      typename BoostGraph::VertexIterator v_it2, v_end2;

      typename BoostGraph::VertexProperty vdata1, vdata2;
      typename BoostGraph::VertexDescriptor v1_remove, v2_remove, current_node;
      typename BoostGraph::VertexProperty label1, label2;

      typename BoostGraph::EdgeIterator ed_it, ed_end;
      typename BoostGraph::EdgeIterator ed_it2, ed_end2;

      typename BoostGraph::EdgeDescriptor e1, e2;
      typename BoostGraph::EdgeProperty tmp, tmp2, tmp3;

      // common image iterator
      typename ImageIn::const_iterator it, iend;
      offset_t o0, current_offset;
      offset_t o1;
      bool in1;

      ImageIn imOut = imIn.getSame();

      // SCAN IMAGE TO FIND THE PIXELS LABELS
      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        current_offset = it.getOffset();
        ;
        imOut.setPixel(current_offset, 0);
      }

      // SCAN IMAGE TO FIND THE PIXELS LABELS
      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        current_offset = it.getOffset();
        ;

        // IF PIXELS IS MARKED
        int marker = imIn.pixelFromOffset(current_offset);
        int pixout = imOut.pixelFromOffset(current_offset);

        if (p[current_offset] != vRoot && p[current_offset] > 0)
          imOut.setPixel(p[current_offset], pixout + 1);
      }

      // SCAN IMAGE TO FIND THE PIXELS LABELS
      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        current_offset = it.getOffset();
        ;

        // IF PIXELS IS MARKED
        int pixout = imOut.pixelFromOffset(current_offset);
        int marker = imIn.pixelFromOffset(current_offset);

        if (pixout == 0) {
          while (current_offset > 0 && current_offset != vRoot && marker == 0) {
            boost::tie(e1, in1) = boost::edge(current_offset, p[current_offset],
                                              Tout.getBoostGraph());

            if (in1 == 0) {
              Tout.addEdge(current_offset, p[current_offset], 1);
            } else {
              Tout.edgeWeight(e1, &tmp);
              Tout.setEdgeWeight(e1, tmp + 1);
            }
          }
        }
      }

      return RES_OK;
    }

    template <class ImageWs, class ImageIn, class ImageGrad, class SE,
              class BoostGraph>
    RES_C t_AverageLinkageTree_MS(const ImageWs &imWs, const ImageIn &imIn,
                                  const ImageGrad &imGrad, const SE &nl,
                                  BoostGraph &Tout)
    {
      MORPHEE_ENTER_FUNCTION("t_AverageLinkageTree_MS");

      /// Marker image:: assume single seed
      if ((!imWs.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      /// Original image.
      if ((!imIn.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      /// Gradient image.
      if ((!imGrad.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      typename BoostGraph::VertexIterator v_it, v_end;
      typename BoostGraph::VertexIterator v_it2, v_end2;

      typename BoostGraph::VertexProperty vdata1, vdata2;
      typename BoostGraph::VertexDescriptor v1_remove, v2_remove, current_node;
      typename BoostGraph::VertexProperty label1, label2;

      typename BoostGraph::EdgeIterator ed_it, ed_end;
      typename BoostGraph::EdgeIterator ed_it2, ed_end2;
      typename BoostGraph::EdgeDescriptor et1, et2;
      typename BoostGraph::EdgeProperty tmp, tmp2, tmp3;

      // common image iterator
      typename ImageIn::const_iterator it, iend;
      morphee::selement::Neighborhood<SE, ImageIn> neighb(imIn, nl);
      typename morphee::selement::Neighborhood<SE, ImageIn>::iterator nit, nend;
      offset_t o0;
      offset_t o1;

      std::queue<int> temp_q;

      //----------------------------------------------------------------
      // Needed for spanning tree computation
      //----------------------------------------------------------------
      typedef boost::adjacency_list<
          boost::vecS, boost::vecS, boost::undirectedS,
          boost::property<boost::vertex_distance_t, double>,
          boost::property<boost::edge_capacity_t, double>>
          Graph_d;

      Graph_d g;

      boost::property_map<Graph_d, boost::edge_capacity_t>::type weightmap =
          boost::get(boost::edge_capacity, g);

      bool in1;
      Graph_d::edge_descriptor e1, e2, e3, e4;
      Graph_d::vertex_descriptor vRoot;
      int numVert = 0;

      //----------------------------------------------------------------
      //----------------------------------------------------------------

      std::cout << "build graph vertices" << std::endl;

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o0 = it.getOffset();
        boost::add_vertex(g);
        numVert++;
      }

      BoostGraph T_temp = morphee::graph::CommonGraph32(numVert);

      vRoot = boost::add_vertex(g);

      std::cout << "number of vertices: " << numVert << std::endl;

      std::cout << "build graph edges" << std::endl;

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1         = it.getOffset();
        double val = imGrad.pixelFromOffset(o1);
        int marker = imWs.pixelFromOffset(o1);

        if (marker > 0) {
          boost::tie(e4, in1) = boost::add_edge(vRoot, o1, g);
          weightmap[e4]       = 0.1;
        }

        neighb.setCenter(o1);

        for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
          const offset_t o2 = nit.getOffset();
          if (o2 <= o1)
            continue;
          if (o2 > o1) {
            double val2         = imGrad.pixelFromOffset(o2);
            double cost         = std::max(val, val2);
            boost::tie(e4, in1) = boost::add_edge(o1, o2, g);
            weightmap[e4]       = cost;
          }
        }
      }

      std::cout << "Compute Minimum Spanning Forest" << std::endl;

      std::vector<boost::graph_traits<Graph_d>::vertex_descriptor> p(
          boost::num_vertices(g));
      boost::property_map<Graph_d, boost::vertex_distance_t>::type distancemap =
          boost::get(boost::vertex_distance, g);
      boost::property_map<Graph_d, boost::vertex_index_t>::type indexmap2 =
          boost::get(boost::vertex_index, g);

      dijkstra_shortest_paths(g, vRoot, &p[0], distancemap, weightmap,
                              indexmap2, std::less<double>(),
                              boost::detail::maximum<double>(),
                              (std::numeric_limits<double>::max)(), 0,
                              boost::default_dijkstra_visitor());

      std::vector<int> p_int;

      for (int i = 0; i < numVert; i++) {
        p_int[i] = (int) p[i];
      }

      t_Centrality_Edges_Weighting(imWs, p_int, (int) vRoot, Tout);

      return RES_OK;
    }

    template <class ImageIn, class ImageGrad, class SE, class BoostGraph>
    RES_C t_NeighborhoodGraphFromMosaic_WithMinValue(const ImageIn &imIn,
                                                     const ImageGrad &imGrad,
                                                     const SE &nl,
                                                     BoostGraph &Gout)
    {
      MORPHEE_ENTER_FUNCTION("t_NeighborhoodGraphFromMosaic_WithMinValue");

      if ((!imIn.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if ((!imGrad.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      // common image iterator
      typename ImageIn::const_iterator it, iend;
      morphee::selement::Neighborhood<SE, ImageIn> neighb(imIn, nl);
      typename morphee::selement::Neighborhood<SE, ImageIn>::iterator nit, nend;
      offset_t o0;
      offset_t o1;

      typename BoostGraph::EdgeIterator ed_it, ed_end;
      typename BoostGraph::EdgeIterator ed_it2, ed_end2;

      bool in1;
      int numVert  = 0;
      int numEdges = 0;
      typename BoostGraph::EdgeDescriptor e1, e2;
      typename BoostGraph::EdgeProperty tmp, tmp2;

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
      std::cout << "number of Vertices:" << numVert << std::endl;

      Gout             = morphee::graph::CommonGraph32(numVert);
      BoostGraph Gtemp = morphee::graph::CommonGraph32(numVert);

      std::cout << "build Region Adjacency Graph Edges" << std::endl;

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1      = it.getOffset();
        int val = imIn.pixelFromOffset(o1);

        if (val > 0) {
          neighb.setCenter(o1);

          for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
            const offset_t o2 = nit.getOffset();
            int val2          = imIn.pixelFromOffset(o2);

            if (o2 > o1) {
              if (val != val2) {
                boost::tie(e1, in1) =
                    boost::edge(val - 1, val2 - 1, Gout.getBoostGraph());
                boost::tie(e2, in1) =
                    boost::edge(val - 1, val2 - 1, Gtemp.getBoostGraph());

                double val3 = imGrad.pixelFromOffset(o1);
                double val4 = imGrad.pixelFromOffset(o2);
                double mini = std::max(val3, val4);

                if (in1 == 0) {
                  Gout.addEdge(val - 1, val2 - 1, mini);
                  Gtemp.addEdge(val - 1, val2 - 1, 1);
                } else {
                  Gout.edgeWeight(e1, &tmp);
                  Gout.setEdgeWeight(e1, tmp + mini);

                  Gtemp.edgeWeight(e2, &tmp2);
                  Gtemp.setEdgeWeight(e2, tmp2 + 1);
                }
              }
            }
          }
        }
      }

      boost::tie(ed_it2, ed_end2) = boost::edges(Gtemp.getBoostGraph());
      // Weights the graph Gout with mean gradient value along boundary
      for (boost::tie(ed_it, ed_end) = boost::edges(Gout.getBoostGraph());
           ed_it != ed_end, ed_it2 != ed_end2; ++ed_it, ++ed_it2) {
        Gout.edgeWeight(*ed_it, &tmp);
        Gtemp.edgeWeight(*ed_it2, &tmp2);

        Gout.setEdgeWeight(*ed_it,
                           ((double) tmp) / ((double) tmp2)); // sum gradient
      }

      return RES_OK;
    }

    template <class ImageIn, class ImageGrad, class ImageVal, typename _alpha,
              class SE, class BoostGraph>
    RES_C t_NeighborhoodGraphFromMosaic_WithMeanGradientValue_AndQuadError(
        const ImageIn &imIn, const ImageGrad &imGrad, const ImageVal &imVal,
        const _alpha alpha, const SE &nl, BoostGraph &Gout)
    {
      MORPHEE_ENTER_FUNCTION(
          "t_NeighborhoodGraphFromMosaic_WithMeanGradientValue_AndQuadError");

      if ((!imIn.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if ((!imGrad.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if ((!imVal.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      // common image iterator
      typename ImageIn::const_iterator it, iend;
      morphee::selement::Neighborhood<SE, ImageIn> neighb(imIn, nl);
      typename morphee::selement::Neighborhood<SE, ImageIn>::iterator nit, nend;
      offset_t o0;
      offset_t o1;

      typename BoostGraph::EdgeIterator ed_it, ed_end;
      typename BoostGraph::VertexIterator v_it, v_end;
      typename BoostGraph::EdgeIterator ed_it2, ed_end2;

      bool in1;
      int numVert  = 0;
      int numEdges = 0;
      typename BoostGraph::EdgeDescriptor e1, e2;
      typename BoostGraph::EdgeProperty tmp, tmp2;
      typename BoostGraph::VertexProperty quad1, quad2;

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

      std::vector<double> mean_values;
      std::vector<double> number_of_values;
      std::vector<double> quad_error;

      for (int i = 0; i < numVert; i++) {
        mean_values.push_back(0);
        number_of_values.push_back(0);
        quad_error.push_back(0);
      }

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o1                        = it.getOffset();
        int val                   = imIn.pixelFromOffset(o1);
        double val_image          = imVal.pixelFromOffset(o1);
        mean_values[val - 1]      = mean_values[val - 1] + val_image;
        number_of_values[val - 1] = number_of_values[val - 1] + 1.0;
      }

      for (int i = 0; i < numVert; i++) {
        mean_values[i] = mean_values[i] / number_of_values[i];
      }

      std::cout << "build Region Adjacency Graph Vertices" << std::endl;
      std::cout << "number of Vertices:" << numVert << std::endl;

      Gout             = morphee::graph::CommonGraph32(numVert);
      BoostGraph Gtemp = morphee::graph::CommonGraph32(numVert);

      std::cout << "build Region Adjacency Graph Edges" << std::endl;

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1               = it.getOffset();
        int val          = imIn.pixelFromOffset(o1);
        double val_image = imVal.pixelFromOffset(o1);

        quad_error[val - 1] =
            (double) quad_error[val - 1] +
            std::pow(std::abs(val_image - (double) mean_values[val - 1]), 2);

        if (val > 0) {
          neighb.setCenter(o1);

          for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
            const offset_t o2 = nit.getOffset();
            int val2          = imIn.pixelFromOffset(o2);

            if (o2 > o1) {
              if (val != val2) {
                boost::tie(e1, in1) =
                    boost::edge(val - 1, val2 - 1, Gout.getBoostGraph());
                boost::tie(e2, in1) =
                    boost::edge(val - 1, val2 - 1, Gtemp.getBoostGraph());

                double val3 = imGrad.pixelFromOffset(o1);
                double val4 = imGrad.pixelFromOffset(o2);
                double maxi = std::max(val3, val4);

                if (in1 == 0) {
                  numEdges++;
                  Gout.addEdge(val - 1, val2 - 1, maxi);
                  Gtemp.addEdge(val - 1, val2 - 1, 1);
                } else {
                  Gout.edgeWeight(e1, &tmp);
                  Gout.setEdgeWeight(e1, (double) tmp + maxi);

                  Gtemp.edgeWeight(e2, &tmp2);
                  Gtemp.setEdgeWeight(e2, (double) tmp2 + 1);
                }
              }
            }
          }
        }
      }

      std::cout << "number of Edges : " << numEdges << std::endl;

      int current_vertex = 0;
      double max_value   = 0;

      for (boost::tie(v_it, v_end) = boost::vertices(Gout.getBoostGraph());
           v_it != v_end; ++v_it) {
        // Gout.setVertexData(*v_it, number_of_values[current_vertex] +
        // quad_error[current_vertex] );

        if (max_value < quad_error[current_vertex])
          max_value = quad_error[current_vertex];

        Gout.setVertexData(*v_it, mean_values[current_vertex]);
        current_vertex = current_vertex + 1;
        // std::cout<< mean_values[current_vertex] <<std::endl;
      }

      boost::tie(ed_it2, ed_end2) = boost::edges(Gtemp.getBoostGraph());

      for (boost::tie(ed_it, ed_end) = boost::edges(Gout.getBoostGraph());
           ed_it != ed_end, ed_it2 != ed_end2; ++ed_it, ++ed_it2) {
        Gout.edgeWeight(*ed_it, &tmp);
        Gtemp.edgeWeight(*ed_it2, &tmp2);

        Gout.vertexData(Gout.edgeSource(*ed_it), &quad1);
        Gout.vertexData(Gout.edgeTarget(*ed_it), &quad2);

        double area_1 = number_of_values[(int) Gout.edgeSource(*ed_it)] +
                        quad_error[(int) Gout.edgeSource(*ed_it)];
        double area_2 = number_of_values[(int) Gout.edgeTarget(*ed_it)] +
                        quad_error[(int) Gout.edgeTarget(*ed_it)];

        double cost =
            (((double) tmp) / ((double) tmp2)); // * std::pow( std::min( quad1 ,
                                                // quad2 ) , (double) alpha ) ;
        // double cost = ( tmp ) * std::pow( std::min( quad1 , quad2 ) ,
        // (double) alpha )   ; double cost = std::abs( ( (double) quad1 -
        // (double) quad2) ) * std::pow( std::min( area_1, area_2 ) , (double)
        // alpha ) ; double cost = ( ((double) tmp) /( (double) tmp2) ) *
        // std::pow( std::min( area_1, area_2 ) , (double) alpha ) ;

        Gout.setEdgeWeight(*ed_it, cost);
      }

      current_vertex = 0;
      for (boost::tie(v_it, v_end) = boost::vertices(Gout.getBoostGraph());
           v_it != v_end; ++v_it) {
        Gout.setVertexData(*v_it, (double) number_of_values[current_vertex] +
                                      (double) quad_error[current_vertex]);
        current_vertex = current_vertex + 1;
      }

      return RES_OK;
    }

    // template<class BoostGraph>
    // 	const BoostGraph t_DendogramFromTree(const BoostGraph &TreeIn)
    // {
    // 	MORPHEE_ENTER_FUNCTION("t_DendogramFromTree");

    // 	typedef	typename BoostGraph::EdgeIterator EdgeIterator;
    // 	typedef	typename BoostGraph::VertexIterator VertexIterator;
    // 	typedef typename BoostGraph::EdgeProperty EdgeProperty;

    // 	typename BoostGraph::VertexDescriptor vs,vt;
    // 	std::vector<double> val_edges;
    // 	EdgeIterator ed_it, ed_end;
    // 	VertexIterator v_it, v_end;
    // 	EdgeProperty tmp;

    // 	BoostGraph Tree_temp(TreeIn.numVertices());
    // 	BoostGraph Tree_label(TreeIn.numVertices());
    // 	BoostGraph Tree_contract(TreeIn.numVertices());
    // 	BoostGraph Dendogram(TreeIn.numVertices());

    // 	// order edges weights to ensure unicity of each weight
    // 	morphee::Morpho_Graph::t_Order_Edges_Weights(TreeIn, Tree_temp);

    // 	typename BoostGraph::VertexProperty vdata1;

    // 	for (boost::tie(ed_it, ed_end)=boost::edges(Tree_temp.getBoostGraph()) ;
    // ed_it != ed_end ; ++ed_it )
    // 	{
    // 		Tree_temp.edgeWeight(*ed_it,&tmp); // PASS VALUE
    // 		val_edges.push_back( tmp );
    // 	}

    // 	// sort edges weights to explore the hierarchy
    // 	std::cout<<"sort edges values"<<std::endl;
    // 	std::sort(val_edges.begin(), val_edges.end(),std::less<double>());

    // 	// clear edges and update edges values
    // 	int num_connected_component;
    // 	t_LabelConnectedComponent(Tree_contract,Tree_label,&num_connected_component);
    // 	std::cout<<"numer of leaves: "<<num_connected_component<<std::endl;

    // 	while ( num_connected_component > 1 ){

    // 		double last_edge_value = val_edges.back();
    // 		val_edges.pop_back();
    // 		std::cout<<"last_edge_value: "<<last_edge_value<<std::endl;

    // 		// add edges in Tree_contract
    // 		for (boost::tie(ed_it, ed_end)=boost::edges(Tree_temp.getBoostGraph())
    // ; ed_it != ed_end; ++ed_it)
    // 		{
    // 			Tree_temp.edgeWeight(*ed_it,&tmp);

    // 			if(tmp == last_edge_value){ // check is  current max weight
    // 				vs = Tree_temp.edgeSource(*ed_it) ;
    // 				vt = Tree_temp.edgeTarget(*ed_it) ;
    // 				Tree_contract.addEdge(vs,vt,1);;
    // 			}
    // 		}

    // 		t_LabelConnectedComponent(Tree_contract,Tree_label,&num_connected_component);
    // 		std::cout<<"number of leaves: "<<num_connected_component<<std::endl;

    // 		int* histogram_nb_nodes = new int[num_connected_component+1];

    // 		for (int i=0;i<num_connected_component+1;i++)
    // 		{
    // 			histogram_nb_nodes[i]=0;
    // 		}

    // 		for (boost::tie(v_it, v_end)=boost::vertices(Tree_temp.getBoostGraph())
    // ; v_it != v_end; ++v_it)
    // 		{
    // 			Tree_label.vertexData( *v_it , &vdata1);
    // 			histogram_nb_nodes[(int) vdata1] = histogram_nb_nodes[(int) vdata1] +
    // 1 ;
    // 		}

    // 		int number_of_nodes_to_add = 0 ;

    // 		for (int i=0;i<num_connected_component+1;i++)
    // 		{
    // 			if( histogram_nb_nodes[i] > 1 ) number_of_nodes_to_add++;
    // 		}

    // 		std::cout<<"number of nodes to add:
    // "<<number_of_nodes_to_add<<std::endl;

    // 	}

    // 	return GCopy;
    // }

    template <class ImageIn, class ImageGradx, class ImageGrady,
              class ImageMarker, class SE, class ImageOut>
    RES_C t_geoCutsParametric(const ImageIn &imIn, const ImageGradx &imGradx,
                              const ImageGrady &imGrady,
                              const ImageMarker &imMarker, const SE &nl,
                              ImageOut &imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_geoCutsParametric");

      std::cout << "Enter function t_geoCutsParametric" << std::endl;

      if ((!imOut.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if ((!imIn.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if ((!imGradx.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if ((!imGrady.isAllocated())) {
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
              boost::edge_capacity_t, float,
              boost::property<boost::edge_residual_capacity_t, float,
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

      int numVert  = 0;
      int numEdges = 0;

      std::cout << "build graph vertices" << std::endl;
      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o0 = it.getOffset();
        boost::add_vertex(g);
        numVert++;
      }

      vSource = boost::add_vertex(g);
      vSink   = boost::add_vertex(g);
      numVert += 2;

      std::cout << "build graph edges" << std::endl;
      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1 = it.getOffset();

        boost::tie(e4, in1) = boost::add_edge(vSource, o1, g);
        boost::tie(e3, in1) = boost::add_edge(o1, vSource, g);
        rev[e4]             = e3;
        rev[e3]             = e4;
        numEdges++;

        boost::tie(e3, in1) = boost::add_edge(o1, vSink, g);
        boost::tie(e4, in1) = boost::add_edge(vSink, o1, g);
        rev[e3]             = e4;
        rev[e4]             = e3;
        numEdges++;

        neighb.setCenter(o1);

        for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
          const offset_t o2 = nit.getOffset();
          if (o2 <= o1)
            continue;
          if (o2 > o1) {
            boost::tie(e4, in1) = boost::add_edge(o1, o2, g);
            boost::tie(e3, in1) = boost::add_edge(o2, o1, g);
            rev[e4]             = e3;
            rev[e3]             = e4;
            numEdges++;
          }
        }
      }

      std::cout << "graph is built" << std::endl;

      std::cout << "number of vertices: " << numVert << std::endl;
      std::cout << "number of edges: " << numEdges << std::endl;

      std::cout << "weight edges" << std::endl;

      double value_object     = 0.7;
      double value_background = 0.0;
      float lambda            = 1.0;

      boost::property_map<Graph_d, boost::vertex_index_t>::type indexmap =
          boost::get(boost::vertex_index, g);
      std::vector<boost::default_color_type> color(boost::num_vertices(g));

      int iter;

      for (lambda = 0.25, iter = 0; lambda <= 0.75, iter <= 500;
           lambda = lambda + 0.001, iter++) {
        for (it = imIn.begin(), iend = imIn.end(); it != iend;
             ++it) // for all pixels in imIn create a vertex and an edge
        {
          o1 = it.getOffset();

          // float val = (1.0/255.0) * (float) imIn.pixelFromOffset(o1);
          float val = (float) imMarker.pixelFromOffset(o1);

          // unary potential
          if (val >= 0.0f) {
            boost::tie(e4, in1) = boost::edge(vSource, o1, g);
            boost::tie(e3, in1) = boost::edge(o1, vSource, g);
            // capacity[e4] = (value_object-val)*(value_object-val);
            // capacity[e3] = (value_object-val)*(value_object-val);
            capacity[e4] = val;
            capacity[e3] = val;
          } else {
            boost::tie(e3, in1) = boost::edge(o1, vSink, g);
            boost::tie(e4, in1) = boost::edge(vSink, o1, g);
            // capacity[e3] = (value_background-val)*(value_background-val);
            // capacity[e4] = (value_background-val)*(value_background-val);
            capacity[e3] = abs(val);
            capacity[e4] = abs(val);
          }

          neighb.setCenter(o1);

          // binary potential
          for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
            const offset_t o2 = nit.getOffset();
            if (o2 <= o1)
              continue;
            if (o2 > o1) {
              float val2 = (1.0 / 255.0) * (float) imIn.pixelFromOffset(o2);
              float cost =
                  lambda / ((1.0 / 255.0) + (val - val2) + (val - val2));
              boost::tie(e4, in1) = boost::edge(o1, o2, g);
              boost::tie(e3, in1) = boost::edge(o2, o1, g);
              capacity[e4]        = cost;
              capacity[e3]        = cost;
            }
          }
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

        if (iter == 0) {
          for (it = imIn.begin(), iend = imIn.end(); it != iend;
               ++it) // for all pixels in imIn create a vertex and an edge
          {
            o1 = it.getOffset();
            if (color[o1] == color[vSource])
              imOut.setPixel(o1, 0);
            if (color[o1] == 1)
              imOut.setPixel(o1, 0);
            if (color[o1] == color[vSink])
              imOut.setPixel(o1, 100.0 * (lambda + 0.1));
          }
        } else {
          for (it = imIn.begin(), iend = imIn.end(); it != iend;
               ++it) // for all pixels in imIn create a vertex and an edge
          {
            o1 = it.getOffset();

            int oldval = imOut.pixelFromOffset(o1);
            int oldval2 =
                std::min((double) 100.0 * (lambda + 0.1), (double) oldval);
            int oldval3 = 100.0 * (lambda + 0.1);

            if (color[o1] == color[vSink] && oldval > 0)
              imOut.setPixel(o1, (int) oldval2);

            else if (color[o1] == color[vSink] && oldval == 0)
              imOut.setPixel(o1, (int) oldval3);
          }
        }
      }

      return RES_OK;
    }

    template <class ImageIn, class ImageMarker, class ImageMarker2, class SE,
              class ImageOut>
    RES_C t_geoCutsBoundary_Constrained_MinSurfaces(
        const ImageIn &imIn, const ImageMarker &imMarker,
        const ImageMarker2 &imMarker2, const SE &nl, ImageOut &imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_geoCutsBoundary_Constrained_MinSurfaces");

      std::cout << "Enter function Geo-Cuts " << std::endl;

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

      if ((!imMarker2.isAllocated())) {
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
      int numVert  = 0;
      int numEdges = 0;

      std::cout << "build graph vertices" << std::endl;
      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o0 = it.getOffset();
        boost::add_vertex(g);
        numVert++;
      }

      std::cout << "number of vertices: " << numVert << std::endl;

      vSource               = boost::add_vertex(g);
      vSink                 = boost::add_vertex(g);
      int source_sink_found = 0;

      std::cout << "build graph edges" << std::endl;
      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1         = it.getOffset();
        double val = imIn.pixelFromOffset(o1);
        int marker = imMarker.pixelFromOffset(o1);

        int valright = 0;
        int valleft  = 0;
        int valup    = 0;
        int valdown  = 0;

        // if(marker==128){
        //	std::cout<<"here 1"<<std::endl;
        //	boost::tie(e4, in1) = boost::add_edge(vSource,o1,g);
        //	boost::tie(e3, in1) = boost::add_edge(o1,vSource,g);
        //	capacity[e4] = (std::numeric_limits<double>::max)();
        //	capacity[e3] = (std::numeric_limits<double>::max)();
        //	rev[e4] = e3;
        //	rev[e3] = e4;
        //}
        // else if(marker==255){
        //	std::cout<<"here 2"<<std::endl;
        //	boost::tie(e4, in1) = boost::add_edge(o1,vSink, g);
        //	boost::tie(e3, in1) = boost::add_edge(vSink,o1, g);
        //	capacity[e4] = (std::numeric_limits<double>::max)();
        //	capacity[e3] = (std::numeric_limits<double>::max)();
        //	rev[e4] = e3;
        //	rev[e3] = e4;
        //}

        neighb.setCenter(o1);

        for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
          const offset_t o2 = nit.getOffset();
          int marker2       = imMarker.pixelFromOffset(o2);

          if (marker == 255 && marker2 == 128) {
            boost::tie(e4, in1) = boost::add_edge(vSource, o2, g);
            boost::tie(e3, in1) = boost::add_edge(o2, vSource, g);
            capacity[e4]        = (std::numeric_limits<double>::max)();
            capacity[e3]        = (std::numeric_limits<double>::max)();
            rev[e4]             = e3;
            rev[e3]             = e4;
            numEdges++;

            boost::tie(e4, in1) = boost::add_edge(vSink, o1, g);
            boost::tie(e3, in1) = boost::add_edge(o1, vSink, g);
            capacity[e4]        = (std::numeric_limits<double>::max)();
            capacity[e3]        = (std::numeric_limits<double>::max)();
            rev[e4]             = e3;
            rev[e3]             = e4;
            numEdges++;
          }

          if (o2 <= o1)
            continue;

          if (o2 > o1 && (marker == 0 || marker2 == 0 || marker == marker2)) {
            numEdges++;
            double val2 = imIn.pixelFromOffset(o2);
            double maxi = std::abs(val2 - val);
            double cost = 256 / (1 + std::pow(maxi, 2));
            // double cost = 1.0 ;
            boost::tie(e4, in1) = boost::add_edge(o1, o2, g);
            boost::tie(e3, in1) = boost::add_edge(o2, o1, g);
            capacity[e4]        = cost;
            capacity[e3]        = cost;
            rev[e4]             = e3;
            rev[e3]             = e4;
          }
        }
      }

      std::cout << "number of Edges : " << numEdges << std::endl;

      boost::property_map<Graph_d, boost::vertex_index_t>::type indexmap =
          boost::get(boost::vertex_index, g);
      std::vector<boost::default_color_type> color(boost::num_vertices(g));

      std::cout << "Compute Max flow " << std::endl;

      double flow2 =
          kolmogorov_max_flow_min_cost(g, capacity, residual_capacity, rev,
                                       &color[0], indexmap, vSink, vSource);
      std::cout << "c  The total flow found :" << std::endl;
      std::cout << "s " << flow2 << std::endl << std::endl;

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1 = it.getOffset();
        if (color[o1] == color[vSource])
          imOut.setPixel(o1, 2);
        else if (color[o1] == color[vSink])
          imOut.setPixel(o1, 3);
        else
          imOut.setPixel(o1, 4);
      }

      return RES_OK;
    }

    template <class ImageIn, class ImageVal, class BoostGraph,
              typename _nbmarkers, class SE, class ImageOut>
    RES_C t_geoCutsStochastic_Watershed_Graph(const ImageIn &imIn,
                                               const ImageVal &imVal,
                                               BoostGraph &Gin,
                                               const _nbmarkers nbmarkers,
                                               const SE &nl, ImageOut &imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_geoCutsStochastic_Watershed_Graph");

      std::cout << "Enter function Geo-Cuts Stochastic Watershed graph"
                << std::endl;

      if ((!imOut.isAllocated())) {
        std::cout << "imOut Not allocated" << std::endl;
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if ((!imIn.isAllocated())) {
        std::cout << "imIn Not allocated" << std::endl;
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if ((!imVal.isAllocated())) {
        std::cout << "imVal Not allocated" << std::endl;
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      double markers = (double) nbmarkers;

      int size = imIn.getXSize() * imIn.getYSize() * imIn.getZSize();
      std::cout << size << std::endl;

      typename ImageOut::const_iterator it, iend;
      typename ImageIn::const_iterator it2, iend2;
      morphee::selement::Neighborhood<SE, ImageOut> neighb(imOut, nl);
      typename morphee::selement::Neighborhood<SE, ImageOut>::iterator nit,
          nend;
      offset_t o0, o1;

      typedef typename BoostGraph::EdgeIterator EdgeIterator;
      typedef typename BoostGraph::VertexIterator VertexIterator;
      typedef typename BoostGraph::EdgeDescriptor EdgeDescriptor;
      typedef typename BoostGraph::VertexDescriptor VertexDescriptor;
      typename morphee::graph::CommonGraph32::EdgeProperty tmp;
      typename morphee::graph::CommonGraph32::VertexProperty vdata1;
      typename morphee::graph::CommonGraph32::VertexProperty label1, label2;
      EdgeIterator ed_it, ed_end, ed_it2, ed_end2;
      VertexIterator v_it, v_end;
      EdgeDescriptor last_edge, e1, e2;

      bool in1;

      std::vector<double> val_edges;
      val_edges.push_back(0.0);
      double last_edge_value = 0.0;

      std::vector<EdgeDescriptor> removed_edges;

      morphee::graph::CommonGraph32 G(0);
      morphee::graph::CommonGraph32 G2(0);
      morphee::graph::CommonGraph32 Tree(0);

      std::cout << " Get NeighborhoodGraph " << std::endl;
      morphee::morphoBase::t_NeighborhoodGraphFromMosaic(imIn, nl, G);

      std::cout << "Copy Minimum Spanning Tree" << std::endl;
      Tree = t_CopyGraph(Gin);

      std::cout << "Done" << std::endl;

      // const morphee::graph::CommonGraph32 t_CopyGraph(const
      // morphee::graph::CommonGraph32 &);

      morphee::graph::t_ProjectMarkersOnGraph(imVal, imIn, Tree);
      typename morphee::graph::CommonGraph32 Tree_temp = t_CopyGraph(Tree);
      typename morphee::graph::CommonGraph32 Tree_out  = t_CopyGraph(Tree);
      typename morphee::graph::CommonGraph32 Tree2     = t_CopyGraph(Tree_temp);

      double volume = 0.0;
      for (boost::tie(v_it, v_end) = boost::vertices(G.getBoostGraph());
           v_it != v_end; ++v_it) {
        Tree.vertexData(*v_it, &vdata1);
        volume = volume + (double) vdata1;
      }

      for (boost::tie(ed_it, ed_end) = boost::edges(G.getBoostGraph());
           ed_it != ed_end; ++ed_it) {
        G.setEdgeWeight(*ed_it, 0);
      }

      for (boost::tie(ed_it, ed_end) = boost::edges(Tree.getBoostGraph());
           ed_it != ed_end; ++ed_it) {
        Tree.edgeWeight(*ed_it, &tmp);

        val_edges.push_back(tmp);
        last_edge_value = (double) tmp;
      }

      std::cout << "sort" << std::endl;
      // std::sort(val_edges.begin(), val_edges.end());
      std::sort(val_edges.begin(), val_edges.end(), std::less<double>());

      last_edge_value      = val_edges.back();
      float max_value      = last_edge_value;
      double last_analyzed = last_edge_value;

      while (val_edges.size() > 1) {
        // std::cout<<val_edges.size()<<std::endl;
        // std::cout<<last_edge_value<<std::endl;

        // remove edge of maximal weight
        for (boost::tie(ed_it, ed_end) = boost::edges(Tree.getBoostGraph());
             ed_it != ed_end; ++ed_it) {
          // std::cout<<"look"<<std::endl;
          Tree.edgeWeight(*ed_it, &tmp);

          if (tmp == last_edge_value) {
            boost::tie(e1, in1) =
                boost::edge(Tree.edgeSource(*ed_it), Tree.edgeTarget(*ed_it),
                            Tree_temp.getBoostGraph());
            removed_edges.push_back(e1);
            Tree_temp.removeEdge(Tree.edgeSource(*ed_it),
                                 Tree.edgeTarget(*ed_it));
          }
        }

        // RES_C t_LabelConnectedComponent(const morphee::graph::CommonGraph32&,
        // morphee::graph::CommonGraph32&);

        t_LabelConnectedComponent(Tree_temp, Tree2);

        while (removed_edges.size() > 0) {
          e1 = removed_edges.back();
          removed_edges.pop_back();

          Tree2.vertexData(Tree.edgeSource(e1), &label1);
          Tree2.vertexData(Tree.edgeTarget(e1), &label2);

          Tree.edgeWeight(e1, &tmp);

          double S1    = 0;
          double S2    = 0;
          double quad1 = 0;
          double quad2 = 0;

          for (boost::tie(v_it, v_end) = boost::vertices(Tree2.getBoostGraph());
               v_it != v_end; ++v_it) {
            Tree2.vertexData(*v_it, &vdata1);
            if (vdata1 == label1) {
              Tree.vertexData(*v_it, &vdata1);
              S1 = S1 + (double) vdata1;
            } else if (vdata1 == label2) {
              Tree.vertexData(*v_it, &vdata1);
              S2 = S2 + (double) vdata1;
            }
          }

          /*double k = markers*((S1+S2)/(double) volume);
          double probability = 1 - std::pow( ( S1/(S1+S2) ) , k ) - std::pow( (
          S2/(S1+S2) ) , k ) ; */

          double probability =
              1.0 - std::pow(1.0 - S1 / ((double) volume), markers) -
              std::pow(1.0 - S2 / ((double) volume), markers) +
              std::pow(1.0 - (S1 + S2) / ((double) volume), markers);

          // double probability = 1.0 - 2 * std::pow( 1.0 - std::min(S1,S2) /(
          // (double) volume)  , markers ) + std::pow( 1.0 - 2 *
          // std::min(S1,S2)/((double) volume)  , markers );

          if (probability > 0.0) {
            RES_C res = Tree_out.edgeFromVertices(Tree.edgeSource(e1),
                                                  Tree.edgeTarget(e1), &e2);
            Tree_out.setEdgeWeight(e2, 65535.0 * probability);
          } else {
            RES_C res = Tree_out.edgeFromVertices(Tree.edgeSource(e1),
                                                  Tree.edgeTarget(e1), &e2);
            Tree_out.setEdgeWeight(e2, 0.0);
          }
        }

        while (last_edge_value == last_analyzed) {
          last_edge_value = val_edges.back();
          val_edges.pop_back();
        }
        last_analyzed = last_edge_value;
      }

      std::cout << "project on graphs" << std::endl;
      Tree2     = t_CopyGraph(Tree_out);
      Tree_temp = t_CopyGraph(Tree_out);

      int lio = 0;

      for (boost::tie(ed_it, ed_end) = boost::edges(Tree_out.getBoostGraph());
           ed_it != ed_end; ++ed_it) {
        Tree_out.edgeWeight(*ed_it, &tmp);
        double value = (double) tmp;

        if (value > 0.0) {
          Tree_temp.removeEdge(Tree_out.edgeSource(*ed_it),
                               Tree_out.edgeTarget(*ed_it));

          // RES_C t_LabelConnectedComponent(const
          // morphee::graph::CommonGraph32&, morphee::graph::CommonGraph32&);
          t_LabelConnectedComponent(Tree_temp, Tree2);

          Tree_temp.addEdge(Tree_out.edgeSource(*ed_it),
                            Tree_out.edgeTarget(*ed_it), tmp);

          for (boost::tie(ed_it2, ed_end2) = boost::edges(G.getBoostGraph());
               ed_it2 != ed_end2; ++ed_it2) {
            Tree2.vertexData(G.edgeSource(*ed_it2), &label1);
            Tree2.vertexData(G.edgeTarget(*ed_it2), &label2);

            if (label1 != label2) {
              G.edgeWeight(*ed_it2, &tmp);
              G.setEdgeWeight(*ed_it2, std::max((double) tmp, value));
            }
          }
        }
        std::cout << lio << " / " << Tree_out.numEdges() << std::endl;
        lio = lio + 1;
      }

      Gin = morphee::graph::MinimumSpanningTreeFromGraph(G);

      std::cout << "project on image the pdf" << std::endl;

      for (it = imOut.begin(), iend = imOut.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o0 = it.getOffset();
        imOut.setPixel(o0, 0);
      }

      std::cout << "init done" << std::endl;

      for (it = imOut.begin(), iend = imOut.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o0             = it.getOffset();
        int val1       = imIn.pixelFromOffset(o0);
        double valout1 = imOut.pixelFromOffset(o0);

        neighb.setCenter(o0);

        for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
          o1 = nit.getOffset();

          if (o1 > o0) {
            int val2       = imIn.pixelFromOffset(o1);
            double valout2 = imOut.pixelFromOffset(o1);

            if (val2 != val1) {
              RES_C res = G.edgeFromVertices(val1 - 1, val2 - 1, &e1);
              if (res == RES_OK) {
                G.edgeWeight(e1, &tmp);
                if (tmp > 0) {
                  imOut.setPixel(o0, std::max(valout1, (double) tmp));
                  imOut.setPixel(o1, std::max(valout2, (double) tmp));
                }
              } else {
                RES_C res = G.edgeFromVertices(val2 - 1, val1 - 1, &e1);
                if (res == RES_OK) {
                  G.edgeWeight(e1, &tmp);
                  if (tmp > 0) {
                    imOut.setPixel(o0, std::max(valout1, (double) tmp));
                    imOut.setPixel(o1, std::max(valout2, (double) tmp));
                  }
                }
              }
            }
          }
        }
      }

      return RES_OK;
    }

    template <class ImageIn, class ImageVal, class BoostGraph,
              typename _nbmarkers, class SE, class ImageOut>
    RES_C t_geoCutsStochastic_Watershed_Graph_NP(const ImageIn &imIn,
                                                  const ImageVal &imVal,
                                                  BoostGraph &Gin,
                                                  const _nbmarkers nbmarkers,
                                                  const SE &nl, ImageOut &imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_geoCutsStochastic_Watershed_Graph_NP");

      std::cout
          << "Enter function Geo-Cuts Stochastic Watershed graph non ponctual"
          << std::endl;

      if ((!imOut.isAllocated())) {
        std::cout << "imOut Not allocated" << std::endl;
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if ((!imIn.isAllocated())) {
        std::cout << "imIn Not allocated" << std::endl;
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if ((!imVal.isAllocated())) {
        std::cout << "imVal Not allocated" << std::endl;
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      double markers = (double) nbmarkers;

      int size = imIn.getXSize() * imIn.getYSize() * imIn.getZSize();
      std::cout << size << std::endl;

      typename ImageOut::const_iterator it, iend;
      typename ImageIn::const_iterator it2, iend2;
      morphee::selement::Neighborhood<SE, ImageOut> neighb(imOut, nl);
      typename morphee::selement::Neighborhood<SE, ImageOut>::iterator nit,
          nend;
      offset_t o0, o1;

      typedef typename BoostGraph::EdgeIterator EdgeIterator;
      typedef typename BoostGraph::VertexIterator VertexIterator;
      typedef typename BoostGraph::EdgeDescriptor EdgeDescriptor;
      typedef typename BoostGraph::VertexDescriptor VertexDescriptor;
      typename morphee::graph::CommonGraph32::EdgeProperty tmp;
      typename morphee::graph::CommonGraph32::VertexProperty vdata1;
      typename morphee::graph::CommonGraph32::VertexProperty label1, label2;
      EdgeIterator ed_it, ed_end, ed_it2, ed_end2;
      VertexIterator v_it, v_end;
      EdgeDescriptor last_edge, e1, e2;

      bool in1;

      std::vector<double> val_edges;
      val_edges.push_back(0.0);
      double last_edge_value = 0.0;

      std::vector<EdgeDescriptor> removed_edges;

      morphee::graph::CommonGraph32 G(0);
      morphee::graph::CommonGraph32 G2(0);
      morphee::graph::CommonGraph32 Tree(0);

      std::cout << " Get NeighborhoodGraph " << std::endl;
      morphee::morphoBase::t_NeighborhoodGraphFromMosaic(imIn, nl, G);

      std::cout << "Copy Minimum Spanning Tree" << std::endl;
      Tree = t_CopyGraph(Gin);

      std::cout << "Done" << std::endl;

      // const morphee::graph::CommonGraph32 t_CopyGraph(const
      // morphee::graph::CommonGraph32 &);

      morphee::graph::t_ProjectMarkersOnGraph(imVal, imIn, Tree);
      morphee::graph::CommonGraph32 Tree_temp = t_CopyGraph(Tree);
      morphee::graph::CommonGraph32 Tree_out  = t_CopyGraph(Tree);
      morphee::graph::CommonGraph32 Tree2     = t_CopyGraph(Tree_temp);

      double volume = 0.0;
      for (boost::tie(v_it, v_end) = boost::vertices(G.getBoostGraph());
           v_it != v_end; ++v_it) {
        Tree.vertexData(*v_it, &vdata1);
        volume = volume + (double) vdata1;
      }

      for (boost::tie(ed_it, ed_end) = boost::edges(G.getBoostGraph());
           ed_it != ed_end; ++ed_it) {
        G.setEdgeWeight(*ed_it, 0);
      }

      for (boost::tie(ed_it, ed_end) = boost::edges(Tree.getBoostGraph());
           ed_it != ed_end; ++ed_it) {
        Tree.edgeWeight(*ed_it, &tmp);

        val_edges.push_back(tmp);
        last_edge_value = (double) tmp;
      }

      std::cout << "sort" << std::endl;
      // std::sort(val_edges.begin(), val_edges.end());
      std::sort(val_edges.begin(), val_edges.end(), std::less<double>());

      last_edge_value      = val_edges.back();
      float max_value      = last_edge_value;
      double last_analyzed = last_edge_value;

      while (val_edges.size() > 1) {
        // std::cout<<val_edges.size()<<std::endl;
        // std::cout<<last_edge_value<<std::endl;

        // remove edge of maximal weight
        for (boost::tie(ed_it, ed_end) = boost::edges(Tree.getBoostGraph());
             ed_it != ed_end; ++ed_it) {
          // std::cout<<"look"<<std::endl;
          Tree.edgeWeight(*ed_it, &tmp);

          if (tmp == last_edge_value) {
            boost::tie(e1, in1) =
                boost::edge(Tree.edgeSource(*ed_it), Tree.edgeTarget(*ed_it),
                            Tree_temp.getBoostGraph());
            removed_edges.push_back(e1);
            Tree_temp.removeEdge(Tree.edgeSource(*ed_it),
                                 Tree.edgeTarget(*ed_it));
          }
        }

        // RES_C t_LabelConnectedComponent(const morphee::graph::CommonGraph32&,
        // morphee::graph::CommonGraph32&);
        t_LabelConnectedComponent(Tree_temp, Tree2);

        while (removed_edges.size() > 0) {
          e1 = removed_edges.back();
          removed_edges.pop_back();

          Tree2.vertexData(Tree.edgeSource(e1), &label1);
          Tree2.vertexData(Tree.edgeTarget(e1), &label2);

          Tree.edgeWeight(e1, &tmp);

          double S1    = 0;
          double S2    = 0;
          double quad1 = 0;
          double quad2 = 0;

          for (boost::tie(v_it, v_end) = boost::vertices(Tree2.getBoostGraph());
               v_it != v_end; ++v_it) {
            Tree2.vertexData(*v_it, &vdata1);
            if (vdata1 == label1) {
              Tree.vertexData(*v_it, &vdata1);
              S1 = S1 + (double) vdata1;
            } else if (vdata1 == label2) {
              Tree.vertexData(*v_it, &vdata1);
              S2 = S2 + (double) vdata1;
            }
          }

          /*double k = markers*((S1+S2)/(double) volume);
          double probability = 1 - std::pow( ( S1/(S1+S2) ) , k ) - std::pow( (
          S2/(S1+S2) ) , k ) ; */

          double probability =
              1.0 - std::pow(1.0 - S1 / ((double) volume), markers) -
              std::pow(1.0 - S2 / ((double) volume), markers) +
              std::pow(1.0 - (S1 + S2) / ((double) volume), markers);

          // double probability = 1.0 - 2 * std::pow( 1.0 - std::min(S1,S2) /(
          // (double) volume)  , markers ) + std::pow( 1.0 - 2 *
          // std::min(S1,S2)/((double) volume)  , markers );

          if (probability > 0.0) {
            RES_C res = Tree_out.edgeFromVertices(Tree.edgeSource(e1),
                                                  Tree.edgeTarget(e1), &e2);
            Tree_out.setEdgeWeight(e2, 65535.0 * probability);
          } else {
            RES_C res = Tree_out.edgeFromVertices(Tree.edgeSource(e1),
                                                  Tree.edgeTarget(e1), &e2);
            Tree_out.setEdgeWeight(e2, 0.0);
          }
        }

        while (last_edge_value == last_analyzed) {
          last_edge_value = val_edges.back();
          val_edges.pop_back();
        }
        last_analyzed = last_edge_value;
      }

      std::cout << "project on graphs" << std::endl;
      // const morphee::graph::CommonGraph32 t_CopyGraph(const
      // morphee::graph::CommonGraph32 &);
      Tree2     = t_CopyGraph(Tree_out);
      Tree_temp = t_CopyGraph(Tree_out);

      int lio = 0;

      for (boost::tie(ed_it, ed_end) = boost::edges(Tree_out.getBoostGraph());
           ed_it != ed_end; ++ed_it) {
        Tree_out.edgeWeight(*ed_it, &tmp);
        double value = (double) tmp;

        if (value > 0.0) {
          Tree_temp.removeEdge(Tree_out.edgeSource(*ed_it),
                               Tree_out.edgeTarget(*ed_it));

          // RES_C t_LabelConnectedComponent(const
          // morphee::graph::CommonGraph32&, morphee::graph::CommonGraph32&);
          t_LabelConnectedComponent(Tree_temp, Tree2);

          Tree_temp.addEdge(Tree_out.edgeSource(*ed_it),
                            Tree_out.edgeTarget(*ed_it), tmp);

          for (boost::tie(ed_it2, ed_end2) = boost::edges(G.getBoostGraph());
               ed_it2 != ed_end2; ++ed_it2) {
            Tree2.vertexData(G.edgeSource(*ed_it2), &label1);
            Tree2.vertexData(G.edgeTarget(*ed_it2), &label2);

            if (label1 != label2) {
              G.edgeWeight(*ed_it2, &tmp);
              G.setEdgeWeight(*ed_it2, std::max((double) tmp, value));
            }
          }
        }
        std::cout << lio << " / " << Tree_out.numEdges() << std::endl;
        lio = lio + 1;
      }

      Gin = morphee::graph::MinimumSpanningTreeFromGraph(G);

      std::cout << "project on image the pdf" << std::endl;

      for (it = imOut.begin(), iend = imOut.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o0 = it.getOffset();
        imOut.setPixel(o0, 0);
      }

      std::cout << "init done" << std::endl;

      for (it = imOut.begin(), iend = imOut.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o0             = it.getOffset();
        int val1       = imIn.pixelFromOffset(o0);
        double valout1 = imOut.pixelFromOffset(o0);

        neighb.setCenter(o0);

        for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
          o1 = nit.getOffset();

          if (o1 > o0) {
            int val2       = imIn.pixelFromOffset(o1);
            double valout2 = imOut.pixelFromOffset(o1);

            if (val2 != val1) {
              RES_C res = G.edgeFromVertices(val1 - 1, val2 - 1, &e1);
              if (res == RES_OK) {
                G.edgeWeight(e1, &tmp);
                if (tmp > 0) {
                  imOut.setPixel(o0, std::max(valout1, (double) tmp));
                  imOut.setPixel(o1, std::max(valout2, (double) tmp));
                }
              } else {
                RES_C res = G.edgeFromVertices(val2 - 1, val1 - 1, &e1);
                if (res == RES_OK) {
                  G.edgeWeight(e1, &tmp);
                  if (tmp > 0) {
                    imOut.setPixel(o0, std::max(valout1, (double) tmp));
                    imOut.setPixel(o1, std::max(valout2, (double) tmp));
                  }
                }
              }
            }
          }
        }
      }

      return RES_OK;
    }

    template <class BoostGraph>
    RES_C t_UpdateSpanningTreeFromForest(const BoostGraph &ForestIn,
                                         const BoostGraph &TIn,
                                         BoostGraph &Tout)
    {
      MORPHEE_ENTER_FUNCTION("t_UpdateSpanningTreeFromForest");

      std::cout << "Enter function t_UpdateSpanningTreeFromForest "
                << std::endl;

      typedef typename BoostGraph::EdgeIterator EdgeIterator;
      typedef typename BoostGraph::VertexIterator VertexIterator;
      typedef typename BoostGraph::EdgeDescriptor EdgeDescriptor;
      typedef typename BoostGraph::VertexDescriptor VertexDescriptor;
      typename BoostGraph::EdgeProperty tmp;
      EdgeIterator ed_it, ed_end, ed_it2, ed_end2;
      VertexIterator v_it, v_end;
      EdgeDescriptor last_edge, e1, e2;

      bool in1;

      Tout = t_CopyGraph(TIn);

      for (boost::tie(ed_it, ed_end) = boost::edges(ForestIn.getBoostGraph());
           ed_it != ed_end; ++ed_it) {
        ForestIn.edgeWeight(*ed_it, &tmp);
        boost::tie(e1, in1) =
            boost::edge(ForestIn.edgeSource(*ed_it),
                        ForestIn.edgeTarget(*ed_it), Tout.getBoostGraph());
        Tout.setEdgeWeight(e1, 0);
      }

      return RES_OK;
    }

    template <class ImageIn, class BoostGraph, class SE, class ImageOut>
    RES_C t_GetUltrametricContourMap(const ImageIn &imIn,
                                     const BoostGraph &Tree, const SE &nl,
                                     ImageOut &imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_GetUltrametricContourMap");

      std::cout << "Enter t_GetUltrametricContourMap " << std::endl;

      if ((!imOut.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if ((!imIn.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      int size = imIn.getXSize() * imIn.getYSize() * imIn.getZSize();
      // std::cout<<size<<std::endl;

      typename BoostGraph::VertexProperty label1, label2;
      typedef typename BoostGraph::EdgeProperty EdgeProperty;
      typedef typename BoostGraph::EdgeDescriptor EdgeDescriptor;
      typedef typename BoostGraph::VertexDescriptor VertexDescriptor;
      EdgeDescriptor e1;
      EdgeProperty tmp;

      typedef typename BoostGraph::EdgeIterator EdgeIterator;
      EdgeIterator ed_it, ed_end, ed_it2, ed_end2;

      std::cout << "project on image the graph" << std::endl;

      typename ImageOut::const_iterator it, iend;
      morphee::selement::Neighborhood<SE, ImageOut> neighb(imOut, nl);
      typename morphee::selement::Neighborhood<SE, ImageOut>::iterator nit,
          nend;
      offset_t o0, o1;

      for (it = imOut.begin(), iend = imOut.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o0 = it.getOffset();
        imOut.setPixel(o0, 0);
      }

      std::cout << "project on graphs" << std::endl;
      BoostGraph G, Tree2, Tree_temp;

      Tree2     = t_CopyGraph(Tree);
      Tree_temp = t_CopyGraph(Tree);

      std::cout << "t_NeighborhoodGraphFromMosaic" << std::endl;
      morphee::morphoBase::t_NeighborhoodGraphFromMosaic(imIn, nl, G);

      for (boost::tie(ed_it, ed_end) = boost::edges(G.getBoostGraph());
           ed_it != ed_end; ++ed_it) {
        G.setEdgeWeight(*ed_it, 0);
      }

      std::cout << "iterate..." << std::endl;

      int lio = 0;

      for (boost::tie(ed_it, ed_end) = boost::edges(Tree.getBoostGraph());
           ed_it != ed_end; ++ed_it) {
        Tree.edgeWeight(*ed_it, &tmp);
        double value = (double) tmp;

        if (value > 0.0) {
          // v1 = BoostGraph::VertexDescriptor( Tree.edgeSource(*ed_it)) ;
          // v2 = BoostGraph::VertexDescriptor( Tree.edgeTarget(*ed_it)) ;

          Tree_temp.removeEdge(
              typename BoostGraph::VertexDescriptor(Tree.edgeSource(*ed_it)),
              typename BoostGraph::VertexDescriptor(Tree.edgeTarget(*ed_it)));

          t_LabelConnectedComponent(Tree_temp, Tree2);

          Tree_temp.addEdge(
              typename BoostGraph::VertexDescriptor(Tree.edgeSource(*ed_it)),
              typename BoostGraph::VertexDescriptor(Tree.edgeTarget(*ed_it)),
              tmp);

          for (boost::tie(ed_it2, ed_end2) = boost::edges(G.getBoostGraph());
               ed_it2 != ed_end2; ++ed_it2) {
            Tree2.vertexData(
                typename BoostGraph::VertexDescriptor(G.edgeSource(*ed_it2)),
                &label1);
            Tree2.vertexData(
                typename BoostGraph::VertexDescriptor(G.edgeTarget(*ed_it2)),
                &label2);

            if (label1 != label2) {
              G.edgeWeight(*ed_it2, &tmp);
              G.setEdgeWeight(*ed_it2, std::max((double) tmp, value));
              // G.setEdgeWeight(*ed_it2,(double)tmp+value);
            }
          }
        }
        std::cout << lio << " / " << Tree.numEdges() << std::endl;
        lio = lio + 1;
      }

      std::cout << "project on image " << std::endl;
      for (it = imOut.begin(), iend = imOut.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o0 = it.getOffset();

        int val1 = imIn.pixelFromOffset(o0);

        double valout1 = imOut.pixelFromOffset(o0);

        neighb.setCenter(o0);

        for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
          o1 = nit.getOffset();

          if (o1 > o0) {
            int val2 = imIn.pixelFromOffset(o1);

            double valout2 = imOut.pixelFromOffset(o1);

            if (val2 != val1) {
              RES_C res = G.edgeFromVertices(val1 - 1, val2 - 1, &e1);

              if (res == RES_OK) {
                G.edgeWeight(e1, &tmp);

                if (tmp > 0) {
                  imOut.setPixel(
                      o0, std::max(valout2, std::max(valout1, (double) tmp)));
                  // imOut.setPixel( o1 , std::max( valout2, (double) tmp));
                }
              }

              else {
                RES_C res = G.edgeFromVertices(val2 - 1, val1 - 1, &e1);
                if (res == RES_OK) {
                  G.edgeWeight(e1, &tmp);
                  if (tmp > 0) {
                    imOut.setPixel(
                        o0, std::max(valout2, std::max(valout1, (double) tmp)));
                    // imOut.setPixel( o1 , std::max( valout2, (double) tmp));
                  }
                }
              }
            }
          }
        }
      }

      return RES_OK;
    }

    template <class ImageIn, class BoostGraph, class SE, class ImageOut>
    RES_C t_GetScaleSetUltrametricContourMap(const ImageIn &imIn,
                                             const BoostGraph &Tree,
                                             const SE &nl, ImageOut &imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_GetScaleSetUltrametricContourMap");

      std::cout << "Enter t_GetScaleSetUltrametricContourMap " << std::endl;

      if ((!imOut.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if ((!imIn.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      int size = imIn.getXSize() * imIn.getYSize() * imIn.getZSize();
      // std::cout<<size<<std::endl;

      typename BoostGraph::VertexProperty label1, label2, vdata1, vdata2;
      typedef typename BoostGraph::EdgeProperty EdgeProperty;
      typedef typename BoostGraph::EdgeDescriptor EdgeDescriptor;
      typedef typename BoostGraph::VertexDescriptor VertexDescriptor;
      EdgeDescriptor e1;
      EdgeProperty tmp;
      VertexDescriptor vs, vt;
      bool in1;

      typedef typename BoostGraph::EdgeIterator EdgeIterator;
      EdgeIterator ed_it, ed_end, ed_it2, ed_end2;

      std::cout << "project on image the graph" << std::endl;

      typename ImageIn::const_iterator it2, iend2;

      typename ImageOut::const_iterator it, iend;
      morphee::selement::Neighborhood<SE, ImageOut> neighb(imOut, nl);
      typename morphee::selement::Neighborhood<SE, ImageOut>::iterator nit,
          nend;
      offset_t o0, o1;

      for (it = imOut.begin(), iend = imOut.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o0 = it.getOffset();
        imOut.setPixel(o0, 0);
      }

      std::cout << "project on graphs" << std::endl;
      BoostGraph G;

      std::cout << "t_NeighborhoodGraphFromMosaic" << std::endl;
      morphee::morphoBase::t_NeighborhoodGraphFromMosaic(imIn, nl, G);

      for (boost::tie(ed_it, ed_end) = boost::edges(G.getBoostGraph());
           ed_it != ed_end; ++ed_it) {
        G.setEdgeWeight(*ed_it, 0);
      }

      std::vector<double> val_edges;

      for (boost::tie(ed_it, ed_end) = boost::edges(Tree.getBoostGraph());
           ed_it != ed_end; ++ed_it) {
        Tree.edgeWeight(*ed_it, &tmp);
        val_edges.push_back(tmp);
      }

      int numVert = 0;

      for (it2 = imIn.begin(), iend2 = imIn.end(); it2 != iend2;
           ++it2) // for all pixels in imIn create a vertex
      {
        o0      = it2.getOffset();
        int val = imIn.pixelFromOffset(o0);

        if (val > numVert) {
          numVert = val;
        }
      }

      std::cout << "nb of regions = " << numVert << std::endl;

      // INIT TREETEMP
      BoostGraph Tree_label = morphee::graph::CommonGraph32(numVert);
      BoostGraph Tree_temp  = morphee::graph::CommonGraph32(numVert);

      std::cout << "sort edges of tree" << std::endl;
      std::sort(val_edges.begin(), val_edges.end(), std::greater<double>());

      double last_edge_value = val_edges.back();
      double last_analyzed   = last_edge_value;
      val_edges.pop_back();

      std::cout << "iterate..." << std::endl;
      int lio = 0;

      while (val_edges.size() > 0) {
        // add edge of minimal weight
        for (boost::tie(ed_it, ed_end) = boost::edges(Tree.getBoostGraph());
             ed_it != ed_end; ++ed_it) {
          Tree.edgeWeight(*ed_it, &tmp);

          if (tmp == last_edge_value) { // check if current min

            vs = Tree.edgeSource(*ed_it);
            vt = Tree.edgeTarget(*ed_it);
            boost::tie(e1, in1) =
                boost::add_edge(vs, vt, Tree_temp.getBoostGraph());
          }
        }

        // AFTER MERGING
        int number_of_connected_components;
        t_LabelConnectedComponent(Tree_temp, Tree_label,
                                  &number_of_connected_components);
        std::cout << "number_of_connected_components = "
                  << number_of_connected_components << std::endl;

        for (boost::tie(ed_it, ed_end) = boost::edges(G.getBoostGraph());
             ed_it != ed_end; ++ed_it) {
          vs = G.edgeSource(*ed_it);
          vt = G.edgeTarget(*ed_it);

          // new label of the regions
          Tree_label.vertexData(vs, &vdata1);
          Tree_label.vertexData(vt, &vdata2);

          G.edgeWeight(*ed_it, &tmp);

          if (vdata1 != vdata2) {
            G.setEdgeWeight(*ed_it,
                            std::max((float) tmp, (float) last_edge_value));
          }
        }

        while (last_edge_value == last_analyzed) {
          last_edge_value = val_edges.back();
          val_edges.pop_back();
        }
        last_analyzed = last_edge_value;
      }

      std::cout << "project on image " << std::endl;
      for (it = imOut.begin(), iend = imOut.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o0 = it.getOffset();

        int val1 = imIn.pixelFromOffset(o0);

        double valout1 = imOut.pixelFromOffset(o0);

        neighb.setCenter(o0);

        for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
          o1 = nit.getOffset();

          if (o1 > o0) {
            int val2 = imIn.pixelFromOffset(o1);

            double valout2 = imOut.pixelFromOffset(o1);

            if (val2 != val1) {
              RES_C res = G.edgeFromVertices(val1 - 1, val2 - 1, &e1);

              if (res == RES_OK) {
                G.edgeWeight(e1, &tmp);

                if (tmp > 0) {
                  imOut.setPixel(
                      o0, std::max(valout2, std::max(valout1, (double) tmp)));
                  // imOut.setPixel( o1 , std::max( valout2, (double) tmp));
                }
              }

              else {
                RES_C res = G.edgeFromVertices(val2 - 1, val1 - 1, &e1);
                if (res == RES_OK) {
                  G.edgeWeight(e1, &tmp);
                  if (tmp > 0) {
                    imOut.setPixel(
                        o0, std::max(valout2, std::max(valout1, (double) tmp)));
                    // imOut.setPixel( o1 , std::max( valout2, (double) tmp));
                  }
                }
              }
            }
          }
        }
      }

      return RES_OK;
    }

    template <class ImageIn, class ImageIn2, class ImageVal,
              typename _nbmarkers, typename _alpha, class SE, class ImageOut>
    RES_C t_geoCutsStochastic_Watershed_Variance(const ImageIn &imIn,
                                                  const ImageIn2 &imVal,
                                                  const ImageVal &imGrad,
                                                  const _nbmarkers nbmarkers,
                                                  const _alpha alpha,
                                                  const SE &nl, ImageOut &imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_geoCutsStochastic_Watershed");

      std::cout << "Enter function Geo-Cuts Stochastic Watershed" << std::endl;

      if ((!imOut.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if ((!imIn.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if ((!imVal.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if ((!imGrad.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      double markers = (double) nbmarkers;

      int size = imIn.getXSize() * imIn.getYSize() * imIn.getZSize();
      // std::cout<<size<<std::endl;

      typedef typename morphee::graph::CommonGraph32::_boostGraph BoostGraph;
      typedef
          typename boost::graph_traits<BoostGraph>::edge_iterator EdgeIterator;
      typedef typename boost::graph_traits<BoostGraph>::vertex_iterator
          VertexIterator;
      typedef typename boost::graph_traits<BoostGraph>::edge_descriptor
          EdgeDescriptor;
      typedef typename boost::graph_traits<BoostGraph>::vertex_descriptor
          VertexDescriptor;
      typename morphee::graph::CommonGraph32::EdgeProperty tmp;
      typename morphee::graph::CommonGraph32::VertexProperty vdata1;
      typename morphee::graph::CommonGraph32::VertexProperty label1, label2,
          label21, label22;
      EdgeIterator ed_it, ed_end, ed_it2, ed_end2;
      VertexIterator v_it, v_end;
      EdgeDescriptor last_edge, e1, e2;

      bool in1;

      std::vector<double> val_edges;
      val_edges.push_back(0.0);
      double last_edge_value = 0.0;

      std::vector<EdgeDescriptor> removed_edges;

      morphee::graph::CommonGraph32 G(0);
      morphee::graph::CommonGraph32 G2(0);
      morphee::graph::CommonGraph32 Tree(0);

      // morphee::morphoBase::t_NeighborhoodGraphFromMosaic_WithPass(imIn,imGrad,nl,G);
      // morphee::graphalgo::t_NeighborhoodGraphFromMosaic_WithMeanGradientValue(imIn,imGrad,nl,G);
      morphee::graphalgo::
          t_NeighborhoodGraphFromMosaic_WithMeanGradientValue_AndQuadError(
              imIn, imGrad, imVal, alpha, nl, G);

      Tree = morphee::graph::MinimumSpanningTreeFromGraph(G);

      ImageIn ImTempSurfaces = imIn.getSame();
      // Image<UINT8>ImTempSurfaces = imIn.t_getSame<UINT8>();
      morphee::morphoBase::t_ImLabelFlatZonesWithArea(imIn, nl, ImTempSurfaces);
      // morphee::morphoBase::t_ImLabelFlatZonesWithVolume(imIn,imVal,nl,ImTempSurfaces);

      // const morphee::graph::CommonGraph32 t_CopyGraph(const
      // morphee::graph::CommonGraph32 &);
      morphee::graph::CommonGraph32 Tree_temp = t_CopyGraph(Tree);
      morphee::graph::CommonGraph32 Tree_out  = t_CopyGraph(Tree);
      morphee::graph::CommonGraph32 Tree2     = t_CopyGraph(Tree_temp);

      morphee::graph::t_ProjectMarkersOnGraph(ImTempSurfaces, imIn, Tree);

      double volume = 0.0;
      for (boost::tie(v_it, v_end) = boost::vertices(Tree.getBoostGraph());
           v_it != v_end; ++v_it) {
        Tree.vertexData(*v_it, &vdata1);
        volume = volume + (double) vdata1;
      }

      for (boost::tie(ed_it, ed_end) = boost::edges(G.getBoostGraph());
           ed_it != ed_end; ++ed_it) {
        G.setEdgeWeight(*ed_it, 0);
      }

      for (boost::tie(ed_it, ed_end) = boost::edges(Tree.getBoostGraph());
           ed_it != ed_end; ++ed_it) {
        Tree.edgeWeight(*ed_it, &tmp);
        // if( (double) tmp > last_edge_value)	{

        val_edges.push_back(tmp);
        last_edge_value = (double) tmp;

        //}
      }

      std::cout << "sort" << std::endl;
      // std::sort(val_edges.begin(), val_edges.end());
      std::sort(val_edges.begin(), val_edges.end(), std::less<int>());

      last_edge_value      = val_edges.back();
      double last_analyzed = last_edge_value;

      while (val_edges.size() > 1) {
        // std::cout<<val_edges.size()<<std::endl;
        // std::cout<<last_edge_value<<std::endl;

        // remove edge of maximal weight
        for (boost::tie(ed_it, ed_end) = boost::edges(Tree.getBoostGraph());
             ed_it != ed_end; ++ed_it) {
          // std::cout<<"look"<<std::endl;
          Tree.edgeWeight(*ed_it, &tmp);

          if (tmp == last_edge_value) {
            boost::tie(e1, in1) =
                boost::edge(Tree.edgeSource(*ed_it), Tree.edgeTarget(*ed_it),
                            Tree_temp.getBoostGraph());
            removed_edges.push_back(e1);
            Tree_temp.removeEdge(
                typename boost::graph_traits<BoostGraph>::vertex_descriptor(
                    Tree.edgeSource(*ed_it)),
                typename boost::graph_traits<BoostGraph>::vertex_descriptor(
                    Tree.edgeTarget(*ed_it)));
          }
        }

        // RES_C t_LabelConnectedComponent(const morphee::graph::CommonGraph32&,
        // morphee::graph::CommonGraph32&);
        t_LabelConnectedComponent(Tree_temp, Tree2);

        while (removed_edges.size() > 0) {
          e1 = removed_edges.back();
          removed_edges.pop_back();

          Tree2.vertexData(
              typename boost::graph_traits<BoostGraph>::vertex_descriptor(
                  Tree.edgeSource(e1)),
              &label1);
          Tree2.vertexData(
              typename boost::graph_traits<BoostGraph>::vertex_descriptor(
                  Tree.edgeTarget(e1)),
              &label2);

          double S1          = 0;
          double S2          = 0;
          double perimeter_1 = 0;
          double perimeter_2 = 0;

          for (boost::tie(v_it, v_end) = boost::vertices(Tree2.getBoostGraph());
               v_it != v_end; ++v_it) {
            Tree2.vertexData(*v_it, &vdata1);
            if (vdata1 == label1) {
              Tree.vertexData(*v_it, &vdata1);
              S1 = S1 + (double) vdata1;
            } else if (vdata1 == label2) {
              Tree.vertexData(*v_it, &vdata1);
              S2 = S2 + (double) vdata1;
            }
          }

          for (boost::tie(ed_it2, ed_end2) = boost::edges(G.getBoostGraph());
               ed_it2 != ed_end2; ++ed_it2) {
            Tree2.vertexData(
                typename boost::graph_traits<BoostGraph>::vertex_descriptor(
                    Tree.edgeSource(*ed_it2)),
                &label21);
            Tree2.vertexData(
                typename boost::graph_traits<BoostGraph>::vertex_descriptor(
                    Tree.edgeTarget(*ed_it2)),
                &label22);

            if ((label21 == label1 && label22 != label1) ||
                (label21 != label1 && label22 == label1)) {
              perimeter_1 = perimeter_1 + 1.0;
            }
            if ((label21 != label2 && label22 == label2) ||
                (label21 == label2 && label22 != label2)) {
              perimeter_2 = perimeter_2 + 1.0;
            }
          }

          // double pie = 3.14159265358979323846;
          // double C1 = perimeter_1*perimeter_1 / (4 * pie * S1);
          // double C2 = perimeter_2*perimeter_2 / (4 * pie * S2);
          // double probability2 = ( std::pow( C1 , markers ) + std::pow( C2 ,
          // markers ) );

          // double k = markers*((S1+S2)/(double) volume);
          // double probability = 1 - std::pow( ( S1/(S1+S2) ) , k ) - std::pow(
          // ( S2/(S1+S2) ) , k ) ; double probability = 1 - std::pow( 1 -
          // S1/((double) volume)  , markers ) - std::pow( 1- S2/((double)
          // volume)  , markers ) + std::pow( 1- (S1+S2)/((double) volume)  ,
          // markers );
          double probability =
              (1 - std::pow(1 - S1 / ((double) volume), markers) -
               std::pow(1 - S2 / ((double) volume), markers) +
               std::pow(1 - (S1 + S2) / ((double) volume), markers));

          // std::cout<<"probability :" <<probability<<std::endl;

          if (probability > 0.0) {
            RES_C res = Tree_out.edgeFromVertices(
                typename boost::graph_traits<BoostGraph>::vertex_descriptor(
                    Tree.edgeSource(e1)),
                typename boost::graph_traits<BoostGraph>::vertex_descriptor(
                    Tree.edgeTarget(e1)),
                &e2);
            Tree_out.setEdgeWeight(e2, 255.0 * probability);
          } else {
            RES_C res = Tree_out.edgeFromVertices(
                typename boost::graph_traits<BoostGraph>::vertex_descriptor(
                    Tree.edgeSource(e1)),
                typename boost::graph_traits<BoostGraph>::vertex_descriptor(
                    Tree.edgeTarget(e1)),
                &e2);
            Tree_out.setEdgeWeight(e2, 0.0);
          }
        }

        while (last_edge_value == last_analyzed) {
          last_edge_value = val_edges.back();
          val_edges.pop_back();
        }
        last_analyzed = last_edge_value;
      }

      std::cout << "project on graphs" << std::endl;

      // const morphee::graph::CommonGraph32 t_CopyGraph(const
      // morphee::graph::CommonGraph32 &);
      Tree2     = t_CopyGraph(Tree_out);
      Tree_temp = t_CopyGraph(Tree_out);

      for (boost::tie(ed_it, ed_end) = boost::edges(Tree_out.getBoostGraph());
           ed_it != ed_end; ++ed_it) {
        Tree_out.edgeWeight(*ed_it, &tmp);
        double value = (double) tmp;

        if (value > 0.0) {
          Tree_temp.removeEdge(
              typename boost::graph_traits<BoostGraph>::vertex_descriptor(
                  Tree_out.edgeSource(*ed_it)),
              typename boost::graph_traits<BoostGraph>::vertex_descriptor(
                  Tree_out.edgeTarget(*ed_it)));

          // RES_C t_LabelConnectedComponent(const
          // morphee::graph::CommonGraph32&, morphee::graph::CommonGraph32&);
          t_LabelConnectedComponent(Tree_temp, Tree2);

          Tree_temp.addEdge(
              typename boost::graph_traits<BoostGraph>::vertex_descriptor(
                  Tree_out.edgeSource(*ed_it)),
              typename boost::graph_traits<BoostGraph>::vertex_descriptor(
                  Tree_out.edgeTarget(*ed_it)),
              tmp);

          for (boost::tie(ed_it2, ed_end2) = boost::edges(G.getBoostGraph());
               ed_it2 != ed_end2; ++ed_it2) {
            Tree2.vertexData(
                typename boost::graph_traits<BoostGraph>::vertex_descriptor(
                    G.edgeSource(*ed_it2)),
                &label1);
            Tree2.vertexData(
                typename boost::graph_traits<BoostGraph>::vertex_descriptor(
                    G.edgeTarget(*ed_it2)),
                &label2);

            if (label1 != label2) {
              G.edgeWeight(*ed_it2, &tmp);
              G.setEdgeWeight(*ed_it2, std::max((double) tmp, value));
              // G.setEdgeWeight(*ed_it2,(double) tmp + value);
            }
          }
        }
      }

      std::cout << "project on image the pdf" << std::endl;

      typename ImageOut::const_iterator it, iend;
      morphee::selement::Neighborhood<SE, ImageOut> neighb(imOut, nl);
      typename morphee::selement::Neighborhood<SE, ImageOut>::iterator nit,
          nend;
      offset_t o0, o1;

      for (it = imOut.begin(), iend = imOut.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o0 = it.getOffset();
        imOut.setPixel(o0, 0);
      }

      for (it = imOut.begin(), iend = imOut.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o0             = it.getOffset();
        int val1       = imIn.pixelFromOffset(o0);
        double valout1 = imOut.pixelFromOffset(o0);

        neighb.setCenter(o0);

        for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
          o1 = nit.getOffset();

          if (o1 > o0) {
            int val2       = imIn.pixelFromOffset(o1);
            double valout2 = imOut.pixelFromOffset(o1);

            if (val2 != val1) {
              RES_C res = G.edgeFromVertices(val1 - 1, val2 - 1, &e1);
              if (res == RES_OK) {
                G.edgeWeight(e1, &tmp);
                if (tmp > 0) {
                  imOut.setPixel(o0, std::max(valout1, (double) tmp));
                  imOut.setPixel(o1, std::max(valout2, (double) tmp));
                }
              } else {
                RES_C res = G.edgeFromVertices(val2 - 1, val1 - 1, &e1);
                if (res == RES_OK) {
                  G.edgeWeight(e1, &tmp);
                  if (tmp > 0) {
                    imOut.setPixel(o0, std::max(valout1, (double) tmp));
                    imOut.setPixel(o1, std::max(valout2, (double) tmp));
                  }
                }
              }
            }
          }
        }
      }

      return RES_OK;
    }

    // ##################################################
    // END FROM STAWIASKI JAN 2012
    // ##################################################

    template <class BoostGraph>
    const BoostGraph t_CopyGraph(const BoostGraph &graphIn)
    {
      MORPHEE_ENTER_FUNCTION("t_CopyGraph");

      typedef typename BoostGraph::EdgeIterator EdgeIterator;
      typedef typename BoostGraph::VertexIterator VertexIterator;

      EdgeIterator ed_it, ed_end;
      VertexIterator v_it, v_end;

      BoostGraph GCopy(graphIn.numVertices());
      typename BoostGraph::EdgeProperty tmp;
      typename BoostGraph::VertexProperty vdata1;

      for (boost::tie(ed_it, ed_end) = boost::edges(graphIn.getBoostGraph());
           ed_it != ed_end; ++ed_it) {
        graphIn.edgeWeight(*ed_it,
                           &tmp); // shouldn't ever happen, but who knows ?
        GCopy.addEdge(
            typename BoostGraph::VertexDescriptor(graphIn.edgeSource(*ed_it)),
            typename BoostGraph::VertexDescriptor(graphIn.edgeTarget(*ed_it)),
            tmp);
      }
      // copying the properties of each vertex
      for (boost::tie(v_it, v_end) = boost::vertices(graphIn.getBoostGraph());
           v_it != v_end; ++v_it) {
        graphIn.vertexData(*v_it, &vdata1);
        GCopy.setVertexData(*v_it, vdata1);
      }

      return GCopy;
    }

    template <class Graph>
    RES_C t_LabelConnectedComponent(const Graph &GIn, Graph &Gout)
    {
      MORPHEE_ENTER_FUNCTION("t_LabelConnectedComponent");

      typedef typename Graph::VertexDescriptor VertexDescriptor;
      typedef typename Graph::_boostGraph BoostGraph;
      typedef
          typename boost::graph_traits<BoostGraph>::edge_iterator EdgeIterator;
      typedef typename boost::graph_traits<BoostGraph>::vertex_iterator
          VertexIterator;
      typedef typename boost::graph_traits<BoostGraph>::edge_descriptor
          edge_descriptor;

      EdgeIterator ed_it, ed_end;
      VertexIterator u_iter, u_end;

      Gout = t_CopyGraph(GIn);

      std::vector<int> component(boost::num_vertices(GIn.getBoostGraph()));
      int num = connected_components(GIn.getBoostGraph(), &component[0]);

      // SCAN IMAGE TO FIND THE PIXELS LABELS
      for (boost::tie(u_iter, u_end) = boost::vertices(GIn.getBoostGraph());
           u_iter != u_end; ++u_iter) {
        Gout.setVertexData(*u_iter, component[*u_iter] + 1);
      }

      return RES_OK;
    }

    // BEGIN FROM stawiaski JAN 2012
    template <class Graph>
    RES_C t_LabelConnectedComponent(const Graph &GIn, Graph &Gout, int *num)
    {
      MORPHEE_ENTER_FUNCTION("t_LabelConnectedComponent");

      typedef typename Graph::VertexDescriptor VertexDescriptor;
      typedef typename Graph::_boostGraph BoostGraph;
      typedef
          typename boost::graph_traits<BoostGraph>::edge_iterator EdgeIterator;
      typedef typename boost::graph_traits<BoostGraph>::vertex_iterator
          VertexIterator;
      typedef typename boost::graph_traits<BoostGraph>::edge_descriptor
          edge_descriptor;

      EdgeIterator ed_it, ed_end;
      VertexIterator u_iter, u_end;

      Gout = t_CopyGraph(GIn);

      std::vector<int> component(boost::num_vertices(GIn.getBoostGraph()));
      *num = connected_components(GIn.getBoostGraph(), &component[0]);

      // SCAN IMAGE TO FIND THE PIXELS LABELS
      for (boost::tie(u_iter, u_end) = boost::vertices(GIn.getBoostGraph());
           u_iter != u_end; ++u_iter) {
        Gout.setVertexData(*u_iter, component[*u_iter] + 1);
      }

      return RES_OK;
    }
    // END FROM stawiaski JAN 2012

    template <class ImageIn, class ImageGradx, class ImageGrady,
              class ImageMarker, class SE, class ImageOut>
    RES_C t_geoCuts(const ImageIn &imIn, const ImageGradx &imGradx,
                    const ImageGrady &imGrady, const ImageMarker &imMarker,
                    const SE &nl, ImageOut &imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_GeoCuts");

      std::cout << "Enter function Geo-Cuts" << std::endl;

      if ((!imOut.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if ((!imIn.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if ((!imGradx.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if ((!imGrady.isAllocated())) {
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

      std::cout << "build graph vertices" << std::endl;
      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o0 = it.getOffset();
        boost::add_vertex(g);
        numVert++;
      }

      std::cout << "number of vertices: " << numVert << std::endl;

      vSource = boost::add_vertex(g);
      vSink   = boost::add_vertex(g);

      std::cout << "build graph edges" << std::endl;
      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1         = it.getOffset();
        double val = imIn.pixelFromOffset(o1);
        int marker = imMarker.pixelFromOffset(o1);

        int valright = 0;
        int valleft  = 0;
        int valup    = 0;
        int valdown  = 0;

        if (marker == 2) {
          boost::tie(e4, in1) = boost::add_edge(vSource, o1, g);
          boost::tie(e3, in1) = boost::add_edge(o1, vSource, g);
          capacity[e4]        = (std::numeric_limits<double>::max)();
          capacity[e3]        = (std::numeric_limits<double>::max)();
          rev[e4]             = e3;
          rev[e3]             = e4;
        } else if (marker == 3) {
          boost::tie(e3, in1) = boost::add_edge(o1, vSink, g);
          boost::tie(e4, in1) = boost::add_edge(vSink, o1, g);
          capacity[e3]        = (std::numeric_limits<double>::max)();
          capacity[e4]        = (std::numeric_limits<double>::max)();
          rev[e3]             = e4;
          rev[e4]             = e3;
        }

        neighb.setCenter(o1);

        for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
          const offset_t o2 = nit.getOffset();
          if (o2 <= o1)
            continue;
          if (o2 > o1) {
            double val2 = imIn.pixelFromOffset(o2);
            double cost = 1000 / (1 + 1.5 * (val - val2) * (val - val2));
            // double cost = std::exp(-((val-val2)*(val-val2))/(1.0));
            boost::tie(e4, in1) = boost::add_edge(o1, o2, g);
            boost::tie(e3, in1) = boost::add_edge(o2, o1, g);
            capacity[e4]        = cost;
            capacity[e3]        = cost;
            rev[e4]             = e3;
            rev[e3]             = e4;
          }
        }

        for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
          const offset_t o2 = nit.getOffset();
          if (o2 == o1)
            continue;
          else if (o2 == o1 - 1)
            valleft = (int) (imGradx.pixelFromOffset(o2));
          else if (o2 == o1 + 1)
            valright = (int) (imGradx.pixelFromOffset(o2));
          else if (o2 > o1 + 1)
            valdown = (int) (imGrady.pixelFromOffset(o2));
          else if (o2 < o1 - 1)
            valup = (int) (imGrady.pixelFromOffset(o2));
        }

        // std::cout<<"  "<<valleft<<"  "<<valright<<"  "<<valdown<<"
        // "<<valup<<std::endl;
        double divergence = 10 * ((valleft - valright) + (valup - valdown));

        if (divergence < 0 && marker == 0) {
          boost::tie(e4, in1) = boost::add_edge(vSource, o1, g);
          boost::tie(e3, in1) = boost::add_edge(o1, vSource, g);
          capacity[e4]        = std::abs(divergence);
          capacity[e3]        = std::abs(divergence);
          rev[e4]             = e3;
          rev[e3]             = e4;
        } else if (divergence > 0 && marker == 0) {
          boost::tie(e4, in1) = boost::add_edge(o1, vSink, g);
          boost::tie(e3, in1) = boost::add_edge(vSink, o1, g);
          capacity[e4]        = divergence;
          capacity[e3]        = divergence;
          rev[e4]             = e3;
          rev[e3]             = e4;
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
        o1 = it.getOffset();
        if (color[o1] == color[vSource])
          imOut.setPixel(o1, 2);
        if (color[o1] == 1)
          imOut.setPixel(o1, 4);
        if (color[o1] == color[vSink])
          imOut.setPixel(o1, 3);
      }

      return RES_OK;
    }

    template <class ImageIn, class ImageMarker, class SE, class ImageOut>
    RES_C t_geoCutsMinSurfaces(const ImageIn &imIn,
                                const ImageMarker &imMarker, const SE &nl,
                                ImageOut &imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_geoCutsMinSurfaces");

      std::cout << "Enter function Geo-Cuts " << std::endl;

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
      int numVert  = 0;
      int numEdges = 0;

      std::cout << "build graph vertices" << std::endl;
      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o0 = it.getOffset();
        boost::add_vertex(g);
        numVert++;
      }

      std::cout << "number of vertices: " << numVert << std::endl;

      vSource = boost::add_vertex(g);
      vSink   = boost::add_vertex(g);

      std::cout << "build graph edges" << std::endl;
      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1         = it.getOffset();
        double val = imIn.pixelFromOffset(o1);
        int marker = imMarker.pixelFromOffset(o1);

        int valright = 0;
        int valleft  = 0;
        int valup    = 0;
        int valdown  = 0;

        if (marker == 2) {
          boost::tie(e4, in1) = boost::add_edge(vSource, o1, g);
          boost::tie(e3, in1) = boost::add_edge(o1, vSource, g);
          capacity[e4]        = (std::numeric_limits<double>::max)();
          capacity[e3]        = (std::numeric_limits<double>::max)();
          rev[e4]             = e3;
          rev[e3]             = e4;
        } else if (marker == 3) {
          boost::tie(e4, in1) = boost::add_edge(o1, vSink, g);
          boost::tie(e3, in1) = boost::add_edge(vSink, o1, g);
          capacity[e4]        = (std::numeric_limits<double>::max)();
          capacity[e3]        = (std::numeric_limits<double>::max)();
          rev[e4]             = e3;
          rev[e3]             = e4;
        }

        neighb.setCenter(o1);

        for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
          const offset_t o2 = nit.getOffset();
          if (o2 <= o1)
            continue;
          if (o2 > o1) {
            numEdges++;
            double val2         = imIn.pixelFromOffset(o2);
            double maxi         = std::abs(val2 - val);
            double cost         = 256 / (1 + std::pow(maxi, 2));
            boost::tie(e4, in1) = boost::add_edge(o1, o2, g);
            boost::tie(e3, in1) = boost::add_edge(o2, o1, g);
            capacity[e4]        = cost;
            capacity[e3]        = cost;
            rev[e4]             = e3;
            rev[e3]             = e4;
          }
        }
      }

      std::cout << "number of Edges : " << numEdges << std::endl;

      boost::property_map<Graph_d, boost::vertex_index_t>::type indexmap =
          boost::get(boost::vertex_index, g);
      std::vector<boost::default_color_type> color(boost::num_vertices(g));

      std::cout << "Compute Max flow " << std::endl;
      /*double flow = kolmogorov_max_flow(g, capacity, residual_capacity, rev,
      &color[0], indexmap, vSource, vSink); std::cout << "c  The total flow:" <<
      std::endl; std::cout << "s " << flow << std::endl << std::endl;*/
#if BOOST_VERSION >= 104700
      double flow2 =
          boykov_kolmogorov_max_flow(g, capacity, residual_capacity, rev,
                                     &color[0], indexmap, vSource, vSink);
#else
      double flow2 = kolmogorov_max_flow(g, capacity, residual_capacity, rev,
                                         &color[0], indexmap, vSource, vSink);
#endif
      std::cout << "c  The total flow found :" << std::endl;
      std::cout << "s " << flow2 << std::endl << std::endl;

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1 = it.getOffset();
        if (color[o1] == color[vSource])
          imOut.setPixel(o1, 2);
        else if (color[o1] == color[vSink])
          imOut.setPixel(o1, 3);
        else if (color[o1] == 1)
          imOut.setPixel(o1, 4);
      }

      return RES_OK;
    }

    template <class ImageIn, class ImageVal, typename _nbmarkers, class SE,
              class ImageOut>
    RES_C t_geoCutsStochastic_Watershed(const ImageIn &imIn,
                                         const ImageVal &imVal,
                                         const _nbmarkers nbmarkers,
                                         const SE &nl, ImageOut &imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_geoCutsStochastic_Watershed");

      std::cout << "Enter function Geo-Cuts Stochastic Watershed" << std::endl;

      if ((!imOut.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if ((!imIn.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if ((!imVal.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      double markers = (double) nbmarkers;

      int size = imIn.getXSize() * imIn.getYSize() * imIn.getZSize();
      // std::cout<<size<<std::endl;

      typedef typename morphee::graph::CommonGraph32::_boostGraph BoostGraph;
      typedef
          typename boost::graph_traits<BoostGraph>::edge_iterator EdgeIterator;
      typedef typename boost::graph_traits<BoostGraph>::vertex_iterator
          VertexIterator;
      typedef typename boost::graph_traits<BoostGraph>::edge_descriptor
          EdgeDescriptor;
      typedef typename boost::graph_traits<BoostGraph>::vertex_descriptor
          VertexDescriptor;
      typename morphee::graph::CommonGraph32::EdgeProperty tmp;
      typename morphee::graph::CommonGraph32::VertexProperty vdata1;
      typename morphee::graph::CommonGraph32::VertexProperty label1, label2;
      EdgeIterator ed_it, ed_end, ed_it2, ed_end2;
      VertexIterator v_it, v_end;
      EdgeDescriptor last_edge, e1, e2;

      bool in1;

      std::vector<double> val_edges;
      val_edges.push_back(0.0);
      double last_edge_value = 0.0;

      std::vector<EdgeDescriptor> removed_edges;

      morphee::graph::CommonGraph32 G(0);
      morphee::graph::CommonGraph32 Tree(0);

      morphee::morphoBase::t_NeighborhoodGraphFromMosaic_WithPass(imIn, imVal,
                                                                  nl, G);
      Tree = morphee::graph::MinimumSpanningTreeFromGraph(G);

      ImageIn ImTempSurfaces = imIn.getSame();
      // Image<UINT8>ImTempSurfaces = imIn.t_getSame<UINT8>();
      // morphee::morphoBase::t_ImLabelFlatZonesWithArea(imIn,nl,ImTempSurfaces);
      morphee::morphoBase::t_ImLabelFlatZonesWithVolume(imIn, imVal, nl,
                                                        ImTempSurfaces);

      // ImageIn ImTempVolumes = imIn.getSame();
      // ImageIn imMinima = imIn.getSame();
      // morphee::morphoBase::t_ImMinima(imVal,nl,1,imMinima);
      // Image<UINT32>imLabel = imIn.t_getSame<UINT32>();
      // Image<UINT32>imOut2 = imIn.t_getSame<UINT32>();
      // morphee::morphoBase::t_ImLabel(imMinima,nl,imLabel);
      // morphee::morphoBase::t_ImHierarchicalSegmentation(imVal,imLabel,nl,morphee::morphoBase::VolumicHierarchicalSegmentation,imOut2,Tree);
      //

      morphee::graph::CommonGraph32 Tree_temp = t_CopyGraph(Tree);
      morphee::graph::CommonGraph32 Tree_out  = t_CopyGraph(Tree);
      morphee::graph::CommonGraph32 Tree2     = t_CopyGraph(Tree_temp);

      morphee::graph::t_ProjectMarkersOnGraph(ImTempSurfaces, imIn, Tree);

      double volume = 0.0;
      for (boost::tie(v_it, v_end) = boost::vertices(G.getBoostGraph());
           v_it != v_end; ++v_it) {
        Tree.vertexData(*v_it, &vdata1);
        volume = volume + (double) vdata1;
      }

      for (boost::tie(ed_it, ed_end) = boost::edges(G.getBoostGraph());
           ed_it != ed_end; ++ed_it) {
        G.setEdgeWeight(*ed_it, 0);
      }

      for (boost::tie(ed_it, ed_end) = boost::edges(Tree.getBoostGraph());
           ed_it != ed_end; ++ed_it) {
        Tree.edgeWeight(*ed_it, &tmp);
        if ((double) tmp > last_edge_value) {
          val_edges.push_back(tmp);
          last_edge_value = (double) tmp;
        }
      }

      while (val_edges.size() > 1) {
        last_edge_value = val_edges.back();
        val_edges.pop_back();

        // remove edge of maximal weight
        for (boost::tie(ed_it, ed_end) = boost::edges(Tree.getBoostGraph());
             ed_it != ed_end; ++ed_it) {
          // std::cout<<"look"<<std::endl;
          Tree.edgeWeight(*ed_it, &tmp);

          if (tmp == last_edge_value) {
            boost::tie(e1, in1) =
                boost::edge(Tree.edgeSource(*ed_it), Tree.edgeTarget(*ed_it),
                            Tree_temp.getBoostGraph());
            removed_edges.push_back(e1);
            Tree_temp.removeEdge(
                typename boost::graph_traits<BoostGraph>::vertex_descriptor(
                    Tree.edgeSource(*ed_it)),
                typename boost::graph_traits<BoostGraph>::vertex_descriptor(
                    Tree.edgeTarget(*ed_it)));
          }
        }

        t_LabelConnectedComponent(Tree_temp, Tree2);

        while (removed_edges.size() > 0) {
          e1 = removed_edges.back();
          removed_edges.pop_back();

          Tree2.vertexData(
              typename boost::graph_traits<BoostGraph>::vertex_descriptor(
                  Tree.edgeSource(e1)),
              &label1);
          Tree2.vertexData(
              typename boost::graph_traits<BoostGraph>::vertex_descriptor(
                  Tree.edgeTarget(e1)),
              &label2);

          double S1 = 0;
          double S2 = 0;

          for (boost::tie(v_it, v_end) = boost::vertices(Tree2.getBoostGraph());
               v_it != v_end; ++v_it) {
            Tree2.vertexData(*v_it, &vdata1);
            if (vdata1 == label1) {
              Tree.vertexData(*v_it, &vdata1);
              S1 = S1 + (double) vdata1;
            } else if (vdata1 == label2) {
              Tree.vertexData(*v_it, &vdata1);
              S2 = S2 + (double) vdata1;
            }
          }

          // double k = markers*((S1+S2)/(double) volume);
          // double probability = 1 - std::pow( ( S1/(S1+S2) ) , k ) - std::pow(
          // ( S2/(S1+S2) ) , k ) ;
          double probability =
              1 - std::pow(1 - S1 / ((double) volume), markers) -
              std::pow(1 - S2 / ((double) volume), markers) +
              std::pow(1 - (S1 + S2) / ((double) volume), markers);

          // std::cout<<"probability :" <<probability<<std::endl;

          if (probability > 0.0) {
            RES_C res = Tree_out.edgeFromVertices(
                typename boost::graph_traits<BoostGraph>::vertex_descriptor(
                    Tree.edgeSource(e1)),
                typename boost::graph_traits<BoostGraph>::vertex_descriptor(
                    Tree.edgeTarget(e1)),
                &e2);
            Tree_out.setEdgeWeight(e2, 255.0 * probability);
          } else {
            RES_C res = Tree_out.edgeFromVertices(
                typename boost::graph_traits<BoostGraph>::vertex_descriptor(
                    Tree.edgeSource(e1)),
                typename boost::graph_traits<BoostGraph>::vertex_descriptor(
                    Tree.edgeTarget(e1)),
                &e2);
            Tree_out.setEdgeWeight(e2, 0.0);
          }
        }
      }

      std::cout << "project on graphs" << std::endl;
      Tree2     = t_CopyGraph(Tree_out);
      Tree_temp = t_CopyGraph(Tree_out);

      for (boost::tie(ed_it, ed_end) = boost::edges(Tree_out.getBoostGraph());
           ed_it != ed_end; ++ed_it) {
        Tree_out.edgeWeight(*ed_it, &tmp);
        double value = (double) tmp;

        if (value > 0.0) {
          Tree_temp.removeEdge(
              typename boost::graph_traits<BoostGraph>::vertex_descriptor(
                  Tree_out.edgeSource(*ed_it)),
              typename boost::graph_traits<BoostGraph>::vertex_descriptor(
                  Tree_out.edgeTarget(*ed_it)));

          t_LabelConnectedComponent(Tree_temp, Tree2);

          Tree_temp.addEdge(
              typename boost::graph_traits<BoostGraph>::vertex_descriptor(
                  Tree_out.edgeSource(*ed_it)),
              typename boost::graph_traits<BoostGraph>::vertex_descriptor(
                  Tree_out.edgeTarget(*ed_it)),
              tmp);

          for (boost::tie(ed_it2, ed_end2) = boost::edges(G.getBoostGraph());
               ed_it2 != ed_end2; ++ed_it2) {
            Tree2.vertexData(
                typename boost::graph_traits<BoostGraph>::vertex_descriptor(
                    G.edgeSource(*ed_it2)),
                &label1);
            Tree2.vertexData(
                typename boost::graph_traits<BoostGraph>::vertex_descriptor(
                    G.edgeTarget(*ed_it2)),
                &label2);

            if (label1 != label2) {
              G.edgeWeight(*ed_it2, &tmp);
              G.setEdgeWeight(*ed_it2, std::max((double) tmp, value));
              // G.setEdgeWeight(*ed_it2,(double) tmp + value);
            }
          }
        }
      }

      std::cout << "project on image the pdf" << std::endl;

      typename ImageOut::const_iterator it, iend;
      morphee::selement::Neighborhood<SE, ImageOut> neighb(imOut, nl);
      typename morphee::selement::Neighborhood<SE, ImageOut>::iterator nit,
          nend;
      offset_t o0, o1;

      for (it = imOut.begin(), iend = imOut.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o0 = it.getOffset();
        imOut.setPixel(o0, 0);
      }

      for (it = imOut.begin(), iend = imOut.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o0             = it.getOffset();
        int val1       = imIn.pixelFromOffset(o0);
        double valout1 = imOut.pixelFromOffset(o0);

        neighb.setCenter(o0);

        for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
          o1 = nit.getOffset();

          if (o1 > o0) {
            int val2       = imIn.pixelFromOffset(o1);
            double valout2 = imOut.pixelFromOffset(o1);

            if (val2 != val1) {
              RES_C res = G.edgeFromVertices(val1 - 1, val2 - 1, &e1);
              if (res == RES_OK) {
                G.edgeWeight(e1, &tmp);
                if (tmp > 0) {
                  imOut.setPixel(o0, std::max(valout1, (double) tmp));
                  imOut.setPixel(o1, std::max(valout2, (double) tmp));
                }
              } else {
                RES_C res = G.edgeFromVertices(val2 - 1, val1 - 1, &e1);
                if (res == RES_OK) {
                  G.edgeWeight(e1, &tmp);
                  if (tmp > 0) {
                    imOut.setPixel(o0, std::max(valout1, (double) tmp));
                    imOut.setPixel(o1, std::max(valout2, (double) tmp));
                  }
                }
              }
            }
          }
        }
      }

      return RES_OK;
    }

    template <class ImageIn, class ImageVal, class ImageMark,
              typename _nbmarkers, class SE, class ImageOut>
    RES_C t_geoCutsStochastic_Watershed_2(const ImageIn &imIn,
                                           const ImageVal &imVal,
                                           const ImageMark &imMark,
                                           const _nbmarkers nbmarkers,
                                           const SE &nl, ImageOut &imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_geoCutsStochastic_Watershed_2");

      std::cout << "Enter function Geo-Cuts Stochastic Watershed_2"
                << std::endl;

      if ((!imOut.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if ((!imIn.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if ((!imVal.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if ((!imMark.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      double markers = (double) nbmarkers;

      int size = imIn.getXSize() * imIn.getYSize() * imIn.getZSize();
      // std::cout<<size<<std::endl;

      typedef typename morphee::graph::CommonGraph32::_boostGraph BoostGraph;
      typedef
          typename boost::graph_traits<BoostGraph>::edge_iterator EdgeIterator;
      typedef typename boost::graph_traits<BoostGraph>::vertex_iterator
          VertexIterator;
      typedef typename boost::graph_traits<BoostGraph>::edge_descriptor
          EdgeDescriptor;
      typedef typename boost::graph_traits<BoostGraph>::vertex_descriptor
          VertexDescriptor;
      typename morphee::graph::CommonGraph32::EdgeProperty tmp;
      typename morphee::graph::CommonGraph32::VertexProperty vdata1, vdata2;
      typename morphee::graph::CommonGraph32::VertexProperty label1, label2;
      EdgeIterator ed_it, ed_end, ed_it2, ed_end2;
      VertexIterator v_it, v_end;
      EdgeDescriptor last_edge, e1, e2;
      bool in1;

      std::vector<double> val_edges;
      val_edges.push_back(0.0);
      double last_edge_value = 0.0;

      std::vector<EdgeDescriptor> removed_edges;

      morphee::graph::CommonGraph32 G(0);
      morphee::graph::CommonGraph32 Tree(0);
      morphee::morphoBase::t_NeighborhoodGraphFromMosaic_WithPass(imIn, imVal,
                                                                  nl, G);
      // morphee::morphoBase::t_NeighborhoodGraphFromMosaic_WithAverageAndDifference(imIn,imVal,nl,G);
      Tree = morphee::graph::MinimumSpanningTreeFromGraph(G);

      ImageIn ImTempSurfaces = imIn.getSame();
      // Image<UINT8>ImTempSurfaces = imIn.t_getSame<UINT8>();
      morphee::morphoBase::t_ImLabelFlatZonesWithArea(imIn, nl, ImTempSurfaces);
      // morphee::morphoBase::t_ImLabelFlatZonesWithVolume(imIn,imVal,nl,ImTempSurfaces);

      // ImageIn imMinima = imIn.getSame();
      // morphee::morphoBase::t_ImMinima(imVal,nl,1,imMinima);
      // Image<UINT32>imLabel = imIn.t_getSame<UINT32>();
      // morphee::morphoBase::t_ImLabel(imMinima,nl,imLabel);
      // morphee::morphoBase::t_ImLabelMarkersWithExtinctionValues_Area(imLabel,imVal,nl,ImTempSurfaces);

      morphee::graph::CommonGraph32 Tree_temp = t_CopyGraph(Tree);
      morphee::graph::CommonGraph32 Tree_out  = t_CopyGraph(Tree);
      morphee::graph::CommonGraph32 Tree2     = t_CopyGraph(Tree_temp);
      morphee::graph::CommonGraph32 TreeMark  = t_CopyGraph(Tree);

      morphee::graph::t_ProjectMarkersOnGraph(ImTempSurfaces, imIn, Tree);

      morphee::graph::t_ProjectMarkersOnGraph(imMark, imIn, TreeMark);

      double volume = 0.0;
      for (boost::tie(v_it, v_end) = boost::vertices(G.getBoostGraph());
           v_it != v_end; ++v_it) {
        Tree.vertexData(*v_it, &vdata1);
        volume = volume + (double) vdata1;
      }

      for (boost::tie(ed_it, ed_end) = boost::edges(G.getBoostGraph());
           ed_it != ed_end; ++ed_it) {
        G.setEdgeWeight(*ed_it, 0);
      }

      for (boost::tie(ed_it, ed_end) = boost::edges(Tree.getBoostGraph());
           ed_it != ed_end; ++ed_it) {
        Tree.edgeWeight(*ed_it, &tmp);
        if ((double) tmp > last_edge_value) {
          val_edges.push_back(tmp);
          last_edge_value = (double) tmp;
        }
      }

      while (val_edges.size() > 1) {
        last_edge_value = val_edges.back();
        val_edges.pop_back();

        // remove edge of maximal weight
        for (boost::tie(ed_it, ed_end) = boost::edges(Tree.getBoostGraph());
             ed_it != ed_end; ++ed_it) {
          // std::cout<<"look"<<std::endl;
          Tree.edgeWeight(*ed_it, &tmp);

          if (tmp == last_edge_value) {
            boost::tie(e1, in1) =
                boost::edge(Tree.edgeSource(*ed_it), Tree.edgeTarget(*ed_it),
                            Tree_temp.getBoostGraph());
            removed_edges.push_back(e1);
            Tree_temp.removeEdge(
                typename boost::graph_traits<BoostGraph>::vertex_descriptor(
                    Tree.edgeSource(*ed_it)),
                typename boost::graph_traits<BoostGraph>::vertex_descriptor(
                    Tree.edgeTarget(*ed_it)));
          }
        }

        t_LabelConnectedComponent(Tree_temp, Tree2);

        while (removed_edges.size() > 0) {
          e1 = removed_edges.back();
          removed_edges.pop_back();

          Tree2.vertexData(
              typename boost::graph_traits<BoostGraph>::vertex_descriptor(
                  Tree.edgeSource(e1)),
              &label1);
          Tree2.vertexData(
              typename boost::graph_traits<BoostGraph>::vertex_descriptor(
                  Tree.edgeTarget(e1)),
              &label2);

          double S1 = 0;
          double S2 = 0;

          bool region1 = false;
          bool region2 = false;

          for (boost::tie(v_it, v_end) = boost::vertices(Tree2.getBoostGraph());
               v_it != v_end; ++v_it) {
            Tree2.vertexData(*v_it, &vdata1);
            TreeMark.vertexData(*v_it, &vdata2);

            if (vdata1 == label1) {
              Tree.vertexData(*v_it, &vdata1);
              S1 = S1 + (double) vdata1;
            } else if (vdata1 == label2) {
              Tree.vertexData(*v_it, &vdata1);
              S2 = S2 + (double) vdata1;
            }

            if (vdata1 == label1 && vdata2 > 0) {
              region1 = true;
            } else if (vdata1 == label2 && vdata2 > 0) {
              region2 = true;
            }
          }

          // double k = markers*((S1+S2)/(double) volume);
          // double probability = 1 - std::pow( ( S1/(S1+S2) ) , k ) - std::pow(
          // ( S2/(S1+S2) ) , k ) ;
          double probability = 0;

          if (region1 == false && region2 == false) {
            probability = 1 - std::pow(1 - S1 / ((double) volume), markers) -
                          std::pow(1 - S2 / ((double) volume), markers) +
                          std::pow(1 - (S1 + S2) / ((double) volume), markers);
          } else if (region1 == true && region2 == false) {
            probability = 1 - std::pow(1 - S2 / ((double) volume), markers);
          } else if (region1 == false && region2 == true) {
            probability = 1 - std::pow(1 - S1 / ((double) volume), markers);
          } else if (region1 == true && region2 == true) {
            probability = 0;
          }

          // std::cout<<"probability :" <<probability<<std::endl;

          if (probability > 0.0) {
            RES_C res = Tree_out.edgeFromVertices(
                typename boost::graph_traits<BoostGraph>::vertex_descriptor(
                    Tree.edgeSource(e1)),
                typename boost::graph_traits<BoostGraph>::vertex_descriptor(
                    Tree.edgeTarget(e1)),
                &e2);
            Tree_out.setEdgeWeight(e2, 255.0 * probability);
          } else {
            RES_C res = Tree_out.edgeFromVertices(
                typename boost::graph_traits<BoostGraph>::vertex_descriptor(
                    Tree.edgeSource(e1)),
                typename boost::graph_traits<BoostGraph>::vertex_descriptor(
                    Tree.edgeTarget(e1)),
                &e2);
            Tree_out.setEdgeWeight(e2, 0.0);
          }
        }
      }

      std::cout << "project on graphs" << std::endl;
      Tree2     = t_CopyGraph(Tree_out);
      Tree_temp = t_CopyGraph(Tree_out);

      for (boost::tie(ed_it, ed_end) = boost::edges(Tree_out.getBoostGraph());
           ed_it != ed_end; ++ed_it) {
        Tree_out.edgeWeight(*ed_it, &tmp);
        double value = (double) tmp;

        if (value > 0.0) {
          Tree_temp.removeEdge(
              typename boost::graph_traits<BoostGraph>::vertex_descriptor(
                  Tree_out.edgeSource(*ed_it)),
              typename boost::graph_traits<BoostGraph>::vertex_descriptor(
                  Tree_out.edgeTarget(*ed_it)));

          t_LabelConnectedComponent(Tree_temp, Tree2);

          Tree_temp.addEdge(
              typename boost::graph_traits<BoostGraph>::vertex_descriptor(
                  Tree_out.edgeSource(*ed_it)),
              typename boost::graph_traits<BoostGraph>::vertex_descriptor(
                  Tree_out.edgeTarget(*ed_it)),
              tmp);

          for (boost::tie(ed_it2, ed_end2) = boost::edges(G.getBoostGraph());
               ed_it2 != ed_end2; ++ed_it2) {
            Tree2.vertexData(
                typename boost::graph_traits<BoostGraph>::vertex_descriptor(
                    G.edgeSource(*ed_it2)),
                &label1);
            Tree2.vertexData(
                typename boost::graph_traits<BoostGraph>::vertex_descriptor(
                    G.edgeTarget(*ed_it2)),
                &label2);

            if (label1 != label2) {
              G.edgeWeight(*ed_it2, &tmp);
              G.setEdgeWeight(*ed_it2, std::max((double) tmp, value));
              // G.setEdgeWeight(*ed_it2,(double) tmp + value);
            }
          }
        }
      }

      std::cout << "project on image the pdf" << std::endl;

      typename ImageOut::const_iterator it, iend;
      morphee::selement::Neighborhood<SE, ImageOut> neighb(imOut, nl);
      typename morphee::selement::Neighborhood<SE, ImageOut>::iterator nit,
          nend;
      offset_t o0, o1;

      for (it = imOut.begin(), iend = imOut.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o0 = it.getOffset();
        imOut.setPixel(o0, 0);
      }

      for (it = imOut.begin(), iend = imOut.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o0             = it.getOffset();
        int val1       = imIn.pixelFromOffset(o0);
        double valout1 = imOut.pixelFromOffset(o0);

        neighb.setCenter(o0);

        for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
          o1 = nit.getOffset();

          if (o1 > o0) {
            int val2       = imIn.pixelFromOffset(o1);
            double valout2 = imOut.pixelFromOffset(o1);

            if (val2 != val1) {
              RES_C res = G.edgeFromVertices(val1 - 1, val2 - 1, &e1);
              if (res == RES_OK) {
                G.edgeWeight(e1, &tmp);
                if (tmp > 0) {
                  imOut.setPixel(o0, std::max(valout1, (double) tmp));
                  imOut.setPixel(o1, std::max(valout2, (double) tmp));
                }
              } else {
                RES_C res = G.edgeFromVertices(val2 - 1, val1 - 1, &e1);
                if (res == RES_OK) {
                  G.edgeWeight(e1, &tmp);
                  if (tmp > 0) {
                    imOut.setPixel(o0, std::max(valout1, (double) tmp));
                    imOut.setPixel(o1, std::max(valout2, (double) tmp));
                  }
                }
              }
            }
          }
        }
      }

      return RES_OK;
    }

    template <class ImageIn, class ImageMarker, typename _Power, class SE,
              class ImageOut>
    RES_C t_geoCutsWatershed_MinCut(const ImageIn &imIn,
                                     const ImageMarker &imMarker,
                                     const _Power Power, const SE &nl,
                                     ImageOut &imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_geoCutsWatershed_MinCut");

      std::cout << "Enter function Geo-Cuts Watershed" << std::endl;

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

      double exposant = (double) Power;
      std::cout << "exposant = " << exposant << std::endl;
      // std::cout<<"test 2^x = "<<std::pow(2.0,exposant)<<std::endl;

      // common image iterator
      typename ImageIn::const_iterator it, iend;
      typename ImageOut::const_iterator ito, ioend;
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
      Graph_d::vertex_descriptor vSource, vSink;
      int numVert = 0;

      std::cout << "build graph vertices" << std::endl;
      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o0 = it.getOffset();
        boost::add_vertex(g);
        numVert++;
      }

      std::cout << "number of vertices: " << numVert << std::endl;

      vSource = boost::add_vertex(g);
      vSink   = boost::add_vertex(g);

      std::cout << "build graph edges" << std::endl;
      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1         = it.getOffset();
        double val = imIn.pixelFromOffset(o1);
        int marker = imMarker.pixelFromOffset(o1);

        int valright = 0;
        int valleft  = 0;
        int valup    = 0;
        int valdown  = 0;

        if (marker == 2) {
          boost::tie(e4, in1) = boost::add_edge(vSource, o1, g);
          boost::tie(e3, in1) = boost::add_edge(o1, vSource, g);
          capacity[e4]        = (std::numeric_limits<double>::max)();
          capacity[e3]        = (std::numeric_limits<double>::max)();
          rev[e4]             = e3;
          rev[e3]             = e4;
        } else if (marker == 3) {
          boost::tie(e4, in1) = boost::add_edge(o1, vSink, g);
          boost::tie(e3, in1) = boost::add_edge(vSink, o1, g);
          capacity[e4]        = (std::numeric_limits<double>::max)();
          capacity[e3]        = (std::numeric_limits<double>::max)();
          rev[e4]             = e3;
          rev[e3]             = e4;
        }

        neighb.setCenter(o1);

        for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
          const offset_t o2 = nit.getOffset();
          if (o2 <= o1)
            continue;
          if (o2 > o1) {
            double val2   = imIn.pixelFromOffset(o2);
            double valeur = (255.0 / (std::abs(val - val2) + 1));
            // double valeur = (std::abs(val-val2)+1) ;
            double cost         = std::pow(valeur, exposant);
            boost::tie(e4, in1) = boost::add_edge(o1, o2, g);
            boost::tie(e3, in1) = boost::add_edge(o2, o1, g);
            capacity[e4]        = cost;
            capacity[e3]        = cost;
            rev[e4]             = e3;
            rev[e3]             = e4;
          }
        }
      }

      boost::property_map<Graph_d, boost::vertex_index_t>::type indexmap =
          boost::get(boost::vertex_index, g);
      std::vector<boost::default_color_type> color(boost::num_vertices(g));
      std::vector<boost::graph_traits<Graph_d>::vertex_descriptor> p(
          boost::num_vertices(g));

      std::cout << "Compute Max flow " << std::endl;
#if BOOST_VERSION >= 104700
      double flow =
          boykov_kolmogorov_max_flow(g, capacity, residual_capacity, rev,
                                     &color[0], indexmap, vSource, vSink);
#else
      double flow = kolmogorov_max_flow(g, capacity, residual_capacity, rev,
                                        &color[0], indexmap, vSource, vSink);
#endif

      std::cout << "c  The total flow found :" << std::endl;
      std::cout << "s " << flow << std::endl << std::endl;

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1 = it.getOffset();
        if (color[o1] == color[vSource])
          imOut.setPixel(o1, 2);
        else if (color[o1] == color[vSink])
          imOut.setPixel(o1, 3);
        else if (color[o1] == 1)
          imOut.setPixel(o1, 4);
      }

      return RES_OK;
    }

    template <class ImageIn, class ImageMarker, class SE, class ImageOut>
    RES_C t_geoCutsWatershed_Prog_MinCut(const ImageIn &imIn,
                                          const ImageMarker &imMarker,
                                          const SE &nl, ImageOut &imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_geoCutsWatershed_Prog_MinCut");

      std::cout << "Enter function geoCutsWatershed_Prog_MinCut" << std::endl;

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
      typename ImageOut::const_iterator ito, ioend;
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
      Graph_d::vertex_descriptor vSource, vSink;
      int numVert = 0;

      std::cout << "build graph vertices" << std::endl;
      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o0 = it.getOffset();
        boost::add_vertex(g);
        numVert++;
      }

      std::cout << "number of vertices: " << numVert << std::endl;

      vSource = boost::add_vertex(g);
      vSink   = boost::add_vertex(g);

      std::cout << "build graph edges" << std::endl;
      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1         = it.getOffset();
        double val = imIn.pixelFromOffset(o1);
        int marker = imMarker.pixelFromOffset(o1);

        int valright = 0;
        int valleft  = 0;
        int valup    = 0;
        int valdown  = 0;

        if (marker == 2) {
          boost::tie(e4, in1) = boost::add_edge(vSource, o1, g);
          boost::tie(e3, in1) = boost::add_edge(o1, vSource, g);
          capacity[e4]        = (std::numeric_limits<double>::max)();
          capacity[e3]        = (std::numeric_limits<double>::max)();
          rev[e4]             = e3;
          rev[e3]             = e4;
        } else if (marker == 3) {
          boost::tie(e4, in1) = boost::add_edge(o1, vSink, g);
          boost::tie(e3, in1) = boost::add_edge(vSink, o1, g);
          capacity[e4]        = (std::numeric_limits<double>::max)();
          capacity[e3]        = (std::numeric_limits<double>::max)();
          rev[e4]             = e3;
          rev[e3]             = e4;
        }

        neighb.setCenter(o1);

        for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
          const offset_t o2 = nit.getOffset();
          if (o2 <= o1)
            continue;
          if (o2 > o1) {
            double val2         = imIn.pixelFromOffset(o2);
            double valeur       = (255.0 / (std::abs(val - val2) + 1));
            boost::tie(e4, in1) = boost::add_edge(o1, o2, g);
            boost::tie(e3, in1) = boost::add_edge(o2, o1, g);
            capacity[e4]        = valeur;
            capacity[e3]        = valeur;
            rev[e4]             = e3;
            rev[e3]             = e4;
          }
        }
      }

      boost::property_map<Graph_d, boost::vertex_index_t>::type indexmap =
          boost::get(boost::vertex_index, g);
      std::vector<boost::default_color_type> color(boost::num_vertices(g));
      std::vector<boost::graph_traits<Graph_d>::vertex_descriptor> p(
          boost::num_vertices(g));
      std::cout << "Compute Max flow " << std::endl;
#if BOOST_VERSION >= 104700
      double flow =
          boykov_kolmogorov_max_flow(g, capacity, residual_capacity, rev,
                                     &color[0], indexmap, vSource, vSink);
#else
      double flow = kolmogorov_max_flow(g, capacity, residual_capacity, rev,
                                        &color[0], indexmap, vSource, vSink);
#endif
      std::cout << "c  The total flow found :" << std::endl;
      std::cout << "s " << flow << std::endl << std::endl;

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1 = it.getOffset();
        if (color[o1] == color[vSource])
          imOut.setPixel(o1, 0);
        if (color[o1] == color[vSink])
          imOut.setPixel(o1, 2);
      }

      for (int i = 2; i <= 15; i++) {
        for (it = imIn.begin(), iend = imIn.end(); it != iend;
             ++it) // for all pixels in imIn create a vertex and an edge
        {
          o1 = it.getOffset();
          neighb.setCenter(o1);

          for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
            const offset_t o2 = nit.getOffset();
            if (o2 <= o1)
              continue;
            if (o2 > o1) {
              boost::tie(e4, in1) = boost::edge(o1, o2, g);
              boost::tie(e3, in1) = boost::edge(o2, o1, g);
              double valeur       = capacity[e4];
              double cost         = std::pow(valeur, i) - residual_capacity[e4];
              capacity[e4]        = cost;
              capacity[e3]        = cost;
              rev[e4]             = e3;
              rev[e3]             = e4;
            }
          }
        }
        std::cout << "Compute Max flow " << std::endl;
#if BOOST_VERSION >= 104700
        double flow =
            boykov_kolmogorov_max_flow(g, capacity, residual_capacity, rev,
                                       &color[0], indexmap, vSource, vSink);
#else
        double flow = kolmogorov_max_flow(g, capacity, residual_capacity, rev,
                                          &color[0], indexmap, vSource, vSink);
#endif
        std::cout << "c  The total flow found :" << std::endl;
        std::cout << "s " << flow << std::endl << std::endl;

        for (it = imIn.begin(), iend = imIn.end(); it != iend;
             ++it) // for all pixels in imIn create a vertex and an edge
        {
          o1      = it.getOffset();
          int val = imOut.pixelFromOffset(o1);
          if (color[o1] == color[vSource])
            imOut.setPixel(o1, 0);
          if (color[o1] == color[vSink] && val == 0)
            imOut.setPixel(o1, 2 + i - 1);
        }
      }

      return RES_OK;
    }

    template <class ImageIn, class ImageMarker, typename _Power, class SE,
              class ImageOut>
    RES_C
    t_geoCutsWatershed_SPF(const ImageIn &imIn, const ImageMarker &imMarker,
                            const _Power Power, const SE &nl, ImageOut &imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_geoCutsWatershed_SPF");

      std::cout << "Enter function Geo-Cuts Watershed SPF" << std::endl;

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

      double exposant = (double) Power;
      std::cout << "exposant = " << exposant << std::endl;

      // common image iterator
      typename ImageIn::const_iterator it, iend;
      morphee::selement::Neighborhood<SE, ImageIn> neighb(imIn, nl);
      typename morphee::selement::Neighborhood<SE, ImageIn>::iterator nit, nend;
      offset_t o0;
      offset_t o1;

      std::queue<int> temp_q;

      //----------------------------------------------------------------
      // Needed for spanning tree computation
      //----------------------------------------------------------------
      typedef boost::adjacency_list<
          boost::vecS, boost::vecS, boost::undirectedS,
          boost::property<boost::vertex_distance_t, double>,
          boost::property<boost::edge_capacity_t, double>>
          Graph_d;

      Graph_d g;

      boost::property_map<Graph_d, boost::edge_capacity_t>::type weightmap =
          boost::get(boost::edge_capacity, g);

      bool in1;
      Graph_d::edge_descriptor e1, e2, e3, e4;
      Graph_d::vertex_descriptor vRoot;
      int numVert = 0;

      //----------------------------------------------------------------
      //----------------------------------------------------------------

      std::cout << "build graph vertices" << std::endl;

      // double mean1 = 0;
      // double mean2 = 0;
      // double nb1= 0;
      // double nb2 = 0;

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o0 = it.getOffset();
        imOut.setPixel(o0, 0);
        boost::add_vertex(g);
        numVert++;

        // double val = imIn.pixelFromOffset(o0);
        // int marker = imMarker.pixelFromOffset(o0);

        // if (marker==2){
        //	mean1 = mean1+val;
        //	nb1++;
        //}
        // if (marker==3){
        //	mean2 = mean2+val;
        //	nb2++;
        //}
      }

      // mean1 = mean1/(nb1);
      // mean2 = mean2/(nb2);
      // std::cout<<"mean region 1 :"<<mean1<<std::endl;
      //   std::cout<<"mean region 2 :"<<mean2<<std::endl;

      // double mean_considered = std::min(mean1,mean2);

      vRoot = boost::add_vertex(g);

      std::cout << "number of vertices: " << numVert << std::endl;

      std::cout << "build graph edges" << std::endl;

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1         = it.getOffset();
        double val = imIn.pixelFromOffset(o1);
        int marker = imMarker.pixelFromOffset(o1);

        if (marker > 0) {
          boost::tie(e4, in1) = boost::add_edge(vRoot, o1, g);
          weightmap[e4]       = 1.0;
        }

        neighb.setCenter(o1);

        for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
          const offset_t o2 = nit.getOffset();
          if (o2 <= o1)
            continue;
          if (o2 > o1) {
            double val2 = imIn.pixelFromOffset(o2);
            // double cost = std::pow(std::abs((val-val2))+1,exposant);
            double cost = std::pow(std::max(val, val2) + 1, exposant);
            // double cost = 1.0 + exposant*(std::abs(val-val2));
            // double cost = std::abs( std::abs(val-mean_considered) -
            // std::abs(val2-mean_considered) ) + 1;
            boost::tie(e4, in1) = boost::add_edge(o1, o2, g);
            weightmap[e4]       = cost;
          }
        }
      }

      std::cout << "Compute Shortest Path Forest" << std::endl;

      std::vector<boost::graph_traits<Graph_d>::vertex_descriptor> p(
          boost::num_vertices(g));
      boost::property_map<Graph_d, boost::vertex_distance_t>::type distancemap =
          boost::get(boost::vertex_distance, g);
      boost::property_map<Graph_d, boost::vertex_index_t>::type indexmap2 =
          boost::get(boost::vertex_index, g);

      // int D[1][1];

      // std::cout<<"start"<<std::endl;

      // johnson_all_pairs_shortest_paths( g, D, indexmap2, weightmap, 0 );

      // std::cout<<"finished"<<std::endl;

      dijkstra_shortest_paths(g, vRoot, &p[0], distancemap, weightmap,
                              indexmap2, std::less<double>(),
                              boost::closed_plus<double>(),
                              (std::numeric_limits<double>::max)(), 0,
                              boost::default_dijkstra_visitor());

      std::cout << "Backward Nodes Labelling" << std::endl;

      int current_offset = 0;
      int marker         = 0;
      int pixout         = 0;
      int label          = 0;

      // SCAN IMAGE TO FIND THE PIXELS LABELS
      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1 = it.getOffset();

        //	marker = imMarker.pixelFromOffset(o1);

        //	 IMOUT TAKE THE VALUE OF THE MARKER
        //	if(marker==0){
        //		imOut.setPixel(o1,(distancemap[o1]));
        //		imOut.setPixel(o1,-10.0 *
        //std::log(1.0/(1.0+0.00001*distancemap[o1])));
        //	}
        //	else{
        //		imOut.setPixel(o1,0.0);
        //	}
        //}

        // IF PIXELS IS MARKED
        marker = imMarker.pixelFromOffset(o1);

        // IMOUT TAKE THE VALUE OF THE MARKER
        if (marker > 0) {
          imOut.setPixel(o1, marker);
        }

        // CHECK IMOUT VALUE
        pixout = imOut.pixelFromOffset(o1);

        // IF IMOUT HAS NOT A VALUE, SCAN THE PREDECESSOR UNTIL REACHING A
        // MARKER
        if (pixout == 0) {
          // CURRENT POSITION
          current_offset = it.getOffset();
          ;

          // std::cout<<current_offset<<std::endl;

          temp_q.push(current_offset);

          // CHECK THE PREDECESSOR
          marker = imMarker.pixelFromOffset(p[current_offset]);
          pixout = imOut.pixelFromOffset(p[current_offset]);

          // IF BOTH MARKERS AND IMOUT HAS NO VALUES
          while ((marker == 0 && pixout == 0)) {
            // GO BACKWARD
            current_offset = p[current_offset];

            if (p[current_offset] == current_offset) {
              marker = 1;
              pixout = 1;
            } else {
              // PUSH THE PREDECESSOT IN THE QUEUE
              temp_q.push(current_offset);

              // CHECK THE PREDECESSOR
              marker = imMarker.pixelFromOffset(p[current_offset]);
              pixout = imOut.pixelFromOffset(p[current_offset]);
            }
          }

          // WE HAVE REACHED A MARKER
          if (marker > 0 && pixout == 0) {
            label = marker;
            imOut.setPixel(p[current_offset], label);
          } else if (marker == 0 && pixout > 0) {
            label = pixout;
          } else if (marker > 0 && pixout > 0) {
            label = pixout;
          }
          // EMPTY THE QUEUE AND LABEL NODES ALONG THE PATH
          while (temp_q.size() > 0) {
            current_offset = temp_q.front();
            temp_q.pop();
            imOut.setPixel(current_offset, label);
          }
        }
      }

      return RES_OK;
    }

    template <class ImageIn, class ImageMarker, class SE, class ImageOut>
    RES_C t_geoCutsMax_Fiability_Forest(const ImageIn &imIn,
                                         const ImageMarker &imMarker,
                                         const SE &nl, ImageOut &imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_geoCutsMax_Fiability_Forest");

      std::cout << "Enter function t_geoCutsMax_Fiability_Forest" << std::endl;

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

      std::queue<int> temp_q;

      //----------------------------------------------------------------
      // Needed for spanning tree computation
      //----------------------------------------------------------------
      typedef boost::adjacency_list<
          boost::vecS, boost::vecS, boost::undirectedS,
          boost::property<boost::vertex_distance_t, double>,
          boost::property<boost::edge_capacity_t, double>>
          Graph_d;

      Graph_d g;

      boost::property_map<Graph_d, boost::edge_capacity_t>::type weightmap =
          boost::get(boost::edge_capacity, g);

      boost::property_map<Graph_d, boost::edge_capacity_t>::type weightmap2 =
          boost::get(boost::edge_capacity, g);

      bool in1;
      Graph_d::edge_descriptor e1, e2, e3, e4;
      Graph_d::vertex_descriptor vRoot;
      int numVert = 0;

      //----------------------------------------------------------------
      //----------------------------------------------------------------

      std::cout << "build graph vertices" << std::endl;

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o0 = it.getOffset();
        imOut.setPixel(o0, 0);
        boost::add_vertex(g);
        numVert++;
      }

      vRoot = boost::add_vertex(g);

      std::cout << "number of vertices: " << numVert << std::endl;

      std::cout << "build graph edges" << std::endl;

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1         = it.getOffset();
        double val = imIn.pixelFromOffset(o1);
        int marker = imMarker.pixelFromOffset(o1);

        if (marker > 0) {
          boost::tie(e4, in1) = boost::add_edge(vRoot, o1, g);
          weightmap[e4]       = 1.0;
        }

        neighb.setCenter(o1);

        for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
          const offset_t o2 = nit.getOffset();
          if (o2 <= o1)
            continue;
          if (o2 > o1) {
            double val2         = imIn.pixelFromOffset(o2);
            double cost         = 1.0 / (1.0 + 0.1 * std::abs(val - val2));
            boost::tie(e4, in1) = boost::add_edge(o1, o2, g);
            weightmap2[e4]      = cost;
          }
        }
      }

      // for(it=imIn.begin(),iend=imIn.end(); it!=iend ; ++it) // for all pixels
      // in imIn create a vertex and an edge
      //{
      //	o1=it.getOffset();
      //
      //	neighb.setCenter(o1);

      //	double totalcost = 0;

      //	for(nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit)
      //			{
      //				const offset_t o2 = nit.getOffset();
      //				if(o2==o1) continue;
      //				else {
      //					boost::tie(e4, in1) = boost::edge(o1,o2, g);
      //					totalcost = totalcost + weightmap[e4] ;
      //				}
      //			}

      //	for(nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit)
      //			{
      //				const offset_t o2 = nit.getOffset();
      //				if(o2==o1) continue;
      //				else {
      //					boost::tie(e4, in1) = boost::edge(o1,o2, g);
      //					weightmap2[e4] =
      //std::min(weightmap[e4]/totalcost,weightmap2[e4]);
      //				}
      //			}
      //}

      std::cout << "Compute Shortest Path Forest" << std::endl;

      std::vector<boost::graph_traits<Graph_d>::vertex_descriptor> p(
          boost::num_vertices(g));
      boost::property_map<Graph_d, boost::vertex_distance_t>::type distancemap =
          boost::get(boost::vertex_distance, g);
      boost::property_map<Graph_d, boost::vertex_index_t>::type indexmap2 =
          boost::get(boost::vertex_index, g);

      dijkstra_shortest_paths(g, vRoot, &p[0], distancemap, weightmap2,
                              indexmap2, std::greater<double>(),
                              boost::detail::multiplication<double>(), 0.0, 1.0,
                              boost::default_dijkstra_visitor());

      std::cout << "Backward Nodes Labelling" << std::endl;

      int current_offset = 0;
      int marker         = 0;
      int pixout         = 0;
      int label          = 0;

      // SCAN IMAGE TO FIND THE PIXELS LABELS
      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1 = it.getOffset();

        // IF PIXELS IS MARKED
        marker = imMarker.pixelFromOffset(o1);
        // imOut.setPixel(o1,distancemap[o1]);
        //}

        // IMOUT TAKE THE VALUE OF THE MARKER
        if (marker > 0) {
          imOut.setPixel(o1, marker);
        }

        // CHECK IMOUT VALUE
        pixout = imOut.pixelFromOffset(o1);

        // IF IMOUT HAS NOT A VALUE, SCAN THE PREDECESSOR UNTIL REACHING A
        // MARKER
        if (pixout == 0) {
          // CURRENT POSITION
          current_offset = it.getOffset();
          ;

          // std::cout<<current_offset<<std::endl;

          temp_q.push(current_offset);

          // CHECK THE PREDECESSOR
          marker = imMarker.pixelFromOffset(p[current_offset]);
          pixout = imOut.pixelFromOffset(p[current_offset]);

          // IF BOTH MARKERS AND IMOUT HAS NO VALUES
          while ((marker == 0 && pixout == 0)) {
            // GO BACKWARD
            current_offset = p[current_offset];

            if (p[current_offset] == current_offset) {
              marker = 1;
              pixout = 1;
            } else {
              // PUSH THE PREDECESSOT IN THE QUEUE
              temp_q.push(current_offset);

              // CHECK THE PREDECESSOR
              marker = imMarker.pixelFromOffset(p[current_offset]);
              pixout = imOut.pixelFromOffset(p[current_offset]);
            }
          }

          // WE HAVE REACHED A MARKER
          if (marker > 0 && pixout == 0) {
            label = marker;
            imOut.setPixel(p[current_offset], label);
          } else if (marker == 0 && pixout > 0) {
            label = pixout;
          } else if (marker > 0 && pixout > 0) {
            label = pixout;
          }
          // EMPTY THE QUEUE AND LABEL NODES ALONG THE PATH
          while (temp_q.size() > 0) {
            current_offset = temp_q.front();
            temp_q.pop();
            imOut.setPixel(current_offset, label);
          }
        }
      }

      return RES_OK;
    }

    template <class ImageIn, class ImageMarker, class SE, class ImageOut>
    RES_C t_geoCutsBiCriteria_Shortest_Forest(const ImageIn &imIn,
                                               const ImageMarker &imMarker,
                                               const SE &nl, ImageOut &imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_geoCutsBiCriteria_Shortest_Forest");

      std::cout << "Enter function t_geoCutsBiCriteria_Shortest_Forest"
                << std::endl;

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

      std::queue<int> temp_q;

      //----------------------------------------------------------------
      // Needed for spanning tree computation
      //----------------------------------------------------------------
      typedef std::pair<double, double> double2;

      typedef boost::adjacency_list<
          boost::vecS, boost::vecS, boost::undirectedS,
          boost::property<boost::vertex_distance_t, double2>,
          boost::property<boost::edge_capacity_t, double2>>
          Graph_d;

      Graph_d g;

      boost::property_map<Graph_d, boost::edge_capacity_t>::type weightmap =
          boost::get(boost::edge_capacity, g);

      bool in1;
      Graph_d::edge_descriptor e1, e2, e3, e4;
      Graph_d::vertex_descriptor vRoot;
      int numVert = 0;

      //----------------------------------------------------------------
      //----------------------------------------------------------------

      std::cout << "build graph vertices" << std::endl;

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o0 = it.getOffset();
        imOut.setPixel(o0, 0);
        boost::add_vertex(g);
        numVert++;
      }

      vRoot = boost::add_vertex(g);

      std::cout << "number of vertices: " << numVert << std::endl;

      std::cout << "build graph edges" << std::endl;

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1         = it.getOffset();
        double val = imIn.pixelFromOffset(o1);
        int marker = imMarker.pixelFromOffset(o1);

        if (marker > 0) {
          boost::tie(e4, in1) = boost::add_edge(vRoot, o1, g);
          weightmap[e4]       = std::make_pair(1.0, 1.0);
        }

        neighb.setCenter(o1);

        for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
          const offset_t o2 = nit.getOffset();
          if (o2 <= o1)
            continue;
          if (o2 > o1) {
            double val2 = imIn.pixelFromOffset(o2);
            // double cost = std::pow(std::abs((val-val2))+1,1.0);
            double cost         = std::max(val, val2) + 1;
            boost::tie(e4, in1) = boost::add_edge(o1, o2, g);
            weightmap[e4]       = std::make_pair(cost, 1.0);
          }
        }
      }

      std::cout << "Compute Shortest Path Forest" << std::endl;

      std::vector<boost::graph_traits<Graph_d>::vertex_descriptor> p(
          boost::num_vertices(g));
      boost::property_map<Graph_d, boost::vertex_distance_t>::type distancemap =
          boost::get(boost::vertex_distance, g);
      boost::property_map<Graph_d, boost::vertex_index_t>::type indexmap2 =
          boost::get(boost::vertex_index, g);

      dijkstra_shortest_paths(
          g, vRoot, &p[0], distancemap, weightmap, indexmap2,
          boost::detail::lexico_less2<double2>(),
          boost::detail::lexico_addition2<double2>(),
          std::make_pair(std::numeric_limits<double>::max(),
                         std::numeric_limits<double>::max()),
          std::make_pair(0, 0), boost::default_dijkstra_visitor());

      std::cout << "Backward Nodes Labelling" << std::endl;

      int current_offset = 0;
      int marker         = 0;
      int pixout         = 0;
      int label          = 0;

      // SCAN IMAGE TO FIND THE PIXELS LABELS
      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1 = it.getOffset();

        // IF PIXELS IS MARKED
        marker = imMarker.pixelFromOffset(o1);

        // IMOUT TAKE THE VALUE OF THE MARKER
        if (marker > 0) {
          imOut.setPixel(o1, marker);
        }

        // CHECK IMOUT VALUE
        pixout = imOut.pixelFromOffset(o1);

        // IF IMOUT HAS NOT A VALUE, SCAN THE PREDECESSOR UNTIL REACHING A
        // MARKER
        if (pixout == 0) {
          // CURRENT POSITION
          current_offset = it.getOffset();
          ;

          // std::cout<<current_offset<<std::endl;

          temp_q.push(current_offset);

          // CHECK THE PREDECESSOR
          marker = imMarker.pixelFromOffset(p[current_offset]);
          pixout = imOut.pixelFromOffset(p[current_offset]);

          // IF BOTH MARKERS AND IMOUT HAS NO VALUES
          while ((marker == 0 && pixout == 0)) {
            // GO BACKWARD
            current_offset = p[current_offset];

            if (p[current_offset] == current_offset) {
              marker = 1;
              pixout = 1;
            } else {
              // PUSH THE PREDECESSOT IN THE QUEUE
              temp_q.push(current_offset);

              // CHECK THE PREDECESSOR
              marker = imMarker.pixelFromOffset(p[current_offset]);
              pixout = imOut.pixelFromOffset(p[current_offset]);
            }
          }

          // WE HAVE REACHED A MARKER
          if (marker > 0 && pixout == 0) {
            label = marker;
            imOut.setPixel(p[current_offset], label);
          } else if (marker == 0 && pixout > 0) {
            label = pixout;
          } else if (marker > 0 && pixout > 0) {
            label = pixout;
          }
          // EMPTY THE QUEUE AND LABEL NODES ALONG THE PATH
          while (temp_q.size() > 0) {
            current_offset = temp_q.front();
            temp_q.pop();
            imOut.setPixel(current_offset, label);
          }
        }
      }

      return RES_OK;
    }

    template <class ImageIn, class ImageMarker, class SE, class ImageOut>
    RES_C t_geoCutsLexicographical_Shortest_Forest(const ImageIn &imIn,
                                                    const ImageMarker &imMarker,
                                                    const SE &nl,
                                                    ImageOut &imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_geoCutsLexicographical_Shortest_Forest");

      std::cout << "Enter function t_geoCutsLexicographical_Shortest_Forest"
                << std::endl;

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

      std::queue<int> temp_q;

      //----------------------------------------------------------------
      // Needed for spanning tree computation
      //----------------------------------------------------------------
      typedef std::vector<double> vdouble;

      typedef boost::adjacency_list<
          boost::vecS, boost::vecS, boost::undirectedS,
          boost::property<boost::vertex_distance_t, vdouble>,
          boost::property<boost::edge_capacity_t, vdouble>>
          Graph_d;

      Graph_d g;

      boost::property_map<Graph_d, boost::edge_capacity_t>::type weightmap =
          boost::get(boost::edge_capacity, g);

      bool in1;
      Graph_d::edge_descriptor e1, e2, e3, e4;
      Graph_d::vertex_descriptor vRoot;
      int numVert = 0;

      std::vector<double> roots_cost;
      roots_cost.push_back(1.0);

      std::vector<double> edges_cost;
      edges_cost.push_back(1.0);
      //----------------------------------------------------------------
      //----------------------------------------------------------------

      std::cout << "build graph vertices" << std::endl;

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o0 = it.getOffset();
        imOut.setPixel(o0, 0);
        boost::add_vertex(g);
        numVert++;
      }

      vRoot = boost::add_vertex(g);

      std::cout << "number of vertices: " << numVert << std::endl;

      std::cout << "build graph edges" << std::endl;

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1         = it.getOffset();
        double val = imIn.pixelFromOffset(o1);
        int marker = imMarker.pixelFromOffset(o1);

        if (marker > 0) {
          boost::tie(e4, in1) = boost::add_edge(vRoot, o1, g);
          weightmap[e4]       = roots_cost;
        }

        neighb.setCenter(o1);

        for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
          const offset_t o2 = nit.getOffset();
          if (o2 <= o1)
            continue;
          if (o2 > o1) {
            double val2 = imIn.pixelFromOffset(o2);
            // double cost = std::abs(val-val2)+1;
            double cost         = std::max(val, val2) + 1;
            edges_cost[0]       = cost;
            boost::tie(e4, in1) = boost::add_edge(o1, o2, g);
            weightmap[e4]       = edges_cost;
          }
        }
      }

      std::cout << "Compute Shortest Path Forest" << std::endl;

      std::vector<boost::graph_traits<Graph_d>::vertex_descriptor> p(
          boost::num_vertices(g));
      boost::property_map<Graph_d, boost::vertex_distance_t>::type distancemap =
          boost::get(boost::vertex_distance, g);
      boost::property_map<Graph_d, boost::vertex_index_t>::type indexmap2 =
          boost::get(boost::vertex_index, g);

      std::vector<double> infinite;
      infinite.push_back(std::numeric_limits<double>::max());

      std::vector<double> zero;
      zero.push_back(0);

      dijkstra_shortest_paths(
          g, vRoot, &p[0], distancemap, weightmap, indexmap2,
          boost::detail::lexico_compare<vdouble>(),
          boost::detail::lexico_addition<vdouble>(), infinite, zero,
          boost::default_dijkstra_visitor());

      std::cout << "Backward Nodes Labelling" << std::endl;

      int current_offset = 0;
      int marker         = 0;
      int pixout         = 0;
      int label          = 0;

      // SCAN IMAGE TO FIND THE PIXELS LABELS
      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1 = it.getOffset();

        // std::cout<<distancemap[o1]<<std::endl;

        // IF PIXELS IS MARKED
        marker = imMarker.pixelFromOffset(o1);

        // IMOUT TAKE THE VALUE OF THE MARKER
        if (marker > 0) {
          imOut.setPixel(o1, marker);
        }

        // CHECK IMOUT VALUE
        pixout = imOut.pixelFromOffset(o1);

        // IF IMOUT HAS NOT A VALUE, SCAN THE PREDECESSOR UNTIL REACHING A
        // MARKER
        if (pixout == 0) {
          // CURRENT POSITION
          current_offset = it.getOffset();
          ;

          // std::cout<<current_offset<<std::endl;

          temp_q.push(current_offset);

          // CHECK THE PREDECESSOR
          marker = imMarker.pixelFromOffset(p[current_offset]);
          pixout = imOut.pixelFromOffset(p[current_offset]);

          // IF BOTH MARKERS AND IMOUT HAS NO VALUES
          while ((marker == 0 && pixout == 0)) {
            // GO BACKWARD
            current_offset = p[current_offset];

            if (p[current_offset] == current_offset) {
              marker = 1;
              pixout = 1;
            } else {
              // PUSH THE PREDECESSOT IN THE QUEUE
              temp_q.push(current_offset);

              // CHECK THE PREDECESSOR
              marker = imMarker.pixelFromOffset(p[current_offset]);
              pixout = imOut.pixelFromOffset(p[current_offset]);
            }
          }

          // WE HAVE REACHED A MARKER
          if (marker > 0 && pixout == 0) {
            label = marker;
            imOut.setPixel(p[current_offset], label);
          } else if (marker == 0 && pixout > 0) {
            label = pixout;
          } else if (marker > 0 && pixout > 0) {
            label = pixout;
          }
          // EMPTY THE QUEUE AND LABEL NODES ALONG THE PATH
          while (temp_q.size() > 0) {
            current_offset = temp_q.front();
            temp_q.pop();
            imOut.setPixel(current_offset, label);
          }
        }
      }

      return RES_OK;
    }

    template <class ImageIn, class ImageMarker, class SE, class ImageOut>
    RES_C t_geoCutsVectorial_Lexicographical_Shortest_Forest(
        const ImageIn &imIn, const ImageMarker &imMarker, const SE &nl,
        ImageOut &imOut)
    {
      MORPHEE_ENTER_FUNCTION(
          "t_geoCutsVectorial_Lexicographical_Shortest_Forest");

      std::cout << "Enter function "
                   "t_geoCutsVectorial_Lexicographical_Shortest_Forest"
                << std::endl;

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

      std::queue<int> temp_q;

      //----------------------------------------------------------------
      // Needed for spanning tree computation
      //----------------------------------------------------------------

      typedef boost::adjacency_list<
          boost::vecS, boost::vecS, boost::undirectedS,
          boost::property<boost::vertex_distance_t,
                          std::vector<std::vector<double>>>,
          boost::property<boost::edge_capacity_t,
                          std::vector<std::vector<double>>>>
          Graph_d;

      Graph_d g;

      boost::property_map<Graph_d, boost::edge_capacity_t>::type weightmap =
          boost::get(boost::edge_capacity, g);

      bool in1;
      Graph_d::edge_descriptor e1, e2, e3, e4;
      Graph_d::vertex_descriptor vRoot;
      int numVert = 0;

      //----------------------------------------------------------------
      //----------------------------------------------------------------

      pixel_3<F_DOUBLE> pxImageIn;
      pixel_3<F_DOUBLE> pxImageNeigh;

      std::vector<std::vector<double>> root;

      std::vector<double> roots_cost;
      roots_cost.push_back(0.1);
      roots_cost.push_back(0.1);
      roots_cost.push_back(0.1);

      root.push_back(roots_cost);

      std::vector<std::vector<double>> test;
      test.push_back(roots_cost);
      test.push_back(roots_cost);
      int sizea = test.size();

      std::cout << "taille des test   " << sizea << std::endl;

      for (int i = 0; i < 2; i++) {
        std::vector<double> temp = test[i];
        std::cout << temp[0] << "	" << temp[1] << "	" << temp[2] << std::endl;
      }

      std::cout << "build graph vertices" << std::endl;

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o0 = it.getOffset();
        imOut.setPixel(o0, 0);
        boost::add_vertex(g);
        numVert++;
      }

      vRoot = boost::add_vertex(g);

      std::cout << "number of vertices: " << numVert << std::endl;

      std::cout << "build graph edges" << std::endl;

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1        = it.getOffset();
        pxImageIn = imIn.pixelFromOffset(o1);

        double valeur1 = pxImageIn.channel1;
        double valeur2 = pxImageIn.channel2;
        double valeur3 = pxImageIn.channel3;

        int marker = imMarker.pixelFromOffset(o1);

        if (marker > 0) {
          boost::tie(e4, in1) = boost::add_edge(vRoot, o1, g);
          weightmap[e4]       = root;
        }

        neighb.setCenter(o1);

        for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
          const offset_t o2 = nit.getOffset();
          if (o2 <= o1)
            continue;
          if (o2 > o1) {
            pxImageNeigh = imIn.pixelFromOffset(o2);
            std::vector<std::vector<double>> cost;
            std::vector<double> ordered_cost_vector;

            double valeurn1 = std::abs(pxImageNeigh.channel1 - valeur1) + 1;
            double valeurn2 = std::abs(pxImageNeigh.channel2 - valeur2) + 1;
            double valeurn3 = std::abs(pxImageNeigh.channel3 - valeur3) + 1;

            // double maxim =  std::max(std::max(valeurn1,valeurn2),valeurn3);

            ordered_cost_vector.push_back(valeurn1);
            ordered_cost_vector.push_back(valeurn2);
            ordered_cost_vector.push_back(valeurn3);

            // double maxim =  std::max(std::max(valeurn1,valeurn2),valeurn3);
            // double minim =  std::min(std::min(valeurn1,valeurn2),valeurn3);

            // if(maxim >= valeurn1 && minim <= valeurn1){
            //	ordered_cost_vector.push_back(maxim);
            //	ordered_cost_vector.push_back(valeurn1);
            //	ordered_cost_vector.push_back(minim);
            //}
            // else if(maxim >= valeurn2 && minim <= valeurn2){
            //	ordered_cost_vector.push_back(maxim);
            //	ordered_cost_vector.push_back(valeurn2);
            //	ordered_cost_vector.push_back(minim);
            //}
            // else if(maxim >= valeurn3 && minim <= valeurn3){
            //	ordered_cost_vector.push_back(maxim);
            //	ordered_cost_vector.push_back(valeurn3);
            //	ordered_cost_vector.push_back(minim);
            //}

            cost.push_back(ordered_cost_vector);
            boost::tie(e4, in1) = boost::add_edge(o1, o2, g);
            weightmap[e4]       = cost;
          }
        }
      }

      std::cout << "Compute Minimum Spanning Forest" << std::endl;

      std::vector<boost::graph_traits<Graph_d>::vertex_descriptor> p(
          boost::num_vertices(g));
      boost::property_map<Graph_d, boost::vertex_distance_t>::type distancemap =
          boost::get(boost::vertex_distance, g);
      boost::property_map<Graph_d, boost::vertex_index_t>::type indexmap2 =
          boost::get(boost::vertex_index, g);

      std::vector<std::vector<double>> inff;
      std::vector<double> infinite;
      infinite.push_back(std::numeric_limits<double>::max());
      infinite.push_back(std::numeric_limits<double>::max());
      infinite.push_back(std::numeric_limits<double>::max());
      inff.push_back(infinite);

      std::vector<std::vector<double>> Zeroo;
      std::vector<double> zero;
      zero.push_back(0.0);
      zero.push_back(0.0);
      zero.push_back(0.0);
      Zeroo.push_back(zero);

      dijkstra_shortest_paths(g, vRoot, &p[0], distancemap, weightmap,
                              indexmap2,
                              boost::detail::lexico_compare_vect_of_vect<
                                  std::vector<std::vector<double>>>(),
                              boost::detail::lexico_addition_vect_of_vect<
                                  std::vector<std::vector<double>>>(),
                              inff, Zeroo, boost::default_dijkstra_visitor());
      std::cout << "Backward Nodes Labelling" << std::endl;

      int current_offset = 0;
      int marker         = 0;
      int pixout         = 0;
      int label          = 0;

      // SCAN IMAGE TO FIND THE PIXELS LABELS
      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1 = it.getOffset();

        // IF PIXELS IS MARKED
        marker = imMarker.pixelFromOffset(o1);

        // IMOUT TAKE THE VALUE OF THE MARKER
        if (marker > 0) {
          imOut.setPixel(o1, marker);
        }

        // CHECK IMOUT VALUE
        pixout = imOut.pixelFromOffset(o1);

        // IF IMOUT HAS NOT A VALUE, SCAN THE PREDECESSOR UNTIL REACHING A
        // MARKER
        if (pixout == 0) {
          // CURRENT POSITION
          current_offset = it.getOffset();
          ;

          // std::cout<<current_offset<<std::endl;

          temp_q.push(current_offset);

          // CHECK THE PREDECESSOR
          marker = imMarker.pixelFromOffset(p[current_offset]);
          pixout = imOut.pixelFromOffset(p[current_offset]);

          // IF BOTH MARKERS AND IMOUT HAS NO VALUES
          while ((marker == 0 && pixout == 0)) {
            // GO BACKWARD
            current_offset = p[current_offset];

            if (p[current_offset] == current_offset) {
              marker = 1;
              pixout = 1;
            } else {
              // PUSH THE PREDECESSOT IN THE QUEUE
              temp_q.push(current_offset);

              // CHECK THE PREDECESSOR
              marker = imMarker.pixelFromOffset(p[current_offset]);
              pixout = imOut.pixelFromOffset(p[current_offset]);
            }
          }

          // WE HAVE REACHED A MARKER
          if (marker > 0 && pixout == 0) {
            label = marker;
            imOut.setPixel(p[current_offset], label);
          } else if (marker == 0 && pixout > 0) {
            label = pixout;
          } else if (marker > 0 && pixout > 0) {
            label = pixout;
          }
          // EMPTY THE QUEUE AND LABEL NODES ALONG THE PATH
          while (temp_q.size() > 0) {
            current_offset = temp_q.front();
            temp_q.pop();
            imOut.setPixel(current_offset, label);
          }
        }
      }

      return RES_OK;
    }

    template <class ImageIn, class ImageMarker, class SE, class ImageOut>
    RES_C t_geoCutsVectorial_Shortest_Forest(const ImageIn &imIn,
                                              const ImageMarker &imMarker,
                                              const SE &nl, ImageOut &imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_geoCutsVectorial_Shortest_Forest");

      std::cout << "Enter function t_geoCutsVectorial_Shortest_Forest"
                << std::endl;

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

      std::queue<int> temp_q;

      //----------------------------------------------------------------
      // Needed for spanning tree computation
      //----------------------------------------------------------------
      typedef std::vector<double> vdouble;

      typedef boost::adjacency_list<
          boost::vecS, boost::vecS, boost::undirectedS,
          boost::property<boost::vertex_distance_t, vdouble>,
          boost::property<boost::edge_capacity_t, vdouble>>
          Graph_d;

      Graph_d g;

      boost::property_map<Graph_d, boost::edge_capacity_t>::type weightmap =
          boost::get(boost::edge_capacity, g);

      bool in1;
      Graph_d::edge_descriptor e1, e2, e3, e4;
      Graph_d::vertex_descriptor vRoot;
      int numVert = 0;

      //----------------------------------------------------------------
      //----------------------------------------------------------------

      pixel_3<F_DOUBLE> pxImageIn;
      pixel_3<F_DOUBLE> pxImageNeigh;

      std::vector<double> roots_cost;
      roots_cost.push_back(0.1);
      roots_cost.push_back(0.1);
      roots_cost.push_back(0.1);

      std::cout << "build graph vertices" << std::endl;

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o0 = it.getOffset();
        imOut.setPixel(o0, 0);
        boost::add_vertex(g);
        numVert++;
      }

      vRoot = boost::add_vertex(g);

      std::cout << "number of vertices: " << numVert << std::endl;

      std::cout << "build graph edges" << std::endl;

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1        = it.getOffset();
        pxImageIn = imIn.pixelFromOffset(o1);

        double valeur1 = pxImageIn.channel1;
        double valeur2 = pxImageIn.channel2;
        double valeur3 = pxImageIn.channel3;

        int marker = imMarker.pixelFromOffset(o1);

        if (marker > 0) {
          boost::tie(e4, in1) = boost::add_edge(vRoot, o1, g);
          weightmap[e4]       = roots_cost;
        }

        neighb.setCenter(o1);

        for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
          const offset_t o2 = nit.getOffset();
          if (o2 <= o1)
            continue;
          if (o2 > o1) {
            pxImageNeigh = imIn.pixelFromOffset(o2);
            std::vector<double> ordered_cost_vector;
            double valeurn1 = std::abs(pxImageNeigh.channel1 - valeur1) + 1;
            double valeurn2 = std::abs(pxImageNeigh.channel2 - valeur2) + 1;
            double valeurn3 = std::abs(pxImageNeigh.channel3 - valeur3) + 1;

            // double maxim =  std::max(std::max(valeurn1,valeurn2),valeurn3);

            ordered_cost_vector.push_back(valeurn1);
            ordered_cost_vector.push_back(valeurn2);
            ordered_cost_vector.push_back(valeurn3);

            // double maxim =  std::max(std::max(valeurn1,valeurn2),valeurn3);
            // double minim =  std::min(std::min(valeurn1,valeurn2),valeurn3);
            //
            // if(maxim >= valeurn1 && minim <= valeurn1){
            //	ordered_cost_vector.push_back(maxim);
            //	ordered_cost_vector.push_back(valeurn1);
            //	ordered_cost_vector.push_back(minim);
            //}
            // else if(maxim >= valeurn2 && minim <= valeurn2){
            //	ordered_cost_vector.push_back(maxim);
            //	ordered_cost_vector.push_back(valeurn2);
            //	ordered_cost_vector.push_back(minim);
            //}
            // else if(maxim >= valeurn3 && minim <= valeurn3){
            //	ordered_cost_vector.push_back(maxim);
            //	ordered_cost_vector.push_back(valeurn3);
            //	ordered_cost_vector.push_back(minim);
            //}

            boost::tie(e4, in1) = boost::add_edge(o1, o2, g);
            weightmap[e4]       = ordered_cost_vector;
          }
        }
      }

      std::cout << "Compute Minimum Spanning Forest" << std::endl;

      std::vector<boost::graph_traits<Graph_d>::vertex_descriptor> p(
          boost::num_vertices(g));
      boost::property_map<Graph_d, boost::vertex_distance_t>::type distancemap =
          boost::get(boost::vertex_distance, g);
      boost::property_map<Graph_d, boost::vertex_index_t>::type indexmap2 =
          boost::get(boost::vertex_index, g);

      std::vector<double> infinite;
      infinite.push_back(std::numeric_limits<double>::max());
      infinite.push_back(std::numeric_limits<double>::max());
      infinite.push_back(std::numeric_limits<double>::max());

      std::vector<double> zero;
      zero.push_back(0.0);
      zero.push_back(0.0);
      zero.push_back(0.0);

      dijkstra_shortest_paths(
          g, vRoot, &p[0], distancemap, weightmap, indexmap2,
          boost::detail::lexico_compare3<vdouble>(),
          boost::detail::lexico_addition3<vdouble>(), infinite, zero,
          boost::default_dijkstra_visitor());
      std::cout << "Backward Nodes Labelling" << std::endl;

      int current_offset = 0;
      int marker         = 0;
      int pixout         = 0;
      int label          = 0;

      // SCAN IMAGE TO FIND THE PIXELS LABELS
      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1 = it.getOffset();

        // IF PIXELS IS MARKED
        marker = imMarker.pixelFromOffset(o1);

        // IMOUT TAKE THE VALUE OF THE MARKER
        if (marker > 0) {
          imOut.setPixel(o1, marker);
        }

        // CHECK IMOUT VALUE
        pixout = imOut.pixelFromOffset(o1);

        // IF IMOUT HAS NOT A VALUE, SCAN THE PREDECESSOR UNTIL REACHING A
        // MARKER
        if (pixout == 0) {
          // CURRENT POSITION
          current_offset = it.getOffset();
          ;

          // std::cout<<current_offset<<std::endl;

          temp_q.push(current_offset);

          // CHECK THE PREDECESSOR
          marker = imMarker.pixelFromOffset(p[current_offset]);
          pixout = imOut.pixelFromOffset(p[current_offset]);

          // IF BOTH MARKERS AND IMOUT HAS NO VALUES
          while ((marker == 0 && pixout == 0)) {
            // GO BACKWARD
            current_offset = p[current_offset];

            if (p[current_offset] == current_offset) {
              marker = 1;
              pixout = 1;
            } else {
              // PUSH THE PREDECESSOT IN THE QUEUE
              temp_q.push(current_offset);

              // CHECK THE PREDECESSOR
              marker = imMarker.pixelFromOffset(p[current_offset]);
              pixout = imOut.pixelFromOffset(p[current_offset]);
            }
          }

          // WE HAVE REACHED A MARKER
          if (marker > 0 && pixout == 0) {
            label = marker;
            imOut.setPixel(p[current_offset], label);
          } else if (marker == 0 && pixout > 0) {
            label = pixout;
          } else if (marker > 0 && pixout > 0) {
            label = pixout;
          }
          // EMPTY THE QUEUE AND LABEL NODES ALONG THE PATH
          while (temp_q.size() > 0) {
            current_offset = temp_q.front();
            temp_q.pop();
            imOut.setPixel(current_offset, label);
          }
        }
      }

      return RES_OK;
    }

    template <class ImageIn, class ImageMarker, class SE, class ImageOut>
    RES_C t_geoCutsWatershed_SpanningForest(const ImageIn &imIn,
                                             const ImageMarker &imMarker,
                                             const SE &nl, ImageOut &imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_geoCutsWatershed_SpanningForest");

      std::cout << "Enter function t_geoCutsWatershed_SpanningForest"
                << std::endl;

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

      std::queue<int> temp_q;

      //----------------------------------------------------------------
      // Needed for spanning tree computation
      //----------------------------------------------------------------
      typedef boost::adjacency_list<
          boost::vecS, boost::vecS, boost::undirectedS,
          boost::property<boost::vertex_distance_t, double>,
          boost::property<boost::edge_capacity_t, double>>
          Graph_d;

      Graph_d g;

      boost::property_map<Graph_d, boost::edge_capacity_t>::type weightmap =
          boost::get(boost::edge_capacity, g);

      bool in1;
      Graph_d::edge_descriptor e1, e2, e3, e4;
      Graph_d::vertex_descriptor vRoot;
      int numVert = 0;

      //----------------------------------------------------------------
      //----------------------------------------------------------------

      std::cout << "build graph vertices" << std::endl;

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o0 = it.getOffset();
        imOut.setPixel(o0, 0);
        boost::add_vertex(g);
        numVert++;
      }

      vRoot = boost::add_vertex(g);

      std::cout << "number of vertices: " << numVert << std::endl;

      std::cout << "build graph edges" << std::endl;

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1         = it.getOffset();
        double val = imIn.pixelFromOffset(o1);
        int marker = imMarker.pixelFromOffset(o1);

        if (marker > 0) {
          boost::tie(e4, in1) = boost::add_edge(vRoot, o1, g);
          weightmap[e4]       = 0.1;
        }

        neighb.setCenter(o1);

        for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
          const offset_t o2 = nit.getOffset();
          if (o2 <= o1)
            continue;
          if (o2 > o1) {
            double val2 = imIn.pixelFromOffset(o2);
            double cost = std::abs(val - val2) + 1;
            // double cost = std::sqrt(std::abs(val-val2)+1);
            boost::tie(e4, in1) = boost::add_edge(o1, o2, g);
            weightmap[e4]       = cost;
          }
        }
      }

      std::cout << "Compute Minimum Spanning Forest" << std::endl;

      std::vector<boost::graph_traits<Graph_d>::vertex_descriptor> p(
          boost::num_vertices(g));
      boost::property_map<Graph_d, boost::vertex_distance_t>::type distancemap =
          boost::get(boost::vertex_distance, g);
      boost::property_map<Graph_d, boost::vertex_index_t>::type indexmap2 =
          boost::get(boost::vertex_index, g);

      dijkstra_shortest_paths(g, vRoot, &p[0], distancemap, weightmap,
                              indexmap2, std::less<double>(),
                              boost::detail::maximum<double>(),
                              (std::numeric_limits<double>::max)(), 0,
                              boost::default_dijkstra_visitor());

      // prim_minimum_spanning_tree(g, vRoot, &p[0], distancemap, weightmap,
      // indexmap2, boost::default_dijkstra_visitor());

      std::cout << "Backward Nodes Labelling" << std::endl;

      int current_offset = 0;
      int marker         = 0;
      int pixout         = 0;
      int label          = 0;

      // SCAN IMAGE TO FIND THE PIXELS LABELS
      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1 = it.getOffset();

        // IF PIXELS IS MARKED
        marker = imMarker.pixelFromOffset(o1);

        // IMOUT TAKE THE VALUE OF THE MARKER
        if (marker > 0) {
          imOut.setPixel(o1, marker);
        }

        // CHECK IMOUT VALUE
        pixout = imOut.pixelFromOffset(o1);

        // IF IMOUT HAS NOT A VALUE, SCAN THE PREDECESSOR UNTIL REACHING A
        // MARKER
        if (pixout == 0) {
          // CURRENT POSITION
          current_offset = it.getOffset();
          ;

          // std::cout<<current_offset<<std::endl;

          temp_q.push(current_offset);

          // CHECK THE PREDECESSOR
          marker = imMarker.pixelFromOffset(p[current_offset]);
          pixout = imOut.pixelFromOffset(p[current_offset]);

          // IF BOTH MARKERS AND IMOUT HAS NO VALUES
          while ((marker == 0 && pixout == 0)) {
            // GO BACKWARD
            current_offset = p[current_offset];

            if (p[current_offset] == current_offset) {
              marker = 1;
              pixout = 1;
            } else {
              // PUSH THE PREDECESSOT IN THE QUEUE
              temp_q.push(current_offset);

              // CHECK THE PREDECESSOR
              marker = imMarker.pixelFromOffset(p[current_offset]);
              pixout = imOut.pixelFromOffset(p[current_offset]);
            }
          }

          // WE HAVE REACHED A MARKER
          if (marker > 0 && pixout == 0) {
            label = marker;
            imOut.setPixel(p[current_offset], label);
          } else if (marker == 0 && pixout > 0) {
            label = pixout;
          } else if (marker > 0 && pixout > 0) {
            label = pixout;
          }
          // EMPTY THE QUEUE AND LABEL NODES ALONG THE PATH
          while (temp_q.size() > 0) {
            current_offset = temp_q.front();
            temp_q.pop();
            imOut.setPixel(current_offset, label);
          }
        }
      }

      return RES_OK;
    }

    template <class ImageIn, class ImageMarker, class SE, class ImageOut>
    RES_C t_geoCutsWatershed_SpanningForest_v2(const ImageIn &imIn,
                                                const ImageMarker &imMarker,
                                                const SE &nl, ImageOut &imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_geoCutsWatershed_SpanningForest_v2");

      std::cout << "Enter function t_geoCutsWatershed_SpanningForest_v2"
                << std::endl;

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

      SE nl2 = morphee::selement::neighborsCross2D;
      morphee::selement::Neighborhood<SE, ImageIn> neighb2(imIn, nl2);
      typename morphee::selement::Neighborhood<SE, ImageIn>::iterator nit2,
          nend2;

      morphee::selement::Neighborhood<SE, ImageIn> neighb(imIn, nl);
      typename morphee::selement::Neighborhood<SE, ImageIn>::iterator nit, nend;

      typename ImageIn::const_iterator it, iend;

      offset_t o0;
      offset_t o1;

      std::queue<int> temp_q;

      //----------------------------------------------------------------
      // Needed for spanning tree computation
      //----------------------------------------------------------------
      typedef boost::adjacency_list<
          boost::vecS, boost::vecS, boost::undirectedS,
          boost::property<boost::vertex_distance_t, double>,
          boost::property<boost::edge_capacity_t, double>>
          Graph_d;

      Graph_d g;

      boost::property_map<Graph_d, boost::edge_capacity_t>::type weightmap =
          boost::get(boost::edge_capacity, g);

      bool in1;
      Graph_d::edge_descriptor e1, e2, e3, e4;
      Graph_d::vertex_descriptor vRoot;
      int numVert = 0;

      //----------------------------------------------------------------
      //----------------------------------------------------------------
      typedef typename s_from_type_to_type<ImageIn, F_SIMPLE>::image_type
          dist_image_type;
      dist_image_type work = imIn.template t_getSame<float>();
      RES_C res            = t_ImSetConstant(work, 0.0);
      //----------------------------------------------------------------
      //----------------------------------------------------------------

      std::cout << "build graph vertices" << std::endl;

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o0 = it.getOffset();
        work.setPixel(o0, 0.0);
        imOut.setPixel(o0, 0);
        boost::add_vertex(g);
        numVert++;
      }

      vRoot = boost::add_vertex(g);

      std::cout << "number of vertices: " << numVert << std::endl;

      //----------------------------------------------------------------
      //------------ COMPUTE LOWER SLOPE AT EACH PIXEL
      //----------------------------

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o0         = it.getOffset();
        double val = imIn.pixelFromOffset(o0);

        neighb2.setCenter(o0);
        double tempval  = val;
        double tempval2 = val;
        double val2     = 0;

        for (nit2 = neighb2.begin(), nend2 = neighb2.end(); nit2 != nend2;
             ++nit2) {
          o1   = nit2.getOffset();
          val2 = imIn.pixelFromOffset(o1);

          if (val2 < tempval) {
            tempval = val2;
          }

          // if(val2>tempval){
          //	tempval = val2;
          //}
          // else if(val2<tempval2){
          //	tempval2 = val2;
          //}
        }

        work.setPixel(o0, (float) val - tempval);
        // work.setPixel(o0,(float) tempval-tempval2);

        // work.setPixel(o0,0.0);
      }

      std::cout << "build graph edges" << std::endl;

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1 = it.getOffset();

        double val = imIn.pixelFromOffset(o1);
        double ls1 = work.pixelFromOffset(o1);

        int marker = imMarker.pixelFromOffset(o1);

        if (marker > 0) {
          boost::tie(e4, in1) = boost::add_edge(vRoot, o1, g);
          weightmap[e4]       = 0;
        }

        neighb.setCenter(o1);

        for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
          const offset_t o2 = nit.getOffset();

          if (o2 <= o1)
            continue;
          if (o2 > o1) {
            double val2 = imIn.pixelFromOffset(o2);
            double ls2  = work.pixelFromOffset(o2);
            double cost = 0;

            if (val > val2) {
              cost = val2 + ls1;
            } else if (val2 > val) {
              cost = val + ls2;
            } else if (val2 == val) {
              cost = val + std::max(ls2, ls1);
            }

            // cost = std::max(val,val2) + std::max(ls2,ls1)/2;
            // cost = std::max(val,val2);

            boost::tie(e4, in1) = boost::add_edge(o1, o2, g);
            weightmap[e4]       = cost + 1;
          }
        }
      }

      std::cout << "Compute Minimum Spanning Forest" << std::endl;

      std::vector<boost::graph_traits<Graph_d>::vertex_descriptor> p(
          boost::num_vertices(g));
      boost::property_map<Graph_d, boost::vertex_distance_t>::type distancemap =
          boost::get(boost::vertex_distance, g);
      boost::property_map<Graph_d, boost::vertex_index_t>::type indexmap2 =
          boost::get(boost::vertex_index, g);

      dijkstra_shortest_paths(g, vRoot, &p[0], distancemap, weightmap,
                              indexmap2, std::less<double>(),
                              boost::detail::maximum<double>(),
                              (std::numeric_limits<double>::max)(), 0,
                              boost::default_dijkstra_visitor());

      // dijkstra_shortest_paths(g, vRoot, &p[0], distancemap, weightmap,
      // indexmap2, std::less<double>(), boost::closed_plus<double>(),
      //	(std::numeric_limits<double>::max)(), 0,
      //boost::default_dijkstra_visitor());

      // prim_minimum_spanning_tree(g, vRoot, &p[0], distancemap, weightmap,
      // indexmap2, boost::default_dijkstra_visitor());

      std::cout << "Backward Nodes Labelling" << std::endl;

      int current_offset = 0;
      int marker         = 0;
      int pixout         = 0;
      int label          = 0;

      // SCAN IMAGE TO FIND THE PIXELS LABELS
      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1 = it.getOffset();
        //	imOut.setPixel(o1,distancemap[o1]);

        //}

        // IF PIXELS IS MARKED
        marker = imMarker.pixelFromOffset(o1);

        // IMOUT TAKE THE VALUE OF THE MARKER
        if (marker > 0) {
          imOut.setPixel(o1, marker);
        }

        // CHECK IMOUT VALUE
        pixout = imOut.pixelFromOffset(o1);

        // IF IMOUT HAS NOT A VALUE, SCAN THE PREDECESSOR UNTIL REACHING A
        // MARKER
        if (pixout == 0) {
          // CURRENT POSITION
          current_offset = it.getOffset();
          ;

          // std::cout<<current_offset<<std::endl;

          temp_q.push(current_offset);

          // CHECK THE PREDECESSOR
          marker = imMarker.pixelFromOffset(p[current_offset]);
          pixout = imOut.pixelFromOffset(p[current_offset]);

          // IF BOTH MARKERS AND IMOUT HAS NO VALUES
          while ((marker == 0 && pixout == 0)) {
            // GO BACKWARD
            current_offset = p[current_offset];

            if (p[current_offset] == current_offset) {
              marker = 1;
              pixout = 1;
            } else {
              // PUSH THE PREDECESSOT IN THE QUEUE
              temp_q.push(current_offset);

              // CHECK THE PREDECESSOR
              marker = imMarker.pixelFromOffset(p[current_offset]);
              pixout = imOut.pixelFromOffset(p[current_offset]);
            }
          }

          // WE HAVE REACHED A MARKER
          if (marker > 0 && pixout == 0) {
            label = marker;
            imOut.setPixel(p[current_offset], label);
          } else if (marker == 0 && pixout > 0) {
            label = pixout;
          } else if (marker > 0 && pixout > 0) {
            label = pixout;
          }
          // EMPTY THE QUEUE AND LABEL NODES ALONG THE PATH
          while (temp_q.size() > 0) {
            current_offset = temp_q.front();
            temp_q.pop();
            imOut.setPixel(current_offset, label);
          }
        }
      }

      return RES_OK;
    }

    template <class ImageIn, class ImageMarker, class SE, class ImageOut>
    RES_C t_geoCutsReg_SpanningForest(const ImageIn &imIn,
                                       const ImageMarker &imMarker,
                                       const SE &nl, ImageOut &imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_geoCutsReg_SpanningForest");

      std::cout << "Enter function geoCutsSpanningForest" << std::endl;

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

      std::queue<int> temp_q;

      //----------------------------------------------------------------
      // needed for spanning tree
      //----------------------------------------------------------------
      typedef boost::adjacency_list<
          boost::vecS, boost::vecS, boost::undirectedS,
          boost::property<boost::vertex_distance_t, double>,
          boost::property<boost::edge_capacity_t, double>>
          Graph_d;

      Graph_d g;

      boost::property_map<Graph_d, boost::edge_capacity_t>::type weightmap =
          boost::get(boost::edge_capacity, g);

      bool in1, in2;
      Graph_d::edge_descriptor e1, e2, e3, e4;
      Graph_d::vertex_descriptor vRoot;
      int numVert = 0;

      //----------------------------------------------------------------
      // needed for max flow: capacit map, rev_capacity map, etc.
      //----------------------------------------------------------------
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
          Graph_flow;

      Graph_flow g_flow;

      boost::property_map<Graph_flow, boost::edge_capacity_t>::type capacity =
          boost::get(boost::edge_capacity, g_flow);

      boost::property_map<Graph_flow, boost::edge_reverse_t>::type rev =
          get(boost::edge_reverse, g_flow);

      boost::property_map<Graph_flow, boost::edge_residual_capacity_t>::type
          residual_capacity = get(boost::edge_residual_capacity, g_flow);

      Graph_flow::edge_descriptor e11, e22;
      Graph_flow::vertex_descriptor vSource, vSink;

      //----------------------------------------------------------------
      //----------------------------------------------------------------

      std::cout << "build graph vertices" << std::endl;

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o0 = it.getOffset();
        imOut.setPixel(o0, 0);
        boost::add_vertex(g);

        //-------------For max_flow-------------
        boost::add_vertex(g_flow);
        //------------------------------------------

        numVert++;
      }

      vRoot = boost::add_vertex(g);

      //-------------For max_flow-------------
      vSource = boost::add_vertex(g_flow);
      vSink   = boost::add_vertex(g_flow);
      //------------------------------------------

      std::cout << "number of vertices: " << numVert << std::endl;

      std::cout << "build graph edges" << std::endl;

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1         = it.getOffset();
        double val = imIn.pixelFromOffset(o1);
        int marker = imMarker.pixelFromOffset(o1);

        if (marker > 0) {
          boost::tie(e4, in1) = boost::add_edge(vRoot, o1, g);
          weightmap[e4]       = 0.5;

          if (marker == 2) {
            boost::tie(e11, in1) = boost::add_edge(vSource, o1, g_flow);
            boost::tie(e22, in2) = boost::add_edge(o1, vSource, g_flow);
            capacity[e11]        = 1000000000;
            capacity[e22]        = 1000000000;
            rev[e11]             = e22;
            rev[e22]             = e11;
          }
          if (marker == 3) {
            boost::tie(e11, in1) = boost::add_edge(vSink, o1, g_flow);
            boost::tie(e22, in2) = boost::add_edge(o1, vSink, g_flow);
            capacity[e11]        = 1000000000;
            capacity[e22]        = 1000000000;
            rev[e11]             = e22;
            rev[e22]             = e11;
          }
        }

        neighb.setCenter(o1);

        for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
          const offset_t o2 = nit.getOffset();
          if (o2 <= o1)
            continue;
          if (o2 > o1) {
            double val2         = imIn.pixelFromOffset(o2);
            double cost         = std::abs((val - val2)) + 1;
            boost::tie(e4, in1) = boost::add_edge(o1, o2, g);
            weightmap[e4]       = cost;

            boost::tie(e11, in1) = boost::add_edge(o1, o2, g_flow);
            boost::tie(e22, in2) = boost::add_edge(o2, o1, g_flow);
            capacity[e11]        = 256 / (cost);
            capacity[e22]        = 256 / (cost);
            rev[e11]             = e22;
            rev[e22]             = e11;
          }
        }
      }

      std::cout << "Compute Minimum Spanning Forest" << std::endl;

      std::vector<boost::graph_traits<Graph_d>::vertex_descriptor> p(
          boost::num_vertices(g));
      boost::property_map<Graph_d, boost::vertex_distance_t>::type distancemap =
          boost::get(boost::vertex_distance, g);
      boost::property_map<Graph_d, boost::vertex_index_t>::type indexmap2 =
          boost::get(boost::vertex_index, g);

      prim_minimum_spanning_tree(g, vRoot, &p[0], distancemap, weightmap,
                                 indexmap2, boost::default_dijkstra_visitor());

      std::cout << "Backward Nodes Labelling" << std::endl;

      int current_offset = 0;
      int first_offset   = 0;
      int marker         = 0;
      int pixout         = 0;
      int label          = 0;

      // SCAN IMAGE TO FIND THE PIXELS LABELS
      for (int i = 0; i < numVert; i++) {
        // IF PIXELS IS MARKED
        marker = imMarker.pixelFromOffset(i);

        // IMOUT TAKE THE VALUE OF THE MARKER
        if (marker > 0) {
          imOut.setPixel(i, marker);
        }

        // CHECK IMOUT VALUE
        pixout = imOut.pixelFromOffset(i);

        // IF IMOUT HAS NOT A VALUE, SCAN THE PREDECESSOR UNTIL REACHING A
        // MARKER
        if (pixout == 0) {
          // CURRENT POSITION
          current_offset = i;
          temp_q.push(current_offset);

          // CHECK THE PREDECESSOR
          marker = imMarker.pixelFromOffset(p[current_offset]);
          pixout = imOut.pixelFromOffset(p[current_offset]);

          // IF BOTH MARKERS AND IMOUT HAS NO VALUES
          while ((marker == 0 && pixout == 0)) {
            // GO BACKWARD
            current_offset = p[current_offset];

            // PUSH THE PREDECESSOT IN THE QUEUE
            temp_q.push(current_offset);

            // CHECK THE PREDECESSOR
            marker = imMarker.pixelFromOffset(p[current_offset]);
            pixout = imOut.pixelFromOffset(p[current_offset]);
          }

          temp_q.push(p[current_offset]);

          // WE HAVE REACHED A MARKER
          if (marker > 0 && pixout == 0) {
            label = marker;
            imOut.setPixel(p[current_offset], label);
          } else if (marker == 0 && pixout > 0) {
            label = pixout;
          } else if (marker > 0 && pixout > 0) {
            label = pixout;
          }
          // EMPTY THE QUEUE AND LABEL NODES ALONG THE PATH

          while (temp_q.size() > 0) {
            first_offset = temp_q.front();
            temp_q.pop();
            imOut.setPixel(first_offset, label);

            if (p[first_offset] != vRoot) {
              current_offset = p[first_offset];
              imOut.setPixel(current_offset, label);

              boost::tie(e11, in1) =
                  boost::edge(first_offset, current_offset, g_flow);
              boost::tie(e22, in2) =
                  boost::edge(current_offset, first_offset, g_flow);

              if (in1 && in2) {
                capacity[e22] = capacity[e22];
                capacity[e11] = capacity[e11];
              }
            }
          }
        }
      }

      int pix1 = 0;
      int pix2 = 0;

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1   = it.getOffset();
        pix1 = imOut.pixelFromOffset(o1);
        neighb.setCenter(o1);

        for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
          const offset_t o2 = nit.getOffset();
          pix2              = imOut.pixelFromOffset(o2);

          if (o2 <= o1)
            continue;
          if (o2 > o1) {
            if (pix2 != pix1) {
              boost::tie(e11, in1) = boost::edge(o1, o2, g_flow);
              boost::tie(e22, in1) = boost::edge(o2, o1, g_flow);

              capacity[e11] = 3.2;
              capacity[e22] = 3.2;
            }
          }
        }
      }

      boost::property_map<Graph_flow, boost::vertex_index_t>::const_type
          indexmap = boost::get(boost::vertex_index, g_flow);
      std::vector<boost::default_color_type> color(boost::num_vertices(g_flow));

      std::cout << "Compute Max flow" << std::endl;
#if BOOST_VERSION >= 104700
      double flow =
          boykov_kolmogorov_max_flow(g_flow, capacity, residual_capacity, rev,
                                     &color[0], indexmap, vSource, vSink);
#else
      double flow =
          kolmogorov_max_flow(g_flow, capacity, residual_capacity, rev,
                              &color[0], indexmap, vSource, vSink);
#endif
      std::cout << "c  The total flow:" << std::endl;
      std::cout << "s " << flow << std::endl << std::endl;

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1 = it.getOffset();
        if (color[o1] == 0)
          imOut.setPixel(o1, 2);
        if (color[o1] == 1)
          imOut.setPixel(o1, 4);
        if (color[o1] == 4)
          imOut.setPixel(o1, 3);
      }

      return RES_OK;
    }

    template <class ImageIn, class ImageMarker, class SE, class ImageOut>
    RES_C t_geoCutsMinSurfaces_With_Line(const ImageIn &imIn,
                                          const ImageMarker &imMarker,
                                          const SE &nl, ImageOut &imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_geoCutsMinSurfaces_With_Line");

      std::cout << "Enter function Geo-Cuts" << std::endl;

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

      std::cout << "build graph vertices" << std::endl;
      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o0 = it.getOffset();
        boost::add_vertex(g);
        numVert++;
      }

      std::cout << "number of vertices: " << numVert << std::endl;

      vSource = boost::add_vertex(g);
      vSink   = boost::add_vertex(g);

      std::cout << "build graph edges" << std::endl;
      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1         = it.getOffset();
        double val = imIn.pixelFromOffset(o1);
        int marker = imMarker.pixelFromOffset(o1);

        int valright = 0;
        int valleft  = 0;
        int valup    = 0;
        int valdown  = 0;

        if (marker == 2) {
          boost::tie(e4, in1) = boost::add_edge(vSource, o1, g);
          boost::tie(e3, in1) = boost::add_edge(o1, vSource, g);
          capacity[e4]        = (std::numeric_limits<double>::max)();
          capacity[e3]        = (std::numeric_limits<double>::max)();
          rev[e4]             = e3;
          rev[e3]             = e4;
        } else if (marker == 3) {
          boost::tie(e4, in1) = boost::add_edge(o1, vSink, g);
          boost::tie(e3, in1) = boost::add_edge(vSink, o1, g);
          capacity[e4]        = (std::numeric_limits<double>::max)();
          capacity[e3]        = (std::numeric_limits<double>::max)();
          rev[e4]             = e3;
          rev[e3]             = e4;
        }

        neighb.setCenter(o1);

        for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
          const offset_t o2 = nit.getOffset();
          if (o2 <= o1)
            continue;
          if (o2 > o1) {
            double val2 = imIn.pixelFromOffset(o2);
            double cost = 10000 / (1 + 1.5 * (val - val2) * (val - val2));
            // double cost = std::exp(-((val-val2)*(val-val2))/(1.0));
            boost::tie(e4, in1) = boost::add_edge(o1, o2, g);
            boost::tie(e3, in1) = boost::add_edge(o2, o1, g);
            capacity[e4]        = cost + 1;
            capacity[e3]        = cost + 1;
            rev[e4]             = e3;
            rev[e3]             = e4;
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
        o1 = it.getOffset();
        if (color[o1] == 0)
          imOut.setPixel(o1, 2);
        if (color[o1] == 1)
          imOut.setPixel(o1, 4);
        if (color[o1] == 4)
          imOut.setPixel(o1, 3);
      }

      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1         = it.getOffset();
        double val = imOut.pixelFromOffset(o1);
        neighb.setCenter(o1);

        for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
          const offset_t o2 = nit.getOffset();
          double val2       = imOut.pixelFromOffset(o2);

          if (val2 != val && val != 0 && val2 != 0) {
            imOut.setPixel(o2, 0);
            imOut.setPixel(o1, 0);
          }
        }
      }

      return RES_OK;
    }

    template <class ImageIn, class ImageMarker, class SE, class ImageOut>
    RES_C t_geoCutsMultiway_MinSurfaces(const ImageIn &imIn,
                                         const ImageMarker &imMarker,
                                         const SE &nl, ImageOut &imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_geoCutsMultiway_MinSurfaces");

      std::cout << "Enter function Multi way Geo-Cuts" << std::endl;

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

      std::cout << "build graph vertices" << std::endl;
      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o0       = it.getOffset();
        int val2 = imMarker.pixelFromOffset(o0);

        imOut.setPixel(o0, 1);

        if (val2 > numLabels) {
          numLabels = val2;
        }

        boost::add_vertex(g);
        numVert++;
      }

      std::cout << "number of Labels: " << numLabels << std::endl;

      std::cout << "number of vertices: " << numVert << std::endl;

      vSource = boost::add_vertex(g);
      vSink   = boost::add_vertex(g);

      std::cout << "build graph edges" << std::endl;
      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1         = it.getOffset();
        double val = imIn.pixelFromOffset(o1);

        neighb.setCenter(o1);

        for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
          const offset_t o2 = nit.getOffset();
          if (o2 <= o1)
            continue;
          if (o2 > o1) {
            double val2 = imIn.pixelFromOffset(o2);
            double cost = 10000 / (1 + 1.5 * (val - val2) * (val - val2));
            // std::cout<<cost<<std::endl;

            // double cost = std::exp(-((val-val2)*(val-val2))/(1.0));
            boost::tie(e4, in1) = boost::add_edge(o1, o2, g);
            boost::tie(e3, in1) = boost::add_edge(o2, o1, g);
            capacity[e4]        = cost + 1;
            capacity[e3]        = cost + 1;
            rev[e4]             = e3;
            rev[e3]             = e4;
          }
        }
      }

      for (int nbk = 2; nbk <= numLabels; nbk++) {
        for (it = imIn.begin(), iend = imIn.end(); it != iend;
             ++it) // for all pixels in imIn create a vertex
        {
          o0      = it.getOffset();
          int val = imMarker.pixelFromOffset(o0);

          if (val == nbk) {
            boost::tie(e4, in1) = boost::edge(vSource, o0, g);
            if (in1 == 0) {
              boost::tie(e4, in1) = boost::add_edge(vSource, o0, g);
              boost::tie(e3, in1) = boost::add_edge(o0, vSource, g);
              capacity[e4]        = (std::numeric_limits<double>::max)();
              capacity[e3]        = (std::numeric_limits<double>::max)();
              rev[e4]             = e3;
              rev[e3]             = e4;
            }
          } else if (val > 1 && val != nbk) {
            boost::tie(e4, in1) = boost::edge(o0, vSink, g);
            if (in1 == 0) {
              boost::tie(e4, in1) = boost::add_edge(o0, vSink, g);
              boost::tie(e3, in1) = boost::add_edge(vSink, o0, g);
              capacity[e4]        = (std::numeric_limits<double>::max)();
              capacity[e3]        = (std::numeric_limits<double>::max)();
              rev[e4]             = e3;
              rev[e3]             = e4;
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
          o1       = it.getOffset();
          int val  = imIn.pixelFromOffset(o1);
          int val2 = imOut.pixelFromOffset(o1);
          int val3 = imMarker.pixelFromOffset(o1);

          if (val2 == 1) {
            if (color[o1] == color[vSource])
              imOut.setPixel(o1, nbk);
          }

          if (val3 == nbk) {
            boost::tie(e4, in1) = boost::edge(vSource, o1, g);
            if (in1 == 1) {
              boost::remove_edge(vSource, o1, g);
              boost::remove_edge(o1, vSource, g);
            }
          } else if (val3 > 1 && val3 != nbk) {
            boost::tie(e4, in1) = boost::edge(o1, vSink, g);
            if (in1 == 1) {
              boost::remove_edge(o1, vSink, g);
              boost::remove_edge(vSink, o1, g);
            }
          }
        }
      }

      return RES_OK;
    }

    template <class ImageIn, class ImageMarker, typename _Power, class SE,
              class ImageOut>
    RES_C t_geoCutsMultiway_Watershed(const ImageIn &imIn,
                                       const ImageMarker &imMarker,
                                       const _Power Power, const SE &nl,
                                       ImageOut &imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_geoCutsMultiway_Watershed");

      std::cout << "Enter function Multi way watershed" << std::endl;

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

      double exposant = (double) Power;
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

      std::cout << "build graph vertices" << std::endl;
      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o0       = it.getOffset();
        int val2 = imMarker.pixelFromOffset(o0);

        imOut.setPixel(o0, 1);

        if (val2 > numLabels) {
          numLabels = val2;
        }

        boost::add_vertex(g);
        numVert++;
      }

      std::cout << "number of Labels: " << numLabels << std::endl;

      std::cout << "number of vertices: " << numVert << std::endl;

      vSource = boost::add_vertex(g);
      vSink   = boost::add_vertex(g);

      std::cout << "build graph edges" << std::endl;
      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1         = it.getOffset();
        double val = imIn.pixelFromOffset(o1);

        neighb.setCenter(o1);

        for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
          const offset_t o2 = nit.getOffset();
          if (o2 <= o1)
            continue;
          if (o2 > o1) {
            double val2   = imIn.pixelFromOffset(o2);
            double valeur = (255.0 / (std::abs(val - val2) + 1));
            double cost   = std::pow(valeur, exposant);

            // double cost = std::exp(-((val-val2)*(val-val2))/(1.0));
            boost::tie(e4, in1) = boost::add_edge(o1, o2, g);
            boost::tie(e3, in1) = boost::add_edge(o2, o1, g);
            capacity[e4]        = cost;
            capacity[e3]        = cost;
            rev[e4]             = e3;
            rev[e3]             = e4;
          }
        }
      }

      for (int nbk = 2; nbk <= numLabels; nbk++) {
        for (it = imIn.begin(), iend = imIn.end(); it != iend;
             ++it) // for all pixels in imIn create a vertex
        {
          o0      = it.getOffset();
          int val = imMarker.pixelFromOffset(o0);

          double cost = (std::numeric_limits<double>::max)();

          if (val == nbk) {
            boost::tie(e4, in1) = boost::edge(vSource, o0, g);
            if (in1 == 0) {
              boost::tie(e4, in1) = boost::add_edge(vSource, o0, g);
              boost::tie(e3, in1) = boost::add_edge(o0, vSource, g);
              capacity[e4]        = cost;
              capacity[e3]        = cost;
              rev[e4]             = e3;
              rev[e3]             = e4;
            }
          } else if (val > 1 && val != nbk) {
            boost::tie(e4, in1) = boost::edge(o0, vSink, g);
            if (in1 == 0) {
              boost::tie(e4, in1) = boost::add_edge(o0, vSink, g);
              boost::tie(e3, in1) = boost::add_edge(vSink, o0, g);
              capacity[e4]        = cost;
              capacity[e3]        = cost;
              rev[e4]             = e3;
              rev[e3]             = e4;
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
          o1       = it.getOffset();
          int val  = imIn.pixelFromOffset(o1);
          int val2 = imOut.pixelFromOffset(o1);
          int val3 = imMarker.pixelFromOffset(o1);

          if (val2 == 1) {
            if (color[o1] == color[vSource])
              imOut.setPixel(o1, nbk);
          }

          if (val3 == nbk) {
            boost::tie(e4, in1) = boost::edge(vSource, o1, g);
            if (in1 == 1) {
              boost::remove_edge(vSource, o1, g);
              boost::remove_edge(o1, vSource, g);
            }
          } else if (val3 > 1 && val3 != nbk) {
            boost::tie(e4, in1) = boost::edge(o1, vSink, g);
            if (in1 == 1) {
              boost::remove_edge(o1, vSink, g);
              boost::remove_edge(vSink, o1, g);
            }
          }
        }
      }

      return RES_OK;
    }

    template <class ImageIn, class ImageMarker, typename _Beta, typename _Sigma,
              class SE, class ImageOut>
    RES_C t_MAP_MRF_Ising(const ImageIn &imIn, const ImageMarker &imMarker,
                          const _Beta Beta, const _Sigma Sigma, const SE &nl,
                          ImageOut &imOut)
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
      Graph_d::vertex_descriptor vSource, vSink;
      int numVert = 0;

      double moy_foreground = 0;
      double moy_background = 0;
      double nb_foreground  = 0;
      double nb_background  = 0;

      double mean_image = 0;
      double nb_pixels  = 0;

      std::cout << "build graph vertices" << std::endl;
      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o0            = it.getOffset();
        int valmarker = imMarker.pixelFromOffset(o0);
        int val       = imIn.pixelFromOffset(o0);
        boost::add_vertex(g);
        numVert++;

        if (valmarker == 2) {
          moy_foreground = moy_foreground + val;
          nb_foreground++;
        } else if (valmarker == 3) {
          moy_background = moy_background + val;
          nb_background++;
        }

        mean_image = mean_image + val;
        nb_pixels++;
      }

      mean_image     = mean_image / nb_pixels;
      moy_foreground = moy_foreground / nb_foreground;
      moy_background = moy_background / nb_background;
      moy_foreground = moy_foreground / 255.0;
      moy_background = moy_background / 255.0;

      std::cout << "number of vertices: " << numVert << std::endl;
      std::cout << "Foreground Mean: " << moy_foreground << std::endl;
      std::cout << "Background Mean: " << moy_background << std::endl;

      vSource = boost::add_vertex(g);
      vSink   = boost::add_vertex(g);

      std::cout << "build graph edges" << std::endl;
      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1 = it.getOffset();

        double val1   = imIn.pixelFromOffset(o1);
        int valmarker = imMarker.pixelFromOffset(o1);

        neighb.setCenter(o1);

        for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
          const offset_t o2 = nit.getOffset();

          if (o2 <= o1)
            continue;
          if (o2 > o1) {
            double cost         = (double) Beta;
            boost::tie(e4, in1) = boost::add_edge(o1, o2, g);
            boost::tie(e3, in1) = boost::add_edge(o2, o1, g);
            capacity[e4]        = cost;
            capacity[e3]        = cost;
            rev[e4]             = e3;
            rev[e3]             = e4;
          }
        }

        if (valmarker == 2) {
          boost::tie(e4, in1) = boost::add_edge(o1, vSink, g);
          boost::tie(e3, in1) = boost::add_edge(vSink, o1, g);
          capacity[e4]        = (std::numeric_limits<double>::max)();
          capacity[e3]        = (std::numeric_limits<double>::max)();
          rev[e4]             = e3;
          rev[e3]             = e4;
        } else if (valmarker == 3) {
          boost::tie(e4, in1) = boost::add_edge(vSource, o1, g);
          boost::tie(e3, in1) = boost::add_edge(o1, vSource, g);
          capacity[e4]        = (std::numeric_limits<double>::max)();
          capacity[e3]        = (std::numeric_limits<double>::max)();
          rev[e4]             = e3;
          rev[e3]             = e4;
        } else {
          val1 = val1 / 255.0;

          double sigma = (double) Sigma;

          double sigmab = 0.2;

          double slink = (val1 - moy_foreground) * (val1 - moy_foreground) /
                         (2 * sigma * sigma);
          double tlink = (val1 - moy_background) * (val1 - moy_background) /
                         (2 * sigmab * sigmab);

          boost::tie(e4, in1) = boost::add_edge(vSource, o1, g);
          boost::tie(e3, in1) = boost::add_edge(o1, vSource, g);
          capacity[e4]        = slink;
          capacity[e3]        = slink;
          rev[e4]             = e3;
          rev[e3]             = e4;

          boost::tie(e4, in1) = boost::add_edge(o1, vSink, g);
          boost::tie(e3, in1) = boost::add_edge(vSink, o1, g);
          capacity[e4]        = tlink;
          capacity[e3]        = tlink;
          rev[e4]             = e3;
          rev[e3]             = e4;
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
        o1 = it.getOffset();
        if (color[o1] == color[vSource])
          imOut.setPixel(o1, 3);
        else if (color[o1] == color[vSink])
          imOut.setPixel(o1, 2);
        else if (color[o1] == 1)
          imOut.setPixel(o1, 4);
      }

      return RES_OK;
    }

    template <class ImageIn, class ImageMarker, typename _Beta, typename _Sigma,
              class SE, class ImageOut>
    RES_C t_MAP_MRF_edge_preserving(const ImageIn &imIn,
                                    const ImageMarker &imMarker,
                                    const _Beta Beta, const _Sigma Sigma,
                                    const SE &nl, ImageOut &imOut)
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
      int numVert = 0;

      double moy_foreground = 0;
      double moy_background = 0;
      double nb_foreground  = 0;
      double nb_background  = 0;

      double mean_image = 0;
      double nb_pixels  = 0;

      double mean_difference = 0;
      double nb_difference   = 0;

      std::cout << "build graph vertices" << std::endl;
      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex
      {
        o0            = it.getOffset();
        int valmarker = imMarker.pixelFromOffset(o0);
        int val       = imIn.pixelFromOffset(o0);
        boost::add_vertex(g);
        numVert++;

        if (valmarker == 2) {
          moy_foreground = moy_foreground + val;
          nb_foreground++;
        } else if (valmarker == 3) {
          moy_background = moy_background + val;
          nb_background++;
        }

        mean_image = mean_image + val;
        nb_pixels++;

        neighb.setCenter(o0);
        double vall1 = (double) val / 255.0;

        for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
          const offset_t o2 = nit.getOffset();
          double val2       = imIn.pixelFromOffset(o2);

          if (o2 <= o0)
            continue;
          if (o2 > o0) {
            double vall2 = (double) val2 / 255.0;
            mean_difference =
                mean_difference + (vall2 - vall1) * (vall2 - vall1);
            nb_difference++;
          }
        }
      }

      mean_difference = mean_difference / nb_difference;
      mean_image      = mean_image / nb_pixels;

      moy_foreground = moy_foreground / nb_foreground;
      moy_background = moy_background / nb_background;
      moy_foreground = moy_foreground / 255.0;
      moy_background = moy_background / 255.0;

      std::cout << "number of vertices: " << numVert << std::endl;
      std::cout << "Foreground Mean: " << moy_foreground << std::endl;
      std::cout << "Background Mean: " << moy_background << std::endl;
      std::cout << "Mean difference : " << mean_difference << std::endl;

      vSource = boost::add_vertex(g);
      vSink   = boost::add_vertex(g);

      std::cout << "build graph edges" << std::endl;
      for (it = imIn.begin(), iend = imIn.end(); it != iend;
           ++it) // for all pixels in imIn create a vertex and an edge
      {
        o1            = it.getOffset();
        double val1   = imIn.pixelFromOffset(o1);
        int valmarker = imMarker.pixelFromOffset(o1);
        val1          = val1 / 255;

        neighb.setCenter(o1);

        for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
          const offset_t o2 = nit.getOffset();
          double val2       = imIn.pixelFromOffset(o2);
          val2              = val2 / 255;

          if (o2 <= o1)
            continue;
          if (o2 > o1) {
            // double cost =   (double) Beta *
            // std::exp(-((val1-val2)*(val1-val2))/(2*mean_difference)) ;
            double cost =
                (double) Beta -
                ((double) Beta) * std::pow(std::abs(val1 - val2), 0.25);
            boost::tie(e4, in1) = boost::add_edge(o1, o2, g);
            boost::tie(e3, in1) = boost::add_edge(o2, o1, g);
            capacity[e4]        = cost;
            capacity[e3]        = cost;
            rev[e4]             = e3;
            rev[e3]             = e4;
          }
        }

        if (valmarker == 2) {
          boost::tie(e4, in1) = boost::add_edge(o1, vSink, g);
          boost::tie(e3, in1) = boost::add_edge(vSink, o1, g);
          capacity[e4]        = (std::numeric_limits<double>::max)();
          capacity[e3]        = (std::numeric_limits<double>::max)();
          rev[e4]             = e3;
          rev[e3]             = e4;
        } else if (valmarker == 3) {
          boost::tie(e4, in1) = boost::add_edge(vSource, o1, g);
          boost::tie(e3, in1) = boost::add_edge(o1, vSource, g);
          capacity[e4]        = (std::numeric_limits<double>::max)();
          capacity[e3]        = (std::numeric_limits<double>::max)();
          rev[e4]             = e3;
          rev[e3]             = e4;
        } else {
          double sigma  = Sigma;
          double sigmab = 0.2;

          double val_f = (val1 - moy_foreground) * (val1 - moy_foreground) /
                         (2 * sigma * sigma);
          double val_b = (val1 - moy_background) * (val1 - moy_background) /
                         (2 * sigmab * sigmab);

          boost::tie(e4, in1) = boost::add_edge(vSource, o1, g);
          boost::tie(e3, in1) = boost::add_edge(o1, vSource, g);
          capacity[e4]        = val_f;
          capacity[e3]        = val_f;
          rev[e4]             = e3;
          rev[e3]             = e4;

          boost::tie(e4, in1) = boost::add_edge(o1, vSink, g);
          boost::tie(e3, in1) = boost::add_edge(vSink, o1, g);
          capacity[e4]        = val_b;
          capacity[e3]        = val_b;
          rev[e4]             = e3;
          rev[e3]             = e4;
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
        o1 = it.getOffset();
        if (color[o1] == color[vSource])
          imOut.setPixel(o1, 3);
        else if (color[o1] == color[vSink])
          imOut.setPixel(o1, 2);
        else if (color[o1] == 1)
          imOut.setPixel(o1, 4);
      }

      return RES_OK;
    }

    // template<class ImageIn,class ImageMarker, class SE, class ImageOut >
    //	RES_C t_MAP_MRF_Potts(const ImageIn& imIn, const ImageMarker& imMarker,
    //const SE& nl, ImageOut &imOut)
    //{
    //	MORPHEE_ENTER_FUNCTION("t_MAP_MRF_Potts");
    //	std::cout << "Enter function t_MAP_MRF_Potts" << std::endl;

    //	if( (!imOut.isAllocated()) )
    //	{
    //		MORPHEE_REGISTER_ERROR("Not allocated");
    //		return RES_NOT_ALLOCATED;
    //	}

    //	if( (!imIn.isAllocated()) )
    //	{
    //		MORPHEE_REGISTER_ERROR("Not allocated");
    //		return RES_NOT_ALLOCATED;
    //	}

    //	if( (!imMarker.isAllocated()) )
    //	{
    //		MORPHEE_REGISTER_ERROR("Not allocated");
    //		return RES_NOT_ALLOCATED;
    //	}

    //	//common image iterator
    //	typename ImageIn::const_iterator it, iend;
    //	morphee::selement::Neighborhood<SE, ImageIn >						neighb(imIn,nl);
    //	typename morphee::selement::Neighborhood<SE, ImageIn >::iterator
    //nit,nend; 	offset_t o0; 	offset_t o1;

    //	//needed for max flow: capacit map, rev_capacity map, etc.
    //	typedef
    //boost::adjacency_list_traits<boost::vecS,boost::vecS,boost::directedS>
    //Traits; 	typedef
    //boost::adjacency_list<boost::listS,boost::vecS,boost::directedS,
    //		boost::property<boost::vertex_name_t, std::string>,
    //		boost::property<boost::edge_capacity_t, double,
    //		boost::property<boost::edge_residual_capacity_t, double,
    //		boost::property<boost::edge_reverse_t, Traits::edge_descriptor> > > >
    //Graph_d;

    //	Graph_d::edge_descriptor e1,e2,e3,e4;
    //	Graph_d::vertex_descriptor vSource,vSink;
    //	int nb_combi = 3;
    //	bool in1, in2;
    //	double sigma[3][2];
    //	sigma[0][0] = 0.5 ;
    //	sigma[0][1] = 0.5 ;

    //	sigma[1][0] = 0.5 ;
    //	sigma[1][1] = 0.5 ;

    //	sigma[2][0] = 0.5 ;
    //	sigma[2][1] = 0.5 ;

    //	double combi_valeur[3][2];
    //	combi_valeur[0][0] = 1.0 ;
    //	combi_valeur[0][1] = 0.75 ;

    //	combi_valeur[1][0] = 0.75 ;
    //	combi_valeur[1][1] = 0.5 ;

    //	combi_valeur[2][0] = 0.5 ;
    //	combi_valeur[2][1] = 0.0 ;

    //	double combi_label[3][2];
    //	combi_label[0][0] = 4 ;
    //	combi_label[0][1] = 3 ;

    //	combi_label[1][0] = 3 ;
    //	combi_label[1][1] = 2 ;

    //	combi_label[2][0] = 2 ;
    //	combi_label[2][1] = 1 ;

    //	for(int nbk=0;nbk<nb_combi;nbk++){

    //		Graph_d g;

    //		boost::property_map<Graph_d, boost::edge_capacity_t>::type
    //			capacity = boost::get(boost::edge_capacity, g);

    //		boost::property_map<Graph_d, boost::edge_reverse_t>::type
    //			rev = get(boost::edge_reverse, g);

    //		boost::property_map<Graph_d, boost::edge_residual_capacity_t>::type
    //			residual_capacity = get(boost::edge_residual_capacity, g);

    //		std::cout<<"build graph vertices"<<std::endl;
    //		int numVert = 0;

    //		for(it=imIn.begin(),iend=imIn.end();it!=iend ; ++it) // for all pixels
    //in imIn create a vertex
    //		{
    //			o0=it.getOffset();
    //			int val = imIn.pixelFromOffset(o0);
    //			boost::add_vertex(g);
    //			numVert++;
    //			if(nbk==0){
    //				imOut.setPixel(o0,0);
    //			}
    //		}

    //		vSource = boost::add_vertex(g);
    //		vSink = boost::add_vertex(g);

    //		std::cout<<"build graph edges"<<std::endl;
    //		boost::property_map<Graph_d, boost::vertex_index_t>::type indexmap =
    //boost::get(boost::vertex_index, g); 		std::vector<boost::default_color_type>
    //color(boost::num_vertices(g)); 		std::cout<<"build neighborhood edges and
    //terminal links"<<std::endl; 		for(it=imIn.begin(),iend=imIn.end(); it!=iend
    //; ++it) // for all pixels in imIn create a vertex and an edge
    //		{
    //			o1=it.getOffset();
    //			neighb.setCenter(o1);
    //			double valCenter = imOut.pixelFromOffset(o1);
    //			double val = 0;
    //			double nbval = 0;

    //			for(nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit)
    //			{
    //				const offset_t o2 = nit.getOffset();
    //				double val2 = imIn.pixelFromOffset(o2);
    //				val = val + val2;
    //				nbval++;
    //				if(o2<=o1) continue;
    //				if(o2>o1 && (valCenter == 0 || valCenter == combi_label[nbk][1] ||
    //valCenter == combi_label[nbk][0]) ){ 					double cost =   0.01; 					boost::tie(e4,
    //in1) = boost::add_edge(o1,o2, g); 					boost::tie(e3, in1) =
    //boost::add_edge(o2,o1, g); 					capacity[e4] = cost; 					capacity[e3] = cost;
    //					rev[e4] = e3;
    //					rev[e3] = e4;
    //				}
    //			}

    //			if(valCenter == 0 || valCenter == combi_label[nbk][1] || valCenter
    //== combi_label[nbk][0]){ 				val = val/nbval; 				double valee = val/255; 				double
    //val1 =    (valee - combi_valeur[nbk][0])*(valee -
    //combi_valeur[nbk][0])/(2*sigma[nbk][0]*sigma[nbk][0]); 				double val2 =
    //(valee - combi_valeur[nbk][1])*(valee -
    //combi_valeur[nbk][1])/(2*sigma[nbk][1]*sigma[nbk][1]);
    //
    //				boost::tie(e4, in1) = boost::add_edge(vSource,o1,g);
    //				boost::tie(e3, in1) = boost::add_edge(o1,vSource,g);
    //				capacity[e4] = val1;
    //				capacity[e3] = val1;
    //				rev[e4] = e3;
    //				rev[e3] = e4;

    //				boost::tie(e4, in1) = boost::add_edge(vSink,o1,g);
    //				boost::tie(e3, in1) = boost::add_edge(o1,vSink,g);
    //				capacity[e4] = val2;
    //				capacity[e3] = val2;
    //				rev[e4] = e3;
    //				rev[e3] = e4;
    //			}
    //		}

    //		std::cout << "Compute Max flow" << std::endl;
    //		double flow = kolmogorov_max_flow(g, capacity, residual_capacity, rev,
    //&color[0], indexmap, vSource, vSink); 		std::cout << "c  The total flow:" <<
    //std::endl; 		std::cout << "s " << flow << std::endl << std::endl;

    //		std::cout << "Source Label:" <<color[vSource]<< std::endl;
    //		std::cout << "Sink  Label:" <<color[vSink]<< std::endl;

    //		for (unsigned int i=0;i<numVert;i++)
    //		{
    //			int valimout =  imOut.pixelFromOffset(i);

    //			if( valimout == 0 || valimout == combi_label[nbk][1] || valimout ==
    //combi_label[nbk][0]){ 				if(color[i]==0)
    //					imOut.setPixel(i,combi_label[nbk][1]);
    //				if(color[i]==4)
    //					imOut.setPixel(i,combi_label[nbk][0]);
    //			}
    //		}

    //	}

    //	return RES_OK;
    //}

    template <class ImageIn, class ImageMarker, typename _Beta, typename _Sigma,
              class SE, class ImageOut>
    RES_C t_MAP_MRF_Potts(const ImageIn &imIn, const ImageMarker &imMarker,
                          const _Beta Beta, const _Sigma Sigma, const SE &nl,
                          ImageOut &imOut)
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
            val3 = (val1 - 0.5) * (val1 - 0.5) /
                   (2 * sigma[nbk][1] * sigma[nbk][1]);
            val4 = (val1) * (val1) / (2 * sigma[nbk][1] * sigma[nbk][1]);
          } else if (nbk == 1) {
            val2 = (val1 - 1.0) * (val1 - 1.0) /
                   (2 * sigma[nbk][1] * sigma[nbk][1]);
            val3 = (val1 - 0.5) * (val1 - 0.5) /
                   (2 * sigma[nbk][1] * sigma[nbk][1]);
            val4 = (val1) * (val1) / (2 * sigma[nbk][1] * sigma[nbk][1]);
          } else if (nbk == 2) {
            val2 = (val1 - 1.0) * (val1 - 1.0) /
                   (2 * sigma[nbk][1] * sigma[nbk][1]);
            val3 = (val1 - 0.75) * (val1 - 0.75) /
                   (2 * sigma[nbk][1] * sigma[nbk][1]);
            val4 = (val1) * (val1) / (2 * sigma[nbk][1] * sigma[nbk][1]);
          } else if (nbk == 3) {
            val2 = (val1 - 1.0) * (val1 - 1.0) /
                   (2 * sigma[nbk][1] * sigma[nbk][1]);
            val3 = (val1 - 0.75) * (val1 - 0.75) /
                   (2 * sigma[nbk][1] * sigma[nbk][1]);
            val4 = (val1 - 0.5) * (val1 - 0.5) /
                   (2 * sigma[nbk][1] * sigma[nbk][1]);
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

  } // namespace graphalgo
} // namespace morphee

#endif // GRAPHALGO_IMPL_T_HPP
