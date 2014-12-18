#ifndef __MAGIC_WAND_T_HPP__
#define __MAGIC_WAND_T_HPP__

#include "Morpho/include/private/DMorphoGraph.hpp"
#include <math.h>

using namespace std;

namespace smil
{

  template <class TMark, class TSeg>
  RES_T magicWandSeg(const Image<TMark> &imMarkers, const Image<TSeg> &imSeg, Graph<TMark, TSeg> &graph)// UINT -> TSeg
    {
        ASSERT(CHECK_ALLOCATED(&imMarkers, &imSeg));
        ASSERT(CHECK_SAME_SIZE(&imMarkers, &imSeg));
        
        typedef typename Image<TMark>::lineType lineTypeMark;
        typedef typename Image<TSeg>::lineType lineTypeSeg;
        lineTypeMark pixMark = imMarkers.getPixels();
        lineTypeSeg pixSeg = imSeg.getPixels();


      // --------------------------------------------------
      // --------------------------------------------------
      // Set all the nodes to zero
      // --------------------------------------------------
      // --------------------------------------------------
        std::map<size_t, size_t> nodeValues;
        //        nodeValues=graph.nodeValues;        
        //        for( std::map<size_t, size_t>::iterator it=nodeValues.begin();it!=nodeValues.end(); it++ ){
        //          it.second=0; // first est l'index, second la valeur
        //        }
        for (int i = 0; i < nodeValues.size(); i ++){
          nodeValues[i]=0;
        }


      // --------------------------------------------------
      // --------------------------------------------------
      // PROJECT IMAGE TO GRAPH 
      // --------------------------------------------------
      // --------------------------------------------------

        for (size_t i=0;i<imMarkers.getPixelCount();i++)
          if (pixMark[i]!=0)
            nodeValues[pixSeg[i]] = 1;

      // --------------------------------------------------
      // --------------------------------------------------
      // COMPUTE HIERARCHICAL LEVEL REQUIRED
      // --------------------------------------------------
      // --------------------------------------------------
        typedef typename Graph<TMark,TSeg>::EdgeWeightType EdgeWeightType;
        typedef typename Graph<TMark,TSeg>::EdgeType EdgeType;

        const vector< EdgeType > &edges = graph.getEdges();
        EdgeWeightType min_edge_val = edges.size();
        EdgeWeightType  edge_val;

        //        for (vector<EdgeType>::iterator e_it=edges.begin();e_it!=edges.end();e_it++)
        for (int i =0; i < edges.size();i++)
          {
            const EdgeType &e = edges[i];
            if(nodeValues[e.source] && nodeValues[e.target] ){
              edge_val = e.weight;
              if(edge_val < min_edge_val){
                min_edge_val = edge_val;
              }
            }// if both nodes selected
            graph.removeHighEdges(min_edge_val);
          }
        return RES_OK;
    }



} // smil


#endif //__MAGIC_WAND_T_HPP__
