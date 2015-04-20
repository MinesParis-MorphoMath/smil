/*
 * Copyright (c) 2011-2015, Matthieu FAESSEL and ARMINES
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


#ifndef _D_MORPHO_GRAPH_HPP
#define _D_MORPHO_GRAPH_HPP

#include "DMorphImageOperations.hpp"
#include "Core/include/private/DGraph.hpp"
#include "Core/include/private/DTraits.hpp"


namespace smil
{
    /**
    * \ingroup Morpho
    * \defgroup Graph
    * @{
    */

    template <class T1, class T2, class graphT=Graph<UINT,UINT> >
    class mosaicToGraphFunct : public MorphImageFunctionBase<T1, T2>
    {
    public:
        typedef MorphImageFunctionBase<T1, T2> parentClass;
        typedef Image<T1> imageInType;
        typedef Image<T2> imageOutType;
        
        mosaicToGraphFunct(/*graphT *externGraph=NULL*/)
          : graphPtr(auto_ptr<graphT>(new graphT())), graph(*graphPtr.get())
        {
            imEdgeValues = NULL;
            imNodeValues = NULL;
        }
        mosaicToGraphFunct(graphT &externGraph)
          : graph(externGraph)
        {
            imEdgeValues = NULL;
            imNodeValues = NULL;
            externGraph.clear();
        }
        virtual RES_T initialize(const imageInType &imIn, imageOutType &imOut, const StrElt &se)
        {
            parentClass::initialize(imIn, imOut, se);
            
            if (imEdgeValues)
              edgeValuePixels = imEdgeValues->getPixels();
            if (imNodeValues)
              nodeValuePixels = imNodeValues->getPixels();
            
            return RES_OK;
        }
        
        virtual inline void processPixel(size_t pointOffset, vector<int> &dOffsetList)
        {
            T1 curVal = parentClass::pixelsIn[pointOffset];
//             bool mixed = false;
            vector<int>::iterator dOffset = dOffsetList.begin();
            while(dOffset!=dOffsetList.end())
            {
                T1 val = parentClass::pixelsIn[pointOffset + *dOffset];
                if (val!=curVal)
                {
                    if (imNodeValues)
                    {
                        graph.addNode(curVal, nodeValuePixels[pointOffset]);
                        graph.addNode(val, nodeValuePixels[pointOffset + *dOffset]);
                    }

//                   mixed = true;
                  // Add an edge between the two basins. 
                  // If the edge already exists, its weight will be the minimum value between the existing and the new one (pixelsOut[pointOffset]).
                    if (imEdgeValues)
                      graph.addEdge(curVal, val, edgeValuePixels[pointOffset]);
                    else
                      graph.addEdge(curVal, val);
                }
                dOffset++;
            }
        }
    protected:
        auto_ptr<graphT> graphPtr;
        typename ImDtTypes<T2>::lineType edgeValuePixels;
        typename ImDtTypes<T2>::lineType nodeValuePixels;
    public:
        graphT &graph;
        const Image<T2> *imEdgeValues;
        const Image<T2> *imNodeValues;
    };
    
    /**
    */ 
    template <class T1, class T2, class graphT>
    RES_T mosaicToGraph(const Image<T1> &imMosaic, graphT &graph, const StrElt &se=DEFAULT_SE)
    {
        ASSERT_ALLOCATED(&imMosaic);
        
        mosaicToGraphFunct<T1, T2, graphT > f(graph);
        
        ASSERT(f._exec(imMosaic, se)==RES_OK);
        
        return RES_OK;
    }
    template <class T1, class T2, class graphT>
    RES_T mosaicToGraph(const Image<T1> &imMosaic, const Image<T2> &imEdgeValues, graphT &graph, const StrElt &se=DEFAULT_SE)
    {
        ASSERT_ALLOCATED(&imMosaic, &imEdgeValues);
        ASSERT( haveSameSize(&imMosaic, &imEdgeValues, NULL) );
        
        mosaicToGraphFunct<T1, T2, graphT > f(graph);
        f.imEdgeValues = &imEdgeValues;
        
        ASSERT(f._exec(imMosaic, se)==RES_OK);
        
        return RES_OK;
    }
    template <class T1, class T2, class graphT>
    RES_T mosaicToGraph(const Image<T1> &imMosaic, const Image<T2> &imEdgeValues, const Image<T2> &imNodeValues, graphT &graph, const StrElt &se=DEFAULT_SE)
    {
        ASSERT_ALLOCATED(&imMosaic, &imEdgeValues, &imNodeValues);
        ASSERT( haveSameSize(&imMosaic, &imEdgeValues, &imNodeValues, NULL) );
        
        mosaicToGraphFunct<T1, T2, graphT > f(graph);
        f.imEdgeValues = &imEdgeValues;
        f.imNodeValues = &imNodeValues;
        
        ASSERT(f._exec(imMosaic, se)==RES_OK);
        
        return RES_OK;
    }
    
    template <class T1>
    Graph<T1,UINT> mosaicToGraph(const Image<T1> &imMosaic, const StrElt &se=DEFAULT_SE)
    {
        Graph<T1,UINT> graph;
        mosaicToGraph(imMosaic, graph, se);
        return graph;
        
    }
    template <class T1, class T2>
    Graph<T1,T2> mosaicToGraph(const Image<T1> &imMosaic, const Image<T2> &imEdgeValues, const StrElt &se=DEFAULT_SE)
    {
        Graph<T1,T2> graph;
        mosaicToGraph(imMosaic, imEdgeValues, graph, se);
        return graph;
        
    }
    template <class T1, class T2>
    Graph<T1,T2> mosaicToGraph(const Image<T1> &imMosaic, const Image<T2> &imEdgeValues, const Image<T2> &imNodeValues, const StrElt &se=DEFAULT_SE)
    {
        Graph<T1,T2> graph;
        mosaicToGraph(imMosaic, imEdgeValues, imNodeValues, graph, se);
        return graph;
        
    }

    template <class T1>
    RES_T mosaicToGraph(const Image<T1> &imMosaic, Graph<T1,UINT> &graph, const StrElt &se=DEFAULT_SE)
    {
        return mosaicToGraph<T1, UINT, Graph<T1,UINT> >(imMosaic, graph, se);
    }
    template <class T1, class T2>
    RES_T mosaicToGraph(const Image<T1> &imMosaic, const Image<T2> &imEdgeValues, Graph<T1,T2> &graph, const StrElt &se=DEFAULT_SE)
    {
        return mosaicToGraph<T1, T2, Graph<T1,T2> >(imMosaic, imEdgeValues, graph, se);
    }
    template <class T1, class T2>
    RES_T mosaicToGraph(const Image<T1> &imMosaic, const Image<T2> &imEdgeValues, const Image<T2> &imNodeValues, Graph<T1,T2> &graph, const StrElt &se=DEFAULT_SE)
    {
        return mosaicToGraph<T1, T2, Graph<T1,T2> >(imMosaic, imEdgeValues, imNodeValues, graph, se);
    }

#ifndef SWIG
    template <class T1>
    ENABLE_IF( !IS_SAME(T1,size_t), RES_T ) // SFINAE Only if T1!=size_t && T2!=size_t
    mosaicToGraph(const Image<T1> &imMosaic, Graph<> &graph, const StrElt &se=DEFAULT_SE)
    {
        return mosaicToGraph<T1, UINT, Graph<> >(imMosaic, graph, se);
    }
    template <class T1, class T2>
    ENABLE_IF( !IS_SAME(T1,size_t) && !IS_SAME(T2,size_t), RES_T ) // SFINAE Only if T1!=size_t && T2!=size_t
    mosaicToGraph(const Image<T1> &imMosaic, const Image<T2> &imEdgeValues, Graph<> &graph, const StrElt &se=DEFAULT_SE)
    {
        return mosaicToGraph<T1, T2, Graph<> >(imMosaic, imEdgeValues, graph, se);
    }
    template <class T1, class T2>
    ENABLE_IF( !IS_SAME(T1,size_t) && !IS_SAME(T2,size_t), RES_T ) // SFINAE Only if T1!=size_t && T2!=size_t
    mosaicToGraph(const Image<T1> &imMosaic, const Image<T2> &imEdgeValues, const Image<T2> &imNodeValues, Graph<> &graph, const StrElt &se=DEFAULT_SE)
    {
        return mosaicToGraph<T1, T2, Graph<> >(imMosaic, imEdgeValues, imNodeValues, graph, se);
    }

    template <class T1>
    ENABLE_IF( !IS_SAME(T1,UINT), RES_T ) // SFINAE Only if T1!=UINT
    mosaicToGraph(const Image<T1> &imMosaic, Graph<UINT,UINT> &graph, const StrElt &se=DEFAULT_SE)
    {
        return mosaicToGraph<T1, UINT, Graph<UINT,UINT> >(imMosaic, graph, se);
    }
    template <class T1, class T2>
    ENABLE_IF( !IS_SAME(T1,UINT), RES_T ) // SFINAE Only if T1!=UINT
    mosaicToGraph(const Image<T1> &imMosaic, const Image<T2> &imEdgeValues, Graph<UINT,T2> &graph, const StrElt &se=DEFAULT_SE)
    {
        return mosaicToGraph<T1, T2, Graph<UINT,T2> >(imMosaic, imEdgeValues, graph, se);
    }
    template <class T1, class T2>
    ENABLE_IF( !IS_SAME(T1,UINT), RES_T ) // SFINAE Only if T1!=UINT
    mosaicToGraph(const Image<T1> &imMosaic, const Image<T2> &imEdgeValues, const Image<T2> &imNodeValues, Graph<UINT,T2> &graph, const StrElt &se=DEFAULT_SE)
    {
        return mosaicToGraph<T1, T2, Graph<UINT,T2> >(imMosaic, imEdgeValues, imNodeValues, graph, se);
    }

    template <class T1, class T2>
    ENABLE_IF( !IS_SAME(T2,UINT), RES_T ) // SFINAE Only if T2!=UINT
    mosaicToGraph(const Image<T1> &imMosaic, const Image<T2> &imEdgeValues, Graph<T1,UINT> &graph, const StrElt &se=DEFAULT_SE)
    {
        return mosaicToGraph<T1, T2, Graph<T1,UINT> >(imMosaic, imEdgeValues, graph, se);
    }
    template <class T1, class T2>
    ENABLE_IF( !IS_SAME(T2,UINT), RES_T ) // SFINAE Only if T2!=UINT
    mosaicToGraph(const Image<T1> &imMosaic, const Image<T2> &imEdgeValues, const Image<T2> &imNodeValues, Graph<T1,UINT> &graph, const StrElt &se=DEFAULT_SE)
    {
        return mosaicToGraph<T1, T2, Graph<T1,UINT> >(imMosaic, imEdgeValues, imNodeValues, graph, se);
    }
#else // SWIG
    template <class T1>
    RES_T mosaicToGraph(const Image<T1> &imMosaic, Graph<> &graph, const StrElt &se=DEFAULT_SE);
    template <class T1, class T2>
    RES_T mosaicToGraph(const Image<T1> &imMosaic, const Image<T2> &imEdgeValues, Graph<> &graph, const StrElt &se=DEFAULT_SE);
    template <class T1, class T2>
    RES_T mosaicToGraph(const Image<T1> &imMosaic, const Image<T2> &imEdgeValues, const Image<T2> &imNodeValues, Graph<> &graph, const StrElt &se=DEFAULT_SE);
    
    template <class T1>
    RES_T mosaicToGraph(const Image<T1> &imMosaic, Graph<UINT,UINT> &graph, const StrElt &se=DEFAULT_SE);
    template <class T1, class T2>
    RES_T mosaicToGraph(const Image<T1> &imMosaic, const Image<T2> &imEdgeValues, Graph<UINT,T2> &graph, const StrElt &se=DEFAULT_SE);
    template <class T1, class T2>
    RES_T mosaicToGraph(const Image<T1> &imMosaic, const Image<T2> &imEdgeValues, const Image<T2> &imNodeValues, Graph<UINT,T2> &graph, const StrElt &se=DEFAULT_SE);

    template <class T1, class T2>
    RES_T mosaicToGraph(const Image<T1> &imMosaic, const Image<T2> &imEdgeValues, Graph<T1,UINT> &graph, const StrElt &se=DEFAULT_SE);
    template <class T1, class T2>
    RES_T mosaicToGraph(const Image<T1> &imMosaic, const Image<T2> &imEdgeValues, const Image<T2> &imNodeValues, Graph<T1,UINT> &graph, const StrElt &se=DEFAULT_SE);
#endif // SWIG

    
    
    template <class T, class graphT>
    RES_T graphToMosaic(const Image<T> &imMosRef, const graphT &graph, Image<T> &imOut)
    {
        ASSERT_ALLOCATED(&imOut);
        
        typedef typename graphT::NodeType NodeType;
        map<NodeType,NodeType> nodeMap = graph.labelizeNodes();
        map<T,T> lut(nodeMap.begin(), nodeMap.end()); 
        
        return applyLookup(imMosRef, lut, imOut);
    }

    template <class T>
    RES_T graphToMosaic(const Image<T> &imMosRef, const Graph<T, T> &graph, Image<T> &imOut)
    {
        return graphToMosaic< T, Graph<T,T> >(imMosRef, graph, imOut);
    }

#ifndef SWIG
    template <class T>
    ENABLE_IF( !IS_SAME(T,UINT), RES_T ) // SFINAE Only if T!=UINT
    graphToMosaic(const Image<T> &imMosRef, const Graph<UINT, T> &graph, Image<T> &imOut)
    {
        return graphToMosaic< T, Graph<UINT,T> >(imMosRef, graph, imOut);
    }

    template <class T>
    ENABLE_IF( !IS_SAME(T,UINT), RES_T ) // SFINAE Only if T!=UINT
    graphToMosaic(const Image<T> &imMosRef, const Graph<T, UINT> &graph, Image<T> &imOut)
    {
        return graphToMosaic< T, Graph<T,UINT> >(imMosRef, graph, imOut);
    }

    template <class T>
    ENABLE_IF( !IS_SAME(T,size_t), RES_T ) // SFINAE Only if T!=size_t
    graphToMosaic(const Image<T> &imMosRef, const Graph<> &graph, Image<T> &imOut)
    {
        return graphToMosaic< T, Graph<> >(imMosRef, graph, imOut);
    }
#else // SWIG
    template <class T>
    RES_T graphToMosaic(const Image<T> &imMosRef, const Graph<UINT, T> &graph, Image<T> &imOut);

    template <class T>
    RES_T graphToMosaic(const Image<T> &imMosRef, const Graph<T, UINT> &graph, Image<T> &imOut);

    template <class T>
    RES_T graphToMosaic(const Image<T> &imMosRef, const Graph<> &graph, Image<T> &imOut);
#endif // SWIG
    
    
    template <class mosImT, class graphT, class imOutT>
    RES_T drawGraph(const Image<mosImT> &imMosaic, const graphT &graph, Image<imOutT> &imOut, imOutT linesValue=ImDtTypes<imOutT>::max())
    {
        ASSERT_ALLOCATED(&imMosaic, &imOut);
        ASSERT_SAME_SIZE(&imMosaic, &imOut);
        
        ImageFreezer freeze(imOut);
        
        map<mosImT, vector<double> > barys = measBarycenters(imMosaic);
        
        typedef typename graphT::EdgeType EdgeType;
        typedef const vector< EdgeType > EdgeListType;
        EdgeListType &edges = graph.getEdges();
        
        for(typename EdgeListType::const_iterator it=edges.begin();it!=edges.end();it++)
        {
            const EdgeType &edge = *it;
            
            if (edge.source==edge.target)
              continue;
            
            vector<double> &p1 = barys[edge.source];
            vector<double> &p2 = barys[edge.target];
            
            if (p1.empty() || p2.empty())
              continue;
            
            ASSERT(drawLine(imOut, int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]), linesValue)==RES_OK);
        }
        
        return RES_OK;
    }

    template <class mosImT, class imOutT>
    RES_T drawGraph(const Image<mosImT> &imMosaic, const Graph<mosImT,imOutT> &graph, Image<imOutT> &imOut, imOutT linesValue=ImDtTypes<imOutT>::max())
    {
        return drawGraph<mosImT, Graph<mosImT,imOutT>, imOutT >(imMosaic, graph, imOut, linesValue);
    }

#ifndef SWIG
    template <class mosImT, class imOutT>
    ENABLE_IF( !IS_SAME(mosImT,size_t) && !IS_SAME(imOutT,size_t), RES_T ) // SFINAE Only if mosImT!=size_t && imOutT!=size_t
    drawGraph(const Image<mosImT> &imMosaic, const Graph<> &graph, Image<imOutT> &imOut, imOutT linesValue=ImDtTypes<imOutT>::max())
    {
        return drawGraph<mosImT, Graph<>, imOutT>(imMosaic, graph, imOut, linesValue);
    }
    
    template <class mosImT, class imOutT>
    ENABLE_IF( !IS_SAME(mosImT,UINT), RES_T ) // SFINAE Only if mosImT!=UINT
    drawGraph(const Image<mosImT> &imMosaic, const Graph<UINT,imOutT> &graph, Image<imOutT> &imOut, imOutT linesValue=ImDtTypes<imOutT>::max())
    {
        return drawGraph<mosImT, Graph<UINT,imOutT>, imOutT >(imMosaic, graph, imOut, linesValue);
    }
    
    template <class mosImT, class imOutT>
    ENABLE_IF( !IS_SAME(imOutT,UINT), RES_T ) // SFINAE Only if imOutT!=UINT
    drawGraph(const Image<mosImT> &imMosaic, const Graph<mosImT,UINT> &graph, Image<imOutT> &imOut, imOutT linesValue=ImDtTypes<imOutT>::max())
    {
        return drawGraph<mosImT, Graph<mosImT,UINT>, imOutT >(imMosaic, graph, imOut, linesValue);
    }
#else //SWIG
    template <class mosImT, class imOutT>
    RES_T drawGraph(const Image<mosImT> &imMosaic, const Graph<> &graph, Image<imOutT> &imOut, imOutT linesValue=ImDtTypes<imOutT>::max());
    
    template <class mosImT, class imOutT>
    RES_T drawGraph(const Image<mosImT> &imMosaic, const Graph<UINT,imOutT> &graph, Image<imOutT> &imOut, imOutT linesValue=ImDtTypes<imOutT>::max());
    
    template <class mosImT, class imOutT>
    RES_T drawGraph(const Image<mosImT> &imMosaic, const Graph<mosImT,UINT> &graph, Image<imOutT> &imOut, imOutT linesValue=ImDtTypes<imOutT>::max());
#endif //SWIG
    
    
/** \} */

} // namespace smil



#endif // _D_MORPHO_GRAPH_HPP

