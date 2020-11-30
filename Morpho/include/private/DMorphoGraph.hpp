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
    * @ingroup Morpho
    * @addtogroup MorphoGraph 
    * @{
    */

    template <class T1, class T2, class graphT=Graph<T1,T2> >
    class mosaicToGraphFunct 
#ifndef SWIG    
      : public MorphImageFunctionBase<T1, T2>
#endif // SWIG    
    {
    public:
        typedef MorphImageFunctionBase<T1, T2> parentClass;
        typedef Image<T1> imageInType;
        typedef Image<T2> imageOutType;
        
        typedef typename graphT::NodeType NodeType;
        typedef typename graphT::EdgeType EdgeType;
        typedef typename graphT::EdgeWeightType EdgeWeightType;
        typedef typename graphT::NodeListType NodeListType;
        typedef typename graphT::EdgeListType EdgeListType;
        
        mosaicToGraphFunct()
        {
            internalGraph = NULL;
            imEdgeValues = NULL;
            imNodeValues = NULL;
        }
        
        virtual ~mosaicToGraphFunct()
        {
            if (internalGraph)
              delete internalGraph;
        }

        
        RES_T operator()(const Image<T1> &imMosaic, const Image<T2> &imEdgeValues, const Image<T2> &imNodeValues, graphT &graph, const StrElt &se=DEFAULT_SE)
        {
            ASSERT_ALLOCATED(&imMosaic, &imEdgeValues, &imNodeValues);
            ASSERT_SAME_SIZE(&imMosaic, &imEdgeValues, &imNodeValues);
        
            this->imEdgeValues = &imEdgeValues;
            this->imNodeValues = &imNodeValues;
            this->graph = &graph;
            return this->_exec(imMosaic, se); 
        }
        const graphT &operator()(const Image<T1> &imMosaic, const Image<T2> &imEdgeValues, const Image<T2> &imNodeValues, const StrElt &se=DEFAULT_SE)
        {
            ASSERT( areAllocated(&imMosaic, &imEdgeValues, &imNodeValues, NULL), "Unallocated input image", *internalGraph );
            ASSERT( haveSameSize(&imMosaic, &imEdgeValues, &imNodeValues, NULL), "Input images must have the same size", *internalGraph );
            
            this->imEdgeValues = &imEdgeValues;
            this->imNodeValues = &imNodeValues;
            this->graph = internalGraph;
            this->_exec(imMosaic, se);
            return *internalGraph;
        }
        RES_T operator()(const Image<T1> &imMosaic, const Image<T2> &imEdgeValues, graphT &graph, const StrElt &se=DEFAULT_SE)
        {
            ASSERT_ALLOCATED(&imMosaic, &imEdgeValues);
            ASSERT_SAME_SIZE(&imMosaic, &imEdgeValues);

            this->imEdgeValues = &imEdgeValues;
            this->imNodeValues = NULL;
            this->graph = &graph;
            return this->_exec(imMosaic, se); 
        }
        const graphT &operator()(const Image<T1> &imMosaic, const Image<T2> &imEdgeValues, const StrElt &se=DEFAULT_SE)
        {
            ASSERT( areAllocated(&imMosaic, &imEdgeValues, NULL), "Unallocated input image", *internalGraph );
            ASSERT( haveSameSize(&imMosaic, &imEdgeValues, NULL), "Input images must have the same size", *internalGraph );
            
            this->imEdgeValues = &imEdgeValues;
            this->imNodeValues = NULL;
            this->graph = internalGraph;
            this->_exec(imMosaic, se);
            return *internalGraph;
        }
        RES_T operator()(const Image<T1> &imMosaic, graphT &graph, const StrElt &se=DEFAULT_SE)
        {
            ASSERT_ALLOCATED(&imMosaic);

            this->imEdgeValues = NULL;
            this->imNodeValues = NULL;
            this->graph = &graph;
            return this->_exec(imMosaic, se); 
        }
        const graphT &operator()(const Image<T1> &imMosaic, const StrElt &se=DEFAULT_SE)
        {
            ASSERT(areAllocated(&imMosaic, NULL), "Unallocated input image", *internalGraph);
            
            this->imEdgeValues = NULL;
            this->imNodeValues = NULL;
            this->graph = internalGraph;
            this->_exec(imMosaic, se);
            return *internalGraph;
        }
        
        

        virtual RES_T initialize(const imageInType &imIn, imageOutType &imOut, const StrElt &se)
        {
            ASSERT( parentClass::initialize(imIn, imOut, se)==RES_OK );
            
            if (graph==NULL)
              graph = internalGraph = new graphT();
            
            graph->clear();
            edges = &graph->getEdges();
            nodes = &graph->getNodes();
            
            imMosaic = &imIn;
            if (imEdgeValues)
              edgeValuePixels = imEdgeValues->getPixels();
            if (imNodeValues)
              nodeValuePixels = imNodeValues->getPixels();
            
            return RES_OK;
        }
        
        virtual RES_T finalize(const imageInType &imIn, imageOutType &imOut, const StrElt &se)
        {
            return parentClass::finalize(imIn, imOut, se);
        }
        
        virtual inline void processPixel(size_t pointOffset, vector<int> &dOffsetList)
        {
            T1 curVal = parentClass::pixelsIn[pointOffset];
            vector<int>::iterator dOffset = dOffsetList.begin();
            
            while(dOffset!=dOffsetList.end())
            {
                T1 val = parentClass::pixelsIn[pointOffset + *dOffset];
                if (val!=curVal)
                {
                    int edgeInd = graph->findEdge(curVal, val);
                    
                    // If the edge already exists, take the min weight value between the existing and the new one (pixelsOut[pointOffset]).
                    if (edgeInd!=-1 && imEdgeValues)
                    {
                        EdgeType &edge = edges->at(edgeInd);
                        edge.weight = min(edge.weight, EdgeWeightType(edgeValuePixels[pointOffset]));
                    }
                    else
                    {
                        if (imNodeValues)
                        {
                            graph->addNode(curVal, nodeValuePixels[pointOffset]);
                            graph->addNode(val, nodeValuePixels[pointOffset + *dOffset]);
                        }
                        if (imEdgeValues)
                          graph->addEdge(curVal, val, edgeValuePixels[pointOffset], false); // false means don't check if the edge exists
                        else
                          graph->addEdge(curVal, val, 0, false);
                    }
                }
                dOffset++;
            }
        }
    protected:
        graphT *internalGraph;
        typename ImDtTypes<T2>::lineType edgeValuePixels;
        typename ImDtTypes<T2>::lineType nodeValuePixels;

        NodeListType *nodes;
        EdgeListType *edges;
        
    public:
        graphT *graph;
        const Image<T1> *imMosaic;
        const Image<T2> *imEdgeValues;
        const Image<T2> *imNodeValues;
    };
    
    // Generic functions
    template <class T1, class T2, class GT1, class GT2>
    RES_T mosaicToGraph(const Image<T1> &imMosaic, const Image<T2> &imEdgeValues, const Image<T2> &imNodeValues, Graph<GT1,GT2> &graph, const StrElt &se=DEFAULT_SE)
    {
        typedef Graph<GT1,GT2> graphT;
        mosaicToGraphFunct<T1, T2, graphT > f;
        
        return f(imMosaic, imEdgeValues, imNodeValues, graph, se);
    }
    template <class T1, class T2>
    Graph<T1,T2> mosaicToGraph(const Image<T1> &imMosaic, const Image<T2> &imEdgeValues, const Image<T2> &imNodeValues, const StrElt &se=DEFAULT_SE)
    {
        typedef Graph<T1,T2> graphT;
        mosaicToGraphFunct<T1, T2, graphT > f;
        
        return f(imMosaic, imEdgeValues, imNodeValues, se);
    }
    template <class T1, class T2, class GT1, class GT2>
    RES_T mosaicToGraph(const Image<T1> &imMosaic, const Image<T2> &imEdgeValues, Graph<GT1,GT2> &graph, const StrElt &se=DEFAULT_SE)
    {
        typedef Graph<GT1,GT2> graphT;
        mosaicToGraphFunct<T1, T2, graphT > f;
        
        return f(imMosaic, imEdgeValues, graph, se);
    }
    template <class T1, class T2>
    Graph<T1,T2> mosaicToGraph(const Image<T1> &imMosaic, const Image<T2> &imEdgeValues, const StrElt &se=DEFAULT_SE)
    {
        typedef Graph<T1,T2> graphT;
        mosaicToGraphFunct<T1, T2, graphT > f;
        
        return f(imMosaic, imEdgeValues, se);
    }
    template <class T1, class GT1, class GT2>
    RES_T mosaicToGraph(const Image<T1> &imMosaic, Graph<GT1,GT2> &graph, const StrElt &se=DEFAULT_SE)
    {
        typedef Graph<GT1,GT2> graphT;
        mosaicToGraphFunct<T1, T1, graphT > f;
        
        return f(imMosaic, graph, se);
    }
    template <class T1>
    Graph<T1,UINT> mosaicToGraph(const Image<T1> &imMosaic, const StrElt &se=DEFAULT_SE)
    {
        typedef Graph<T1,UINT> graphT;
        mosaicToGraphFunct<T1, T1, graphT > f;
        
        return f(imMosaic, se);
    }
    
    
    
    
    
    
#ifndef SWIG
    template <class T1>
    ENABLE_IF( !IS_SAME(T1,size_t), RES_T ) // SFINAE Only if T1!=size_t && T2!=size_t
    mosaicToGraph(const Image<T1> &imMosaic, Graph<> &graph, const StrElt &se=DEFAULT_SE)
    {
        return mosaicToGraph<T1, size_t, size_t>(imMosaic, graph, se);
    }
    template <class T1, class T2>
    ENABLE_IF( !IS_SAME(T1,size_t) && !IS_SAME(T2,size_t), RES_T ) // SFINAE Only if T1!=size_t && T2!=size_t
    mosaicToGraph(const Image<T1> &imMosaic, const Image<T2> &imEdgeValues, Graph<> &graph, const StrElt &se=DEFAULT_SE)
    {
        return mosaicToGraph<T1, T2, size_t, size_t>(imMosaic, imEdgeValues, graph, se);
    }
    template <class T1, class T2>
    ENABLE_IF( !IS_SAME(T1,size_t) && !IS_SAME(T2,size_t), RES_T ) // SFINAE Only if T1!=size_t && T2!=size_t
    mosaicToGraph(const Image<T1> &imMosaic, const Image<T2> &imEdgeValues, const Image<T2> &imNodeValues, Graph<> &graph, const StrElt &se=DEFAULT_SE)
    {
        return mosaicToGraph<T1, T2, size_t, size_t>(imMosaic, imEdgeValues, imNodeValues, graph, se);
    }

#ifdef USE_64BIT_IDS
    template <class T1>
    ENABLE_IF( !IS_SAME(T1,UINT), RES_T ) // SFINAE Only if T1!=UINT
    mosaicToGraph(const Image<T1> &imMosaic, Graph<UINT,UINT> &graph, const StrElt &se=DEFAULT_SE)
    {
        return mosaicToGraph<T1, UINT,UINT>(imMosaic, graph, se);
    }
#endif // USE_64BIT_IDS
    template <class T1, class T2>
    ENABLE_IF( !IS_SAME(T1,UINT), RES_T ) // SFINAE Only if T1!=UINT
    mosaicToGraph(const Image<T1> &imMosaic, const Image<T2> &imEdgeValues, Graph<UINT,T2> &graph, const StrElt &se=DEFAULT_SE)
    {
        return mosaicToGraph<T1, T2, UINT,T2>(imMosaic, imEdgeValues, graph, se);
    }
    template <class T1, class T2>
    ENABLE_IF( !IS_SAME(T1,UINT), RES_T ) // SFINAE Only if T1!=UINT
    mosaicToGraph(const Image<T1> &imMosaic, const Image<T2> &imEdgeValues, const Image<T2> &imNodeValues, Graph<UINT,T2> &graph, const StrElt &se=DEFAULT_SE)
    {
        return mosaicToGraph<T1, T2, UINT,T2>(imMosaic, imEdgeValues, imNodeValues, graph, se);
    }

    template <class T1, class T2>
    ENABLE_IF( !IS_SAME(T2,UINT), RES_T ) // SFINAE Only if T2!=UINT
    mosaicToGraph(const Image<T1> &imMosaic, const Image<T2> &imEdgeValues, Graph<T1,UINT> &graph, const StrElt &se=DEFAULT_SE)
    {
        return mosaicToGraph<T1, T2, T1,UINT>(imMosaic, imEdgeValues, graph, se);
    }
    template <class T1, class T2>
    ENABLE_IF( !IS_SAME(T2,UINT), RES_T ) // SFINAE Only if T2!=UINT
    mosaicToGraph(const Image<T1> &imMosaic, const Image<T2> &imEdgeValues, const Image<T2> &imNodeValues, Graph<T1,UINT> &graph, const StrElt &se=DEFAULT_SE)
    {
        return mosaicToGraph<T1, T2, T1,UINT>(imMosaic, imEdgeValues, imNodeValues, graph, se);
    }
#else // SWIG
    template <class T1>
    RES_T mosaicToGraph(const Image<T1> &imMosaic, Graph<T1,T1> &graph, const StrElt &se=DEFAULT_SE);
    template <class T1, class T2>
    RES_T mosaicToGraph(const Image<T1> &imMosaic, const Image<T2> &imEdgeValues, Graph<T1,T2> &graph, const StrElt &se=DEFAULT_SE);
    
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
        
        map<mosImT, vector<double> > barys = blobsBarycenter(imMosaic);
        
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
    
    
/** @} */

} // namespace smil



#endif // _D_MORPHO_GRAPH_HPP

