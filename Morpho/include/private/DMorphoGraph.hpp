/*
 * Copyright (c) 2011, Matthieu FAESSEL and ARMINES
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
#include "DGraph.hpp"


namespace smil
{
    /**
    * \ingroup Morpho
    * \defgroup Graph
    * @{
    */

    template <class T1, class T2>
    class mosaicToGraphFunct : public unaryMorphImageFunctionBase<T1, T2>
    {
    public:
	typedef unaryMorphImageFunctionBase<T1, T2> parentClass;
	
	virtual inline void processPixel(size_t &pointOffset, vector<int>::iterator dOffset, vector<int>::iterator dOffsetEnd)
	{
	    T1 curVal = parentClass::pixelsIn[pointOffset];
	    bool mixed = false;
	    while(dOffset!=dOffsetEnd)
	    {
		T1 val = parentClass::pixelsIn[pointOffset + *dOffset];
		if (val!=curVal)
		{
		  mixed = true;
		  graph.addEdge(val, curVal, parentClass::pixelsOut[pointOffset]);
		}
		dOffset++;
	    }
	}
	Graph graph;
    };
    
    /**
    */ 
    template <class T1, class T2>
    Graph mosaicToGraph(const Image<T1> &imMosaic, const Image<T2> &imValues, const StrElt &se=DEFAULT_SE)
    {
	ASSERT(imMosaic.isAllocated() && imValues.isAllocated(), Graph());
	
	mosaicToGraphFunct<T1, T2> f;
	
	ASSERT(f._exec(imMosaic, (Image<T2>&)imValues, se)==RES_OK, Graph());
	
	return f.graph;
	
    }

    template <class T>
    RES_T graphToMosaic(const Image<T> &imMosRef, const Graph &graph, Image<T> &imOut)
    {
	ASSERT_ALLOCATED(&imOut);
	
	return applyLookup<T>(imMosRef, graph.labelizeNodes(), imOut);
    }

/** \} */

} // namespace smil



#endif // _D_MORPHO_GRAPH_HPP

