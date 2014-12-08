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

#ifndef _D_MORPHO_WATERSHED_EXTINCTION_HPP
#define _D_MORPHO_WATERSHED_EXTINCTION_HPP

#include "DMorphoWatershed.hpp"
#include "DMorphoGraph.hpp"

namespace smil
{

    /**
     * \ingroup Morpho
     * \defgroup Watershed
     * @{
     */
    

    #ifndef SWIG
    template <class T>
    struct CrossVectorComp
    {
	CrossVectorComp(const vector<T> &vec)
	  : compVec(vec)
	{
	}
	const vector<T> &compVec;
	inline bool operator() (const T &i, const T &j)
	{
	    return ( compVec[i] > compVec[j] );
	}
    };
    #endif // SWIG
    
  /**
    * Generic extinction flooding process
    * 
    * Can be derivated in wrapped languages thanks to Swig directors.
    * 
    * \demo{custom_extinction_value.py]
    */
    template <class T, class labelT, class extValType=UINT, class HQ_Type=HierarchicalQueue<T> >
    class ExtinctionFlooding 
#ifndef SWIG    
	: public BaseFlooding<T, labelT, HQ_Type>
#endif // SWIG    
    {
      public:
	virtual ~ExtinctionFlooding() {}
	
	UINT labelNbr, basinNbr;
	T currentLevel;
	vector<labelT> equivalentLabels;
	vector<extValType> extinctionValues;
	size_t lastOffset;
	
	Graph<labelT, extValType> *graph;

	virtual void createBasins(const UINT &nbr)
	{
	    equivalentLabels.resize(nbr);
	    extinctionValues.resize(nbr, 0);
	    
	    for (UINT i=0;i<nbr;i++)
	      equivalentLabels[i] = i;
	    
	    basinNbr = nbr;
	}
	virtual void deleteBasins()
	{
	    equivalentLabels.clear();
	    extinctionValues.clear();
	    
	    basinNbr = 0;
	}
	
	inline virtual void insertPixel(const size_t &offset, const labelT &lbl) {}
	inline virtual void raiseLevel(const labelT &lbl) {}
	inline virtual labelT mergeBasins(const labelT &lbl1, const labelT &lbl2) { return 0; };
	inline virtual void finalize(const labelT &lbl) {}
	
	
	virtual RES_T flood(const Image<T> &imIn, const Image<labelT> &imMarkers, Image<labelT> &imBasinsOut, const StrElt &se=DEFAULT_SE)
	{
	    return BaseFlooding<T, labelT, HQ_Type>::flood(imIn, imMarkers, imBasinsOut, se);
	}
	
	virtual RES_T flood(const Image<T> &imIn, const Image<labelT> &imMarkers, Image<labelT> &imBasinsOut, Graph<labelT, extValType> &_graph, const StrElt &se=DEFAULT_SE)
	{
	    ASSERT_ALLOCATED(&imIn, &imMarkers, &imBasinsOut);
	    ASSERT_SAME_SIZE(&imIn, &imMarkers, &imBasinsOut);
	    
	    ImageFreezer freeze(imBasinsOut);
	    
	    copy(imMarkers, imBasinsOut);

	    initialize(imIn, imBasinsOut, se);
	    this->graph = &_graph;
	    graph->clear();
	    processImage(imIn, imBasinsOut, se);
	    
	    return RES_OK;
	}
	  
	template <class outT>
	RES_T floodWithExtValues(const Image<T> &imIn, const Image<labelT> &imMarkers, Image<outT> &imExtValOut, Image<labelT> &imBasinsOut, const StrElt & se=DEFAULT_SE)
	{
	    ASSERT_ALLOCATED (&imExtValOut);
	    ASSERT_SAME_SIZE (&imIn, &imExtValOut);
	    
	    ASSERT(this->flood(imIn, imMarkers, imBasinsOut, se)==RES_OK);
	    
	    ImageFreezer freezer (imExtValOut);
	    
	    typename ImDtTypes < outT >::lineType pixOut = imExtValOut.getPixels ();
	    typename ImDtTypes < labelT >::lineType pixMarkers = imMarkers.getPixels ();

	    fill(imExtValOut, outT(0));
	    
	    for (size_t i=0; i<imIn.getPixelCount (); i++, pixMarkers++, pixOut++) 
	    {
		if(*pixMarkers != labelT(0))
		  *pixOut = extinctionValues[*pixMarkers] ;
	    }

	    return RES_OK;
	}
	template <class outT>
	RES_T floodWithExtValues(const Image<T> &imIn, const Image<labelT> &imMarkers, Image<outT> &imExtValOut, const StrElt & se=DEFAULT_SE)
	{
	    Image<labelT> imBasinsOut(imMarkers);
	    return floodWithExtValues(imIn, imMarkers, imExtValOut, imBasinsOut, se);
	}
	
	template <class outT>
	RES_T floodWithExtRank(const Image<T> &imIn, const Image<labelT> &imMarkers, Image<outT> &imExtRankOut, Image<labelT> &imBasinsOut, const StrElt & se=DEFAULT_SE)
	{
	    ASSERT_ALLOCATED (&imExtRankOut);
	    ASSERT_SAME_SIZE (&imIn, &imExtRankOut);
	    
	    ASSERT(this->flood(imIn, imMarkers, imBasinsOut, se)==RES_OK);

	    typename ImDtTypes < outT >::lineType pixOut = imExtRankOut.getPixels ();
	    typename ImDtTypes < labelT >::lineType pixMarkers = imMarkers.getPixels ();

	    ImageFreezer freezer (imExtRankOut);
	    
	    // Sort by extinctionValues
	    vector<outT> rank(labelNbr);
	    for (UINT i=0;i<labelNbr;i++)
	      rank[i] = i+1;
	    
	    CrossVectorComp<extValType> comp(this->extinctionValues);
	    sort(rank.begin(), rank.end(), comp);
	    for (UINT i=0;i<labelNbr;i++)
	      extinctionValues[rank[i]] = i+1;
	    
	    fill(imExtRankOut, outT(0));
	    
	    for (size_t i=0; i<imIn.getPixelCount (); i++, pixMarkers++, pixOut++) 
	    {
		if(*pixMarkers != labelT(0))
		  *pixOut = extinctionValues[*pixMarkers] ;
	    }

	    return RES_OK;
	}
	template <class outT>
	RES_T floodWithExtRank(const Image<T> &imIn, const Image<labelT> &imMarkers, Image<outT> &imExtRankOut, const StrElt & se=DEFAULT_SE)
	{
	    Image<labelT> imBasinsOut(imMarkers);
	    return floodWithExtRank(imIn, imMarkers, imExtRankOut, imBasinsOut, se);
	}
	
      protected:
	virtual RES_T initialize(const Image<T> &imIn, Image<labelT> &imLbl, const StrElt &se)
	{
	    BaseFlooding<T, labelT, HQ_Type>::initialize(imIn, imLbl, se);
	    
	    labelNbr = maxVal(imLbl);
	    createBasins(labelNbr+1);
	    currentLevel = ImDtTypes<T>::min();
	    
	    graph = NULL;
            
            return RES_OK;
	}
	    
	virtual RES_T processImage(const Image<T> &imIn, Image<labelT> &imLbl, const StrElt &se)
	{
	    BaseFlooding<T, labelT, HQ_Type>::processImage(imIn, imLbl, se);
	    
	    // Update last level of flooding.
	    labelT l2 = this->lblPixels[lastOffset];
	    while (l2 != equivalentLabels[l2]) 
		l2 = equivalentLabels[l2];
	    finalize(l2);
            
            return RES_OK;
	}
	inline virtual void processPixel(const size_t &curOffset)
	{
	    if (this->inPixels[curOffset] > currentLevel) 
	    {
		currentLevel = this->inPixels[curOffset];
		for (labelT i = 1; i < labelNbr + 1 ; ++i) 
			raiseLevel(i);
	    }

	    BaseFlooding<T, labelT, HQ_Type>::processPixel(curOffset);
	    
	    lastOffset = curOffset;
	}
	inline virtual void processNeighbor(const size_t &curOffset, const size_t &nbOffset)
	{
	    labelT l2 = this->lblPixels[nbOffset];
	    
	    if (l2==0) 
	    {
		this->hq.push(this->inPixels[nbOffset], nbOffset);
 		this->lblPixels[nbOffset] = labelNbr+1;
// 		this->insertPixel(nbOffset, l1);
	    }
	    else if (l2<labelNbr+1)
	    {
		labelT l1 = this->lblPixels[curOffset];
		
		if (l1==0 || l1==labelNbr+1)
		{
		    this->lblPixels[curOffset] = l2;
		    this->insertPixel(curOffset, l2);
		}
		else if (l1!=l2)
		{
		    labelT l1_orig = l1;
		    labelT l2_orig = l2;
		    
		    while (l1!=equivalentLabels[l1])
		      l1 = equivalentLabels[l1];
		    while (l2!=equivalentLabels[l2])
		      l2 = equivalentLabels[l2];
		    
		    if (l1 != l2)
		    {
			// merge basins
			labelT eater = (mergeBasins(l1, l2)==l1) ? l1_orig : l2_orig;
			labelT eaten = (eater==l1_orig) ? l2_orig : l1_orig;
			
			if (graph)
			  graph->addEdge(eaten, eater, this->extinctionValues[eater==l1_orig ? l2 : l1]);
		    }
		}
	    }
	}

	
    };


    template <class T, class labelT, class extValType=UINT, class HQ_Type=HierarchicalQueue<T> >
    class AreaExtinctionFlooding : public ExtinctionFlooding<T, labelT, extValType, HQ_Type>
    {
      public:
	vector<UINT> areas;
	vector<T> minValues;
	
	virtual void createBasins(const UINT &nbr)
	{
	    areas.resize(nbr, 0);
	    minValues.resize(nbr, ImDtTypes<T>::max());
	    
	    ExtinctionFlooding<T, labelT, extValType, HQ_Type>::createBasins(nbr);
	}
	
	virtual void deleteBasins()
	{
	    areas.clear();
	    minValues.clear();
	    
	    ExtinctionFlooding<T, labelT, extValType, HQ_Type>::deleteBasins();
	}
	
	inline virtual void insertPixel(const size_t &offset, const labelT &lbl)
	{
	    if (this->inPixels[offset] < minValues[lbl])
	      minValues[lbl] = this->inPixels[offset];
	    
	    areas[lbl]++;
	}
	virtual labelT mergeBasins(const labelT &lbl1, const labelT &lbl2)
	{
	    labelT eater, eaten;
	    
	    if (areas[lbl1] > areas[lbl2] || (areas[lbl1] == areas[lbl2] && minValues[lbl1] < minValues[lbl2]))
	    {
		eater = lbl1;
		eaten = lbl2;
	    }
	    else 
	    {
		eater = lbl2;
		eaten = lbl1;
	    }
	    
	    this->extinctionValues[eaten] = areas[eaten];
	    areas[eater] += areas[eaten];
	    this->equivalentLabels[eaten] = eater;
	    
	    return eater;
	}
	virtual void finalize(const labelT &lbl)
	{
	    this->extinctionValues[lbl] += areas[lbl];
	}
    };

    template <class T, class labelT, class extValType=UINT, class HQ_Type=HierarchicalQueue<T> >
    class VolumeExtinctionFlooding : public ExtinctionFlooding<T, labelT, extValType, HQ_Type>
    {
	vector<UINT> areas, volumes;
	vector<T> floodLevels;
	
	virtual void createBasins(const UINT &nbr)
	{
	    areas.resize(nbr, 0);
	    volumes.resize(nbr, 0);
	    floodLevels.resize(nbr, 0);
	    
	    ExtinctionFlooding<T, labelT, extValType, HQ_Type>::createBasins(nbr);
	}
	
	virtual void deleteBasins()
	{
	    areas.clear();
	    volumes.clear();
	    floodLevels.clear();
	    
	    ExtinctionFlooding<T, labelT, extValType, HQ_Type>::deleteBasins();
	}
	
	
	virtual void insertPixel(const size_t &offset, const labelT &lbl)
	{
	    if (floodLevels[lbl]!=this->inPixels[offset])
	      floodLevels[lbl] = this->inPixels[offset];
	    areas[lbl]++;
	}
	virtual void raiseLevel(const labelT &lbl)
	{
	    if (floodLevels[lbl] < this->currentLevel) 
	    {
		volumes[lbl] += areas[lbl] * (this->currentLevel - floodLevels[lbl]);
		floodLevels[lbl] = this->currentLevel;
	    }
	}
	virtual labelT mergeBasins(const labelT &lbl1, const labelT &lbl2)
	{
	    labelT eater, eaten;
	    
	    if (volumes[lbl1] > volumes[lbl2] || (volumes[lbl1] == volumes[lbl2] && floodLevels[lbl1] < floodLevels[lbl2]))
	    {
		eater = lbl1;
		eaten = lbl2;
	    }
	    else 
	    {
		eater = lbl2;
		eaten = lbl1;
	    }
	    
	    this->extinctionValues[eaten] = volumes[eaten];
	    volumes[eater] += volumes[eaten];
	    areas[eater] += areas[eaten];
	    this->equivalentLabels[eaten] = eater;
	    
	    return eater;
	}
	virtual void finalize(const labelT &lbl)
	{
	    volumes[lbl] += areas[lbl];
	    this->extinctionValues[lbl] += volumes[lbl];
	}
	
    };

    template <class T, class labelT, class extValType=UINT, class HQ_Type=HierarchicalQueue<T> >
    class DynamicExtinctionFlooding : public AreaExtinctionFlooding<T, labelT, extValType, HQ_Type>
    {
	virtual labelT mergeBasins(const labelT &lbl1, const labelT &lbl2)
	{
	    labelT eater, eaten;
	    
	    if (this->minValues[lbl1] < this->minValues[lbl2] || (this->minValues[lbl1] == this->minValues[lbl2] && this->areas[lbl1] > this->areas[lbl2]))
	    {
		eater = lbl1;
		eaten = lbl2;
	    }
	    else 
	    {
		eater = lbl2;
		eaten = lbl1;
	    }
	    
	    this->extinctionValues[eaten] = this->currentLevel - this->minValues[eaten];
	    this->areas[eater] += this->areas[eaten];
	    this->equivalentLabels[eaten] = eater;
	    
	    return eater;
	}
    };

    
    
    //*******************************************
    //******** GENERAL EXPORTED FUNCTIONS
    //*******************************************

    
    template < class T, class labelT, class outT > 
    RES_T watershedExtinction (const Image<T> 	&imIn,
			       const Image<labelT> &imMarkers,
			       Image<outT> 	&imOut,
			       Image<labelT> 	&imBasinsOut,
			       const char *  	extinctionType="v",
			       const StrElt 	&se = DEFAULT_SE,
			       bool 		rankOutput=true)
    {
	RES_T res = RES_ERR;
	
	if (rankOutput)
	{
	    ExtinctionFlooding<T, labelT, UINT> *flooding = NULL; // outT type may be smaller than the required flooding type
	    
	    if(strcmp(extinctionType, "v")==0)
		flooding = new VolumeExtinctionFlooding<T, labelT, UINT>();
	    else if(strcmp(extinctionType, "a")==0)
		flooding = new AreaExtinctionFlooding<T, labelT, UINT>();
	    else if(strcmp(extinctionType, "d")==0)
		flooding = new DynamicExtinctionFlooding<T, labelT, UINT>();
	    
	    if (flooding)
	    {
	      flooding->floodWithExtRank(imIn, imMarkers, imOut, imBasinsOut, se);
	      delete flooding;
	    }
	}
	
	else
	{
	    ExtinctionFlooding<T, labelT, outT> *flooding = NULL; // outT type may be smaller than the required flooding type
	    
	    if(strcmp(extinctionType, "v")==0)
		flooding = new VolumeExtinctionFlooding<T, labelT, outT>();
	    else if(strcmp(extinctionType, "a")==0)
		flooding = new AreaExtinctionFlooding<T, labelT, outT>();
	    else if(strcmp(extinctionType, "d")==0)
		flooding = new DynamicExtinctionFlooding<T, labelT, outT>();
	    
	    if (flooding)
	    {
	      flooding->floodWithExtValues(imIn, imMarkers, imOut, imBasinsOut, se);
	      delete flooding;
	    }
	}
	
	return res;
    }
    
    template < class T,	class labelT, class outT > 
    RES_T watershedExtinction (const Image < T > &imIn,
				Image < labelT > &imMarkers,
				Image < outT > &imOut,
				const char *  	extinctionType="v", 
				const StrElt & se = DEFAULT_SE,
				bool rankOutput=true) 
    {
	ASSERT_ALLOCATED (&imIn, &imMarkers, &imOut);
	ASSERT_SAME_SIZE (&imIn, &imMarkers, &imOut);
	Image < labelT > imBasinsOut (imMarkers);
	return watershedExtinction (imIn, imMarkers, imOut, imBasinsOut, extinctionType, se, rankOutput);
    }
    
    template < class T,	class outT > 
    RES_T watershedExtinction (const Image < T > &imIn,
				Image < outT > &imOut,
				const char *  	extinctionType="v", 
				const StrElt & se = DEFAULT_SE,
				bool rankOutput=true) 
    {
	ASSERT_ALLOCATED (&imIn, &imOut);
	ASSERT_SAME_SIZE (&imIn, &imOut);
	Image < T > imMin (imIn);
	minima (imIn, imMin, se);
	Image < UINT > imLbl (imIn);
	label (imMin, imLbl, se);
	return watershedExtinction (imIn, imLbl, imOut,extinctionType,  se, rankOutput);
    }
    
     /**
     * Calculation of the minimum spanning tree, simultaneously to the image flooding, with edges weighted according to volume extinction values.
     */
    template < class T,	class labelT, class outT > 
    RES_T watershedExtinctionGraph (const Image < T > &imIn,
				    const Image < labelT > &imMarkers,
				    Image < labelT > &imBasinsOut,
				    Graph < labelT, outT > &graph,
				    const char *  	extinctionType="v", 
				    const StrElt & se = DEFAULT_SE) 
    {
	
	ExtinctionFlooding<T, labelT, outT> *flooding = NULL; // outT type may be smaller than the required flooding type
	
	if(strcmp(extinctionType, "v")==0)
	    flooding = new VolumeExtinctionFlooding<T, labelT, outT>();
	else if(strcmp(extinctionType, "a")==0)
	    flooding = new AreaExtinctionFlooding<T, labelT, outT>();
	else if(strcmp(extinctionType, "d")==0)
	    flooding = new DynamicExtinctionFlooding<T, labelT, outT>();
	else return RES_ERR;
	
	RES_T res = flooding->flood(imIn, imMarkers, imBasinsOut, graph, se);
	
	delete flooding;

	return res;
	
    }
    template < class T,	class labelT, class outT > 
    RES_T watershedExtinctionGraph (const Image < T > &imIn,
				    Image < labelT > &imBasinsOut,
				    Graph < labelT, outT > &graph,
				    const char *  	extinctionType="v", 
				    const StrElt & se = DEFAULT_SE) 
    {
	ASSERT_ALLOCATED (&imIn, &imBasinsOut);
	ASSERT_SAME_SIZE (&imIn, &imBasinsOut);
	
	Image<T> imMin (imIn);
	minima(imIn, imMin, se);
	Image<labelT> imLbl(imBasinsOut);
	label(imMin, imLbl, se);
	
	return watershedExtinctionGraph(imIn, imLbl, imBasinsOut, graph, extinctionType, se);
    }
    
    /**
     * Warning: returns a graph with ranking values 
     */
    template < class T,	class labelT> 
    Graph<labelT,labelT> watershedExtinctionGraph (const Image < T > &imIn,
				    const Image < labelT > &imMarkers,
				    Image < labelT > &imBasinsOut,
				    const char *  	extinctionType="v", 
				    const StrElt & se = DEFAULT_SE) 
    {
	Graph<labelT,UINT> graph;
	Graph<labelT,labelT> rankGraph;
	ASSERT(watershedExtinctionGraph(imIn, imMarkers, imBasinsOut, graph, extinctionType, se)==RES_OK, rankGraph);
	
	// Sort edges by extinctionValues
	graph.sortEdges();
	
	
	typedef typename Graph<labelT,UINT>::EdgeType EdgeType;
	vector<EdgeType> &edges = graph.getEdges();
	UINT edgesNbr = edges.size();
	
	for (UINT i=0;i<edgesNbr;i++)
	  rankGraph.addEdge(edges[i].source, edges[i].target, i+1, false);

	return rankGraph;
    }
    template < class T,	class labelT> 
    Graph<labelT,labelT> watershedExtinctionGraph (const Image < T > &imIn,
				    Image < labelT > &imBasinsOut,
				    const char *  	extinctionType="v", 
				    const StrElt & se = DEFAULT_SE) 
    {
	Graph<labelT,UINT> graph;
	Graph<labelT,labelT> rankGraph;
	ASSERT(watershedExtinctionGraph(imIn, imBasinsOut, graph, extinctionType, se)==RES_OK, rankGraph);
	
	// Sort edges by extinctionValues
	graph.sortEdges();
	
	
	typedef typename Graph<labelT,UINT>::EdgeType EdgeType;
	vector<EdgeType> &edges = graph.getEdges();
	UINT edgesNbr = edges.size();
	
	for (UINT i=0;i<edgesNbr;i++)
	  rankGraph.addEdge(edges[i].source, edges[i].target, i+1, false);

	return rankGraph;
    }
}				// namespace smil


#endif // _D_MORPHO_WATERSHED_EXTINCTION_HPP
