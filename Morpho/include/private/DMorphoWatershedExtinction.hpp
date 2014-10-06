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
    template <class T, class extValType> class Flooding;

    template <class T, class extValType>
    struct extinctionValuesComp
    {
	extinctionValuesComp(Flooding<T,extValType> *_flooding)
	{
	    flooding = _flooding;
	}
	inline bool operator() (const UINT &i, const UINT &j)
	{
	    return (flooding->extinctionValues[i] > flooding->extinctionValues[j] );
	}
	Flooding<T,extValType> *flooding;
    };
#endif // SWIG

    /**
     * Generic flooding process
     * 
     * Can be derivated in wrapped languages thanks to Swig directors.
     * 
     * Python example:
     * \code{.py}
     * from smilPython import *
     * 
     * class myAreaExtinction(Flooding_UINT8):
     *     def createBasins(self, nbr):
     *       self.areas = [0]*nbr
     *       Flooding_UINT8.createBasins(self, nbr)
     *     def insertPixel(self, lbl):
     *       self.areas[lbl] += 1
     *     def mergeBasins(self, lbl1, lbl2):
     *       if self.areas[lbl1] > self.areas[lbl2]:
     * 	eater = lbl1
     * 	eaten = lbl2
     *       else:
     * 	eater = lbl2
     * 	eaten = lbl1
     *       self.extinctionValues[eaten] =  self.areas[eaten]
     *       self.areas[eater] += self.areas[eaten]
     *       self.equivalents[eaten] = eater
     *       return eater
     *     def finalize(self, lbl):
     *       self.extinctionValues[lbl] += self.areas[lbl]
     * 
     * 
     * imIn = Image("http://cmm.ensmp.fr/~faessel/smil/images/lena.png")
     * imGrad = Image(imIn)
     * imMark = Image(imIn, "UINT16")
     * imExtRank = Image(imIn, "UINT16")
     * 
     * gradient(imIn, imGrad)
     * hMinimaLabeled(imGrad, 20, imMark)
     * 
     * aExt = myAreaExtinction()
     * aExt.floodWithExtRank(imIn, imMark, imExtRank)
     * 
     * imExtRank.showLabel()
     * \endcode

     */
    template <class T, class extValType=UINT>
    class Flooding
    {
      public:
	Flooding()
	  : basinNbr(0)
	{
	}
	virtual ~Flooding()
	{
	    if (basinNbr!=0)
	      deleteBasins();
	}
	
	friend extinctionValuesComp<T,extValType>;

	 
	vector<UINT> equivalents;
	vector<extValType> extinctionValues;
	UINT labelNbr, basinNbr;
	T currentLevel, currentPixVal;
	size_t currentOffset;
	
  protected:
	HierarchicalQueue < T > hq;
	
	
	virtual void insertPixel(const UINT &lbl) {}
	virtual void raiseLevel(const UINT &lbl) {}
	virtual UINT mergeBasins(const UINT &lbl1, const UINT &lbl2) {};
	virtual void finalize(const UINT &lbl) {}
	
	virtual void createBasins(const UINT nbr)
	{
	    equivalents.resize(nbr);
	    extinctionValues.resize(nbr, 0);
	    
	    for (UINT i=0;i<nbr;i++)
	      equivalents[i] = i;
	    
	    basinNbr = nbr;
	}
	virtual void deleteBasins()
	{
	    equivalents.clear();
	    extinctionValues.clear();
	    
	    basinNbr = 0;
	}
	

      public:
	
	template <class labelT>
	RES_T flood(const Image<T> &imIn, const Image<labelT> &imMarkers, Image<labelT> &imBasinsOut, Graph<labelT,extValType> *graph, const StrElt & se=DEFAULT_SE)
	{
	    ASSERT_ALLOCATED (&imIn, &imMarkers, &imBasinsOut);
	    ASSERT_SAME_SIZE (&imIn, &imMarkers, &imBasinsOut);
	    
	    ImageFreezer freezer (imBasinsOut);

	    copy (imMarkers, imBasinsOut);

	    labelNbr = maxVal (imMarkers);
	    
	    if (basinNbr!=0)
	      deleteBasins();
	    
	    createBasins(labelNbr + 1);

	    initHierarchicalQueue(imIn, imBasinsOut);
	    processHierarchicalQueue(imIn, imBasinsOut, se, graph);
	    
	    return RES_OK;
	}
	template <class labelT>
	RES_T flood(const Image<T> &imIn, const Image<labelT> &imMarkers, Image<labelT> &imBasinsOut, const StrElt & se=DEFAULT_SE)
	{
	    Graph<labelT,extValType> *nullGraph = NULL;
	    return flood(imIn, imMarkers, imBasinsOut, nullGraph, se);
	}
	
	template <class labelT, class outT>
	RES_T floodWithExtValues(const Image<T> &imIn, const Image<labelT> &imMarkers, Image<outT> &imExtValOut, Image<labelT> &imBasinsOut, const StrElt & se=DEFAULT_SE)
	{
	    ASSERT_ALLOCATED (&imExtValOut);
	    ASSERT_SAME_SIZE (&imIn, &imExtValOut);
	    
	    ASSERT(flood(imIn, imMarkers, imBasinsOut, se)==RES_OK);
	    
	    ImageFreezer freezer (imExtValOut);
	    
	    typename ImDtTypes < outT >::lineType pixOut = imExtValOut.getPixels ();
	    typename ImDtTypes < labelT >::lineType pixMarkers = imMarkers.getPixels ();

	    fill(imExtValOut, outT(0));
	    
	    for (size_t i=0; i<imIn.getPixelCount (); i++, *pixMarkers++, pixOut++) 
	    {
		if(*pixMarkers != labelT(0))
		  *pixOut = extinctionValues[*pixMarkers] ;
	    }

	    return RES_OK;
	}
	template <class labelT, class outT>
	RES_T floodWithExtValues(const Image<T> &imIn, const Image<labelT> &imMarkers, Image<outT> &imExtValOut, const StrElt & se=DEFAULT_SE)
	{
	    Image<labelT> imBasinsOut(imMarkers);
	    return floodWithExtValues(imIn, imMarkers, imExtValOut, imBasinsOut, se);
	}
	
	template <class labelT, class outT>
	RES_T floodWithExtRank(const Image<T> &imIn, const Image<labelT> &imMarkers, Image<outT> &imExtRankOut, Image<labelT> &imBasinsOut, const StrElt & se=DEFAULT_SE)
	{
	    ASSERT_ALLOCATED (&imExtRankOut);
	    ASSERT_SAME_SIZE (&imIn, &imExtRankOut);
	    
	    ASSERT(flood(imIn, imMarkers, imBasinsOut, se)==RES_OK);

	    typename ImDtTypes < outT >::lineType pixOut = imExtRankOut.getPixels ();
	    typename ImDtTypes < labelT >::lineType pixMarkers = imMarkers.getPixels ();

	    ImageFreezer freezer (imExtRankOut);
	    
	    // Sort by extinctionValues
	    vector<UINT> rank(labelNbr);
	    for (UINT i=0;i<labelNbr;i++)
	      rank[i] = i+1;
	    
	    extinctionValuesComp<T,extValType> comp(this);
	    sort(rank.begin(), rank.end(), comp);
	    for (UINT i=0;i<labelNbr;i++)
	      extinctionValues[rank[i]] = i+1;
	    
	    fill(imExtRankOut, outT(0));
	    
	    for (size_t i=0; i<imIn.getPixelCount (); i++, *pixMarkers++, pixOut++) 
	    {
		if(*pixMarkers != labelT(0))
		  *pixOut = extinctionValues[*pixMarkers] ;
	    }

	    return RES_OK;
	}
	template <class labelT, class outT>
	RES_T floodWithExtRank(const Image<T> &imIn, const Image<labelT> &imMarkers, Image<outT> &imExtRankOut, const StrElt & se=DEFAULT_SE)
	{
	    Image<labelT> imBasinsOut(imMarkers);
	    return floodWithExtRank(imIn, imMarkers, imExtRankOut, imBasinsOut, se);
	}

	
      protected:
	
	template <class labelT>
	void initHierarchicalQueue(const Image<T> &imIn, Image<labelT> &imLbl)
	{
	    // Empty the priority queue
	    hq.initialize (imIn);
	    typename ImDtTypes < T >::lineType inPixels = imIn.getPixels ();
	    typename ImDtTypes < labelT >::lineType lblPixels = imLbl.getPixels ();
	    size_t s[3];

	    imIn.getSize (s);
	    currentOffset = 0;

	    for (size_t k = 0; k < s[2]; k++)
		for (size_t j = 0; j < s[1]; j++)
		    for (size_t i = 0; i < s[0]; i++) {
			if (*lblPixels != 0) {
			    currentPixVal = *inPixels;
			    hq.push (T (*inPixels), currentOffset);
			    insertPixel(*lblPixels);
			}

			inPixels++;
			lblPixels++;
			currentOffset++;
		    }
	}
	
	template < class labelT >
	RES_T processHierarchicalQueue (const Image < T > &imIn, 
					Image < labelT > &imLbl,
					const StrElt & se, Graph<labelT,extValType> *graph=NULL)
	{
	    typename ImDtTypes < T >::lineType inPixels = imIn.getPixels ();
	    typename ImDtTypes < labelT >::lineType lblPixels = imLbl.getPixels ();
	    vector < int >dOffsets;

	    vector < IntPoint >::const_iterator it_start = se.points.begin ();
	    vector < IntPoint >::const_iterator it_end = se.points.end ();
	    vector < IntPoint >::const_iterator it;
	    vector < UINT > tmpOffsets;
	    size_t s[3];

	    imIn.getSize (s);

	    currentLevel = 0;
	    currentOffset = 0;

	    // set an offset distance for each se point
	    for (it = it_start; it != it_end; it++) 
	    {
		dOffsets.push_back (it->x + it->y * s[0] + it->z * s[0] * s[1]);
	    }
	    vector < int >::iterator it_off_start = dOffsets.begin ();
	    vector < int >::iterator it_off;

	    while (!hq.isEmpty ()) 
	    {
		currentOffset = hq.pop ();
		currentPixVal = inPixels[currentOffset];

		// Rising the elevation of the basins
		if (currentPixVal > currentLevel) 
		{
		    currentLevel = currentPixVal;
		    for (labelT i = 1; i < labelNbr + 1 ; ++i) 
			    raiseLevel(i);
		}

		size_t x0, y0, z0;

		imIn.getCoordsFromOffset (currentOffset, x0, y0, z0);
		bool oddLine = se.odd && ((y0) % 2);
		int x, y, z;
		size_t nbOffset;
		UINT l1 = lblPixels[currentOffset], l2;

		for (it = it_start, it_off = it_off_start; it != it_end; it++, it_off++)
		    if (it->x != 0 || it->y != 0 || it->z != 0)	// useless if x=0 & y=0 & z=0
		    {
			x = x0 + it->x;
			y = y0 + it->y;
			z = z0 + it->z;
			if (oddLine)
			    x += (((y + 1) % 2) != 0);
			if (x >= 0 && x < (int) s[0] && y >= 0 && y < (int) s[1]
			    && z >= 0 && z < (int) s[2]) 
			{
			    nbOffset = currentOffset + *it_off;
			    if (oddLine)
				nbOffset += (((y + 1) % 2) != 0);
			    l2 = lblPixels[nbOffset];
			    if (l2 > 0 && l2 != labelNbr + 1 ) 
			    {
				while (l2 != equivalents[l2]) 
				{
				    l2 = equivalents[l2];
				}

				if (l1 == 0 || l1 == labelNbr + 1 ) 
				{
				    l1 = l2; // current pixel takes the label of its first labelled ngb  found
				    lblPixels[currentOffset] = l1;
				    insertPixel(l1);
				}
				else if (l1 != l2) 
				{
				    while (l1 != equivalents[l1]) 
				    {
					l1 = equivalents[l1];
				    }
				    if (l1 != l2)
				    {
					// mergeBasins basins
					UINT eater = mergeBasins(l1, l2);
					UINT eaten = (eater==l1) ? l2 : l1;
					
					if (graph)
					  graph->addEdge(eaten, eater, extinctionValues[eaten]);
					
					if (eater==l2)
					  l1 = l2;
				    }
					  
				}
			    }
			    else if (l2 == 0) 	// Add it to the tmp offsets queue
			    {
				tmpOffsets.push_back (nbOffset);
				lblPixels[nbOffset] = labelNbr + 1 ;
			    }
			}
		    }
		if (!tmpOffsets.empty ()) 
		{
		    typename vector < UINT >::iterator t_it = tmpOffsets.begin ();

		    while (t_it != tmpOffsets.end ()) 
		    {
			hq.push (inPixels[*t_it], *t_it);
			t_it++;
		    }
		}
		tmpOffsets.clear ();
	    }

	    // Update Last level of flooding.
	    finalize(lblPixels[currentOffset]);

	    return RES_OK;
	}
	
    };


    template <class T, class extValType=UINT>
    struct AreaFlooding : public Flooding<T,extValType>
    {
	vector<UINT> areas;
	vector<T> minValues;
	
	virtual void createBasins(const UINT nbr)
	{
	    areas.resize(nbr, 0);
	    minValues.resize(nbr, ImDtTypes<T>::max());
	    
	    Flooding<T,extValType>::createBasins(nbr);
	}
	
	virtual void deleteBasins()
	{
	    areas.clear();
	    minValues.clear();
	    
	    Flooding<T,extValType>::deleteBasins();
	}
	
	
	virtual void insertPixel(const UINT &lbl)
	{
	    if (this->currentPixVal < minValues[lbl])
	      minValues[lbl] = this->currentPixVal;
	    
	    areas[lbl]++;
	}
	virtual UINT mergeBasins(const UINT &lbl1, const UINT &lbl2)
	{
	    UINT eater, eaten;
	    
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
	    this->equivalents[eaten] = eater;
	    
	    return eater;
	}
	virtual void finalize(const UINT &lbl)
	{
	    this->extinctionValues[lbl] += areas[lbl];
	}
	
    };


    template <class T, class extValType=UINT>
    struct VolumeFlooding : public Flooding<T,extValType>
    {
	vector<UINT> areas, volumes;
	vector<T> floodLevels;
	
	virtual void createBasins(const UINT nbr)
	{
	    areas.resize(nbr, 0);
	    volumes.resize(nbr, 0);
	    floodLevels.resize(nbr, 0);
	    
	    Flooding<T,extValType>::createBasins(nbr);
	}
	
	virtual void deleteBasins()
	{
	    areas.clear();
	    volumes.clear();
	    floodLevels.clear();
	    
	    Flooding<T,extValType>::deleteBasins();
	}
	
	
	virtual void insertPixel(const UINT &lbl)
	{
	    if (floodLevels[lbl]!=this->currentPixVal)
	      floodLevels[lbl] = this->currentPixVal;
	    areas[lbl]++;
	}
	virtual void raiseLevel(const UINT &lbl)
	{
	    if (floodLevels[lbl] < this->currentLevel) 
	    {
		volumes[lbl] += areas[lbl] * (this->currentLevel - floodLevels[lbl]);
		floodLevels[lbl] = this->currentLevel;
	    }
	}
	virtual UINT mergeBasins(const UINT &lbl1, const UINT &lbl2)
	{
	    UINT eater, eaten;
	    
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
	    this->equivalents[eaten] = eater;
	    
	    return eater;
	}
	virtual void finalize(const UINT &lbl)
	{
	    volumes[lbl] += areas[lbl];
	    this->extinctionValues[lbl] += volumes[lbl];
	}
	
    };

    template <class T, class extValType=UINT>
    struct DynamicFlooding : public AreaFlooding<T,extValType>
    {
	virtual UINT mergeBasins(const UINT &lbl1, const UINT &lbl2)
	{
	    UINT eater, eaten;
	    
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
	    this->equivalents[eaten] = eater;
	    
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
	    Flooding<T, UINT> *flooding = NULL; // outT type may be smaller than the required flooding type
	    
	    if(strcmp(extinctionType, "v")==0)
		flooding = new VolumeFlooding<T, UINT>();
	    else if(strcmp(extinctionType, "a")==0)
		flooding = new AreaFlooding<T, UINT>();
	    else if(strcmp(extinctionType, "d")==0)
		flooding = new DynamicFlooding<T, UINT>();
	    
	    if (flooding)
	    {
	      flooding->floodWithExtRank(imIn, imMarkers, imOut, imBasinsOut, se);
	      delete flooding;
	    }
	}
	
	else
	{
	    Flooding<T, outT> *flooding = NULL; // outT type may be smaller than the required flooding type
	    
	    if(strcmp(extinctionType, "v")==0)
		flooding = new VolumeFlooding<T, outT>();
	    else if(strcmp(extinctionType, "a")==0)
		flooding = new AreaFlooding<T, outT>();
	    else if(strcmp(extinctionType, "d")==0)
		flooding = new DynamicFlooding<T, outT>();
	    
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
	
	Flooding<T, outT> *flooding = NULL; // outT type may be smaller than the required flooding type
	
	if(strcmp(extinctionType, "v")==0)
	    flooding = new VolumeFlooding<T, outT>();
	else if(strcmp(extinctionType, "a")==0)
	    flooding = new AreaFlooding<T, outT>();
	else if(strcmp(extinctionType, "d")==0)
	    flooding = new DynamicFlooding<T, outT>();
	else return RES_ERR;
	
	RES_T res = flooding->flood(imIn, imMarkers, imBasinsOut, &graph, se);
	
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
    
    template < class T,	class labelT> 
    Graph<labelT,UINT> watershedExtinctionGraph (const Image < T > &imIn,
				    const Image < labelT > &imMarkers,
				    Image < labelT > &imBasinsOut,
				    const char *  	extinctionType="v", 
				    const StrElt & se = DEFAULT_SE) 
    {
	Graph<labelT,UINT> graph;
	ASSERT(watershedExtinctionGraph(imIn, imMarkers, imBasinsOut, graph, extinctionType, se)==RES_OK, graph);
	return graph;
    }
    template < class T,	class labelT> 
    Graph<labelT,UINT> watershedExtinctionGraph (const Image < T > &imIn,
				    Image < labelT > &imBasinsOut,
				    const char *  	extinctionType="v", 
				    const StrElt & se = DEFAULT_SE) 
    {
	Graph<labelT,UINT> graph;
	ASSERT(watershedExtinctionGraph(imIn, imBasinsOut, graph, extinctionType, se)==RES_OK, graph);
	return graph;
    }
}				// namespace smil


#endif // _D_MORPHO_WATERSHED_EXTINCTION_HPP
