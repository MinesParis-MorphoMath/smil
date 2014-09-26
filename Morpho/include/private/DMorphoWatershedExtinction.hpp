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

    template < class labelT, class outT > 
    struct basin
    {
	basin()
	  : extinctionValue(0)
	{
	}
	
	size_t extinctionValue;
	labelT equivalent;
	virtual bool operator< (const basin<labelT,outT> &b2) const 
	{
		return extinctionValue < b2.extinctionValue; 
	}
	virtual void initialize(const labelT &pixVal) {}
	virtual void increment(size_t inc=1) {}
	virtual void raise(const size_t &pixValue)
	{
	}
	virtual void finalize() {}
    };

    template < class labelT, class outT > 
    struct areaBasin : public basin<labelT,outT>
    {
	areaBasin() 
	  : area(0), 
	    minPixVal(numeric_limits<size_t >::max())
	{
	}
	size_t area, minPixVal;
	virtual void increment(size_t inc=1) 
	{
	  this->area += inc;
	}
	void merge(areaBasin<labelT,outT> &other, labelT &l1, labelT &l2, const size_t &currentLevel, Graph<labelT,outT> *graph=NULL) 
	{
	    if (this->area > other.area /*|| (this->area == other.area && this->minValue < other.minValue)*/) 
	    {
		/* cur basin absorbs nb basin */
		other.extinctionValue = other.area;
		this->area += other.area;
		other.equivalent = l1;
	    }
	    else {
		/* nb basin absorbs cur basin */
		this->extinctionValue = this->area;
		other.area += this->area;
		this->equivalent = l2;
		l1 = l2;
	    }
	}
	virtual void finalize()
	{
	    this->extinctionValue += this->area;
	}
    };
    
    template < class labelT, class outT > 
    struct volBasin : public basin<labelT,outT>
    {
	volBasin() 
	  : area(0),
	    volume(0), 
	    floodLevel(0)
	{
	}
	size_t area, volume;
	size_t floodLevel;
	virtual void initialize(const labelT &pixVal)
	{
	    floodLevel = pixVal;
	}
	virtual void increment(size_t inc=1)
	{
	    area++;
	}
	virtual void raise(const size_t &pixValue) 
	{
	    if (floodLevel < pixValue) 
	    {
		this->volume += this->area * (pixValue - this->floodLevel);
		this->floodLevel = pixValue;
	    }
	}
	void merge(volBasin<labelT,outT> &other, labelT &l1, labelT &l2, const size_t &currentLevel, Graph<labelT,outT> *graph=NULL)
	{
	    if (volume > other.volume || (volume == other.volume && floodLevel < other.floodLevel)) 
	    {
		/* cur basin absorbs nb basin */
		other.extinctionValue = other.volume;
		volume += other.volume;
		area += other.area;
		other.equivalent = l1;
	    }
	    else {
		/* nb basin absorbs cur basin */
		this->extinctionValue = volume;
		other.volume += volume;
		other.area += area;
		this->equivalent = l2;
		l1 = l2;
	    }
	    
	}
	virtual void finalize()
	{
	    this->volume += area;
	    this->extinctionValue += this->volume;
	}
    };

    template < class labelT, class outT > 
    struct dynamicBasin : public basin<labelT,outT>
    {
	dynamicBasin() 
	  : area(0),
	    minPixVal(std::numeric_limits<size_t>::max())
	{
	}
	size_t minPixVal;
	size_t area;
	virtual void increment(size_t inc=1)
	{
	    area++;
	}
	virtual void raise(const size_t &pixValue)
	{
	    cout ;
	}
	virtual void initialize(const labelT &pixVal) 
	{
	    if (pixVal < minPixVal)
	      minPixVal = pixVal;
	}
	void merge(dynamicBasin<labelT,outT> &other, labelT &l1, labelT &l2, const size_t &currentLevel, Graph<labelT,outT> *graph=NULL) 
	{
	    if (minPixVal < other.minPixVal || (minPixVal == other.minPixVal && area > other.area)) 
	    {
		/* cur basin absorbs nb basin */
		other.extinctionValue = currentLevel - other.minPixVal;
		area += other.area;
		other.equivalent = l1;
	    }
	    else {
		/* nb basin absorbs cur basin */
		this->extinctionValue = currentLevel - minPixVal;
		other.area += area;
		this->equivalent = l2;
		l1 = l2;
	    }
	}
	virtual void finalize()
	{
	    this->extinctionValue = numeric_limits< size_t >::max();
	}
    };
    
      template < class labelT, class outT >
	void print_equivalences (basin < labelT,outT > *e, UINT nbr_lbl)
    {
	for (UINT i = 1; i < nbr_lbl + 1; ++i)
	{
	    cout << "LABEL " << i << endl;
	    cout << "  vol        : " << e[i].vol << endl;
	    cout << "  area       : " << e[i].area << endl;
	    cout << "  extinctionValue : " << (int) e[i].extinctionValue << endl;
	    cout << "  minValue : " << (int) e[i].minValue << endl;
	    cout << "  equivalent : " << (int) e[i].equivalent << endl;
	}
    }

    template < class T, class labelT, class HQ_Type, class basinT >
	RES_T initWatershedHierarchicalQueueExtinction (const Image < T > &imIn,
							Image < labelT > &imLbl,
							HQ_Type & hq,
							basinT *e, 
							UINT nbr_label)
    {

#pragma omp parallel for
	for (UINT i = 0; i < nbr_label + 1; ++i)
	  e[i].equivalent = i;

	// Empty the priority queue
	hq.initialize (imIn);
	typename ImDtTypes < T >::lineType inPixels = imIn.getPixels ();
	typename ImDtTypes < labelT >::lineType lblPixels =
	    imLbl.getPixels ();
	size_t s[3];

	imIn.getSize (s);
	size_t offset = 0;

	for (size_t k = 0; k < s[2]; k++)
	    for (size_t j = 0; j < s[1]; j++)
		for (size_t i = 0; i < s[0]; i++) {
		    if (*lblPixels != 0) {
			hq.push (T (*inPixels), offset);
			e[*lblPixels].initialize(*inPixels);  
			e[*lblPixels].increment(*inPixels);  
// 			e[*lblPixels].area++;
// 			e[*lblPixels].minValue = *inPixels;
		    }

		    inPixels++;
		    lblPixels++;
		    offset++;
		}

	return RES_OK;
    }
    
    template < class T, class labelT, class HQ_Type, class outT, class basinT >
    RES_T processWatershedExtinctionHierarchicalQueue (const Image < T > &imIn,
							   Image < labelT > &imLbl,
							   HQ_Type & hq,
							   const StrElt & se,
							   basinT *e,
							   UINT nbr_label,
							   Graph<labelT, outT> *graph=NULL) 
    {
	typename ImDtTypes < T >::lineType inPixels = imIn.getPixels ();
	typename ImDtTypes < labelT >::lineType lblPixels =
	    imLbl.getPixels ();
	vector < int >dOffsets;

	vector < IntPoint >::const_iterator it_start = se.points.begin ();
	vector < IntPoint >::const_iterator it_end = se.points.end ();
	vector < IntPoint >::const_iterator it;
	vector < UINT > tmpOffsets;
	size_t s[3];

	imIn.getSize (s);

	T level_before = 0, current_level;
	size_t curOffset;

	// set an offset distance for each se point
	for (it = it_start; it != it_end; it++) 
	{
	    dOffsets.push_back (it->x + it->y * s[0] + it->z * s[0] * s[1]);
	}
	vector < int >::iterator it_off_start = dOffsets.begin ();
	vector < int >::iterator it_off;

	while (!hq.isEmpty ()) 
	{
	    curOffset = hq.pop ();
	    current_level = inPixels[curOffset];

	    // Rising the elevation of the basins
	    if (inPixels[curOffset] > level_before) 
	    {
		for (labelT i = 1; i < nbr_label + 1; ++i) 
		{
			e[i].raise(inPixels[curOffset]);
		}
		level_before = inPixels[curOffset];
	    }

	    size_t x0, y0, z0;

	    imIn.getCoordsFromOffset (curOffset, x0, y0, z0);
	    bool oddLine = se.odd && ((y0) % 2);
	    int x, y, z;
	    size_t nbOffset;
	    labelT l1 = lblPixels[curOffset], l2;

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
			nbOffset = curOffset + *it_off;
			if (oddLine)
			    nbOffset += (((y + 1) % 2) != 0);
			l2 = lblPixels[nbOffset];
			if (l2 > 0 && l2 != nbr_label + 1) 
			{
			    while (l2 != e[l2].equivalent) 
			    {
				l2 = e[l2].equivalent;
			    }

			    if (l1 == 0 || l1 == nbr_label + 1) 
			    {
				l1 = l2; // current pixel takes the label of its first labelled ngb  found
				lblPixels[curOffset] = l1;
				e[l1].increment();
			    }
			    else if (l1 != l2) 
			    {
				while (l1 != e[l1].equivalent) 
				{
				    l1 = e[l1].equivalent;
				}
				if (l1 != l2) 
				  e[l1].merge(e[l2], l1, l2, current_level, graph);
			    }
			}
			else if (l2 == 0) 	// Add it to the tmp offsets queue
			{
			    tmpOffsets.push_back (nbOffset);
			    lblPixels[nbOffset] = nbr_label + 1;
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
	e[lblPixels[curOffset]].finalize();

	return RES_OK;
    }

    /**
     * Watershed using volume extinction values
     *
     * \param[in] imIn Input image.
     * \param[in] imMarkers Label image containing the markers.
     * \param[out] imBasinsOut (optional) Output image containing the volumic extinction order.
     * \param[out] imBasinsOut (optional) Output image containing the basins.
     * After processing, this image will contains a value on each initial markers. This value is the rank of the volumic extinction of the corresponding basin among all the other values of volumic extinction
     */
    template < class T, class labelT, class outT, class basinT > 
    RES_T _generic_watershedExtinction (const Image < T > &imIn,
						const Image < labelT >
						&imMarkers,
						Image < outT > &imOut,
						Image < labelT > &imBasinsOut,
						const StrElt & se = DEFAULT_SE) 
    {
	ASSERT_ALLOCATED (&imIn, &imMarkers, &imOut, &imBasinsOut);
	ASSERT_SAME_SIZE (&imIn, &imMarkers, &imOut, &imBasinsOut);
	ImageFreezer freezer (imOut);
	ImageFreezer freezer2 (imBasinsOut);

	copy (imMarkers, imBasinsOut);

	labelT nbr_label = maxVal (imBasinsOut);

	basinT *e;
	e = new basinT[nbr_label + 1];

	HierarchicalQueue < T > pq;
	initWatershedHierarchicalQueueExtinction(imIn, imBasinsOut, pq, e, nbr_label);
	processWatershedExtinctionHierarchicalQueue<T,labelT,HierarchicalQueue<T>,outT,basinT> (imIn, imBasinsOut, pq, se, e, nbr_label);
	
	typename ImDtTypes < outT >::lineType pixOut = imOut.getPixels ();
	typename ImDtTypes < labelT >::lineType pixMarkers = imMarkers.getPixels ();

	fill (imOut, outT(0));
	// Create the image containing the ws lines
	T wsVal = ImDtTypes < outT >::max ();
	size_t max_ext=0;
	for (int i=0; i<nbr_label; ++i) {
		if (max_ext<e[i].extinctionValue) max_ext = e[i].extinctionValue ;
	}
	basinT *e_cpy = new basinT[nbr_label+1]; 
	memcpy (e_cpy, e, (nbr_label+1)*sizeof(basinT));
	for (int i=1; i<nbr_label+1; ++i)
		e_cpy[i].equivalent = i;
	vector<basinT> ve (e_cpy+1, e_cpy+nbr_label+1);
	// Sorting the vol_ev_val.
	sort (ve.begin(), ve.end());
	for (int i=0; i<nbr_label; ++i){
		e[ve[i].equivalent].extinctionValue = nbr_label - i;
	}
	delete[]e_cpy;
	for (size_t i=0; i<imIn.getPixelCount (); i++, *pixMarkers++, pixOut++) {
		if(*pixMarkers != labelT(0))
			*pixOut = e[*pixMarkers].extinctionValue ;
	}

	delete[]e;
	return RES_OK;
    }
    
    template < class T, class labelT, class outT > 
    RES_T watershedExtinction (const Image < T > &imIn,
						const Image < labelT > &imMarkers,
						Image < outT > &imOut,
						Image < labelT > &imBasinsOut,
						const char *  	extinctionType="v", 
						const StrElt & se = DEFAULT_SE) 
    {
	if(strcmp(extinctionType, "v")==0)
	{
	    return _generic_watershedExtinction<T, labelT, outT, volBasin<labelT,outT> >(imIn, imMarkers, imOut, imBasinsOut, se);
	}
	else if(strcmp(extinctionType, "a")==0)
	{
	    return _generic_watershedExtinction<T, labelT, outT, areaBasin<labelT,outT> >(imIn, imMarkers, imOut, imBasinsOut, se);
	}
	else if(strcmp(extinctionType, "d")==0)
	{
	    return _generic_watershedExtinction<T, labelT, outT, dynamicBasin<labelT,outT> >(imIn, imMarkers, imOut, imBasinsOut, se);
	}
	else{
	  return RES_ERR;
	}
    }
    
    template < class T,
	class labelT,
	class outT > RES_T watershedExtinction (const Image < T > &imIn,
						Image < labelT > &imMarkers,
						Image < outT > &imOut,
						const char *  	extinctionType="v", 
						const StrElt & se = DEFAULT_SE) 
    {
	ASSERT_ALLOCATED (&imIn, &imMarkers, &imOut);
	ASSERT_SAME_SIZE (&imIn, &imMarkers, &imOut);
	Image < labelT > imBasinsOut (imMarkers);
	return watershedExtinction (imIn, imMarkers, imOut, imBasinsOut, extinctionType, se);
    }
    
    template < class T,
	class outT > RES_T watershedExtinction (const Image < T > &imIn,
						Image < outT > &imOut,
						const char *  	extinctionType="v", 
						const StrElt & se = DEFAULT_SE) 
    {
	ASSERT_ALLOCATED (&imIn, &imOut);
	ASSERT_SAME_SIZE (&imIn, &imOut);
	Image < T > imMin (imIn);
	minima (imIn, imMin, se);
	Image < UINT > imLbl (imIn);
	label (imMin, imLbl, se);
	return watershedExtinction (imIn, imLbl, imOut,extinctionType,  se);
    }
    
    template < class T, class labelT, class outT, class HQ_Type >
    RES_T processWatershedExtinctionHierarchicalQueue (const Image < T > &imIn,
							   Image < labelT > &imLbl,
							   HQ_Type & hq,
							   const StrElt & se,
							   volBasin < labelT,outT > *e,
							   UINT nbr_label,
							   Graph < labelT, outT> &graph)
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

	T level_before = 0;
	size_t curOffset;

	// set an offset distance for each se point
	for (it = it_start; it != it_end; it++) {
	    dOffsets.push_back (it->x + it->y * s[0] + it->z * s[0] * s[1]);
	}
	vector < int >::iterator it_off_start = dOffsets.begin ();
	vector < int >::iterator it_off;

	while (!hq.isEmpty ()) {
	    curOffset = hq.pop ();

	    // Rising the elevation of the basins
// 	    if (inPixels[curOffset] > level_before) {
// 		for (labelT i = 1; i < nbr_label + 1; ++i) {
// 		    if (e[i].minValue < inPixels[curOffset]) {
// 			e[i].volume +=
// 			    e[i].area * (inPixels[curOffset] -
// 					 e[i].minValue);
// 			e[i].minValue = inPixels[curOffset];
// 		    }
// 		}
// 		level_before = inPixels[curOffset];
// 	    }

	    size_t x0, y0, z0;

	    imIn.getCoordsFromOffset (curOffset, x0, y0, z0);
	    bool oddLine = se.odd && ((y0) % 2);
	    int x, y, z;
	    size_t nbOffset;
	    labelT l1 = lblPixels[curOffset], l2;

	    for (it = it_start, it_off = it_off_start; it != it_end;
		 it++, it_off++)
		if (it->x != 0 || it->y != 0 || it->z != 0)	// useless if x=0 & y=0 & z=0
		{
		    x = x0 + it->x;
		    y = y0 + it->y;
		    z = z0 + it->z;
		    if (oddLine)
			x += (((y + 1) % 2) != 0);
		    if (x >= 0 && x < (int) s[0] && y >= 0 && y < (int) s[1]
			&& z >= 0 && z < (int) s[2]) {
			nbOffset = curOffset + *it_off;
			if (oddLine)
			    nbOffset += (((y + 1) % 2) != 0);
			l2 = lblPixels[nbOffset];
			if (l2 > 0 && l2 != nbr_label + 1) {
			    while (l2 != e[l2].equivalent) {
				l2 = e[l2].equivalent;
			    }

			    if (l1 == 0 || l1 == nbr_label + 1) {
				l1 = l2;
				lblPixels[curOffset] = l1;
				e[l1].area++;
			    }
			    else if (l1 != l2) {

				while (l1 != e[l1].equivalent) {
				    l1 = e[l1].equivalent;
				}
// 				if (l1 != l2) {
// 				    if (e[l1].volume > e[l2].volume
// 					|| (e[l1].volume == e[l2].volume
// 					    && e[l1].minValue <
// 					    e[l2].minValue)) {
// 					/* cur basin absorbs nb basin */
// 					e[l2].extinctionValue = e[l2].volume;
// 				    	graph.addEdge (l2, l1, e[l2].extinctionValue) ;
// 					e[l1].volume += e[l2].volume;
// 					e[l1].area += e[l2].area;
// 					e[l2].equivalent = l1;
// 				    }
// 				    else {
// 					/* nb basin absorbs cur basin */
// 					e[l1].extinctionValue = e[l1].volume;
// 				    	graph.addEdge (l1, l2, e[l1].extinctionValue) ;
// 					e[l2].volume += e[l1].volume;
// 					e[l2].area += e[l1].area;
// 					e[l1].equivalent = l2;
// 					l1 = l2;
// 				    }
// 				}
			    }
			}
			else if (l2 == 0) {	// Add it to the tmp offsets queue
			    tmpOffsets.push_back (nbOffset);
			    lblPixels[nbOffset] = nbr_label + 1;
			}
		    }
		}
	    if (!tmpOffsets.empty ()) {
		typename vector < UINT >::iterator t_it = tmpOffsets.begin ();

		while (t_it != tmpOffsets.end ()) {
		    hq.push (inPixels[*t_it], *t_it);
		    t_it++;
		}
	    }
	    tmpOffsets.clear ();
	}

	// Update Last level of flooding.
//	e[lblPixels[curOffset]].volume += e[lblPixels[curOffset]].area;
	e[lblPixels[curOffset]].extinctionValue = e[lblPixels[curOffset]].volume;

	return RES_OK;
    }

    /**
     * Calculation of the minimum spanning tree, simultaneously to the image flooding, with edges weighted according to volume extinction values.
     */
    template < class T,	class labelT, class outT > 
    RES_T watershedExtinctionGraph (const Image < T > &imIn,
						const Image < labelT > &imMarkers,
						Image < outT > &imOut,
						Image < labelT > &imBasinsOut,
						Graph < labelT, outT > &graph,
						const StrElt & se = DEFAULT_SE) 
    {
	ASSERT (imIn.isAllocated() && imMarkers.isAllocated() && imOut.isAllocated() && imBasinsOut.isAllocated());
	ASSERT_SAME_SIZE (&imIn, &imMarkers, &imOut, &imBasinsOut);
	ImageFreezer freezer (imOut);
	ImageFreezer freezer2 (imBasinsOut);

	copy (imMarkers, imBasinsOut);

	labelT nbr_label = maxVal (imBasinsOut);

	volBasin < labelT,outT > *e;
	e = new volBasin < labelT,outT >[nbr_label + 1];

	HierarchicalQueue < T > pq;

	initWatershedHierarchicalQueueExtinction (imIn, imBasinsOut,
						  pq, e, nbr_label);
	processWatershedExtinctionHierarchicalQueue (imIn, imBasinsOut,
						     pq, se, e, nbr_label, graph);

	typename ImDtTypes < outT >::lineType pixOut = imOut.getPixels ();
	typename ImDtTypes < labelT >::lineType pixMarkers = imMarkers.getPixels ();

	fill (imOut, outT(0));
	// Create the image containing the ws lines
	T wsVal = ImDtTypes < outT >::max ();
	size_t max_ext=0;
	for (int i=0; i<nbr_label; ++i) {
		if (max_ext<e[i].extinctionValue) max_ext = e[i].extinctionValue ;
	}
	basin <labelT,outT> *e_cpy = new basin<labelT,outT>[nbr_label+1]; 
	memcpy (e_cpy, e, nbr_label*sizeof(basin<labelT,outT>));
	for (int i=1; i<nbr_label+1; ++i)
		e_cpy[i].equivalent = i;
	vector<basin<labelT,outT> > ve (e_cpy+1, e_cpy+nbr_label+1);
	// Sorting the vol_ev_val.
	sort (ve.begin(), ve.end());
	for (int i=0; i<nbr_label; ++i){
		e[ve[i].equivalent].extinctionValue = nbr_label - i;
	}
	delete[]e_cpy;

	for (size_t i=0; i<imIn.getPixelCount (); i++, *pixMarkers++, pixOut++) {
		if(*pixMarkers != labelT(0))
			*pixOut = e[*pixMarkers].extinctionValue ;
	}

	delete[]e;
	return RES_OK;
    }
#ifndef SWIG    
    template < class T,
	class labelT,
	class outT > RES_T watershedExtinctionGraph (const Image < T > &imIn,
						Image < labelT > &imMarkers,
						Image < outT > &imOut,
						Graph < labelT, outT > &graph,
						const StrElt & se =
						DEFAULT_SE) {
	ASSERT_ALLOCATED (&imIn, &imMarkers, &imOut);
	ASSERT_SAME_SIZE (&imIn, &imMarkers, &imOut);
	Image < labelT > imBasinsOut (imMarkers);
	return watershedExtinctionGraph (imIn, imMarkers, imOut, imBasinsOut, graph, se);
    }
    template < class T,
	class outT > RES_T watershedExtinctionGraph (const Image < T > &imIn,
						Image < outT > &imOut,
						Graph < UINT, outT > &graph,
						const StrElt & se =
						DEFAULT_SE) {
	ASSERT_ALLOCATED (&imIn, &imOut);
	ASSERT_SAME_SIZE (&imIn, &imOut);
	Image < T > imMin (imIn);
	minima (imIn, imMin, se);
	Image < UINT > imLbl (imIn);
	label (imMin, imLbl, se);
	return watershedExtinctionGraph (imIn, imLbl, imOut, graph, se);
    }
#endif // SWIG    
    template < class T,
	class outT > Graph < UINT, outT > watershedExtinctionGraph (const Image < T > &imIn,
						Image < outT > &imOut,
						const StrElt & se =
						DEFAULT_SE) {

	Graph < UINT, outT > graph;
	Image < T > imMin (imIn);
	minima (imIn, imMin, se);
	Image < UINT > imLbl (imIn);
	label (imMin, imLbl, se);
	watershedExtinctionGraph (imIn, imLbl, imOut, graph, se);
	return graph;
    }

}				// namespace smil


#endif // _D_MORPHO_WATERSHED_EXTINCTION_HPP
