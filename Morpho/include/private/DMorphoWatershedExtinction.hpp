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

    template < class labelT > struct basin
    {
	size_t vol;
	size_t area;
	size_t val_of_min;
	size_t vol_ex_val;
	labelT equivalent;
	bool operator< (const basin<labelT> b2) const {
		return vol_ex_val < b2.vol_ex_val; 
	}
    };

      template < class labelT >
	void print_equivalences (basin < labelT > *e, UINT nbr_lbl)
    {
	for (UINT i = 1; i < nbr_lbl + 1; ++i)
	{
	    cout << "LABEL " << i << endl;
	    cout << "  vol        : " << e[i].vol << endl;
	    cout << "  area       : " << e[i].area << endl;
	    cout << "  vol_ex_val : " << (int) e[i].vol_ex_val << endl;
	    cout << "  val_of_min : " << (int) e[i].val_of_min << endl;
	    cout << "  equivalent : " << (int) e[i].equivalent << endl;
	}
    }

    template < class T, class labelT,
	class HQ_Type >
	RES_T initWatershedHierarchicalQueueExtinction (const Image < T >
							&imIn,
							Image < labelT >
							&imLbl,
							HQ_Type & hq,
							basin < labelT >
							*e, UINT nbr_label)
    {

#pragma omp parallel for
	for (UINT i = 0; i < nbr_label + 1; ++i) {
	    e[i].equivalent = i;
	    e[i].vol = 0;
	    e[i].area = 0;
	    e[i].vol_ex_val = 0;
	}

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
			++e[*lblPixels].area;
			e[*lblPixels].val_of_min = *inPixels;
		    }

		    inPixels++;
		    lblPixels++;
		    offset++;
		}

	return RES_OK;
    }
    template < class T, class labelT,
	class HQ_Type >
	RES_T processWatershedHierarchicalQueueExtinction (const Image < T >
							   &imIn,
							   Image < labelT >
							   &imLbl,
							   HQ_Type & hq,
							   const StrElt & se,
							   basin < labelT >
							   *e,
							   UINT nbr_label) {
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
	    if (inPixels[curOffset] > level_before) {
		for (labelT i = 1; i < nbr_label + 1; ++i) {
		    if (e[i].val_of_min < inPixels[curOffset]) {
			e[i].vol +=
			    e[i].area * (inPixels[curOffset] -
					 e[i].val_of_min);
			e[i].val_of_min = inPixels[curOffset];
		    }
		}
		level_before = inPixels[curOffset];
	    }

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
				if (l1 != l2) {
				    if (e[l1].vol > e[l2].vol
					|| (e[l1].vol == e[l2].vol
					    && e[l1].val_of_min <
					    e[l2].val_of_min)) {
					/* cur basin absorbs nb basin */
					e[l2].vol_ex_val = e[l2].vol;
					e[l1].vol += e[l2].vol;
					e[l1].area += e[l2].area;
					e[l2].equivalent = l1;
				    }
				    else {
					/* nb basin absorbs cur basin */
					e[l1].vol_ex_val = e[l1].vol;
					e[l2].vol += e[l1].vol;
					e[l2].area += e[l1].area;
					e[l1].equivalent = l2;
					l1 = l2;
				    }
				}
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
	e[lblPixels[curOffset]].vol += e[lblPixels[curOffset]].area;
	e[lblPixels[curOffset]].vol_ex_val += e[lblPixels[curOffset]].vol;

	return RES_OK;
    }
    template < class T,
	class labelT,
	class outT > RES_T watershedExtinction (const Image < T > &imIn,
						const Image < labelT >
						&imMarkers,
						Image < outT > &imOut,
						Image < labelT > &imBasinsOut,
						const StrElt & se =
						DEFAULT_SE) {
	ASSERT_ALLOCATED (&imIn, &imMarkers, &imOut, &imBasinsOut);
	ASSERT_SAME_SIZE (&imIn, &imMarkers, &imOut, &imBasinsOut);
	ImageFreezer freezer (imOut);
	ImageFreezer freezer2 (imBasinsOut);

	copy (imMarkers, imBasinsOut);

	labelT nbr_label = maxVal (imBasinsOut);

	basin < labelT > *e;
	e = new basin < labelT >[nbr_label + 1];

	HierarchicalQueue < T > pq;
	initWatershedHierarchicalQueueExtinction (imIn, imBasinsOut,
						  pq, e, nbr_label);
	processWatershedHierarchicalQueueExtinction (imIn, imBasinsOut,
						     pq, se, e, nbr_label);

	typename ImDtTypes < outT >::lineType pixOut = imOut.getPixels ();
	typename ImDtTypes < labelT >::lineType pixMarkers = imMarkers.getPixels ();

	fill (imOut, outT(0));
	// Create the image containing the ws lines
	T wsVal = ImDtTypes < outT >::max ();
	size_t max_ext=0;
	for (int i=0; i<nbr_label; ++i) {
		if (max_ext<e[i].vol_ex_val) max_ext = e[i].vol_ex_val ;
	}
	if (max_ext > wsVal) {
		cout << "WARNING : Extinction value is too high for the output type. Ranking the Extinction values." << endl;
		basin <labelT> *e_cpy = new basin<labelT>[nbr_label+1]; 
		memcpy (e_cpy, e, nbr_label*sizeof(basin<labelT>));
		for (int i=1; i<nbr_label+1; ++i)
			e_cpy[i].equivalent = i;
		vector<basin<labelT> > ve (e_cpy+1, e_cpy+nbr_label+1);
		// Sorting the vol_ev_val.
		sort (ve.begin(), ve.end());
		for (int i=0; i<nbr_label; ++i){
			e[ve[i].equivalent].vol_ex_val = nbr_label - i;
		}
		delete[]e_cpy;
	}
	for (size_t i=0; i<imIn.getPixelCount (); i++, *pixMarkers++, pixOut++) {
		if(*pixMarkers != labelT(0))
			*pixOut = e[*pixMarkers].vol_ex_val ;
	}

	delete[]e;
	return RES_OK;
    }
    template < class T,
	class labelT,
	class outT > RES_T watershedExtinction (const Image < T > &imIn,
						Image < labelT > &imMarkers,
						Image < outT > &imOut,
						const StrElt & se =
						DEFAULT_SE) {
	ASSERT_ALLOCATED (&imIn, &imMarkers, &imOut);
	ASSERT_SAME_SIZE (&imIn, &imMarkers, &imOut);
	Image < labelT > imBasinsOut (imMarkers);
	return watershedExtinction (imIn, imMarkers, imOut, imBasinsOut, se);
    }
    template < class T,
	class outT > RES_T watershedExtinction (const Image < T > &imIn,
						Image < outT > &imOut,
						const StrElt & se =
						DEFAULT_SE) {
	ASSERT_ALLOCATED (&imIn, &imOut);
	ASSERT_SAME_SIZE (&imIn, &imOut);
	Image < T > imMin (imIn);
	minima (imIn, imMin, se);
	Image < UINT > imLbl (imIn);
	label (imMin, imLbl, se);
	return watershedExtinction (imIn, imLbl, imOut, se);
    }
    template < class T, class labelT, class outT,
	class HQ_Type >
	RES_T processWatershedHierarchicalQueueExtinction (const Image < T >
							   &imIn,
							   Image < labelT >
							   &imLbl,
							   HQ_Type & hq,
							   const StrElt & se,
							   basin < labelT >
							   *e,
							   UINT nbr_label,
							   Graph < labelT, outT> &graph){
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
	    if (inPixels[curOffset] > level_before) {
		for (labelT i = 1; i < nbr_label + 1; ++i) {
		    if (e[i].val_of_min < inPixels[curOffset]) {
			e[i].vol +=
			    e[i].area * (inPixels[curOffset] -
					 e[i].val_of_min);
			e[i].val_of_min = inPixels[curOffset];
		    }
		}
		level_before = inPixels[curOffset];
	    }

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
				if (l1 != l2) {
				    if (e[l1].vol > e[l2].vol
					|| (e[l1].vol == e[l2].vol
					    && e[l1].val_of_min <
					    e[l2].val_of_min)) {
					/* cur basin absorbs nb basin */
					e[l2].vol_ex_val = e[l2].vol;
				    	graph.addEdge (l2, l1, e[l2].vol_ex_val) ;
					e[l1].vol += e[l2].vol;
					e[l1].area += e[l2].area;
					e[l2].equivalent = l1;
				    }
				    else {
					/* nb basin absorbs cur basin */
					e[l1].vol_ex_val = e[l1].vol;
				    	graph.addEdge (l1, l2, e[l1].vol_ex_val) ;
					e[l2].vol += e[l1].vol;
					e[l2].area += e[l1].area;
					e[l1].equivalent = l2;
					l1 = l2;
				    }
				}
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
//	e[lblPixels[curOffset]].vol += e[lblPixels[curOffset]].area;
	e[lblPixels[curOffset]].vol_ex_val = e[lblPixels[curOffset]].vol;

	return RES_OK;
    }
 
    template < class T,
	class labelT,
	class outT > RES_T watershedExtinctionGraph (const Image < T > &imIn,
						const Image < labelT >
						&imMarkers,
						Image < outT > &imOut,
						Image < labelT > &imBasinsOut,
						Graph < labelT, outT > &graph,
						const StrElt & se =
						DEFAULT_SE) {
	ASSERT (imIn.isAllocated() && imMarkers.isAllocated() && imOut.isAllocated() && imBasinsOut.isAllocated());
	ASSERT_SAME_SIZE (&imIn, &imMarkers, &imOut, &imBasinsOut);
	ImageFreezer freezer (imOut);
	ImageFreezer freezer2 (imBasinsOut);

	copy (imMarkers, imBasinsOut);

	labelT nbr_label = maxVal (imBasinsOut);

	basin < labelT > *e;
	e = new basin < labelT >[nbr_label + 1];

	HierarchicalQueue < T > pq;

	initWatershedHierarchicalQueueExtinction (imIn, imBasinsOut,
						  pq, e, nbr_label);
	processWatershedHierarchicalQueueExtinction (imIn, imBasinsOut,
						     pq, se, e, nbr_label, graph);

	typename ImDtTypes < outT >::lineType pixOut = imOut.getPixels ();
	typename ImDtTypes < labelT >::lineType pixMarkers = imMarkers.getPixels ();

	fill (imOut, outT(0));
	// Create the image containing the ws lines
	T wsVal = ImDtTypes < outT >::max ();
	size_t max_ext=0;
	for (int i=0; i<nbr_label; ++i) {
		if (max_ext<e[i].vol_ex_val) max_ext = e[i].vol_ex_val ;
	}
	if (max_ext > wsVal) {
		cout << "WARNING : Extinction value is too high for the output type. Ranking the Extinction values." << endl;
		basin <labelT> *e_cpy = new basin<labelT>[nbr_label+1]; 
		memcpy (e_cpy, e, nbr_label*sizeof(basin<labelT>));
		for (int i=1; i<nbr_label+1; ++i)
			e_cpy[i].equivalent = i;
		vector<basin<labelT> > ve (e_cpy+1, e_cpy+nbr_label+1);
		// Sorting the vol_ev_val.
		sort (ve.begin(), ve.end());
		for (int i=0; i<nbr_label; ++i){
			e[ve[i].equivalent].vol_ex_val = nbr_label - i;
		}
		delete[]e_cpy;
	}

	for (size_t i=0; i<imIn.getPixelCount (); i++, *pixMarkers++, pixOut++) {
		if(*pixMarkers != labelT(0))
			*pixOut = e[*pixMarkers].vol_ex_val ;
	}

	delete[]e;
	return RES_OK;
    }
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
}				// namespace smil


#endif // _D_MORPHO_WATERSHED_EXTINCTION_HPP
