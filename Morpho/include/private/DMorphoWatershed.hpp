/*
 * Copyright (c) 2011-2014, Matthieu FAESSEL and ARMINES
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


#ifndef _D_MORPHO_WATERSHED_HPP
#define _D_MORPHO_WATERSHED_HPP


#include "DMorphoHierarQ.hpp"
#include "DMorphoExtrema.hpp"
#include "DMorphoLabel.hpp"
#include "DMorphoResidues.hpp"
#include "Core/include/DTypes.h"


namespace smil
{
    /**
     * \ingroup Morpho
     * \defgroup Watershed
     * @{
     */

  
    template <class T, class labelT, class HQ_Type >
    RES_T initWatershedHierarchicalQueue(const Image<T> &imIn, Image<labelT> &imLbl, Image<UINT8> &imStatus, HQ_Type &hq)
    {
	// Empty the priority queue
	hq.initialize(imIn);
	
	typename ImDtTypes<T>::lineType inPixels = imIn.getPixels();
	typename ImDtTypes<labelT>::lineType lblPixels = imLbl.getPixels();
	typename ImDtTypes<UINT8>::lineType statPixels = imStatus.getPixels();
	
	size_t s[3];
	
	imIn.getSize(s);
	size_t offset = 0;
	
	for (size_t k=0;k<s[2];k++)
	  for (size_t j=0;j<s[1];j++)
	    for (size_t i=0;i<s[0];i++)
	    {
	      if (*lblPixels!=0)
	      {
		  hq.push(T(*inPixels), offset);
		  *statPixels = HQ_LABELED;
	      }
	      else 
	      {
		  *statPixels = HQ_CANDIDATE;
	      }
	      inPixels++;
	      lblPixels++;
	      statPixels++;
	      offset++;
	    }
	
	return RES_OK;
    }

    template <class T, class labelT, class HQ_Type>
    RES_T processBasinsHierarchicalQueue(const Image<T> &imIn, Image<labelT> &imLbl, Image<UINT8> &imStatus, HQ_Type &hq, const StrElt &se)
    {
	typename ImDtTypes<T>::lineType inPixels = imIn.getPixels();
	typename ImDtTypes<labelT>::lineType lblPixels = imLbl.getPixels();
	typename ImDtTypes<UINT8>::lineType statPixels = imStatus.getPixels();
	
	vector<int> dOffsets;
	
	vector<IntPoint>::const_iterator it_start = se.points.begin();
	vector<IntPoint>::const_iterator it_end = se.points.end();
	vector<IntPoint>::const_iterator it;
	
	size_t s[3];
	imIn.getSize(s);
	
	// Create a copy of se without (potential) center point
	StrElt cpSe;
	cpSe.odd = se.odd;
	
	// set an offset distance for each se point (!=0,0,0)
	for(it=it_start;it!=it_end;it++)
	  if (it->x!=0 || it->y!=0 || it->z!=0)
	{
	    cpSe.addPoint(*it);
	    dOffsets.push_back(it->x + it->y*s[0] + it->z*s[0]*s[1]);
	}
	
	it_start = cpSe.points.begin();
	it_end = cpSe.points.end();
	
	vector<int>::iterator it_off_start = dOffsets.begin();
	vector<int>::iterator it_off;
	
	
	while(!hq.isEmpty())
	{
	    
	    size_t curOffset = hq.pop();
	    size_t x0, y0, z0;
	    
	    
	    
	    imIn.getCoordsFromOffset(curOffset, x0, y0, z0);
	    
	    bool oddLine = se.odd && ((y0)%2);
	    
	    int x, y, z;
	    size_t nbOffset;
	    
	    
	    for(it=it_start,it_off=it_off_start;it!=it_end;it++,it_off++)
	    {
		
		x = x0 + it->x;
		y = y0 + it->y;
		z = z0 + it->z;
		
		if (oddLine)
		  x += (((y+1)%2)!=0);
	      
		if (x>=0 && x<(int)s[0] && y>=0 && y<(int)s[1] && z>=0 && z<(int)s[2])
		{
		    nbOffset = curOffset + *it_off;
		    
		    if (oddLine)
		      nbOffset += (((y+1)%2)!=0);
		    
		    if (statPixels[nbOffset]==HQ_CANDIDATE)
		    {
			lblPixels[nbOffset] = lblPixels[curOffset];
			statPixels[nbOffset] = HQ_QUEUED;
			hq.push(inPixels[nbOffset], nbOffset);
		    }
		    
		}
	    }
	}
	    
	return RES_OK;
    }

    /**
    * Constrained basins.
    * 
    * Hierachical queue based algorithm as described by S. Beucher (2011) \cite beucher_hierarchical_2011
    * \param[in] imIn Input image.
    * \param[in] imMarkers Label image containing the markers. 
    * \param[out] imBasinsOut (optional) Output image containing the basins.
    * After processing, this image will contain the basins with the same label values as the initial markers.
    * 
    * \demo{constrained_watershed.py}
    */
    template <class T, class labelT>
    RES_T basins(const Image<T> &imIn, const Image<labelT> &imMarkers, Image<labelT> &imBasinsOut, const StrElt &se=DEFAULT_SE)
    {
	ASSERT_ALLOCATED(&imIn, &imMarkers, &imBasinsOut, NULL);
	ASSERT_SAME_SIZE(&imIn, &imMarkers, &imBasinsOut, NULL);
	
	ImageFreezer freeze(imBasinsOut);
	
	Image<UINT8> imStatus(imIn);
	copy(imMarkers, imBasinsOut);

 	HierarchicalQueue<T,UINT,FIFO_Queue<UINT> > pq; // preallocated HQ
//  	HierarchicalQueue<T> pq;

	initWatershedHierarchicalQueue(imIn, imBasinsOut, imStatus, pq);
	processBasinsHierarchicalQueue(imIn, imBasinsOut, imStatus, pq, se);

	return RES_OK;
    }

    template <class T, class labelT>
    RES_T basins(const Image<T> &imIn, Image<labelT> &imBasinsInOut, const StrElt &se=DEFAULT_SE)
    {
	ASSERT_ALLOCATED(&imIn, &imBasinsInOut);
	ASSERT_SAME_SIZE(&imIn, &imBasinsInOut);
	
	Image<T> imMin(imIn);
	minima(imIn, imMin, se);
	Image<labelT> imLbl(imIn);
	label(imMin, imLbl, se);
	
	return basins(imIn, imLbl, imBasinsInOut);
    }

    template <class T, class labelT, class HQ_Type>
    RES_T processWatershedHierarchicalQueue(const Image<T> &imIn, Image<labelT> &imLbl, Image<UINT8> &imStatus, HQ_Type &hq, const StrElt &se)
    {
	typename ImDtTypes<T>::lineType inPixels = imIn.getPixels();
	typename ImDtTypes<labelT>::lineType lblPixels = imLbl.getPixels();
	typename ImDtTypes<UINT8>::lineType statPixels = imStatus.getPixels();
	
	vector<int> _dOffsets;
	vector<UINT> tmpOffsets;
	
	size_t s[3];
	imIn.getSize(s);
	
	int nPts = se.points.size();
	const IntPoint *sePts = se.points.data();
	
	// set an offset distance for each se point
	for(int i=0;i<nPts;i++)
	    _dOffsets.push_back(sePts[i].x + sePts[i].y*s[0] + sePts[i].z*s[0]*s[1]);
	const int *dOffsets = _dOffsets.data();
	
	
#pragma omp parallel 
{
	int x, y, z;
	size_t nbOffset;
	UINT8 nbStat;
	
    #pragma omp single
    {
	while(!hq.isEmpty())
	{
	    
	    size_t curOffset = hq.pop();
	    size_t x0, y0, z0;
	    
	    imIn.getCoordsFromOffset(curOffset, x0, y0, z0);
	    
	    bool oddLine = se.odd && ((y0)%2);
	    
	    
	    statPixels[curOffset] = HQ_LABELED;
	    
	    
	    for(int i=0;i<nPts;i++)
// #pragma omp task //firstprivate(curOffset,x0,y0,z0) shared(statPixels,tmpOffsets)
	    {
		const IntPoint &p = sePts[i];
		if (p.x!=0 || p.y!=0 || p.z!=0) // useless if x=0 & y=0 & z=0
		{
		    
		    x = x0 + p.x;
		    y = y0 + p.y;
		    z = z0 + p.z;
		    
		    if (oddLine)
		      x += (((y+1)%2)!=0);
		  
		    if (x>=0 && x<(int)s[0] && y>=0 && y<(int)s[1] && z>=0 && z<(int)s[2])
		    {
			nbOffset = curOffset + dOffsets[i];
			
			if (oddLine)
			  nbOffset += (((y+1)%2)!=0);
			
			nbStat = statPixels[nbOffset];
			
			if (nbStat==HQ_CANDIDATE) // Add it to the tmp offsets queue
			{
#pragma omp critical
			    tmpOffsets.push_back(nbOffset);
			}
			else if (nbStat==HQ_LABELED)
			{
			    if (lblPixels[curOffset]==0)
				lblPixels[curOffset] = lblPixels[nbOffset];
			    else if (lblPixels[curOffset]!=lblPixels[nbOffset])
			      statPixels[curOffset] = HQ_WS_LINE;
			}
			
		    }
		}
	    }
	    
#pragma omp taskwait
	    if (statPixels[curOffset]!=HQ_WS_LINE && !tmpOffsets.empty())
	    {
		typename vector<UINT>::iterator t_it = tmpOffsets.begin();
		while (t_it!=tmpOffsets.end())
		{
		    hq.push(inPixels[*t_it], *t_it);
		    statPixels[*t_it] = HQ_QUEUED;
		    
		    t_it++;
		}
	    }
	    
	    tmpOffsets.clear();
	}
    } // omp single
} // omp parallel

	// Potential remaining candidate points (points surrounded by WS_LINE points)
	// Put their state to WS_LINE
	for (size_t i=0;i<imLbl.getPixelCount();i++)
	  if (statPixels[i]==HQ_CANDIDATE)
	    statPixels[i] = HQ_WS_LINE;
	return RES_OK;
    }

    /**
    * Constrained watershed.
    * 
    * Hierachical queue based algorithm as described by S. Beucher (2011) \cite beucher_hierarchical_2011
    * \param[in] imIn Input image.
    * \param[in] imMarkers Label image containing the markers. 
    * \param[out] imOut Output image containing the watershed lines.
    * \param[out] imBasinsOut (optional) Output image containing the basins.
    * After processing, this image will contain the basins with the same label values as the initial markers.
    * 
    * \demo{constrained_watershed.py}
    */

    template <class T, class labelT>
    RES_T watershed(const Image<T> &imIn, const Image<labelT> &imMarkers, Image<T> &imOut, Image<labelT> &imBasinsOut, const StrElt &se=DEFAULT_SE)
    {
	ASSERT_ALLOCATED(&imIn, &imMarkers, &imOut, &imBasinsOut);
	ASSERT_SAME_SIZE(&imIn, &imMarkers, &imOut, &imBasinsOut);
	
	ImageFreezer freezer(imOut);
	ImageFreezer freezer2(imBasinsOut);
	
	Image<UINT8> imStatus(imIn);
	copy(imMarkers, imBasinsOut);

	HierarchicalQueue<T> pq;

	initWatershedHierarchicalQueue(imIn, imBasinsOut, imStatus, pq);
	processWatershedHierarchicalQueue(imIn, imBasinsOut, imStatus, pq, se);

	ImDtTypes<UINT8>::lineType pixStat = imStatus.getPixels();
	typename ImDtTypes<T>::lineType pixOut = imOut.getPixels();

	// Create the image containing the ws lines
	fill(imOut, T(0));
	T wsVal = ImDtTypes<T>::max();
	for (size_t i=0;i<imIn.getPixelCount();i++,pixStat++,pixOut++)
	  if (*pixStat==HQ_WS_LINE) 
	    *pixOut = wsVal;
	  
	return RES_OK;
    }

    template <class T, class labelT>
    RES_T watershed(const Image<T> &imIn, Image<labelT> &imMarkers, Image<T> &imOut, const StrElt &se=DEFAULT_SE)
    {
	ASSERT_ALLOCATED(&imIn, &imMarkers, &imOut);
	ASSERT_SAME_SIZE(&imIn, &imMarkers, &imOut);
	
	Image<labelT> imBasinsOut(imMarkers);
	return watershed(imIn, imMarkers, imOut, imBasinsOut, se);
    }

    template <class T>
    RES_T watershed(const Image<T> &imIn, Image<T> &imOut, const StrElt &se=DEFAULT_SE)
    {
	ASSERT_ALLOCATED(&imIn, &imOut);
	ASSERT_SAME_SIZE(&imIn, &imOut);
	
	Image<T> imMin(imIn);
	minima(imIn, imMin, se);
	Image<UINT> imLbl(imIn);
	label(imMin, imLbl, se);
	return watershed(imIn, imLbl, imOut, se);
    }

    /**
     * Skiz on label image
     * 
     * Performs the influence zones on a label image as described by S. Beucher (2011) \cite beucher_algorithmes_2002
     * If a maskIm is provided, the skiz is geodesic.
     * 
     */ 
    template <class T>
    RES_T lblSkiz(Image<T> &labelIm1, Image<T> &labelIm2, const Image<T> &maskIm, const StrElt &se=DEFAULT_SE)
    {
	ASSERT_ALLOCATED(&labelIm1, &labelIm2, &maskIm);
	ASSERT_SAME_SIZE(&labelIm1, &labelIm2, &maskIm);
	
	Image<T> tmpIm1(labelIm1);
	Image<T> tmpIm2(labelIm1);
	Image<T> sumIm(labelIm1);
	
	double vol1 = -1, vol2 = 0;
	
	ImageFreezer freeze1(labelIm1);
	ImageFreezer freeze2(labelIm2);
	
	T threshMin = ImDtTypes<T>::min(), threshMax = ImDtTypes<T>::max()-T(1);
	
	while(vol1<vol2)
	{
	    dilate(labelIm1, tmpIm1, se);
	    dilate(labelIm2, tmpIm2, se);
	    addNoSat(labelIm1, labelIm2, sumIm);
	    threshold(sumIm, threshMin, threshMax, sumIm);
	    if (&maskIm)
	      inf(maskIm, sumIm, sumIm);
	    mask(tmpIm1, sumIm, tmpIm1);
	    mask(tmpIm2, sumIm, tmpIm2);
	    sup(labelIm1, tmpIm1, labelIm1);
	    sup(labelIm2, tmpIm2, labelIm2);

	    vol1 = vol2;
	    vol2 = vol(labelIm1);
	}
	return RES_OK;
    }

    /**
     * Influences basins
     * 
     * Performs the influence basins using the lblSkiz function.
     * Input image is supposed to be binary.
     * 
     */ 
    template <class T1, class T2>
    RES_T inflBasins(const Image<T1> &imIn, Image<T2> &basinsOut, const StrElt &se=DEFAULT_SE)
    {
	ASSERT_ALLOCATED(&imIn, &basinsOut);
	ASSERT_SAME_SIZE(&imIn, &basinsOut);
	
	Image<T2> imLbl2(basinsOut);
	Image<T2> *nullIm = NULL;
	
	// Create the label images
	label(imIn, basinsOut, se);
	inv(basinsOut, imLbl2);
	mask(imLbl2, basinsOut, imLbl2);
	
	ASSERT(lblSkiz(basinsOut, imLbl2, *nullIm, se)==RES_OK);
	
	// Clean result image
	open(basinsOut, basinsOut);
	
	return RES_OK;
    }

    /**
     * Influences zones
     * 
     * Performs the influence zones using the lblSkiz function.
     * Input image is supposed to be binary.
     * 
     */ 
    template <class T>
    RES_T inflZones(const Image<T> &imIn, Image<T> &imOut, const StrElt &se=DEFAULT_SE)
    {
	ASSERT_ALLOCATED(&imIn, &imOut);
	ASSERT_SAME_SIZE(&imIn, &imOut);
	
	Image<UINT16> basinsIm(imIn);
	
	ASSERT(inflBasins(imIn, basinsIm, se)==RES_OK);
	gradient(basinsIm, basinsIm, se, StrElt());
	threshold(basinsIm, UINT16(ImDtTypes<T>::min()+T(1)), UINT16(ImDtTypes<T>::max()), basinsIm);
	copy(basinsIm, imOut);
	
	return RES_OK;
    }
    
    /**
     * Waterfall
     * 
     */ 
    template <class T>
    RES_T waterfall(const Image<T> &gradIn, const Image<T> &wsIn, Image<T> &imGradOut, Image<T> &imWsOut, const StrElt &se=DEFAULT_SE)
    {
	ASSERT_ALLOCATED(&gradIn, &wsIn, &imGradOut, &imWsOut);
	ASSERT_SAME_SIZE(&gradIn, &wsIn, &imGradOut, &imWsOut);
	
	ImageFreezer freeze(imWsOut);
	
	test(wsIn, gradIn, ImDtTypes<T>::max(), imWsOut);
	dualBuild(imWsOut, gradIn, imGradOut, se);
	watershed(imGradOut, imWsOut, se);
	
	return RES_OK;
    }
    
    /**
     * Waterfall
     * 
     */ 
    template <class T>
    RES_T waterfall(const Image<T> &gradIn, UINT nLevel, Image<T> &imWsOut, const StrElt &se=DEFAULT_SE)
    {
	ASSERT_ALLOCATED(&gradIn, &imWsOut);
	ASSERT_SAME_SIZE(&gradIn, &imWsOut);
	
	ImageFreezer freeze(imWsOut);
	
	Image<T> tmpGradIm(gradIn, 1); //clone
	Image<T> tmpGradIm2(gradIn); //clone
	
	watershed(gradIn, imWsOut, se);
	
	for (UINT i=0;i<nLevel;i++)
	{
	    waterfall(tmpGradIm, imWsOut, tmpGradIm2, imWsOut, se);
	    copy(tmpGradIm2, tmpGradIm);
	}
	
	return RES_OK;
    }
    
/** @}*/

} // namespace smil


#endif // _D_MORPHO_WATERSHED_HPP

