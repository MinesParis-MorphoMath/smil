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


#ifndef _D_MORPHO_WATERSHED_HPP
#define _D_MORPHO_WATERSHED_HPP

/**
 * \ingroup HierarQ
 * @{
 */

#include "DMorphoHierarQ.hpp"
#include "DMorphoExtrema.hpp"
#include "DMorphoLabel.hpp"
#include "DImage.hpp"

template <class T, class labelT>
RES_T initWatershedHierarchicalQueue(Image<T> &imIn, Image<labelT> &imLbl, Image<UINT8> &imStatus, HierarchicalQueue<T> &hq)
{
    // Empty the priority queue
    hq.reset();
    
    typename ImDtTypes<T>::lineType inPixels = imIn.getPixels();
    typename ImDtTypes<labelT>::lineType lblPixels = imLbl.getPixels();
    typename ImDtTypes<UINT8>::lineType statPixels = imStatus.getPixels();
    
    UINT x, y, z;
    UINT s[3];
    
    imIn.getSize(s);
    UINT offset = 0;
    
    for (UINT k=0;k<s[2];k++)
      for (UINT j=0;j<s[1];j++)
	for (UINT i=0;i<s[0];i++)
	{
	  if (*lblPixels!=0)
	  {
	      hq.push(*inPixels, offset);
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

template <class T, class labelT>
RES_T processWatershedHierarchicalQueue(Image<T> &imIn, Image<labelT> &imLbl, Image<UINT8> &imStatus, HierarchicalQueue<T> &hq, StrElt &se)
{
    typename ImDtTypes<T>::lineType inPixels = imIn.getPixels();
    typename ImDtTypes<labelT>::lineType lblPixels = imLbl.getPixels();
    typename ImDtTypes<UINT8>::lineType statPixels = imStatus.getPixels();
    
    vector<int> dOffsets;
    
    vector<Point>::iterator it_start = se.points.begin();
    vector<Point>::iterator it_end = se.points.end();
    vector<Point>::iterator it;
    
    vector<UINT> tmpOffsets;
    
    UINT s[3];
    imIn.getSize(s);
    
    // set an offset distance for each se point
    for(it=it_start;it!=it_end;it++)
    {
	dOffsets.push_back(it->x - it->y*s[0] + it->z*s[0]*s[1]);
    }
    
    vector<int>::iterator it_off_start = dOffsets.begin();
    vector<int>::iterator it_off;
    
    
    while(!hq.empty())
    {
	
	HQToken<T> token = hq.top();
	hq.pop();
	UINT x0, y0, z0;
	
	UINT curOffset = token.offset;
	
	
	imIn.getCoordsFromOffset(curOffset, x0, y0, z0);
	
	int x, y, z;
	UINT nbOffset;
	UINT8 nbStat;
	
	int oddLine = se.odd * y0%2;
	
	for(it=it_start,it_off=it_off_start;it!=it_end;it++,it_off++)
	    if (it->x!=0 || it->y!=0 || it->z!=0) // useless if x=0 & y=0 & z=0
	{
	    
	    x = x0 + it->x;
	    y = y0 - it->y;
	    z = z0 + it->z;
	    
	    if (oddLine)
	      x += (y+1)%2;
	  
	    if (x>=0 && x<s[0] && y>=0 && y<s[1] && z>=0 && z<s[2])
	    {
		nbOffset = curOffset + *it_off;
		
		if (oddLine)
		  nbOffset += (y+1)%2;
		
		nbStat = statPixels[nbOffset];
		
		if (nbStat==HQ_CANDIDATE) // Add it to the tmp offsets queue
		    tmpOffsets.push_back(nbOffset);
		else if (nbStat==HQ_LABELED)
		{
		    if (statPixels[curOffset]==HQ_LABELED)
		    {
			if (lblPixels[curOffset]!=lblPixels[nbOffset])
			    statPixels[curOffset] = HQ_WS_LINE;
		    }
		    else if (statPixels[curOffset]!=HQ_WS_LINE)
		    {
		      statPixels[curOffset] = HQ_LABELED;
		      lblPixels[curOffset] = lblPixels[nbOffset];
		    }
		}
		
	    }
	}

	if (statPixels[curOffset]==HQ_LABELED && !tmpOffsets.empty())
	{
	    typename vector<UINT>::iterator t_it = tmpOffsets.begin();
	    while (t_it!=tmpOffsets.end())
	    {
		hq.push(inPixels[*t_it], *t_it);
		statPixels[*t_it] = HQ_QUEUED;
		
		t_it++;
	    }
	    
	    tmpOffsets.clear();
	}
    }
    
    // Potential remaining candidate points (points surrounded by WS_LINE points)
    // Put their state to WS_LINE
    for (int i=0;i<imLbl.getPixelCount();i++)
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
RES_T watershed(Image<T> &imIn, Image<labelT> &imMarkers, Image<T> &imOut, Image<labelT> &imBasinsOut, StrElt se=DEFAULT_SE())
{
    if (!areAllocated(&imIn, &imMarkers, &imOut, &imBasinsOut, NULL))
      return RES_ERR_BAD_ALLOCATION;
    
    if (!haveSameSize(&imIn, &imMarkers, &imOut, &imBasinsOut, NULL))
      return RES_ERR_BAD_SIZE;
    
    Image<UINT8> imStatus(imIn);
    copy(imMarkers, imBasinsOut);

    HierarchicalQueue<T> pq;

    initWatershedHierarchicalQueue<T,labelT>(imIn, imBasinsOut, imStatus, pq);
    processWatershedHierarchicalQueue(imIn, imBasinsOut, imStatus, pq, se);

    ImDtTypes<UINT8>::lineType pixStat = imStatus.getPixels();
    typename ImDtTypes<T>::lineType pixOut = imOut.getPixels();

    // Create the image containing the ws lines
    fill(imOut, T(0));
    T wsVal = ImDtTypes<T>::max();
    for (int i=0;i<imIn.getPixelCount();i++,pixStat++,pixOut++)
      if (*pixStat==HQ_WS_LINE) 
	*pixOut = wsVal;
      
    imBasinsOut.modified();
    imOut.modified();
    return RES_OK;
}

template <class T, class labelT>
RES_T watershed(Image<T> &imIn, Image<labelT> &imMarkers, Image<T> &imOut, StrElt se=DEFAULT_SE())
{
    if (!areAllocated(&imIn, &imMarkers, &imOut, NULL))
      return RES_ERR_BAD_ALLOCATION;
    
    if (!haveSameSize(&imIn, &imMarkers, &imOut, NULL))
      return RES_ERR_BAD_SIZE;
    
    Image<labelT> imBasinsOut(imMarkers);
    return watershed(imIn, imMarkers, imOut, imBasinsOut);
}

template <class T>
RES_T watershed(Image<T> &imIn, Image<T> &imOut, StrElt se=DEFAULT_SE())
{
    if (!areAllocated(&imIn, &imOut, NULL))
      return RES_ERR_BAD_ALLOCATION;
    
    if (!haveSameSize(&imIn, &imOut, NULL))
      return RES_ERR_BAD_SIZE;
    
    Image<T> imMin(imIn);
    minima(imIn, imMin, se);
    Image<UINT> imLbl(imIn);
    label(imMin, imLbl, se);
    return watershed(imIn, imLbl, imOut, se);
}

/** @}*/

#endif // _D_MORPHO_WATERSHED_HPP

