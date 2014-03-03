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


#ifndef _D_MORPHO_GEODESIC_HPP
#define _D_MORPHO_GEODESIC_HPP

#include "DMorphImageOperations.hpp"
#include "DMorphoHierarQ.hpp"
#include "Base/include/private/DImageDraw.hpp"
#include "Base/include/private/DImageHistogram.hpp"
#include "Morpho/include/private/DMorphoBase.hpp"

namespace smil
{
    /**
    * \ingroup Morpho
    * \defgroup Geodesic
    * @{
    */

    // Geodesy

    template <class T>
    RES_T geoDil(const Image<T> &imIn, const Image<T> &imMask, Image<T> &imOut, const StrElt &se=DEFAULT_SE)
    {
	ASSERT_ALLOCATED(&imIn, &imMask, &imOut);
	ASSERT_SAME_SIZE(&imIn, &imMask, &imOut);

	ImageFreezer freeze(imOut);
	
	ASSERT((inf(imIn, imMask, imOut)==RES_OK));
	StrElt tmpSE(se(1));
	
	for (UINT i=0;i<se.size;i++)
	{
	    ASSERT((dilate<T>(imOut, imOut, tmpSE)==RES_OK));
	    ASSERT((inf(imOut, imMask, imOut)==RES_OK));
	}
	return RES_OK;
    }

    template <class T>
    RES_T geoEro(const Image<T> &imIn, const Image<T> &imMask, Image<T> &imOut, const StrElt &se=DEFAULT_SE)
    {
	ASSERT_ALLOCATED(&imIn, &imMask, &imOut);
	ASSERT_SAME_SIZE(&imIn, &imMask, &imOut);

	ImageFreezer freeze(imOut);
	
	ASSERT((sup(imIn, imMask, imOut)==RES_OK));
	StrElt tmpSE(se(1));
	
	for (UINT i=0;i<se.size;i++)
	{
	    ASSERT((erode(imOut, imOut, tmpSE)==RES_OK));
	    ASSERT((sup(imOut, imMask, imOut)==RES_OK));
	}
	return RES_OK;
    }

    template <class T>
    RES_T geoBuild(const Image<T> &imIn, const Image<T> &imMask, Image<T> &imOut, const StrElt &se=DEFAULT_SE)
    {
	ASSERT_ALLOCATED(&imIn, &imMask, &imOut);
	ASSERT_SAME_SIZE(&imIn, &imMask, &imOut);

	ImageFreezer freeze(imOut);
	
	ASSERT((inf(imIn, imMask, imOut)==RES_OK));
	
	double vol1 = vol(imOut), vol2;
	while (true)
	{
	    ASSERT((dilate<T>(imOut, imOut, se)==RES_OK));
	    ASSERT((inf(imOut, imMask, imOut)==RES_OK));
	    vol2 = vol(imOut);
	    if (vol2==vol1)
	      break;
	    vol1 = vol2;
	}
	return RES_OK;
    }

    template <class T>
    RES_T geoDualBuild(const Image<T> &imIn, const Image<T> &imMask, Image<T> &imOut, const StrElt &se=DEFAULT_SE)
    {
	ASSERT_ALLOCATED(&imIn, &imMask, &imOut);
	ASSERT_SAME_SIZE(&imIn, &imMask, &imOut);

	ImageFreezer freeze(imOut);
	
	ASSERT((sup(imIn, imMask, imOut)==RES_OK));

	double vol1 = vol(imOut), vol2;
	
	while (true)
	{
	    ASSERT((erode(imOut, imOut, se)==RES_OK));
	    ASSERT((sup(imOut, imMask, imOut)==RES_OK));
	    vol2 = vol(imOut);
	    if (vol2==vol1)
	      break;
	    vol1 = vol2;
	}
	return RES_OK;
    }






    template <class T>
    RES_T initBuildHierarchicalQueue(const Image<T> &imIn, HierarchicalQueue<T> &hq)
    {
	// Initialize the priority queue
	hq.initialize(imIn);
	
	typename ImDtTypes<T>::lineType inPixels = imIn.getPixels();
	
	size_t s[3];
	
	imIn.getSize(s);
	size_t offset = 0;
	
	for (size_t i=0;i<imIn.getPixelCount();i++)
	{
	    hq.push(*inPixels, offset);
	    inPixels++;
	    offset++;
	}
	
    //     hq.printSelf();
	return RES_OK;
    }

    template <class T>
    RES_T initBuildHierarchicalQueue(const Image<T> &imIn, HierarchicalQueue<T> &hq, const T noPushValue)
    {
	// Initialize the priority queue
	hq.initialize(imIn);
	
	typename ImDtTypes<T>::lineType inPixels = imIn.getPixels();
	
	size_t s[3];
	
	imIn.getSize(s);
	size_t offset = 0;
	
	for (size_t i=0;i<imIn.getPixelCount();i++)
	{

	    if (*inPixels != noPushValue) {
	      hq.push(*inPixels, offset);
	    }
	    inPixels++;
	    offset++;
	}
	
    //     hq.printSelf();
	return RES_OK;
    }



    template <class T, class operatorT>
    RES_T processBuildHierarchicalQueue(Image<T> &imIn, const Image<T> &imMark, Image<UINT8> &imStatus, HierarchicalQueue<T> &hq, const StrElt &se)
    {
	typename ImDtTypes<T>::lineType inPixels = imIn.getPixels();
	typename ImDtTypes<T>::lineType markPixels = imMark.getPixels();
	typename ImDtTypes<UINT8>::lineType statPixels = imStatus.getPixels();
	
	vector<int> dOffsets;
	operatorT oper;
	
	vector<IntPoint>::const_iterator it_start = se.points.begin();
	vector<IntPoint>::const_iterator it_end = se.points.end();
	vector<IntPoint>::const_iterator it;
	
	vector<size_t> tmpOffsets;
	
	size_t s[3];
	imIn.getSize(s);
	
	// set an offset distance for each se point
	for(it=it_start;it!=it_end;it++)
	{
	    dOffsets.push_back(it->x + it->y*s[0] + it->z*s[0]*s[1]);
	}
	
	vector<int>::iterator it_off_start = dOffsets.begin();
	vector<int>::iterator it_off;
	
	size_t x0, y0, z0;
	size_t curOffset;
	
	int x, y, z;
	size_t nbOffset;
	UINT8 nbStat;
	
	while(!hq.isEmpty())
	{
	    
	    curOffset = hq.pop();
	    
	    // Give the point the label "FINAL" in the status image
	    statPixels[curOffset] = HQ_FINAL;
	    
	    imIn.getCoordsFromOffset(curOffset, x0, y0, z0);
	    
	    
	    bool oddLine = se.odd && (y0)%2;
	    
	    for(it=it_start,it_off=it_off_start;it!=it_end;it++,it_off++)
		if (it->x!=0 || it->y!=0 || it->z!=0) // useless if x=0 & y=0 & z=0
	    {
		
		x = x0 + it->x;
		y = y0 + it->y;
		z = z0 + it->z;
		
		if (oddLine)
		  x += ((y+1)%2!=0);
	      
		if (x>=0 && x<(int)s[0] && y>=0 && y<(int)s[1] && z>=0 && z<(int)s[2])
		{
		    nbOffset = curOffset + *it_off;
		    
		    if (oddLine)
		      nbOffset += ((y+1)%2!=0);
		    
		    nbStat = statPixels[nbOffset];
		    
		    if (nbStat==HQ_CANDIDATE)
		    {
			inPixels[nbOffset] = oper(inPixels[curOffset], markPixels[nbOffset]);
			statPixels[nbOffset] = HQ_QUEUED;
			hq.push(inPixels[nbOffset], nbOffset);
		    }
		    
		}
	    }

	}
	return RES_OK;    
    }

    template <class T>
    struct minFunctor 
    {
      inline T operator()(T a, T b) { return min(a, b); }
    };

    template <class T>
    struct maxFunctor 
    {
      inline T operator()(T a, T b) { return max(a, b); }
    };

    /**
    * Dual reconstruction (using hierarchical queues).
    */
    template <class T>
    RES_T dualBuild(const Image<T> &imIn, const Image<T> &imMark, Image<T> &imOut, const StrElt &se=DEFAULT_SE)
    {
	ASSERT_ALLOCATED(&imIn, &imMark, &imOut);
	ASSERT_SAME_SIZE(&imIn, &imMark, &imOut);

	ImageFreezer freeze(imOut);
	
	Image<UINT8> imStatus(imIn);
	HierarchicalQueue<T,UINT> pq;
	
	// Make sure that imIn >= imMark
	ASSERT((sup(imIn, imMark, imOut)==RES_OK));
	
	// Set all pixels in the status image to CANDIDATE
	ASSERT((fill(imStatus, (UINT8)HQ_CANDIDATE)==RES_OK));
	
	// Initialize the PQ
	initBuildHierarchicalQueue(imOut, pq);
	processBuildHierarchicalQueue<T, maxFunctor<T> >(imOut, imMark, imStatus, pq, se);
	
	return RES_OK;
    }

    /**
    * Reconstruction (using hierarchical queues).
    */
    template <class T>
    RES_T build(const Image<T> &imIn, const Image<T> &imMark, Image<T> &imOut, const StrElt &se=DEFAULT_SE)
    {
	ASSERT_ALLOCATED(&imIn, &imMark, &imOut);
	ASSERT_SAME_SIZE(&imIn, &imMark, &imOut);

	ImageFreezer freeze(imOut);
	
	Image<UINT8> imStatus(imIn);
	
	// Reverse hierarchical queue (the highest token corresponds to the highest gray value)
	HierarchicalQueue<T> rpq(true);
	
	// Make sure that imIn <= imMark
	ASSERT((inf(imIn, imMark, imOut)==RES_OK));
	
	// Set all pixels in the status image to CANDIDATE
	ASSERT((fill(imStatus, (UINT8)HQ_CANDIDATE)==RES_OK));
	
	// Initialize the PQ
	initBuildHierarchicalQueue(imOut, rpq);
	processBuildHierarchicalQueue<T, minFunctor<T> >(imOut, imMark, imStatus, rpq, se);
	
	return RES_OK;
    }

    /**
    * Reconstruction (using hierarchical queues).
    */
    template <class T>
    RES_T binBuild(const Image<T> &imIn, const Image<T> &imMark, Image<T> &imOut, const StrElt &se=DEFAULT_SE)
    {
	ASSERT_ALLOCATED(&imIn, &imMark, &imOut);
	ASSERT_SAME_SIZE(&imIn, &imMark, &imOut);
	//	T noPushValue = NUMERIC_LIMITS<T>::min();
	//T maxValue =  NUMERIC_LIMITS<T>::max();
	ImageFreezer freeze(imOut);
	
	Image<UINT8> imStatus(imIn);
	
	// Reverse hierarchical queue (the highest token corresponds to the highest gray value)
	HierarchicalQueue<T> rpq(true);
	
	// Make sure that imIn <= imMark
	ASSERT((inf(imIn, imMark, imOut)==RES_OK));
	
	// make a status image with all foreground pixels as CANDIDATE, otherwise as FINAL
	
	ASSERT((copy(imMark, imStatus) == RES_OK));
	ASSERT((threshold<UINT8>(imStatus, imStatus.getDataTypeMin()+1, imStatus.getDataTypeMax(), (UINT8)HQ_CANDIDATE, (UINT8)HQ_FINAL, imStatus)==RES_OK));
    
	// Initialize the PQ
	initBuildHierarchicalQueue(imOut, rpq, imOut.getDataTypeMin());
	processBuildHierarchicalQueue<T, minFunctor<T> >(imOut, imMark, imStatus, rpq, se);
	
	return RES_OK;
    }


    /**
    * h-Reconstuction
    * 
    * Performs a subtraction of size \b height followed by a reconstruction
    */
    template <class T>
    RES_T hBuild(const Image<T> &imIn, const T &height, Image<T> &imOut, const StrElt &se=DEFAULT_SE)
    {
	ASSERT_ALLOCATED(&imIn, &imOut);
	ASSERT_SAME_SIZE(&imIn, &imOut);
	
	if (&imIn==&imOut)
	{
	    Image<T> tmpIm = imIn;
	    return hBuild(tmpIm, height, imOut, se);
	}
	
	ImageFreezer freeze(imOut);
	
	ASSERT((sub(imIn, T(height), imOut)==RES_OK));
	ASSERT((build(imOut, imIn, imOut, se)==RES_OK));
	
	return RES_OK;
    }

    /**
    * Dual h-Reconstuction
    * 
    * Performs an addition of size \b height followed by a dual reconstruction
    */
    template <class T>
    RES_T hDualBuild(const Image<T> &imIn, const T &height, Image<T> &imOut, const StrElt &se=DEFAULT_SE)
    {
	ASSERT_ALLOCATED(&imIn, &imOut);
	ASSERT_SAME_SIZE(&imIn, &imOut);
	
	if (&imIn==&imOut)
	{
	    Image<T> tmpIm = imIn;
	    return hDualBuild(tmpIm, height, imOut, se);
	}
	
	ImageFreezer freeze(imOut);
	
	ASSERT((add(imIn, T(height), imOut)==RES_OK));
	ASSERT((dualBuild(imOut, imIn, imOut, se)==RES_OK));
	
	return RES_OK;
    }

    /**
    * Opening by reconstruction
    */
    template <class T>
    RES_T buildOpen(const Image<T> &imIn, Image<T> &imOut, const StrElt &se=DEFAULT_SE)
    {
	ASSERT_ALLOCATED(&imIn, &imOut);
	ASSERT_SAME_SIZE(&imIn, &imOut);
	
	ImageFreezer freeze(imOut);
	
	Image<T> tmpIm(imIn);
	ASSERT((erode(imIn, tmpIm, se)==RES_OK));
	ASSERT((build(tmpIm, imIn, imOut, se)==RES_OK));
	
	return RES_OK;
    }

    /**
    * Closing by reconstruction
    */
    template <class T>
    RES_T buildClose(const Image<T> &imIn, Image<T> &imOut, const StrElt &se=DEFAULT_SE)
    {
	ASSERT_ALLOCATED(&imIn, &imOut);
	ASSERT_SAME_SIZE(&imIn, &imOut);
	
	ImageFreezer freeze(imOut);
	
	Image<T> tmpIm(imIn);
	ASSERT((dilate(imIn, tmpIm, se)==RES_OK));
	ASSERT((dualBuild(tmpIm, imIn, imOut, se)==RES_OK));
	
	return RES_OK;
    }

    /**
    * Hole filling
    */
    template <class T>
    RES_T fillHoles(const Image<T> &imIn, Image<T> &imOut, const StrElt &se=DEFAULT_SE)
    {
	ASSERT_ALLOCATED(&imIn, &imOut);
	ASSERT_SAME_SIZE(&imIn, &imOut);
	
	ImageFreezer freeze(imOut);
	
	Image<T> tmpIm(imIn);
	
	ASSERT((fill(tmpIm, numeric_limits<T>::max())==RES_OK));
	ASSERT((drawRectangle(tmpIm, 0, 0, tmpIm.getWidth(), tmpIm.getHeight(), ImDtTypes<T>::min())==RES_OK));
	ASSERT((dualBuild(tmpIm, imIn, imOut, se)==RES_OK));
	
	return RES_OK;
    }

    /**
    * Dual hole filling
    */
    template <class T>
    RES_T levelPics(const Image<T> &imIn, Image<T> &imOut, const StrElt &se=DEFAULT_SE)
    {
	ASSERT_ALLOCATED(&imIn, &imOut);
	ASSERT_SAME_SIZE(&imIn, &imOut);
	
	ImageFreezer freeze(imOut);
	
	Image<T> tmpIm(imIn);
	ASSERT((inv(imIn, tmpIm)==RES_OK));
	ASSERT((fillHoles(tmpIm, imOut, se)==RES_OK));
	ASSERT((inv(imOut, imOut)==RES_OK));
	
    //     return res;
	return RES_OK;
    }

    // Multi-source label-correcting algorithm for ALSP problem.
    template <class T1, class T2>
    RES_T dist(const Image<T1> &imIn, Image<T2> &imOut, const StrElt &se=DEFAULT_SE) 
    {
    	ASSERT_ALLOCATED(&imIn, &imOut);
	ASSERT_SAME_SIZE(&imIn, &imOut);

	ImageFreezer freeze(imOut);

        typedef Image<T1> imageInType;
        typedef typename imageInType::lineType lineInType;
        typedef Image<T2> imageOutType;
        typedef typename imageOutType::lineType lineOutType;

        lineInType pixelsIn ;
        lineOutType pixelsOut = imOut.getPixels () ; 

        Image<T1> tmp(imIn);
        Image<T1> tmp2(imIn);

        // Set image to 1 when pixels are !=0
        ASSERT(inf (imIn, T1(1), tmp)==RES_OK);

        ASSERT(copy(tmp, imOut)==RES_OK);
        ASSERT(mul (tmp, T1(255), tmp)==RES_OK);

        // Demi-Gradient to remove sources inside cluster of sources.
        ASSERT(dilate (tmp, tmp2, se)==RES_OK); 
        ASSERT(inv(tmp2, tmp2)==RES_OK);
        ASSERT(add(tmp, tmp2, tmp)==RES_OK);

        queue<size_t> *level = new queue<size_t>();
        queue<size_t> *next_level = new queue<size_t>();
        queue<size_t> *swap ;
        T2 cur_level=T2(0);

        size_t size[3];
        imIn.getSize (size) ;

        pixelsIn = tmp.getPixels ();

        size_t i=0;
        
        for (i=0; i<size[2]*size[1]*size[0]; ++i)
        {
            if (pixelsIn[i] == T1(0))
                level->push (i);
        }

        size_t cur;
        int x,y,z;

        vector<IntPoint> sePoints = se.points;
        vector<IntPoint>::iterator pt ;

        do {
            while (!level->empty()) {
                cur = level->front();
                pt = sePoints.begin();

                z = cur / (size[1] * size[0]);
                y = (cur - z * size[1] * size[0]) / size[0];
                x = cur - y *size[0] - z * size[1] * size[0];

                while (pt!=sePoints.end()) {
                    if (    x+pt->x >= 0 && x+pt->x < size[0] &&
                            y+pt->y >= 0 && y+pt->y < size[1] &&
                            z+pt->z >= 0 && z+pt->z < size[2] && 
                            pixelsOut [x+pt->x + (y+pt->y)*size[0] + (z+pt->z)*size[1]*size[0]] > T2(cur_level))
                    {
                        pixelsOut [x+pt->x + (y+pt->y)*size[0] + (z+pt->z)*size[1]*size[0]] = T2(cur_level); 
                        next_level->push (x+pt->x + (y+pt->y)*size[0] + (z+pt->z)*size[1]*size[0]);
                    }
                    ++pt;
                }
                level->pop();
            }
            ++cur_level;

            swap = level;
            level = next_level;
            next_level = swap;
        } while (!level->empty());

        return RES_OK;
    }

    /**
    * Ugly temporary distance function
    */
    template <class T>
    RES_T dist_v0(const Image<T> &imIn, Image<T> &imOut, const StrElt &se=DEFAULT_SE)
    {
	ASSERT_ALLOCATED(&imIn, &imOut);
	ASSERT_SAME_SIZE(&imIn, &imOut);
	
	ImageFreezer freeze(imOut);
	
	Image<T> tmpIm(imIn);
	
	// Set image to 1 when pixels are !=0
	ASSERT((inf(imIn, T(1), tmpIm)==RES_OK));
	
	ASSERT((copy(tmpIm, imOut)==RES_OK));
	
	do
	{
	    ASSERT((erode(tmpIm, tmpIm, se)==RES_OK));
	    ASSERT((add(tmpIm, imOut, imOut)==RES_OK));
	    
	} while (vol(tmpIm)!=0);

	return RES_OK;
    }

/** \} */

} // namespace smil



#endif // _D_MORPHO_GEODESIC_HPP

