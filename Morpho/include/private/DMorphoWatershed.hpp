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
     * \defgroup Watershed Watershed Segmentation
     * @{
     */

    template <class T, class labelT, class HQ_Type=HierarchicalQueue<T> >
    class BaseFlooding
    {
      public:
        BaseFlooding()
          : STAT_QUEUED(ImDtTypes<labelT>::max())
        {
        }
        virtual ~BaseFlooding()
        {
        }
        
       protected:
        
        const labelT STAT_QUEUED;
        
        typename ImDtTypes<T>::lineType inPixels;
        typename ImDtTypes<labelT>::lineType lblPixels;
            
        size_t imSize[3], pixPerSlice;
        inline void getCoordsFromOffset(size_t off, size_t &x, size_t &y, size_t &z) const
        {
            z = off / pixPerSlice;
            y = (off % pixPerSlice) / imSize[0];
            x = off % imSize[0];
        }
            
        vector<IntPoint> sePts;
        UINT sePtsNbr;
        bool oddSE;
        vector<int> dOffsets;
        
        T currentLevel;
        
      public:
      
        const Image<T> *imgIn;
        Image<labelT> *imgLbl;
        
        virtual RES_T flood(const Image<T> &imIn, const Image<labelT> &imMarkers, Image<labelT> &imBasinsOut, const StrElt &se=DEFAULT_SE)
        {
            ASSERT_ALLOCATED(&imIn, &imMarkers, &imBasinsOut);
            ASSERT_SAME_SIZE(&imIn, &imMarkers, &imBasinsOut);
            
            ASSERT(maxVal(imMarkers)<STAT_QUEUED);
              
            ImageFreezer freeze(imBasinsOut);
            
            copy(imMarkers, imBasinsOut);

            initialize(imIn, imBasinsOut, se);
            processImage(imIn, imBasinsOut, se);
            
            return RES_OK;
        }
        
        virtual RES_T initialize(const Image<T> &imIn, Image<labelT> &imLbl, const StrElt &se)
        {
            imgIn = &imIn;
            imgLbl = &imLbl;
            
            // Empty the priority queue
            hq.initialize(imIn);
            
            inPixels = imIn.getPixels();
            lblPixels = imLbl.getPixels();
            
            imIn.getSize(imSize);
            pixPerSlice = imSize[0]*imSize[1];
            
            dOffsets.clear();
            sePts.clear();
            oddSE = se.odd;
            
            // set an offset distance for each se point (!=0,0,0)
            for(vector<IntPoint>::const_iterator it = se.points.begin() ; it!=se.points.end() ; it++)
              if (it->x!=0 || it->y!=0 || it->z!=0)
            {
                sePts.push_back(*it);
                dOffsets.push_back(it->x + it->y*imSize[0] + it->z*imSize[0]*imSize[1]);
            }
            
            sePtsNbr = sePts.size();
            
            return RES_OK;
        }
        
        virtual RES_T processImage(const Image<T> &/*imIn*/, Image<labelT> &/*imLbl*/, const StrElt &/*se*/)
        {
            // Put the marker pixels in the HQ
            size_t offset = 0;
            for (size_t k=0;k<imSize[2];k++)
              for (size_t j=0;j<imSize[1];j++)
                for (size_t i=0;i<imSize[0];i++)
                {
                  if (lblPixels[offset]!=0)
                  {
                      hq.push(inPixels[offset], offset);
                  }
                  offset++;
                }
                
                
        
            currentLevel = ImDtTypes<T>::min();
            
            while(!hq.isEmpty())
            {
                this->processPixel(hq.pop());
            }
                
            return RES_OK;
        }
        
        inline virtual void processPixel(const size_t &curOffset)
        {
                size_t x0, y0, z0;
                
                getCoordsFromOffset(curOffset, x0, y0, z0);
                
                bool oddLine = oddSE && ((y0)%2);
                
                int x, y, z;
                size_t nbOffset;
                
                for(UINT i=0;i<sePtsNbr;i++)
                {
                    IntPoint &pt = sePts[i];
                    x = x0 + pt.x;
                    y = y0 + pt.y;
                    z = z0 + pt.z;
                    
                    if (oddLine)
                      x += (((y+1)%2)!=0);
                  
                    if (x>=0 && x<(int)imSize[0] && y>=0 && y<(int)imSize[1] && z>=0 && z<(int)imSize[2])
                    {
                        nbOffset = curOffset + dOffsets[i];
                        
                        if (oddLine)
                          nbOffset += (((y+1)%2)!=0);
                        
                        processNeighbor(curOffset, nbOffset);
                        
                    }
                }
        }
        
        inline virtual void processNeighbor(const size_t &curOffset, const size_t &nbOffset)
        {
            labelT nbLbl = this->lblPixels[nbOffset];
            labelT curLbl = lblPixels[curOffset];//==STAT_QUEUED ? 0 : lblPixels[curOffset];
            
            if (nbLbl==0) // Add it to the tmp offsets queue
            {
                hq.push(inPixels[nbOffset], nbOffset);
		this->lblPixels[nbOffset] = curLbl;
	    }
        }
        
      protected:
        HQ_Type hq;
        
    };
  

    template <class T, class labelT, class HQ_Type=HierarchicalQueue<T> >
    class WatershedFlooding 
#ifndef SWIG    
        : public BaseFlooding<T, labelT, HQ_Type>
#endif // SWIG    
    {
      protected:
        vector<size_t> tmpOffsets;
        typename ImDtTypes<T>::lineType wsPixels;
        const T STAT_LABELED, STAT_QUEUED, STAT_CANDIDATE, STAT_WS_LINE;
        
      public:
        WatershedFlooding()
          : STAT_LABELED(1), 
            STAT_QUEUED(2), 
            STAT_CANDIDATE(ImDtTypes<T>::max()-1), 
            STAT_WS_LINE(ImDtTypes<T>::max())
        {
        }
        virtual ~WatershedFlooding()
        {
        }
        
        Image<T> *imgWS;
#ifdef SWIG
        const Image<T> *imgIn;
        Image<labelT> *imgLbl;
#endif // SWIG

        virtual RES_T flood(const Image<T> &imIn, const Image<labelT> &imMarkers, Image<T> &imOut, Image<labelT> &imBasinsOut, const StrElt &se)
        {
            ASSERT_ALLOCATED(&imIn, &imMarkers, &imBasinsOut);
            ASSERT_SAME_SIZE(&imIn, &imMarkers, &imBasinsOut);
            
            ImageFreezer freeze(imBasinsOut);
            ImageFreezer freeze2(imOut);
            
            copy(imMarkers, imBasinsOut);

            initialize(imIn, imBasinsOut, imOut, se);
            this->processImage(imIn, imBasinsOut, se);
            
            // Finalize the image containing the ws lines
            // Potential remaining candidate points (points surrounded by WS_LINE points) with status STAT_CANDIDATE are added to the WS_LINE
            for (size_t i=0;i<imIn.getPixelCount();i++)
            {
              if (wsPixels[i]>=STAT_CANDIDATE)
                wsPixels[i] = STAT_WS_LINE;
              else
                wsPixels[i] = 0;
            }
            return RES_OK;
        }
        
        virtual RES_T initialize(const Image<T> &imIn, Image<labelT> &imLbl, Image<T> &imOut, const StrElt &se)
        {
            BaseFlooding<T, labelT, HQ_Type>::initialize(imIn, imLbl, se);
            
            imgWS = &imOut;
            imgWS->setSize(this->imSize);
            test(imLbl, STAT_LABELED, STAT_CANDIDATE, *imgWS);
            wsPixels = imgWS->getPixels();
            
            tmpOffsets.clear();
            
            return RES_OK;
        }
        
        inline virtual void processPixel(const size_t &curOffset)
        {
            wsPixels[curOffset] = STAT_LABELED;
            
            BaseFlooding<T, labelT, HQ_Type>::processPixel(curOffset);
            
            if (!tmpOffsets.empty())
            {
                if (this->wsPixels[curOffset]!=STAT_WS_LINE)
                {
                    size_t *offsets = tmpOffsets.data();
                    for (UINT i=0;i<tmpOffsets.size();i++)
                    {
                        this->hq.push(this->inPixels[*offsets], *offsets);
                        this->wsPixels[*offsets] = STAT_QUEUED;
                        
                        offsets++;
                    }
                }
                tmpOffsets.clear();
            }
            
        }
        inline virtual void processNeighbor(const size_t &curOffset, const size_t &nbOffset)
        {
            T nbStat = this->wsPixels[nbOffset];
            
            if (nbStat==STAT_CANDIDATE) // Add it to the tmp offsets queue
            {
                tmpOffsets.push_back(nbOffset);
            }
            else if (nbStat==STAT_LABELED)
            {
                if (this->lblPixels[curOffset]==0)
                {
                    this->lblPixels[curOffset] = this->lblPixels[nbOffset];
                }
                else if (this->lblPixels[curOffset]!=this->lblPixels[nbOffset])
                  this->wsPixels[curOffset] = STAT_WS_LINE;
            }
        }
#ifndef SWIG
      using BaseFlooding<T, labelT, HQ_Type>::flood;
      using BaseFlooding<T, labelT, HQ_Type>::initialize;
#endif // SWIG
    };
    
    /**
    * Constrained basins.
    * 
    * Hierachical queue based algorithm as described by S. Beucher (2011) \cite beucher_hierarchical_2011
    * \param[in] imIn Input image.
    * \param[in] imMarkers Label image containing the markers. 
    * \param[out] imBasinsOut (optional) Output image containing the basins.
    * \param[in] se Structuring element
    * After processing, this image will contain the basins with the same label values as the initial markers.
    * 
    * \demo{constrained_watershed.py}
    */
    template <class T, class labelT>
    RES_T basins(const Image<T> &imIn, const Image<labelT> &imMarkers, Image<labelT> &imBasinsOut, const StrElt &se=DEFAULT_SE)
    {
        BaseFlooding<T, labelT> flooding;
        return flooding.flood(imIn, imMarkers, imBasinsOut, se);
    }

    template <class T, class labelT>
    RES_T basins(const Image<T> &imIn, Image<labelT> &imBasinsInOut, const StrElt &se=DEFAULT_SE)
    {
        ASSERT_ALLOCATED(&imIn);
        ASSERT_SAME_SIZE(&imIn, &imBasinsInOut);
        
        Image<labelT> imLbl(imIn);
        minimaLabeled(imIn, imLbl, se);
        
        return basins(imIn, imLbl, imBasinsInOut);
    }


    /**
    * Constrained watershed.
    * 
    * Hierachical queue based algorithm as described by S. Beucher (2011) \cite beucher_hierarchical_2011
    * \param[in] imIn Input image.
    * \param[in] imMarkers Label image containing the markers. 
    * \param[in] se Structuring element
    * \param[out] imOut Output image containing the watershed lines.
    * \param[out] imBasinsOut (optional) Output image containing the basins.
    * After processing, this image will contain the basins with the same label values as the initial markers.
    * 
    * \demo{constrained_watershed.py}
    */

    template <class T, class labelT>
    RES_T watershed(const Image<T> &imIn, const Image<labelT> &imMarkers, Image<T> &imOut, Image<labelT> &imBasinsOut, const StrElt &se=DEFAULT_SE)
    {
        ASSERT_ALLOCATED(&imIn, &imMarkers);
        ASSERT_SAME_SIZE(&imIn, &imMarkers, &imOut, &imBasinsOut);
 
        WatershedFlooding<T,labelT> flooding;
        return flooding.flood(imIn, imMarkers, imOut, imBasinsOut, se);
    }

    template <class T, class labelT>
    RES_T watershed(const Image<T> &imIn, Image<labelT> &imMarkers, Image<T> &imOut, const StrElt &se=DEFAULT_SE)
    {
        ASSERT_ALLOCATED(&imIn, &imMarkers);
        ASSERT_SAME_SIZE(&imIn, &imMarkers, &imOut);
        
        Image<labelT> imBasinsOut(imMarkers);
        return watershed(imIn, imMarkers, imOut, imBasinsOut, se);
    }

    template <class T>
    RES_T watershed(const Image<T> &imIn, Image<T> &imOut, const StrElt &se=DEFAULT_SE)
    {
        ASSERT_ALLOCATED(&imIn);
        ASSERT_SAME_SIZE(&imIn, &imOut);
        
        Image<UINT> imLbl(imIn);
        minimaLabeled(imIn, imLbl, se);
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
            // if (&maskIm) // a reference cannot be NULL
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
        Image<T2> nullIm(basinsOut);
        fill(nullIm, ImDtTypes<T2>::max()); // invariant to liblSkiz inf
        
        // Create the label images
        label(imIn, basinsOut, se);
        inv(basinsOut, imLbl2);
        mask(imLbl2, basinsOut, imLbl2);
        
        ASSERT(lblSkiz(basinsOut, imLbl2, nullIm, se)==RES_OK);
        
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


