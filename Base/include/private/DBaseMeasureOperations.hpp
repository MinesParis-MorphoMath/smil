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


#ifndef _D_BASE_MEASURE_OPERATIONS_HPP
#define _D_BASE_MEASURE_OPERATIONS_HPP

#include "Core/include/private/DImage.hpp"
#include "Base/include/private/DBlob.hpp"

namespace smil
{
  
/**
 * \ingroup Base
 * \defgroup Measures Base measures
 * @{
 */

    template <class T, class _retType>
    struct MeasureFunctionBase
    {
        virtual ~MeasureFunctionBase() {}
        typedef typename Image<T>::lineType lineType;
        typedef _retType retType;
        retType retVal;
        
        virtual void initialize(const Image<T> &/*imIn*/)
        {
            retVal = retType();
        }
        virtual void processSequence(lineType /*lineIn*/, size_t /*size*/) {}
        virtual void finalize(const Image<T> &/*imIn*/) {}
        
        virtual RES_T processImage(const Image<T> &imIn, bool onlyNonZero=false)
        {
            initialize(imIn);
            ASSERT(CHECK_ALLOCATED(&imIn), RES_ERR_BAD_ALLOCATION);
            
            lineType pixels = imIn.getPixels();
            size_t pixCount = imIn.getPixelCount();
            
            if (!onlyNonZero)
                processSequence(pixels, pixCount);
            else
            {
                size_t curSize = 0;
                size_t curStart = 0;
                
                for (size_t i=0;i<pixCount;i++)
                {
                    if (pixels[i]!=T(0))
                    {
                        if (curSize==0 && curStart==0)
                          curStart = i;
                        curSize++;
                    }
                    else if (curSize>0)
                    {
                      processSequence(pixels + curStart, curSize);
                      curStart = 0;
                      curSize = 0;
                    }
                }
                if (curSize>0)
                  processSequence(pixels + curStart, curSize);
            }
            finalize(imIn);
            return RES_OK;
        }
        virtual retType operator()(const Image<T> &imIn, bool onlyNonZero=false)
        {
            processImage(imIn, onlyNonZero);
            return retVal;
        }
        
        virtual retType processImage(const Image<T> &imIn, const Blob &blob)
        {
            initialize(imIn);
            
            ASSERT(CHECK_ALLOCATED(&imIn), RES_ERR_BAD_ALLOCATION, retVal);
            
            lineType pixels = imIn.getPixels();
            Blob::sequences_const_iterator it = blob.sequences.begin();
            Blob::sequences_const_iterator it_end = blob.sequences.end();
            for (;it!=it_end;it++)
              processSequence(pixels + (*it).offset, (*it).size);
            finalize(imIn);
            return retVal;
        }
        virtual retType operator()(const Image<T> &imIn, const Blob &blob)
        {
            processImage(imIn, blob);
            return retVal;
        }
        
    };
    
    template <class T, class _retType>
    struct MeasureFunctionWithPos : public MeasureFunctionBase<T, _retType>
    {
        typedef typename Image<T>::lineType lineType;
        typedef _retType retType;
        virtual void processSequence(lineType /*lineIn*/, size_t /*size*/, size_t /*x*/, size_t /*y*/, size_t /*z*/) {}
        virtual RES_T processImage(const Image<T> &imIn, bool onlyNonZero=false)
        {
            this->initialize(imIn);
            ASSERT(CHECK_ALLOCATED(&imIn), RES_ERR_BAD_ALLOCATION);
            
            typename Image<T>::volType slices = imIn.getSlices();
            typename Image<T>::sliceType lines;
            typename Image<T>::lineType pixels;
            size_t dims[3];
            imIn.getSize(dims);
            
            if (!onlyNonZero)
            {
                for (size_t z=0;z<dims[2];z++)
                {
                    lines = slices[z];
                    for (size_t y=0;y<dims[1];y++)
                      processSequence(lines[y], dims[0], 0, y, z);
                }
            }
            else
            {
                for (size_t z=0;z<dims[2];z++)
                {
                    lines = slices[z];
                    for (size_t y=0;y<dims[1];y++)
                    {
                        pixels = lines[y];
                        size_t curSize = 0;
                        size_t curStart = 0;
                        
                        for (size_t x=0;x<dims[0];x++)
                        {
                            if (pixels[x]!=0)
                            {
                              if (curSize++==0)
                                curStart = x;
                            }
                            else if (curSize>0)
                            {
                              processSequence(pixels+curStart, curSize, curStart, y, z);
                              curSize = 0;
                            }
                        }
                        if (curSize>0)
                          processSequence(pixels+curStart, curSize, curStart, y, z);
                    }
                }
                
            }
            this->finalize(imIn);
            return RES_OK;
        }
        virtual retType processImage(const Image<T> &imIn, const Blob &blob)
        {
            this->initialize(imIn);
            
            ASSERT(CHECK_ALLOCATED(&imIn), RES_ERR_BAD_ALLOCATION, this->retVal);
            
            lineType pixels = imIn.getPixels();
            Blob::sequences_const_iterator it = blob.sequences.begin();
            Blob::sequences_const_iterator it_end = blob.sequences.end();
            size_t x, y, z;
            for (;it!=it_end;it++)
            {
              imIn.getCoordsFromOffset((*it).offset, x, y, z);
              this->processSequence(pixels + (*it).offset, (*it).size, x, y, z);
            }
            this->finalize(imIn);
            return this->retVal;
        }
    private:
        virtual void processSequence(lineType /*lineIn*/, size_t /*size*/) {}
    };
    
    
    
    template <class T, class labelT, class funcT>
    map<labelT, typename funcT::retType> processBlobMeasure(const Image<T> &imIn, const map<labelT, Blob> &blobs)
    {
        typedef typename funcT::retType retType;
        map<labelT, typename funcT::retType> res;
        
        ASSERT(CHECK_ALLOCATED(&imIn), RES_ERR_BAD_ALLOCATION, res);
        
        size_t blobNbr = blobs.size();
        std::vector<labelT> _keys;
        std::vector<retType> _results(blobNbr);

        for (typename map<labelT, Blob>::const_iterator it=blobs.begin();it!=blobs.end();it++)
            _keys.push_back(it->first);
        
        labelT *keys = _keys.data();
        retType *results = _results.data();
        
        size_t i;
        
        #ifdef USE_OPEN_MP
            int nthreads = Core::getInstance()->getNumberOfThreads();
            #pragma omp parallel private(i) num_threads(nthreads)
        #endif // USE_OPEN_MP
        {
            #ifdef USE_OPEN_MP
                #pragma omp for
            #endif // USE_OPEN_MP
            for (i=0;i<blobNbr;i++)
            {
                funcT func;
                const Blob &blob = blobs.at(keys[i]);
                results[i] = func(imIn, blob);
            }
        }
        for (i=0;i<blobNbr;i++)
          res[keys[i]] = results[i];
        return res;
    }
    
    template <class T, class labelT, class funcT>
    map<labelT, typename funcT::retType> processBlobMeasure(const Image<T> &imIn, bool onlyNonZero=true)
    {
        map<labelT, Blob> blobs = computeBlobs(imIn, onlyNonZero);
        return processBlobMeasure<T,labelT,funcT>(imIn, blobs);
    }


/** @}*/

} // namespace smil


#endif // _D_BASE_MEASURE_OPERATIONS_HPP

