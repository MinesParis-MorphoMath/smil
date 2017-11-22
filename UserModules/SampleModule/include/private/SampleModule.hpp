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


#ifndef _SAMPLE_MODULE_HPP
#define _SAMPLE_MODULE_HPP

#include "Core/include/DCore.h"
#include "Morpho/include/DMorpho.h"

#include <unistd.h> // For usleep


namespace smil
{
    // Sample inv function
    template <class T>
    RES_T samplePixelFunction(const Image<T> &imIn, Image<T> &imOut)
    {
        ASSERT_ALLOCATED(&imIn)
        ASSERT_SAME_SIZE(&imIn, &imOut)
        
        ImageFreezer freeze(imOut);
        
        typename Image<T>::lineType pixelsIn = imIn.getPixels();
        typename Image<T>::lineType pixelsOut = imOut.getPixels();
        
        for (size_t i=0;i<imIn.getPixelCount();i++)
          pixelsOut[i] = ImDtTypes<T>::max() - pixelsIn[i];
        
        return RES_OK;
    }
    
    
    // Sample Morpho functor
    template <class T>
    struct SampleMorphoFunctor: public MorphImageFunctionBase<T>
    {
        virtual inline void processPixel(size_t pointOffset, vector<int> &dOffsets)
        {
            double pixSum = 0;
            
            for (vector<int>::iterator it=dOffsets.begin();it!=dOffsets.end();it++)
              pixSum += this->pixelsIn[ pointOffset + *it ];
            
            this->pixelsOut[ pointOffset ] = T( pixSum / dOffsets.size() );
            
        }
    };

    template <class T>
    RES_T sampleMorphoFunction(const Image<T> &imIn, Image<T> &imOut, const StrElt &se=DEFAULT_SE)
    {
        SampleMorphoFunctor<T> func;
        return func(imIn, imOut, se);
    }  
    
    
    
    // Sample (silly) Flooding functor
    template <class T, class labelT>
    struct SampleFloodingFunctor: public BaseFlooding<T, labelT>
    {
        virtual RES_T initialize(const Image<T> &imIn, Image<labelT> &imLbl, const StrElt &se)
        {
             BaseFlooding<T, labelT>::initialize(imIn, imLbl, se);
             this->imgLbl->updatesEnabled = true; // Enable image repaint
        }
        
        virtual inline void processPixel(const size_t &curOffset)
        {
            this->lblPixels[ curOffset ] = UINT16(255);
            
            BaseFlooding<T, labelT>::processPixel(curOffset);
            
            this->imgLbl->modified(); // Trigger image repaint
            Gui::processEvents();  // Refresh display
            
            usleep(1000);
        }
    };

    template <class T>
    RES_T sampleFloodingFunction(const Image<T> &imIn, Image<T> &imOut, const StrElt &se=DEFAULT_SE)
    {
        SampleFloodingFunctor<T, T> func;
        fill(imOut, T(0));
        size_t pixNbr = imOut.getPixelCount();
        // Put two "random" markers
        imOut.setPixel(pixNbr/3, 1);
        imOut.setPixel(pixNbr*2/3, 2);
        return func.flood(imIn, imOut, imOut, se);
    }  
    
    
}

#endif // _SAMPLE_MODULE_HPP
