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


#ifndef _IMAGE_RGB_H
#define _IMAGE_RGB_H


#include "DImage.hpp"
#include "DImage.hxx"
#include "DTypes.h"
#include "DColor.h"
#include "Base/include/private/DImageTransform.hpp"

namespace smil
{
  
    template <>
    void Image<RGB>::init();

    
    template <>
    void* Image<RGB>::getVoidPointer(void);


    template <>
    RES_T Image<RGB>::allocate();

    template <>
    RES_T Image<RGB>::deallocate();

    template <>
    Image<RGB>::operator bool();
    
    template <>
    void Image<RGB>::printSelf(ostream &os, bool displayPixVals, bool hexaGrid, string indent) const;
    
    #ifdef USE_QWT
    template <>
    void QtImageViewer<RGB>::displayHistogram(bool update);
    template <>
    void QtImageViewer<RGB>::displayProfile(bool update);
#endif // USE_QWT
    
    template <class T, UINT N>
    RES_T mul(const Image< MultichannelType<T,N> > &imIn, const double &dValue, Image< MultichannelType<T,N> > &imOut)
    {
    }

    template <class T, UINT N, class T2>
    RES_T stretchHist(const Image< MultichannelType<T,N> > &imIn, Image<T2> &imOut, T2 outMinVal, T2 outMaxVal)
    {
    }
    
    template <class T, UINT N, class T1>
    RES_T stretchHist(const Image<T1> &imIn, Image< MultichannelType<T,N> > &imOut, MultichannelType<T,N> outMinVal, MultichannelType<T,N> outMaxVal)
    {
    }
    
    
} // namespace smil

#endif // _IMAGE_RGB_H
