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


#ifndef _IMAGE_RGB_H
#define _IMAGE_RGB_H


#include "Core/include/private/DImage.hpp"
#include "Base/include/private/DImageArith.hpp"
#include "Core/include/DTypes.h"
#include "Base/include/private/DImageTransform.hpp"
#include "Base/include/private/DMeasures.hpp"

namespace smil
{
  
    template <>
    inline void Image<RGB>::init() 
    { 
        className = "Image";
        
        slices = NULL;
        lines = NULL;
    //     pixels = NULL;

        dataTypeSize = sizeof(pixelType); 
        
        allocatedSize = 0;
        
        viewer = NULL;
        name = "";
        
        updatesEnabled = true;
        
        parentClass::init();
    }

    template <>
    inline RES_T Image< RGB >::restruct(void)
    {
        if (this->slices)
            delete[] this->slices;
        if (this->lines)
            delete[] this->lines;

        this->lines =  new lineType[lineCount];
        this->slices = new sliceType[sliceCount];

        sliceType cur_line = this->lines;
        volType cur_slice = this->slices;

        size_t pixelsPerSlice = this->width * this->height;

        for (size_t k=0; k<depth; k++, cur_slice++)
        {
          *cur_slice = cur_line;

          for (size_t j=0; j<height; j++, cur_line++)
          {
              for (UINT n=0;n<3;n++)
                 (*cur_line).arrays[n] = pixels.arrays[n] + k*pixelsPerSlice + j*width;
          }
        }


        return RES_OK;
    }
    
        
//     template <>
//     inline const char* Image<RGB>::getTypeAsString()
//     {
//         if (this->allocated)
//             return this->pixels.subtypeName.c_str();
//         else return "RGB";
//     }
    
    template <>
    inline void* Image<RGB>::getVoidPointer() 
        {
        return &pixels;
    }

    template <>
    inline RES_T Image<RGB>::allocate()
    {
        if (this->allocated)
            return RES_ERR_BAD_ALLOCATION;

//         this->pixels = ImDtTypes<RGB>::createLine(pixelCount);
    //     pixels = new pixelType[pixelCount];
        this->pixels.createArrays(this->pixelCount);
//         this->pixels.subtypeName = "RGB";

        ASSERT((this->pixels.isAllocated()), "Can't allocate image", RES_ERR_BAD_ALLOCATION);

        this->allocated = true;
        this->allocatedSize = 3*this->pixelCount*sizeof(UINT8);

        this->restruct();

        return RES_OK;
    }
    
    template <>
    inline RES_T Image<RGB>::deallocate()
    {
        if (!this->allocated)
            return RES_OK;

        if (this->slices)
            delete[] this->slices;
        if (this->lines)
            delete[] this->lines;
        if (this->pixels.isAllocated())
          this->pixels.deleteArrays();
//           ImDtTypes<RGB>::deleteLine(pixels);
        
        this->slices = NULL;
        this->lines = NULL;
        for (UINT n=0;n<3;n++)
          this->pixels.arrays[n] = NULL;

        this->allocated = false;
        this->allocatedSize = 0;

        return RES_OK;
    }

    template <>
    inline double vol(const Image<RGB> &imIn)
    {
        if (!imIn.isAllocated())
            return RES_ERR_BAD_ALLOCATION;

        int npix = imIn.getPixelCount();
        ImDtTypes<UINT8>::lineType pixels;
        double vol = 0;

        for (UINT n=0;n<3;n++)
        {
            pixels = imIn.getPixels().arrays[n];
            for (int i=0;i<npix;i++)
                vol += double(pixels[i]);
        }

        return vol;
    }
    
    template <>
    inline Image<RGB>::operator bool()
    { 
        return vol(*this)==255 * pixelCount * 3; 
    }
    
//     template <>
//     inline std::map<RGB, UINT> histogram(const Image<RGB> &imIn)
//     {
//         map<T, UINT> h;
//         for (T i=ImDtTypes<T>::min();;i++)
//         {
//             h.insert(pair<T,UINT>(i, 0));
//             if (i==ImDtTypes<T>::max())
//               break;
//         }
// 
//         typename Image<T>::lineType pixels = imIn.getPixels();
//         for (size_t i=0;i<imIn.getPixelCount();i++)
//             h[pixels[i]]++;
//         
//         return h;
//     }

//     template <>
//     inline void Image<RGB>::printSelf(ostream &os, bool displayPixVals, bool hexaGrid, string indent) const
//     {
//     }
    
#ifndef SWIGPYTHON        
    template <>
    inline char *Image<RGB>::toCharArray()
    {
        cout << "Not implemented for RGB images" << endl;
        return NULL;
    }
#endif // SWIGPYTHON        
    
    
#ifdef USE_QWT
// #include "Gui/Qt/DQtImageViewer.hpp"
#endif // USE_QWT

    template <class T, UINT N>
    inline RES_T
    mul(__attribute__((__unused__)) const Image<MultichannelType<T, N> > &imIn,
        __attribute__((__unused__)) const double &dValue,
        __attribute__((__unused__)) Image<MultichannelType<T, N> > &imOut) {
      return RES_ERR_NOT_IMPLEMENTED;
    }

    template <class T, UINT N, class T2>
    inline RES_T stretchHist(__attribute__((__unused__))
                             const Image<MultichannelType<T, N> > &imIn,
                             __attribute__((__unused__)) Image<T2> &imOut,
                             __attribute__((__unused__)) T2 outMinVal,
                             __attribute__((__unused__)) T2 outMaxVal) {
      return RES_ERR_NOT_IMPLEMENTED;
    }

    template <class T, UINT N, class T1>
    inline RES_T
    stretchHist(__attribute__((__unused__)) const Image<T1> &imIn,
                __attribute__((__unused__))
                Image<MultichannelType<T, N> > &imOut,
                __attribute__((__unused__)) MultichannelType<T, N> outMinVal,
                __attribute__((__unused__)) MultichannelType<T, N> outMaxVal) {
      return RES_ERR_NOT_IMPLEMENTED;
    }
    
    
} // namespace smil

#endif // _IMAGE_RGB_H
