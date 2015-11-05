/*
 * Copyright (c) 2011-2015, Matthieu FAESSEL and ARMINES
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


#ifndef _D_SHARED_IMAGE_HPP
#define _D_SHARED_IMAGE_HPP

#include "Core/include/private/DImage.hpp"
#include "Core/include/DErrors.h"



namespace smil
{
    /**
    * \ingroup Core
    * @{
    */

    /**
    * Image that uses an existing (1D) data pointer.
    * 
    */  
    template <class T>
    class SharedImage : public Image<T>
    {
      public:
        typedef Image<T> parentClass;
        typedef typename Image<T>::lineType lineType;

        //! Default constructor
        SharedImage()
          : Image<T>()
        {
            this->className = "SharedImage";
            attached = false;
        }
        
        SharedImage(const Image<T> &img)
          : Image<T>()
        {
            this->className = "SharedImage";
            attached = false;
            
            this->attach(img);
        }
        
        SharedImage(lineType dataPtr, size_t width, size_t height, size_t depth=1)
          : Image<T>()
        {
            this->className = "SharedImage";
            attached = false;
            
            this->attach(dataPtr, width, height, depth);
        }
        
        SharedImage(const SharedImage<T> &img)
          : Image<T>()
        {
            this->className = "SharedImage";
            attached = false;
            
            this->clone(img);
        }
        
        virtual ~SharedImage()
        {
            this->detach();
        }
        
        SharedImage<T>& operator = (const SharedImage<T> &rhs)
        {
            attached = false;
            
            this->clone(rhs);
            return *this;
        }
        
        virtual RES_T attach(lineType dataPtr, size_t width, size_t height, size_t depth=1)
        {
            if (dataPtr==NULL)
            {
                ERR_MSG("Source image isn't allocated");
                return RES_ERR;
            }
            else if (dataPtr==this->pixels && width==this->width 
                                        && height==this->height
                                        && depth==this->depth)
            {
                return RES_OK;
            }
            else
            {
                if (attached)
                    detach();
                
                this->pixels = dataPtr;
                this->setSize(width, height, depth);

                this->restruct();
                
                this->attached = true;
                this->allocated = true;
                
                return RES_OK;
            }
        }
        
        virtual RES_T attach(const Image<T> &im)
        {
            return attach(im.getPixels(), im.getWidth(), im.getHeight(), im.getDepth());
        }
        
        virtual RES_T attach(lineType dataPtr)
        {
            return attach(dataPtr, this->width, this->height, this->depth);
        }
        
        virtual RES_T detach()
        {
            if (!attached)
                return RES_OK;
            
            if (this->slices)
                delete[] this->slices;
            if (this->lines)
                delete[] this->lines;

            this->slices = NULL;
            this->lines = NULL;
            this->pixels = NULL;
            
            this->width = 0;
            this->height = 0;
            this->depth = 0;
            
            this->attached = false;
            this->allocated = false;
            
            return RES_OK;
        }
        
        virtual RES_T clone(const SharedImage<T> &rhs)
        {
            return attach(rhs.getPixels(), rhs.getWidth(), rhs.getHeight(), rhs.getDepth());
        }
        
        
      protected:
        bool attached;
        
        virtual RES_T setSize(size_t w, size_t h, size_t d = 1, bool /*doAllocate*/ = true)
        {
            this->width = w;
            this->height = h;
            this->depth = d;

            this->sliceCount = d;
            this->lineCount = d * h;
            this->pixelCount = this->lineCount * w;

            if (this->viewer)
                this->viewer->setImage(*this);

            this->modified();

            return RES_OK;
        }
        
        virtual RES_T allocate()
        {
            return RES_ERR_BAD_ALLOCATION;
        }
        
        virtual RES_T deallocate()
        {
            return RES_ERR_BAD_ALLOCATION;
        }
    };
  
/** @}*/

} // namespace smil


#endif // _D_SHARED_IMAGE_HPP
