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


#ifndef _D_SHARED_IMAGE_HPP
#define _D_SHARED_IMAGE_HPP

#include "DImage.hpp"
#include "DErrors.h"



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
	SharedImage(const Image<T> &img)
	{
	    this->className = "SharedImage";
	    parentClass::init();
	    if (!img.isAllocated())
		ERR_MSG("Source image isn't allocated");
	    else
	    {
		this->pixels = img.getPixels();
		this->setSize(img);
	    }
	}
      
	SharedImage(lineType dataPtr, size_t width, size_t height, size_t depth=1)
	{
	    this->className = "SharedImage";
	    parentClass::init();
	    
	    this->pixels = dataPtr;
	    this->setSize(width, height, depth);
	}
      
	SharedImage(const SharedImage<T> &img)
	{
	    this->className = "SharedImage";
	    parentClass::init();
	    this->clone(img);
	}
      
	virtual ~SharedImage()
	{
	    this->deallocate();
	}
	
	virtual void clone(const SharedImage<T> &rhs)
	{
	    this->pixels = rhs.getPixels();
	    this->setSize(rhs);
	}
	
    protected:
	SharedImage() {}
	virtual RES_T allocate()
	{
	    if (this->allocated)
		return RES_ERR_BAD_ALLOCATION;

	    if (this->pixels==NULL)
		return RES_ERR_BAD_ALLOCATION;
	    
	    this->allocated = true;
	    this->allocatedSize = this->pixelCount*sizeof(T);

	    this->restruct();

	    return RES_OK;
	}
	
	virtual RES_T deallocate()
	{
	    if (!this->allocated)
		return RES_OK;

	    if (this->slices)
		delete[] this->slices;
	    if (this->lines)
		delete[] this->lines;
	    
	    this->slices = NULL;
	    this->lines = NULL;
	    this->pixels = NULL;

	    this->allocated = false;
	    this->allocatedSize = 0;

	    return RES_OK;
	}
    };
  
/** @}*/

} // namespace smil


#endif // _D_SHARED_IMAGE_HPP
