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


#ifndef _D_SHADOW_IMAGE_HPP
#define _D_SHADOW_IMAGE_HPP

#include "DImage.hpp"
#include "DImage.hxx"
#include "DSignal.h"
#include "DErrors.h"

#ifdef USE_MORPHEE
#include <morphee/image/include/private/image_T.hpp>
#include <morphee/image/include/imageInterface.hpp>
#endif // USE_MORPHEE    


/**
 * \ingroup Core
 * @{
 */


/**
 * Image that uses an existing (1D) data pointer.
 * 
 */  
template <class T>
class ExtImage : public Image<T>
{
    typedef Image<T> parentClass;
public:

    //! Default constructor
    ExtImage(const Image<T> &img)
    {
	BaseObject::className = "ExtImage";
	parentClass::init();
	if (!img.isAllocated())
	    ERR_MSG("Source image isn't allocated");
	else
	{
	    this->pixels = img.getPixels();
	    this->setSize(img);
	}
    }
  
    ExtImage(const ExtImage<T> &img)
    {
	BaseObject::className = "ExtImage";
	parentClass::init();
	this->clone(img);
    }
  
    virtual ~ExtImage()
    {
	this->deallocate();
    }
    
    virtual void clone(const ExtImage<T> &rhs)
    {
	this->pixels = rhs.getPixels();
	this->setSize(rhs);
    }
    
protected:
    ExtImage() {}
    virtual RES_T allocate()
    {
	cout << "alloc" << endl;
	if (this->allocated)
	    return RES_ERR_BAD_ALLOCATION;

	if (!this->pixels)
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
  
#ifdef USE_MORPHEE
#include <morphee/image/include/private/image_T.hpp>
#include <morphee/image/include/imageInterface.hpp>


template <class T>
class morphmImage : public ExtImage<T>
{
public:
    typedef ExtImage<T> parentClass;
    
    morphmImage(morphee::Image<T> &img)
    {
	BaseObject::className = "morphmImage";
	parentClass::init();
	if (!img.isAllocated())
	    ERR_MSG("Source image isn't allocated");
	else
	{
	    typename morphee::Image<T>::i_coordinate_system s = img.getSize();
	    this->pixels = img.rawPointer();
	    this->setSize(s[0], s[1], s[2]);
	}
    }
    morphmImage(morphee::ImageInterface &imgInt)
    {
	BaseObject::className = "morphmImage";
	parentClass::init();
	if (!imgInt.isAllocated())
	    ERR_MSG("Source image isn't allocated");
	else
	{
	    typename morphee::Image<T>::i_coordinate_system s = imgInt.getSize();
	    cout << s[0] << "," << s[1] << "," << s[2] << endl;
	    morphee::Image<T> *mIm = dynamic_cast< morphee::Image<T>* >(&imgInt);
	    if (!mIm)
	      ERR_MSG("Error in morphM dynamic_cast");
	    else
	    {
		this->pixels = mIm->rawPointer();
		this->setSize(mIm->getXSize(), mIm->getYSize(), mIm->getZSize());
	    }
	}
    }
protected:
};

#endif // USE_MORPHEE    

/** @}*/

#endif // _D_SHADOW_IMAGE_HPP
