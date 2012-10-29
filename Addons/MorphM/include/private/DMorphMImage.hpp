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


#ifndef _D_MORPHM_IMAGE_HPP
#define _D_MORPHM_IMAGE_HPP

#include "DImage.h"
#include "DExtImage.hpp"

#include <morphee/image/include/private/image_T.hpp>
#include <morphee/image/include/imageInterface.hpp>

#ifdef WRAP_PYTHON
#include <boost/python.hpp>
#endif // WRAP_PYTHON

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
	    morphee::Image<T> *mIm = dynamic_cast< morphee::Image<T>* >(&imgInt);
	    if (!mIm)
	      ERR_MSG("Error in morphM dynamic_cast");
	    else
	    {
		typename morphee::Image<T>::i_coordinate_system s = mIm->getSize();
		this->pixels = mIm->rawPointer();
		this->setSize(s[0], s[1], s[2]);
	    }
	}
    }
#if defined(WRAP_PYTHON) || defined(SWIGPYTHON)
    morphmImage(PyObject *obj)
    {
	morphee::ImageInterface *imInt = boost::python::extract<morphee::ImageInterface *>(obj);
	if (imInt)
	{
	    morphee::Image<T> * mIm = dynamic_cast<morphee::Image<T>* >(imInt);
	    if (mIm)
	    {
		if (!mIm->isAllocated())
		    ERR_MSG("Image isn't allocated");
		typename morphee::Image<T>::i_coordinate_system s = mIm->Size();
		this->pixels = mIm->rawPointer();
		this->setSize(mIm->getXSize(), mIm->getYSize(), mIm->getZSize());
	    }
	    else 
	    {
		ERR_MSG("Error in dynamic_cast");
	    }
	}
    }
#endif // defined(WRAP_PYTHON) || defined(SWIGPYTHON)protected:
};

template <class T>
Image<T> fromMorphm(PyObject *obj)
{
    morphee::ImageInterface *imInt = boost::python::extract<morphee::ImageInterface *>(obj);
    if (imInt)
    {
	morphee::Image<T> * mIm = dynamic_cast<morphee::Image<T>* >(imInt);
	if (mIm)
	{
	    morphmImage<T> extIm(*mIm);
	    return extIm;
	}
	else 
	{
	    ERR_MSG("Error in dynamic_cast");
	}
    }
    return NULL;
}

#endif // _D_MORPHM_IMAGE_HPP
