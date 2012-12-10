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


#ifndef _D_BASE_IMAGE_H
#define _D_BASE_IMAGE_H

#include "DBaseObject.h"
#include "DSignal.h"
#include "DSlot.h"
#include "DCommon.h"
#include "DErrors.h"
#include "DCoreInstance.h"

class BaseImageViewer;

/**
 * Base Image class
 */
class _DCORE BaseImage : public BaseObject
{
    typedef BaseObject parentClass;
public:
    BaseImage(const string _className="BaseImage")
      :	BaseObject(_className),
    updatesEnabled(true),
    onModified(this),
    width(0), height(0), depth(0),
    pixelCount(0), lineCount(0), sliceCount(0),
    allocated(false),
    allocatedSize(0)
    {
    }
    
    BaseImage(const BaseImage &rhs)
      :	BaseObject(rhs),
    updatesEnabled(true),
    onModified(this),
    width(0), height(0), depth(0),
    pixelCount(0), lineCount(0), sliceCount(0),
    allocated(false),
    allocatedSize(0)
    {
    }
    
    
private:
    // Forbid implicit assignment operator
    BaseImage& operator=(const BaseImage &rhs);

public:
  
    virtual void init();
    
    inline size_t getWidth() const {
        return width;
    }
    inline size_t getHeight() const {
        return height;
    }
    inline size_t getDepth() const {
        return depth;
    }

    inline size_t getAllocatedSize() const {
        return allocatedSize;
    }
    
    inline UINT getDimension()
    {
	if (depth > 1)
	  return 3;
	else if (height > 1)
	  return 2;
	else return 1;
    }
    
    virtual RES_T setSize(size_t w, size_t h, size_t d = 1, bool doAllocate = true) = 0;
    
    inline void getSize(size_t *w, size_t *h, size_t *d) const
    {
	*w = this->width;
	*h = this->height;
	*d = this->depth;
    }
    
    inline void getSize(size_t s[3]) const
    {
	s[0] = this->width;
	s[1] = this->height;
	s[2] = this->depth;
    }
    
    inline size_t getPixelCount() const {
        return this->pixelCount;
    }
    inline size_t getLineCount() const {
        return this->lineCount;
    }
    inline size_t getSliceCount() const {
        return this->sliceCount;
    }

    inline bool isAllocated() const {
        return this->allocated;
    }

    virtual void* getVoidPointer() = 0;
    virtual void modified() = 0;

    inline size_t getOffsetFromCoords(size_t x, size_t y, size_t z) const
    {
	if (x>=this->width) return -1;
	if (y>=this->height) return -1;
	if (z>=this->depth) return -1;
	return z*this->width*this->height + y*this->width + x;
    }

    inline void getCoordsFromOffset(size_t off, size_t &x, size_t &y, size_t &z) const
    {
	z = off / (this->width*this->height);
	y = (off % (this->width*this->height))/this->width;
	x = off % this->width;
    }

    virtual const char *getInfoString(const char * = "") const { return NULL; }
    virtual const char* getTypeAsString() = 0;
    
    virtual bool isVisible() { return false; }
    virtual void show(const char* = NULL, bool = false) {}
    virtual void showLabel(const char * = NULL) {}
    virtual void hide() = 0;
    
    virtual BaseImageViewer *getViewer() = 0;
    
    bool updatesEnabled;
    Signal onModified;
protected:
    size_t dataTypeSize;

    size_t width;
    size_t height;
    size_t depth;

    size_t pixelCount;
    size_t lineCount;
    size_t sliceCount;

    bool allocated;
    
    size_t allocatedSize;
    

};


class ImageFreezer
{
public:
    ImageFreezer(BaseImage &im, bool updateOnDelete=true)
      : image(&im),
	update(updateOnDelete)
    {
	imState = im.updatesEnabled;
	im.updatesEnabled = false;
    }
    ~ImageFreezer()
    {
	image->updatesEnabled = imState;
	if (update)
	  image->modified();
	
    }
protected:
    BaseImage *image;
    bool imState;
    bool update;
};

/**
 * Check if all images in a list have the same size.
 * The list of images must be finished by NULL.
 */
inline bool haveSameSize(const BaseImage *im, ...)
{
    va_list vargs;

    va_start(vargs, im);
    size_t w = im->getWidth();
    size_t h = im->getHeight();
    size_t d = im->getDepth();

    BaseImage *obj;
    while ((obj = va_arg(vargs, BaseImage*)))
    {
	if (obj->getWidth()!=w) return false;
	if (obj->getHeight()!=h) return false;
	if (obj->getDepth()!=d) return false;
    }
    va_end(vargs);
    return true;
}

inline bool setSameSize(const BaseImage *im, ...)
{
    if (!im->isAllocated())
      return false;
    
    va_list vargs;

    va_start(vargs, im);
    size_t w = im->getWidth();
    size_t h = im->getHeight();
    size_t d = im->getDepth();

    BaseImage *obj;
    
    while ((obj = va_arg(vargs, BaseImage*)))
    {
	if (obj->getWidth()!=w || obj->getHeight()!=h || obj->getDepth()!=d)
	  if (obj->setSize(w, h, d)!=RES_OK)
	    return false;
    }
    return true;
}

/**
 * Check if all images in a list are allocated.
 * The list of images must be finished by NULL.
 */

inline bool areAllocated(const BaseImage *im, ...)
{
    va_list vargs;

    va_start(vargs, im);
    if (!im->isAllocated())
        return false;

    const BaseImage *obj;
    while ((obj = va_arg(vargs, const BaseImage*)))
        if (!obj->isAllocated()) return false;
    va_end(vargs);
    return true;
}

#define CHECK_ALLOCATED(...) (areAllocated(__VA_ARGS__, NULL))
// #define CHECK_ALLOCATED(...) (Core::getInstance()->autoResizeImages ? setSameSize(__VA_ARGS__, NULL) : areAllocated(__VA_ARGS__, NULL))
#define ASSERT_ALLOCATED(...) ASSERT(CHECK_ALLOCATED(__VA_ARGS__), RES_ERR_BAD_ALLOCATION)

#define CHECK_SAME_SIZE(...) (Core::getInstance()->autoResizeImages ? setSameSize(__VA_ARGS__, NULL) : haveSameSize(__VA_ARGS__, NULL))
#define ASSERT_SAME_SIZE(...) ASSERT(CHECK_SAME_SIZE(__VA_ARGS__), RES_ERR_BAD_SIZE)

#endif // _DBASE_IMAGE_H

