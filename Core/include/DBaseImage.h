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
 *     * Neither the name of the University of California, Berkeley nor the
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

#include "DCommon.h"
#include "DBaseObject.h"


class _SMIL baseImage : public baseObject
{
public:
    baseImage()
      :	width(0), height(0), depth(0),
	allocated(false), baseObject("baseImage")
    {
    }
      
    inline UINT getWidth() const {
        return width;
    }
    inline UINT getHeight() const {
        return height;
    }
    inline UINT getDepth() const {
        return depth;
    }

    inline UINT getAllocatedSize() const {
        return allocatedSize;
    }
    
    inline void getSize(UINT *w, UINT *h, UINT *d)
    {
	*w = width;
	*h = height;
	*d = depth;
    }
    
    inline UINT getPixelCount() const {
        return pixelCount;
    }
    inline UINT getLineCount() const {
        return lineCount;
    }
    inline UINT getSliceCount() const {
        return sliceCount;
    }

    inline bool isAllocated() const {
        return allocated;
    }

    virtual void* getVoidPointer() = 0;
    virtual void modified() = 0;

    inline int getOffsetFromCoords(int x, int y, int z)
    {
	if (x<0 || x>=width) return -1;
	if (y<0 || y>=height) return -1;
	if (z<0 || z>=depth) return -1;
	return z*width*height + y*width + x;
    }

protected:
    UINT dataTypeSize;

    UINT width;
    UINT height;
    UINT depth;

    UINT sliceCount;
    UINT lineCount;
    UINT pixelCount;

    bool allocated;
    
    UINT allocatedSize;

};

/**
 * Check if all images in a list have the same size.
 * The list of images must be finished by NULL.
 */
inline bool haveSameSize(const baseImage *im, ...)
{
    va_list vargs;

    va_start(vargs, im);
    int w = im->getWidth();
    int h = im->getHeight();
    int d = im->getDepth();

    const baseImage *obj;
    while (obj = va_arg(vargs, const baseImage*))
    {
        if (obj->getWidth()!=w) return false;
        if (obj->getHeight()!=h) return false;
        if (obj->getDepth()!=d) return false;
    }
    va_end(vargs);
    return true;
}

/**
 * Check if all images in a list are allocated.
 * The list of images must be finished by NULL.
 */

inline bool areAllocated(const baseImage *im, ...)
{
    va_list vargs;

    va_start(vargs, im);
    if (!im->isAllocated())
        return false;

    const baseImage *obj;
    while (obj = va_arg(vargs, const baseImage*))
        if (!obj->isAllocated()) return false;
    va_end(vargs);
    return true;
}


#endif // _DBASE_IMAGE_H

