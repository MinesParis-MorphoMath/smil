/*
 * Smil
 * Copyright (c) 2010 Matthieu Faessel
 *
 * This file is part of Smil.
 *
 * Smil is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * Smil is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Smil.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 */


#ifndef _D_BASE_IMAGE_H
#define _D_BASE_IMAGE_H

#include "DCommon.h"


class _SMIL baseImage
{
public:
    baseImage();
    inline UINT getWidth() const {
        return width;
    }
    inline UINT getHeight() const {
        return height;
    }
    inline UINT getDepth() const {
        return depth;
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

};

// Check if images have the same size
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

// Check if images are allocated
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

