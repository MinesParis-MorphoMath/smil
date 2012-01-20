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


#ifndef _IMAGE_BIN_HXX
#define _IMAGE_BIN_HXX

#include "DImage.hpp"
#include "DBinary.hpp"


template <>
struct ImDtTypes<bool>
{
    typedef bool pixelType;
    typedef pixelType *lineType;
    typedef lineType *sliceType;
    typedef sliceType *volType;

    static pixelType min() { return numeric_limits<bool>::min(); }
    static pixelType max() { return numeric_limits<bool>::max(); }
    static lineType createLine(UINT lineLen) 
    { 
	UINT realSize = BIN::binLen(lineLen);
	return ((bool*) createAlignedBuffer<BIN_TYPE>(realSize));
    }
    static void deleteLine(lineType line) { if (line) deleteAlignedBuffer<BIN::Type>((BIN::Type*)line); }
    static unsigned long ptrOffset(lineType p, unsigned long n=SIMD_VEC_SIZE) { return ((unsigned long)p) & (n-1); }
};


// template<> 
// inline bool *createAlignedBuffer<bool>(int size) 
// {
//     UINT realSize = BIN::binLen(size);
//     return ((bool*) createAlignedBuffer<BIN_TYPE>(realSize));
// }


template <>
inline RES_T Image<bool>::restruct(void)
{
    if (slices)
	delete[] slices;
    if (lines)
	delete[] lines;
    
    lines =  (lineType*) new BIN::lineType[lineCount];
    slices = (sliceType*) new BIN::sliceType[sliceCount];
    
    BIN::Type *bPixels = (BIN::Type*)pixels;
    BIN::lineType *cur_line = (BIN::lineType*)lines;
    BIN::sliceType *cur_slice = (BIN::sliceType*)slices;
    
    UINT realWidth = BIN::binLen(width);
    UINT pixelsPerSlice = realWidth * height;
    
    for (int k=0; k<(int)depth; k++, cur_slice++)
    {
      *cur_slice = cur_line;
      
      for (int j=0; j<(int)height; j++, cur_line++)
	*cur_line = bPixels + k*pixelsPerSlice + j*realWidth;
    }
    
    return RES_OK;
}

template <>
inline RES_T Image<bool>::allocate(void)
{
    if (allocated)
	return RES_ERR_BAD_ALLOCATION;
    
    UINT realWidth = BIN::binLen(width);
    UINT realPixelCount = realWidth*height*depth;
    
    pixels = (bool*)createAlignedBuffer<BIN>(realPixelCount);
//     pixels = new pixelType[pixelCount];
    
    allocated = true;
    allocatedSize = realPixelCount*sizeof(BIN_TYPE);
    
    restruct();
    
    return RES_OK;
}

template <>
inline RES_T Image<bool>::deallocate(void)
{
    if (!allocated)
	return RES_OK;
    
    if (slices)
	delete[] slices;
    if (lines)
	delete[] lines;
    if (pixels)
// 		delete[] pixels;
		deleteAlignedBuffer<BIN>((BIN*)pixels)    ;
    slices = NULL;
    lines = NULL;
    pixels = NULL;

    allocated = false;
    allocatedSize = 0;
    
    return RES_OK;
}


template <>
inline bool Image<bool>::getPixel(UINT x, UINT y, UINT z)
{
    if (x>=width || y>=height || z>=depth)
	return RES_ERR;
    UINT byteCount = width / BIN::SIZE + 1;
    BIN *bLine = (BIN*)slices[z][y];
    return (bLine[int(x/BIN::SIZE)].val & (1UL<<(x%BIN::SIZE)))!=0;
}

template <>
inline bool Image<bool>::getPixel(UINT offset)
{
    if (offset>=pixelCount)
      return RES_ERR;
    UINT realWidth = ((width-1)/BIN::SIZE + 1)*BIN::SIZE;
    UINT lNum = offset / width;
    UINT realOffset = offset + lNum*(realWidth-width);
    BIN *bPixels = (BIN*)pixels;
    return (bPixels[realOffset/BIN::SIZE].val & (1UL<<(realOffset%BIN::SIZE)))!=0;
}

template <>
inline RES_T Image<bool>::setPixel(UINT x, UINT y, UINT z, bool value)
{
    if (x>=width || y>=height || z>=depth)
	return RES_ERR;
    UINT byteCount = width / BIN::SIZE + 1;
    BIN *bLine = (BIN*)slices[z][y];
    if (value)
      bLine[int(x/BIN::SIZE)].val |= (1UL<<(x%BIN::SIZE));
    else
      bLine[int(x/BIN::SIZE)].val &= ~(1UL<<(x%BIN::SIZE));
    modified();
    return RES_OK;
}

template <>
inline RES_T Image<bool>::setPixel(UINT x, UINT y, bool value)
{
    return setPixel(x, y, 0, value);
}


// 
// inline RES_T setPixel(UINT x, UINT y, UINT z, pixelType value)
// {
//     if (x>=width || y>=height || z>=depth)
// 	return RES_ERR;
//     pixels[z*width*height+y*width+x] = value;
//     modified();
//     return RES_OK;
// }
// inline RES_T setPixel(UINT x, UINT y, pixelType value)
// {
//     if (x>=width || y>=height)
// 	return RES_ERR;
//     pixels[height+y*width+x] = value;
//     modified();
//     return RES_OK;
// }
// inline RES_T setPixel(UINT offset, pixelType value)
// {
//     if (offset >= pixelCount)
// 	return RES_ERR;
//     pixels[offset] = value;
//     modified();
//     return RES_OK;
// }



#endif // _IMAGE_BIN_HXX
