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
inline RES_T Image<BIN>::allocate(void)
{
    if (allocated)
	return RES_ERR_BAD_ALLOCATION;
    
    UINT realWidth = (width-1)/BIN::SIZE + 1;
    UINT realPixelCount = realWidth*height*depth;
    
    pixels = createAlignedBuffer<BIN>(realPixelCount);
//     pixels = new pixelType[pixelCount];
    
    allocatedWidth = realWidth;
    
    restruct();
    
    allocated = true;
    allocatedSize = realPixelCount;
    
    return RES_OK;
}

template <>
inline BIN Image<BIN>::getPixel(UINT x, UINT y, UINT z)
{
    if (x>=width || y>=height || z>=depth)
	return RES_ERR;
    UINT byteCount = width / BIN::SIZE + 1;
    return (slices[z][y][int(x/BIN::SIZE)].val & (1<<(x%BIN::SIZE)))!=0;
}

template <>
inline BIN Image<BIN>::getPixel(UINT offset)
{
    if (offset>=pixelCount)
      return RES_ERR;
    UINT realWidth = ((width-1)/BIN::SIZE + 1)*BIN::SIZE;
    UINT lNum = offset / width;
    UINT realOffset = offset + lNum*(realWidth-width);
    return (pixels[realOffset/BIN::SIZE].val & (1<<(realOffset%BIN::SIZE)))!=0;
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
