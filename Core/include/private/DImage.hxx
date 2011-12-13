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


#ifndef _IMAGE_HXX
#define _IMAGE_HXX

// template <>
// const char *getImageDataTypeAsString<UINT8>(Image<UINT8> &im)
// {
//     return "UINT8 (unsigned char)";
// }
#include "DMemory.hpp"


template <class T>
Image<T>::Image()
  : dataTypeMin(numeric_limits<T>::min()),
    dataTypeMax(numeric_limits<T>::max())
{ 
    init(); 
}

template <class T>
Image<T>::Image(UINT w, UINT h, UINT d)
  : dataTypeMin(numeric_limits<T>::min()),
    dataTypeMax(numeric_limits<T>::max())
{ 
    init(); 
    setSize(w, h, d);
}

template <class T>
Image<T>::Image(const Image<T> &rhs, bool cloneit)
  : dataTypeMin(numeric_limits<T>::min()),
    dataTypeMax(numeric_limits<T>::max())
{ 
    init();
    if (cloneit) clone(rhs);
    else setSize(rhs.getWidth(), rhs.getHeight(), rhs.getDepth());
}

template <class T>
template <class T2>
Image<T>::Image(const Image<T2> &rhs, bool cloneit)
  : dataTypeMin(numeric_limits<T>::min()),
    dataTypeMax(numeric_limits<T>::max())
{ 
    init();
    if (cloneit) clone(rhs);
    else setSize(rhs.getWidth(), rhs.getHeight(), rhs.getDepth());
}


template <class T>
Image<T>::~Image()
{ 
    deallocate();
    if (viewer)
	delete viewer;
    
}



template <class T>
void Image<T>::init() 
{ 
    slices = NULL;
    lines = NULL;
    pixels = NULL;

    dataTypeSize = sizeof(pixelType); 
    
    allocatedWidth = 0;
    allocatedSize = 0;
    
     viewer = NULL;
     name = NULL;
}

template <class T>
inline void Image<T>::modified()
{ 
    if (viewer && viewer->isVisible())
      updateViewerData();
}



template <class T>
inline void Image<T>::setName(const char *_name)
{ 	
    name = _name;
    if (viewer)
	viewer->setName(_name);
}

template <class T>
inline void Image<T>::updateViewerData()
{ 
    if (viewer)
	viewer->loadFromData(pixels, width, height);
}



template <class T>
inline Image<T>& Image<T>::clone(const Image<T> &rhs)
{ 
    bool isAlloc = rhs.isAllocated();
    setSize(rhs.getWidth(), rhs.getHeight(), rhs.getDepth(), isAlloc);
    if (isAlloc)
      memcpy(pixels, rhs.getPixels(), allocatedSize*sizeof(T));
    modified();
    return *this;
}

template <class T>
template <class T2>
inline Image<T>& Image<T>::clone(const Image<T2> &rhs)
{ 
    bool isAlloc = rhs.isAllocated();
    setSize(rhs.getWidth(), rhs.getHeight(), rhs.getDepth(), isAlloc);
    if (isAlloc)
      copy(rhs, *this);
    modified();
    return *this;
}

template <class T>
inline Image<T>& Image<T>::clone(void)
{ 
    static Image<T> newIm(*this, true);
    return newIm;
}

template <class T>
void Image<T>::setSize(int w, int h, int d, bool doAllocate)
{
    if (w==width && h==height && d==depth)
	return;
    
    if (allocated) deallocate();
    
    width = w;
    height = h;
    depth = d;
    
    sliceCount = d;
    lineCount = sliceCount * h;
    pixelCount = lineCount * w;
    
    if (doAllocate) allocate();
    modified();
}

template <class T>
inline RES_T Image<T>::allocate(void)
{
    if (allocated)
	return RES_ERR_BAD_ALLOCATION;
    
    pixels = createAlignedBuffer<T>(pixelCount);
//     pixels = new pixelType[pixelCount];
    
    
    allocated = true;
    allocatedWidth = width;
    allocatedSize = pixelCount*sizeof(T);
    
    restruct();
    
    return RES_OK;
}

template <class T>
RES_T Image<T>::restruct(void)
{
    if (slices)
	delete[] slices;
    if (lines)
	delete[] lines;
    
    lines =  new lineType[lineCount];
    slices = new sliceType[sliceCount];
    
    lineType *cur_line = lines;
    sliceType *cur_slice = slices;
    
    int pixelsPerSlice = allocatedWidth * height;
    
    for (int k=0; k<(int)depth; k++, cur_slice++)
    {
      *cur_slice = cur_line;
      
      for (int j=0; j<(int)height; j++, cur_line++)
	*cur_line = pixels + k*pixelsPerSlice + j*allocatedWidth;
    }
	
    // Calc. line (mis)alignment
    int n = SIMD_VEC_SIZE / sizeof(T);
    int w = width%SIMD_VEC_SIZE;
    for (int i=0;i<n;i++)
    {
      lineAlignment[i] = (SIMD_VEC_SIZE - (i*w)%SIMD_VEC_SIZE)%SIMD_VEC_SIZE;
//       cout << i << " " << lineAlignment[i] << endl;
    }
    
    return RES_OK;
}

template <class T>
inline int Image<T>::getLineAlignment(UINT l)
{
    return lineAlignment[l%(SIMD_VEC_SIZE/sizeof(T))];
}

template <class T>
RES_T Image<T>::deallocate(void)
{
    if (!allocated)
	return RES_OK;
    
    if (slices)
	delete[] slices;
    if (lines)
	delete[] lines;
    if (pixels)
// 		delete[] pixels;
		deleteAlignedBuffer<T>(pixels);
    slices = NULL;
    lines = NULL;
    pixels = NULL;

    allocated = false;
    allocatedWidth = 0;
    allocatedSize = 0;
    
    return RES_OK;
}

template <class T>
void Image<T>::printSelf(ostream &os, bool displayPixVals)
{
    if (name)
      os << "Image name: " << name << endl;
    
    if (depth>1)
      os << "3D image" << endl;
    else
      os << "2D image" << endl;

    T val;
    os << "Data type: " << getDataTypeAsString(val) << endl;
    
    if (depth>1)
      os << "Size: " << width << "x" << height << "x" << depth << endl;
    else
      os << "Size: " << width << "x" << height << endl;
    
    if (allocated) os << "Allocated (" << allocatedSize << " bytes)" << endl;
    else os << "Not allocated" << endl;
    
   
    if (displayPixVals)
    {
	os << "Pixel values:" << endl;
	UINT i, j, k;
	
	for (k=0;k<depth;k++)
	{
	  for (j=0;j<height;j++)
	  {
	    for (i=0;i<width;i++)
	      os << getPixel(i,j,k) << "  ";
	    os << endl;
	  }
	  os << endl;
	}
	os << endl;
    }
    
    cout << endl;   
}

template <class T>
void Image<T>::printSelf(bool displayPixVals)
{
    printSelf(std::cout, displayPixVals);
}



// OPERATORS

template <class T>
void operator << (ostream &os, Image<T> &im)
{
    im.printSelf(os);
}

template <class T>
Image<T>& Image<T>::operator = (Image<T> &rhs)
{
    cout << "= op" << endl;
    this->clone(rhs);
    return *this;
}

template <class T>
Image<T>& Image<T>::operator << (Image<T> &rhs)
{
    copy(rhs, *this);
    return *this;
}

template <class T>
Image<T>& Image<T>::operator << (T value)
{
    fill(*this, value);
    return *this;
}

template <class T>
Image<T>& Image<T>::operator ~()
{
    static Image<T> newIm(*this);
    inv(*this, newIm);
    return newIm;
}

template <class T>
Image<T>& Image<T>::operator + (Image<T> &rhs)
{
    static Image<T> newIm(*this);
    add(*this, rhs, newIm);
    return newIm;
}

template <class T>
Image<T>& Image<T>::operator + (T value)
{
    static Image<T> newIm(*this);
    add(*this, value, newIm);
    return newIm;
}

template <class T>
Image<T>& Image<T>::operator += (Image<T> &rhs)
{
    add(*this, rhs, *this);
    return *this;
}

template <class T>
Image<T>& Image<T>::operator += (T value)
{
    add(*this, value, *this);
    return *this;
}

template <class T>
Image<T>& Image<T>::operator - (Image<T> &rhs)
{
    static Image<T> newIm(*this);
    sub(*this, rhs, newIm);
    return newIm;
}

template <class T>
Image<T>& Image<T>::operator - (T value)
{
    static Image<T> newIm(*this);
    sub(*this, value, newIm);
    return newIm;
}

template <class T>
Image<T>& Image<T>::operator -= (Image<T> &rhs)
{
    sub(*this, rhs, *this);
    return *this;
}

template <class T>
Image<T>& Image<T>::operator -= (T value)
{
    sub(*this, value, *this);
    return *this;
}

template <class T>
Image<T>& Image<T>::operator * (Image<T> &rhs)
{
    static Image<T> newIm(*this);
    mul(*this, rhs, newIm);
    return newIm;
}

template <class T>
Image<T>& Image<T>::operator * (T value)
{
    static Image<T> newIm(*this);
    mul(*this, value, newIm);
    return newIm;
}

template <class T>
Image<T>& Image<T>::operator *= (Image<T> &rhs)
{
    mul(*this, rhs, *this);
    return *this;
}

template <class T>
Image<T>& Image<T>::operator *= (T value)
{
    mul(*this, value, *this);
    return *this;
}

template <class T>
Image<T>& Image<T>::operator / (Image<T> &rhs)
{
    static Image<T> newIm(*this);
    div(*this, rhs, newIm);
    return newIm;
}

template <class T>
Image<T>& Image<T>::operator / (T value)
{
    static Image<T> newIm(*this);
    div(*this, value, newIm);
    return newIm;
}

template <class T>
Image<T>& Image<T>::operator /= (Image<T> &rhs)
{
    div(*this, rhs, *this);
    return *this;
}

template <class T>
Image<T>& Image<T>::operator /= (T value)
{
    div(*this, value, *this);
    return *this;
}

template <class T>
Image<T>& Image<T>::operator < (Image<T> &rhs)
{
    static Image<T> newIm(*this);
    low(*this, rhs, newIm);
    return newIm;
}

template <class T>
Image<T>& Image<T>::operator < (T value)
{
    static Image<T> newIm(*this);
    low(*this, value, newIm);
    return newIm;
}

template <class T>
Image<T>& Image<T>::operator <= (Image<T> &rhs)
{
    static Image<T> newIm(*this);
    lowOrEqu(*this, rhs, newIm);
    return newIm;
}

template <class T>
Image<T>& Image<T>::operator <= (T value)
{
    static Image<T> newIm(*this);
    lowOrEqu(*this, value, newIm);
    return newIm;
}

template <class T>
Image<T>& Image<T>::operator > (Image<T> &rhs)
{
    static Image<T> newIm(*this);
    grt(*this, rhs, newIm);
    return newIm;
}

template <class T>
Image<T>& Image<T>::operator > (T value)
{
    static Image<T> newIm(*this);
    grt(*this, value, newIm);
    return newIm;
}

template <class T>
Image<T>& Image<T>::operator >= (Image<T> &rhs)
{
    static Image<T> newIm(*this);
    grtOrEqu(*this, rhs, newIm);
    return newIm;
}

template <class T>
Image<T>& Image<T>::operator >= (T value)
{
    static Image<T> newIm(*this);
    grtOrEqu(*this, value, newIm);
    return newIm;
}

template <class T>
Image<T>& Image<T>::operator | (Image<T> &rhs)
{
    static Image<T> newIm(*this);
    sup(*this, rhs, newIm);
    return newIm;
}

template <class T>
Image<T>& Image<T>::operator | (T value)
{
    static Image<T> newIm(*this);
    sup(*this, value, newIm);
    return newIm;
}

template <class T>
Image<T>& Image<T>::operator |= (Image<T> &rhs)
{
    sup(*this, rhs, *this);
    return *this;
}

template <class T>
Image<T>& Image<T>::operator |= (T value)
{
    sup(*this, value, *this);
    return *this;
}

template <class T>
Image<T>& Image<T>::operator & (Image<T> &rhs)
{
    static Image<T> newIm(*this);
    inf(*this, rhs, newIm);
    return newIm;
}

template <class T>
Image<T>& Image<T>::operator & (T value)
{
    static Image<T> newIm(*this);
    inf(*this, value, newIm);
    return newIm;
}

template <class T>
Image<T>& Image<T>::operator &= (Image<T> &rhs)
{
    inf(*this, rhs, *this);
    return *this;
}

template <class T>
Image<T>& Image<T>::operator &= (T value)
{
    inf(*this, value, *this);
    return *this;
}

template <class T>
Image<T>& Image<T>::operator << (const T *tab)
{
    for (int i=0;i<pixelCount;i++)
      pixels[i] = tab[i];
    modified();
}



#endif // _IMAGE_HXX
