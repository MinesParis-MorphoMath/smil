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

#include "DIO.h"

// template <>
// const char *getImageDataTypeAsString<UINT8>(Image<UINT8> &im)
// {
//     return "UINT8 (unsigned char)";
// }


template <class T>
Image<T>::Image(bool _triggerEvents)
  : baseImage("Image"),
    dataTypeMin(numeric_limits<T>::min()),
    dataTypeMax(numeric_limits<T>::max())
{ 
    triggerEvents = _triggerEvents;
    init();
}

template <class T>
Image<T>::Image(UINT w, UINT h, UINT d)
  : baseImage("Image"),
    dataTypeMin(numeric_limits<T>::min()),
    dataTypeMax(numeric_limits<T>::max())
{ 
    triggerEvents = true;
    init(); 
    setSize(w, h, d);
}

template <class T>
Image<T>::Image(const Image<T> &rhs, bool cloneit)
  : baseImage("Image"),
    dataTypeMin(numeric_limits<T>::min()),
    dataTypeMax(numeric_limits<T>::max())
{ 
    triggerEvents = true;
    init();
    if (cloneit) clone(rhs);
    else setSize(rhs.getWidth(), rhs.getHeight(), rhs.getDepth());
}

template <class T>
template <class T2>
Image<T>::Image(const Image<T2> &rhs, bool cloneit)
  : baseImage("Image"),
    dataTypeMin(numeric_limits<T>::min()),
    dataTypeMax(numeric_limits<T>::max())
{ 
    triggerEvents = true;
    init();
    if (cloneit) clone(rhs);
    else setSize(rhs.getWidth(), rhs.getHeight(), rhs.getDepth());
}

template <class T>
Image<T>::Image(const char *fileName)
  : baseImage("Image"),
    dataTypeMin(numeric_limits<T>::min()),
    dataTypeMax(numeric_limits<T>::max())
{ 
    triggerEvents = true;
    init();
    read(fileName, *this);
}

template <class T>
Image<T>::~Image()
{ 
    hide();
    deallocate();
    if (viewer)
	delete viewer;
    if (operIm)
      delete operIm;
}



template <class T>
void Image<T>::init() 
{ 
    className = "Image";
    
    slices = NULL;
    lines = NULL;
    pixels = NULL;

    dataTypeSize = sizeof(pixelType); 
    
    allocatedSize = 0;
    
     viewer = NULL;
     name = "";
     
     operIm = NULL;
     
     updatesEnabled = true;
     
     parentClass::init();
}

template <class T>
void Image<T>::updateOperIm()
{
    if (!operIm)
      operIm = new Image<T>(false);
    operIm->setSize(*this);
}

template <class T>
void Image<T>::modified()
{ 
    if (updatesEnabled)
      updateViewerData();
//     getCoreInstance()->processEvents();
}



template <class T>
void Image<T>::setName(const char *_name)
{ 	
    parentClass::setName(_name);
    
    if (viewer)
	viewer->setName(_name);
}

template <class T>
void Image<T>::updateViewerData(bool force)
{ 
    if ((viewer && viewer->isVisible()) || force)
	viewer->update();
}

template <class T>
imageViewer<T> *Image<T>::getViewer()
{
    if (!viewer)
        viewer = createViewer<T>(this);
    return viewer;
}

template <class T>
void Image<T>::show(const char* _name, bool labelImage)
{
    if (!viewer)
        viewer = createViewer<T>(this);
    
    if (_name)
        setName(_name);
    
    if (!viewer)
      return;
    
    updateViewerData(true);
    
    if (!labelImage)
      viewer->show();
    else
      viewer->showLabel();
}



template <class T>
Image<T>& Image<T>::clone(const Image<T> &rhs)
{ 
    bool isAlloc = rhs.isAllocated();
    setSize(rhs.getWidth(), rhs.getHeight(), rhs.getDepth(), isAlloc);
    if (isAlloc)
      memcpy(this->pixels, rhs.getPixels(), allocatedSize*sizeof(T));
    modified();
    return *this;
}

template <class T>
template <class T2>
Image<T>& Image<T>::clone(const Image<T2> &rhs)
{ 
    bool isAlloc = rhs.isAllocated();
    setSize(rhs.getWidth(), rhs.getHeight(), rhs.getDepth(), isAlloc);
    if (isAlloc)
      copy(rhs, *this);
    modified();
    return *this;
}

template <class T>
Image<T>& Image<T>::clone(void)
{ 
    updateOperIm();
    return *operIm;
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
RES_T Image<T>::allocate(void)
{
    if (allocated)
	return RES_ERR_BAD_ALLOCATION;
    
    pixels = createAlignedBuffer<T>(pixelCount);
//     pixels = new pixelType[pixelCount];
    
    
    allocated = true;
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
    
    int pixelsPerSlice = width * height;
    
    for (int k=0; k<(int)depth; k++, cur_slice++)
    {
      *cur_slice = cur_line;
      
      for (int j=0; j<(int)height; j++, cur_line++)
	*cur_line = pixels + k*pixelsPerSlice + j*width;
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
    allocatedSize = 0;
    
    return RES_OK;
}



template <class T>
void Image<T>::printSelf(ostream &os, bool displayPixVals)
{
    if (name!="")
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
	      os << (int)getPixel(i,j,k) << ", ";
	    os << endl;
	  }
	  os << endl;
	}
	os << endl;
    }
    
    os << endl;   
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
Image<T>& Image<T>::operator << (const char *s) 
{ 
    read(s, *this); 
    return *this; 
}

template <class T>
Image<T>& Image<T>::operator >> (const char *s) 
{ 
    write(*this, s); 
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
    updateOperIm();
    inv(*this, *operIm);
    return *operIm;
}

template <class T>
Image<T>& Image<T>::operator + (Image<T> &rhs)
{
    updateOperIm();
    add(*this, rhs, *operIm);
    return *operIm;
}

template <class T>
Image<T>& Image<T>::operator + (T value)
{
    updateOperIm();
    add(*this, value, *operIm);
    return *operIm;
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
    updateOperIm();
    sub(*this, rhs, *operIm);
    return *operIm;
}

template <class T>
Image<T>& Image<T>::operator - (T value)
{
    updateOperIm();
    sub(*this, value, *operIm);
    return *operIm;
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
    updateOperIm();
    mul(*this, rhs, *operIm);
    return *operIm;
}

template <class T>
Image<T>& Image<T>::operator * (T value)
{
    updateOperIm();
    mul(*this, value, *operIm);
    return *operIm;
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
    updateOperIm();
    div(*this, rhs, *operIm);
    return *operIm;
}

template <class T>
Image<T>& Image<T>::operator / (T value)
{
    updateOperIm();
    div(*this, value, *operIm);
    return *operIm;
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
Image<T>& Image<T>::operator == (Image<T> &rhs)
{
    updateOperIm();
    equ(*this, rhs, *operIm);
    return *operIm;
}

template <class T>
Image<T>& Image<T>::operator < (Image<T> &rhs)
{
    updateOperIm();
    low(*this, rhs, *operIm);
    return *operIm;
}

template <class T>
Image<T>& Image<T>::operator < (T value)
{
    updateOperIm();
    low(*this, value, *operIm);
    return *operIm;
}

template <class T>
Image<T>& Image<T>::operator <= (Image<T> &rhs)
{
    updateOperIm();
    lowOrEqu(*this, rhs, *operIm);
    return *operIm;
}

template <class T>
Image<T>& Image<T>::operator <= (T value)
{
    updateOperIm();
    lowOrEqu(*this, value, *operIm);
    return *operIm;
}

template <class T>
Image<T>& Image<T>::operator > (Image<T> &rhs)
{
    updateOperIm();
    grt(*this, rhs, *operIm);
    return *operIm;
}

template <class T>
Image<T>& Image<T>::operator > (T value)
{
    updateOperIm();
    grt(*this, value, *operIm);
    return *operIm;
}

template <class T>
Image<T>& Image<T>::operator >= (Image<T> &rhs)
{
    updateOperIm();
    grtOrEqu(*this, rhs, *operIm);
    return *operIm;
}

template <class T>
Image<T>& Image<T>::operator >= (T value)
{
    updateOperIm();
    grtOrEqu(*this, value, *operIm);
    return *operIm;
}

template <class T>
Image<T>& Image<T>::operator | (Image<T> &rhs)
{
    updateOperIm();
    sup(*this, rhs, *operIm);
    return *operIm;
}

template <class T>
Image<T>& Image<T>::operator | (T value)
{
    updateOperIm();
    sup(*this, value, *operIm);
    return *operIm;
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
    updateOperIm();
    inf(*this, rhs, *operIm);
    return *operIm;
}

template <class T>
Image<T>& Image<T>::operator & (T value)
{
    updateOperIm();
    inf(*this, value, *operIm);
    return *operIm;
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
Image<T>& Image<T>::operator << (lineType tab)
{
    for (int i=0;i<pixelCount;i++)
      pixels[i] = tab[i];
    modified();
}

template <class T>
Image<T>& Image<T>::operator << (vector<T> vect)
{
    typename vector<T>::iterator it = vect.begin();
    typename vector<T>::iterator it_end = vect.end();
    
    for (int i=0;i<pixelCount, it!=it_end;i++, it++)
      pixels[i] = *it;
    modified();
}


#endif // _IMAGE_HXX
