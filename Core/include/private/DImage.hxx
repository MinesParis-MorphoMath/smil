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


#ifndef _IMAGE_HXX
#define _IMAGE_HXX

#include "DIO.h"
#include "DImageViewer.h"

// template <>
// const char *getImageDataTypeAsString<UINT8>(Image<UINT8> &im)
// {
//     return "UINT8 (unsigned char)";
// }


template <class T>
Image<T>::Image()
  : baseImage("Image"),
    dataTypeMin(numeric_limits<T>::min()),
    dataTypeMax(numeric_limits<T>::max())
{ 
    init();
}

template <class T>
Image<T>::Image(UINT w, UINT h, UINT d)
  : baseImage("Image"),
    dataTypeMin(numeric_limits<T>::min()),
    dataTypeMax(numeric_limits<T>::max())
{ 
    init(); 
    setSize(w, h, d);
}

template <class T>
Image<T>::Image(const Image<T> &rhs, bool cloneData)
  : baseImage(rhs),
    dataTypeMin(numeric_limits<T>::min()),
    dataTypeMax(numeric_limits<T>::max())
{ 
    init();
    if (cloneData)
      this->clone(rhs);
    else this->setSize(rhs);
}

template <class T>
template <class T2>
Image<T>::Image(const Image<T2> &rhs, bool cloneData)
  : baseImage(rhs),
    dataTypeMin(numeric_limits<T>::min()),
    dataTypeMax(numeric_limits<T>::max())
{ 
    init();
    if (cloneData) 
      this->clone(rhs);
    else setSize(rhs);
}

template <class T>
void Image<T>::clone(const Image<T> &rhs)
{ 
    bool isAlloc = rhs.isAllocated();
    this->setSize(rhs, isAlloc);
    if (isAlloc)
      memcpy(this->pixels, rhs.getPixels(), this->allocatedSize);
    modified();
}

template <class T>
template <class T2>
void Image<T>::clone(const Image<T2> &rhs)
{ 
    bool isAlloc = rhs.isAllocated();
    this->setSize(rhs, isAlloc);
    if (isAlloc)
      copy(rhs, *this);
    modified();
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
    if (viewer)      
	delete viewer;
    
    this->deallocate();
}



template <class T>
void Image<T>::init() 
{ 
    this->slices = NULL;
    this->lines = NULL;
    this->pixels = NULL;

    this->dataTypeSize = sizeof(pixelType); 
    
    this->viewer = NULL;
    this->updatesEnabled = true;
     
    parentClass::init();
}

template <class T>
void Image<T>::modified()
{ 
    if (viewer)
      viewer->update();
    
    onModified.trigger();
}



template <class T>
void Image<T>::setName(const char *_name)
{ 	
    parentClass::setName(_name);
    
    if (viewer)
	viewer->setName(_name);
}

template <class T>
void Image<T>::createViewer()
{
    if (viewer)
      return;
    
    viewer = getDefaultViewer<T>(this);

}

template <class T>
const imageViewer<T> *Image<T>::getViewer()
{
    createViewer();
    return viewer;
}


template <class T>
void Image<T>::show(const char *_name, bool labelImage)
{
    createViewer();
  
    if (_name)
        setName(_name);
    
    if (!viewer)
      return;
    
    modified();
    
    if (!labelImage)
      viewer->show();
    else
      viewer->showLabel();
}




template <class T>
void Image<T>::setSize(UINT w, UINT h, UINT d, bool doAllocate)
{
    if (w==this->width && h==this->height && d==this->depth)
	return;
    
    if (this->allocated) 
      this->deallocate();
    
    this->width = w;
    this->height = h;
    this->depth = d;
    
    this->sliceCount = d;
    this->lineCount = d * h;
    this->pixelCount = this->lineCount * w;
    
    if (doAllocate) 
      this->allocate();
    
    this->modified();
}

template <class T>
RES_T Image<T>::allocate(void)
{
    if (this->allocated)
	return RES_ERR_BAD_ALLOCATION;
    
//     this->pixels = createAlignedBuffer<T>(pixelCount);
    pixels = new pixelType[pixelCount];
    
    
    this->allocated = true;
    this->allocatedSize = this->pixelCount*sizeof(T);
    
    this->restruct();
    
    return RES_OK;
}

template <class T>
RES_T Image<T>::restruct(void)
{
    if (this->slices)
	delete[] this->slices;
    if (this->lines)
	delete[] this->lines;
    
    this->lines =  new lineType[lineCount];
    this->slices = new sliceType[sliceCount];
    
    lineType *cur_line = this->lines;
    sliceType *cur_slice = this->slices;
    
    int pixelsPerSlice = this->width * this->height;
    
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
      this->lineAlignment[i] = (SIMD_VEC_SIZE - (i*w)%SIMD_VEC_SIZE)%SIMD_VEC_SIZE;
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
    if (!this->allocated)
	return RES_OK;
    
    if (this->slices)
	delete[] this->slices;
    if (this->lines)
	delete[] this->lines;
    if (this->pixels)
		delete[] pixels;
// 		deleteAlignedBuffer<T>(pixels);
    this->slices = NULL;
    this->lines = NULL;
    this->pixels = NULL;

    this->allocated = false;
    this->allocatedSize = 0;
    
    return RES_OK;
}



template <class T>
void Image<T>::printSelf(ostream &os, bool displayPixVals) const
{
#if DEBUG_LEVEL > 1
    cout << "Image::printSelf: " << this << endl;
#endif // DEBUG_LEVEL > 1
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
void operator << (ostream &os, const Image<T> &im)
{
    im.printSelf(os);
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
Image<T>& Image<T>::operator << (const Image<T> &rhs)
{
    copy(rhs, *this);
    return *this;
}

template <class T>
Image<T>& Image<T>::operator << (const T &value)
{
    fill(*this, value);
    return *this;
}

template <class T>
Image<T> Image<T>::operator ~() const
{
    Image<T> im(*this);
    inv(*this, im);
    return im;
}

template <class T>
Image<T> Image<T>::operator + (const Image<T> &rhs)
{
    Image<T> im(*this);
    add(*this, rhs, im);
    return im;
}

template <class T>
Image<T> Image<T>::operator + (const T &value)
{
    Image<T> im(*this);
    add(*this, value, im);
    return im;
}

template <class T>
Image<T>& Image<T>::operator += (const Image<T> &rhs)
{
    add(*this, rhs, *this);
    return *this;
}

template <class T>
Image<T>& Image<T>::operator += (const T &value)
{
    add(*this, value, *this);
    return *this;
}

template <class T>
Image<T> Image<T>::operator - (const Image<T> &rhs)
{
    Image<T> im(*this);
    sub(*this, rhs, im);
    return im;
}

template <class T>
Image<T> Image<T>::operator - (const T &value)
{
    Image<T> im(*this);
    sub(*this, value, im);
    return im;
}

template <class T>
Image<T>& Image<T>::operator -= (const Image<T> &rhs)
{
    sub(*this, rhs, *this);
    return *this;
}

template <class T>
Image<T>& Image<T>::operator -= (const T &value)
{
    sub(*this, value, *this);
    return *this;
}

template <class T>
Image<T> Image<T>::operator * (const Image<T> &rhs)
{
    Image<T> im(*this);
    mul(*this, rhs, im);
    return im;
}

template <class T>
Image<T> Image<T>::operator * (const T &value)
{
    Image<T> im(*this);
    mul(*this, value, im);
    return im;
}

template <class T>
Image<T>& Image<T>::operator *= (const Image<T> &rhs)
{
    mul(*this, rhs, *this);
    return *this;
}

template <class T>
Image<T>& Image<T>::operator *= (const T &value)
{
    mul(*this, value, *this);
    return *this;
}

template <class T>
Image<T> Image<T>::operator / (const Image<T> &rhs)
{
    Image<T> im(*this);
    div(*this, rhs, im);
    return im;
}

template <class T>
Image<T> Image<T>::operator / (const T &value)
{
    Image<T> im(*this);
    div(*this, value, im);
    return im;
}

template <class T>
Image<T>& Image<T>::operator /= (const Image<T> &rhs)
{
    div(*this, rhs, *this);
    return *this;
}

template <class T>
Image<T>& Image<T>::operator /= (const T &value)
{
    div(*this, value, *this);
    return *this;
}

template <class T>
Image<T> Image<T>::operator == (const Image<T> &rhs)
{
    Image<T> im(*this);
    equ(*this, rhs, im);
    return im;
}

template <class T>
Image<T> Image<T>::operator < (const Image<T> &rhs)
{
    Image<T> im(*this);
    low(*this, rhs, im);
    return im;
}

template <class T>
Image<T> Image<T>::operator < (const T &value)
{
    Image<T> im(*this);
    low(*this, value, im);
    return im;
}

template <class T>
Image<T> Image<T>::operator <= (const Image<T> &rhs)
{
    Image<T> im(*this);
    lowOrEqu(*this, rhs, im);
    return im;
}

template <class T>
Image<T> Image<T>::operator <= (const T &value)
{
    Image<T> im(*this);
    lowOrEqu(*this, value, im);
    return im;
}

template <class T>
Image<T> Image<T>::operator > (const Image<T> &rhs)
{
    Image<T> im(*this);
    grt(*this, rhs, im);
    return im;
}

template <class T>
Image<T> Image<T>::operator > (const T &value)
{
    Image<T> im(*this);
    grt(*this, value, im);
    return im;
}

template <class T>
Image<T> Image<T>::operator >= (const Image<T> &rhs)
{
    Image<T> im(*this);
    grtOrEqu(*this, rhs, im);
    return im;
}

template <class T>
Image<T> Image<T>::operator >= (const T &value)
{
    Image<T> im(*this);
    grtOrEqu(*this, value, im);
    return im;
}

template <class T>
Image<T> Image<T>::operator | (const Image<T> &rhs)
{
    Image<T> im(*this);
    sup(*this, rhs, im);
    return im;
}

template <class T>
Image<T> Image<T>::operator | (const T &value)
{
    Image<T> im(*this);
    sup(*this, value, im);
    return im;
}

template <class T>
Image<T>& Image<T>::operator |= (const Image<T> &rhs)
{
    sup(*this, rhs, *this);
    return *this;
}

template <class T>
Image<T>& Image<T>::operator |= (const T &value)
{
    sup(*this, value, *this);
    return *this;
}

template <class T>
Image<T> Image<T>::operator & (const Image<T> &rhs)
{
    Image<T> im(*this);
    inf(*this, rhs, im);
    return im;
}

template <class T>
Image<T> Image<T>::operator & (const T &value)
{
    Image<T> im(*this);
    inf(*this, value, im);
    return im;
}

template <class T>
Image<T>& Image<T>::operator &= (const Image<T> &rhs)
{
    inf(*this, rhs, *this);
    return *this;
}

template <class T>
Image<T>& Image<T>::operator &= (const T &value)
{
    inf(*this, value, *this);
    return *this;
}

template <class T>
Image<T>& Image<T>::operator << (const lineType &tab)
{
    for (UINT i=0;i<pixelCount;i++)
      pixels[i] = tab[i];
    modified();
    return *this;
}

template <class T>
Image<T>& Image<T>::operator << (vector<T> &vect)
{
    typename vector<T>::iterator it = vect.begin();
    typename vector<T>::iterator it_end = vect.end();
    
    for (UINT i=0;i<pixelCount;i++, it++)
    {
      if (it==it_end)
	break;
      pixels[i] = *it;
    }
    modified();
    return *this;
}

#if defined SWIGPYTHON && defined USE_NUMPY
#include "DNumpy.h"

template <class T>
PyObject * Image<T>::getNumArray(bool c_contigous)
{
    npy_intp d[] = { this->getHeight(), this->getWidth(), this->getDepth() }; // axis are inverted...
    PyObject *array = PyArray_SimpleNewFromData(this->getDimension(), d, getNumpyType(*this), this->getPixels());
    
    if (c_contigous)
    {
	return array;
    }
    else
    {
	npy_intp t[] = { 1, 0, 2 };
	PyArray_Dims trans_dims;
	trans_dims.ptr = t;
	trans_dims.len = this->getDimension();
	
	PyObject *res = PyArray_Transpose((PyArrayObject*) array, &trans_dims);
	Py_DECREF(array);
	return res;
    }
}
#endif // defined SWIGPYTHON && defined USE_NUMPY


#endif // _IMAGE_HXX
