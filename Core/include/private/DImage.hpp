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


#ifndef _DIMAGE_HPP
#define _DIMAGE_HPP

#include "DCommon.h"

#include "DTypes.hpp"
#include "DBaseObject.h"
#include "DBaseImageOperations.hpp"



#include "DBaseImage.h"

/**
 * \defgroup Core
 * @{
 */

/**
 * Base image viewer.
 * 
 */  
class baseImageViewer
{
public:
    virtual void show() = 0;
    virtual bool isVisible() =0;
    virtual void setName(const char* name) = 0;
    virtual void loadFromData(void *pixels, UINT w, UINT h) = 0;
};



/**
 * Main Image class.
 * 
 */  
template <class T>
class Image : public baseImage
{
public:

    //! Default constructor
    Image();
    Image(UINT w, UINT h, UINT d = 1);
    Image(const Image<T> &rhs, bool cloneit=false);
    template <class T2>
    Image(const Image<T2> &rhs, bool cloneit=false);

    ~Image();
    
    const char* getTypeAsString()
    {
	T val;
	return getDataTypeAsString(val);
    }
    typedef T pixelType;
    typedef pixelType *lineType;
    typedef lineType *sliceType;

    pixelType *getPixels() const {
        return pixels;
    }
    lineType *getLines() const {
        return lines;
    }
    sliceType *getSlices() const {
        return slices;
    }
    
    //! Return the value of the pixel at pos x,y(,z)
    inline pixelType getPixel(UINT x, UINT y, UINT z=0)
    {
	if (x>=width || y>=height || z>=depth)
	    return RES_ERR;
	return pixels[z*width*height+y*width+x];
    }
    inline pixelType getPixel(UINT offset)
    {
	if (offset >= pixelCount)
	    return RES_ERR;
	return pixels[offset];
    }

    inline RES_T setPixel(UINT x, UINT y, UINT z, pixelType value)
    {
	if (x>=width || y>=height || z>=depth)
	    return RES_ERR;
	pixels[z*width*height+y*width+x] = value;
	modified();
	return RES_OK;
    }
    inline RES_T setPixel(UINT x, UINT y, pixelType value)
    {
	return setPixel(x, y, 0, value);
    }
    inline RES_T setPixel(UINT offset, pixelType value)
    {
	if (offset >= pixelCount)
	    return RES_ERR;
	pixels[offset] = value;
	modified();
	return RES_OK;
    }

    void init();
    inline Image<T>& clone(const Image<T> &rhs);
    template <class T2>
    inline Image<T>& clone(const Image<T2> &rhs);
    inline Image<T>& clone(void);
    void setSize(int w, int h, int d = 1, bool doAllocate = true);
    void setSize(baseImage &rhs, bool doAllocate = true) { setSize(rhs.getWidth(), rhs.getHeight(), rhs.getDepth(), doAllocate); }
    RES_T allocate(void);
    RES_T deallocate(void);

    void printSelf(ostream &os, bool displayPixVals = false);
    void printSelf(bool displayPixVals = false);

    inline void* getVoidPointer(void) {
        return pixels;
    }

    inline int getLineAlignment(UINT l);

    inline void modified();

    const T dataTypeMax;
    const T dataTypeMin;

    Image<T>& operator = (Image<T> &rhs);
    Image<T>& operator << (Image<T> &rhs);
    //! Fill operator
    Image<T>& operator << (T value);
    Image<T>& operator ~ ();
    Image<T>& operator + (Image<T> &rhs);
    Image<T>& operator + (T value);
    Image<T>& operator += (Image<T> &rhs);
    Image<T>& operator += (T value);
    Image<T>& operator - (Image<T> &rhs);
    Image<T>& operator - (T value);
    Image<T>& operator -= (Image<T> &rhs);
    Image<T>& operator -= (T value);
    Image<T>& operator * (Image<T> &rhs);
    Image<T>& operator * (T value);
    Image<T>& operator *= (Image<T> &rhs);
    Image<T>& operator *= (T value);
    Image<T>& operator / (Image<T> &rhs);
    Image<T>& operator / (T value);
    Image<T>& operator /= (Image<T> &rhs);
    Image<T>& operator /= (T value);
    Image<T>& operator == (Image<T> &rhs);
    Image<T>& operator < (Image<T> &rhs);
    Image<T>& operator < (T value);
    Image<T>& operator <= (Image<T> &rhs);
    Image<T>& operator <= (T value);
    Image<T>& operator > (Image<T> &rhs);
    Image<T>& operator > (T value);
    Image<T>& operator >= (Image<T> &rhs);
    Image<T>& operator >= (T value);

    Image<T>& operator | (Image<T> &rhs);
    Image<T>& operator | (T value);
    Image<T>& operator |= (Image<T> &rhs);
    Image<T>& operator |= (T value);
    Image<T>& operator & (Image<T> &rhs);
    Image<T>& operator & (T value);
    Image<T>& operator &= (Image<T> &rhs);
    Image<T>& operator &= (T value);
    
    operator bool() { return vol(*this)==numeric_limits<T>::max()*pixelCount; }
    
    Image<T>& operator << (const T *tab);
    
    Image<T>& operator << (const char *s) { cout << "Not implemented" << endl; return *this; };
    Image<T>& operator >> (const char *s) { cout << "Not implemented" << endl; return *this; };
protected:
    pixelType *pixels;
    lineType  *lines;
    sliceType *slices;

    UINT lineAlignment[SIMD_VEC_SIZE];

    RES_T restruct(void);

    baseImageViewer *viewer;
//     ImageViewerWidget *viewer;
    
    const char* name;
    inline void updateViewerData();
public:
    inline void setName(const char* name);
    void show(const char* name=NULL) {  cout << "Not implemented" << endl; }

};
  
#include "DImage.hxx"
#include "DImage_BIN.hxx"





enum DType
{
    dtUINT8, dtUINT16
};

template <class T>
Image<T> *createImage(Image<T> &src)
{
    return new Image<T>(src);
}


/** @}*/

#endif // _DIMAGE_HPP
