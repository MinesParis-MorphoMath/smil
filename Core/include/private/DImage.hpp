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
    
    inline pixelType getPixel(UINT x, UINT y, UINT z=0)
    {
	if (x>=width || y>=height || z>=depth)
	    return RES_ERR;
	return pixels[z*width*height+y*width+x];
    }

    inline RES_T setPixel(pixelType value, UINT x, UINT y, UINT z=0)
    {
	if (x>=width || y>=height || z>=depth)
	    return RES_ERR;
	pixels[z*width*height+y*width+x] = value;
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
    Image<T>& operator < (Image<T> &rhs);
    Image<T>& operator < (T value);
    Image<T>& operator <= (Image<T> &rhs);
    Image<T>& operator <= (T value);
    Image<T>& operator > (Image<T> &rhs);
    Image<T>& operator > (T value);
    Image<T>& operator >= (Image<T> &rhs);
    Image<T>& operator >= (T value);

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
