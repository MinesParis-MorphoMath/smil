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


#ifndef _DIMAGE_HPP
#define _DIMAGE_HPP

#include "DBaseImage.h"
#include "DSignal.h"

template <class T> class imageViewer;

/**
 * \defgroup Core
 * @{
 */


/**
 * Main Image class.
 * 
 */  
template <class T>
class Image : public baseImage
{
    typedef baseImage parentClass;
public:

    //! Default constructor
    Image(bool _triggerEvents=true);
    Image(UINT w, UINT h, UINT d = 1);
    Image(const Image<T> &rhs, bool cloneit=false);
    template <class T2>
    Image(const Image<T2> &rhs, bool cloneit=false);
    Image(const char *fileName);
  
public:
    ~Image();
    
    const char* getTypeAsString()
    {
	T val;
	return getDataTypeAsString(val);
    }
    typedef typename ImDtTypes<T>::pixelType pixelType;
    typedef typename ImDtTypes<T>::lineType lineType;
    typedef typename ImDtTypes<T>::sliceType sliceType;
    typedef typename ImDtTypes<T>::volType volType;
    
    lineType getPixels() const {
        return pixels;
    }
    sliceType getLines() const {
        return lines;
    }
    volType getSlices() const {
        return slices;
    }
    
    //! Return the value of the pixel at pos x,y(,z)
    inline T getPixel(UINT x, UINT y, UINT z=0)
    {
	if (x>=width || y>=height || z>=depth)
	    return T(NULL);
	return pixels[z*width*height+y*width+x];
    }
    inline T getPixel(UINT offset)
    {
	if (offset >= pixelCount)
	    return RES_ERR;
	return pixels[offset];
    }

    inline RES_T setPixel(UINT x, UINT y, UINT z, T value)
    {
	if (x>=width || y>=height || z>=depth)
	    return RES_ERR;
	pixels[z*width*height+y*width+x] = value;
	modified();
	return RES_OK;
    }
    inline RES_T setPixel(UINT x, UINT y, T value)
    {
	return setPixel(x, y, 0, value);
    }
    inline RES_T setPixel(UINT offset, T value)
    {
	if (offset >= pixelCount)
	    return RES_ERR;
	pixels[offset] = value;
	modified();
	return RES_OK;
    }

    const imageViewer<T> *getViewer();
    
    bool updatesEnabled;
    bool isVisible() { return (viewer && viewer->isVisible()); }
    
    virtual void init();
    Image<T>& clone(const Image<T> &rhs);
    template <class T2>
    Image<T>& clone(const Image<T2> &rhs);
    Image<T>& clone(void);
    void setSize(int w, int h, int d = 1, bool doAllocate = true);
    void setSize(baseImage &rhs, bool doAllocate = true) { setSize(rhs.getWidth(), rhs.getHeight(), rhs.getDepth(), doAllocate); }
    RES_T allocate(void);
    RES_T deallocate(void);

    void printSelf(ostream &os, bool displayPixVals);
    virtual void printSelf(ostream &os=std::cout)
    {
	printSelf(os, false);
    }
    void printSelf(bool displayPixVals)
    {
	printSelf(std::cout, displayPixVals);
    }

    void* getVoidPointer(void) {
        return pixels;
    }

#if defined SWIGPYTHON && defined USE_NUMPY
    /**
     * Return a NumPy array containing the \b real image pixel values (not a copy).
     * 
     * \param c_contigous If true, return an array corresponding to the aligment of C data (C_CONTIGOUS NumPy array flag). 
     * If false (default), return a F_CONTIGOUS array.
     * 
     * See \ref numpy_page "NumPy interface page".
     */
    PyObject * getNumArray(bool c_contigous=false);
#endif // defined SWIGPYTHON && defined USE_NUMPY
    
    inline int getLineAlignment(UINT l);

    void modified();

    const T dataTypeMax;
    const T dataTypeMin;

    Image<T>& operator = (Image<T> &rhs);
    //! Copy image
    Image<T>& operator << (Image<T> &rhs);
    //! Fill image
    Image<T>& operator << (T value);
    //! Negate image
    Image<T>& operator ~ ();
    //! Add image
    Image<T>& operator + (Image<T> &rhs);
    //! Add value
    Image<T>& operator + (T value);
    //! Image addition assignment
    Image<T>& operator += (Image<T> &rhs);
    //! Value addition assignment
    Image<T>& operator += (T value);
    //! Sub image
    Image<T>& operator - (Image<T> &rhs);
    //! Sub value
    Image<T>& operator - (T value);
    //! Image subtraction assignment
    Image<T>& operator -= (Image<T> &rhs);
    //! Value subtraction assignment
    Image<T>& operator -= (T value);
    //! Multiply by image
    Image<T>& operator * (Image<T> &rhs);
    //! Multiply by value
    Image<T>& operator * (T value);
    //! Image multiplication assignment
    Image<T>& operator *= (Image<T> &rhs);
    //! Value multiplication assignment
    Image<T>& operator *= (T value);
    //! Divide by image
    Image<T>& operator / (Image<T> &rhs);
    //! Divide by value
    Image<T>& operator / (T value);
    //! Image division assignment
    Image<T>& operator /= (Image<T> &rhs);
    //! Value division assignment
    Image<T>& operator /= (T value);
    //! Equal boolean operator (see \ref equ).
    Image<T>& operator == (Image<T> &rhs);
    //! Inferior boolean operator (see \ref low)
    Image<T>& operator < (Image<T> &rhs);
    //! Inferior boolean operator (see \ref low)
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
    
    Image<T>& operator << (const lineType tab);
    Image<T>& operator << (vector<T> vect);
    
    Image<T>& operator << (const char *s);
    Image<T>& operator >> (const char *s);
protected:
  
    Image<T> *operIm;
    void updateOperIm();
    
    lineType pixels;
    sliceType  lines;
    volType slices;

    UINT lineAlignment[SIMD_VEC_SIZE];

    RES_T restruct(void);

    imageViewer<T> *viewer;
    void createViewer();
    // Specify if the viewer has been created internally
//     ImageViewerWidget *viewer;
    
public:
    virtual void setName(const char *_name);
    virtual void show(const char *_name=NULL, bool labelImage=false);
    virtual void showLabel(const char *_name=NULL) { show(_name, true); }
    virtual void hide() {  if (viewer) viewer->hide(); }

};
  

template <class T>
Image<T> *createImage(Image<T> &src)
{
    return new Image<T>(src);
}


#define SLEEP(im) \
bool im##savedUpdateState = im.updatesEnabled; \
im.updatesEnabled = false;

#define WAKE_UP(im) \
im.updatesEnabled = im##savedUpdateState;


/** @}*/

#endif // _DIMAGE_HPP
