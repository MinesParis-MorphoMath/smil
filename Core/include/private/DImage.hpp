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
 * \tparam T Image data type (UINT8, UINT16, ...)
 */  
template <class T>
class Image : public baseImage
{
    typedef baseImage parentClass;
public:

    //! Default constructor
    Image();
    Image(UINT w, UINT h, UINT d = 1);
    Image(const char *fileName);
  
    ~Image();
    
    // Provide explicit copy constructor and assignment operator
    // Copy constructor
    Image(const Image<T> & rhs, bool cloneData=true);
    template <class T2>
    Image(const Image<T2> &rhs, bool cloneData=true);
    // Assignment operator
    Image<T>& operator = (const Image<T> &rhs)
    {
	this->clone(rhs);
	return *this;
    }
    
  
    //! Get the image type.
    //! \return The type of the image data as a string ("UINT8", "UINT16", ...)
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
    inline T getPixel(UINT x, UINT y, UINT z=0) const
    {
	if (x>=width || y>=height || z>=depth)
	    return T(NULL);
	return pixels[z*width*height+y*width+x];
    }
    //! Return the value of the pixel at a given offset
    inline T getPixel(UINT offset) const
    {
	if (offset >= pixelCount)
	    return RES_ERR;
	return pixels[offset];
    }

    //! Set the value of the pixel at pos x,y,z (for 3D image)
    inline RES_T setPixel(UINT x, UINT y, UINT z, const T &value)
    {
	if (x>=width || y>=height || z>=depth)
	    return RES_ERR;
	pixels[z*width*height+y*width+x] = value;
	modified();
	return RES_OK;
    }
    
    //! Set the value of the pixel at pos x,y
    inline RES_T setPixel(UINT x, UINT y, const T &value)
    {
	return setPixel(x, y, 0, value);
    }
    
    //! Set the value of the pixel at a given offset
    inline RES_T setPixel(UINT offset, const T &value)
    {
	if (offset >= pixelCount)
	    return RES_ERR;
	pixels[offset] = value;
	modified();
	return RES_OK;
    }

    //! Get the image viewer (create one if needed)
    imageViewer<T> *getViewer();
    
    bool updatesEnabled;
    
    //! Check if the image is visible
    //! \return \b true if the viewer is visible, \b false otherwise
    bool isVisible() { return (viewer && viewer->isVisible()); }
    
    virtual void init();
    inline void clone(const Image<T> &rhs);
    template <class T2>
    inline void clone(const Image<T2> &rhs);
//     Image<T> clone(void);
    inline void setSize(UINT w, UINT h, UINT d = 1, bool doAllocate = true);
    inline void setSize(const baseImage &rhs, bool doAllocate = true) 
    { 
	setSize(rhs.getWidth(), rhs.getHeight(), rhs.getDepth(), doAllocate); 
    }
    RES_T allocate(void);
    RES_T deallocate(void);

    void printSelf(ostream &os, bool displayPixVals) const;
    virtual void printSelf(ostream &os=std::cout) const
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

    T dataTypeMin;
    T dataTypeMax;

    //! Copy image
    Image<T>& operator << (const Image<T> &rhs);
    //! Fill image
    Image<T>& operator << (const T &value);
    //! Negate image
    Image<T> operator ~ () const;
    //! Add image
    Image<T> operator + (const Image<T> &rhs);
    //! Add value
    Image<T> operator + (const T &value);
    //! Image addition assignment
    Image<T>& operator += (const Image<T> &rhs);
    //! Value addition assignment
    Image<T>& operator += (const T &value);
    //! Sub image
    Image<T> operator - (const Image<T> &rhs);
    //! Sub value
    Image<T> operator - (const T &value);
    //! Image subtraction assignment
    Image<T>& operator -= (const Image<T> &rhs);
    //! Value subtraction assignment
    Image<T>& operator -= (const T &value);
    //! Multiply by image
    Image<T> operator * (const Image<T> &rhs);
    //! Multiply by value
    Image<T> operator * (const T &value);
    //! Image multiplication assignment
    Image<T>& operator *= (const Image<T> &rhs);
    //! Value multiplication assignment
    Image<T>& operator *= (const T &value);
    //! Divide by image
    Image<T> operator / (const Image<T> &rhs);
    //! Divide by value
    Image<T> operator / (const T &value);
    //! Image division assignment
    Image<T>& operator /= (const Image<T> &rhs);
    //! Value division assignment
    Image<T>& operator /= (const T &value);
    //! Equal boolean operator (see \ref equ).
    Image<T> operator == (const Image<T> &rhs);
    //! Lower boolean operator (see \ref low)
    Image<T> operator < (const Image<T> &rhs);
    //! Lower boolean operator (see \ref low)
    Image<T> operator < (const T &value);
    //! Lower or equal boolean operator (see \ref lowOrEqu)
    Image<T> operator <= (const Image<T> &rhs);
    //! Lower or equal boolean operator (see \ref lowOrEqu)
    Image<T> operator <= (const T &value);
    //! Greater boolean operator (see \ref grt)
    Image<T> operator > (const Image<T> &rhs);
    //! Greater boolean operator (see \ref grt)
    Image<T> operator > (const T &value);
    //! Greater or equal boolean operator (see \ref grt)
    Image<T> operator >= (const Image<T> &rhs);
    //! Greater or equal boolean operator (see \ref grt)
    Image<T> operator >= (const T &value);

    Image<T> operator | (const Image<T> &rhs);
    Image<T> operator | (const T &value);
    Image<T>& operator |= (const Image<T> &rhs);
    Image<T>& operator |= (const T &value);
    Image<T> operator & (const Image<T> &rhs);
    Image<T> operator & (const T &value);
    Image<T>& operator &= (const Image<T> &rhs);
    Image<T>& operator &= (const T &value);
    
    operator bool() { return vol(*this)==numeric_limits<T>::max()*pixelCount; }
    
    //! Import image data from an array
    Image<T>& operator << (const lineType &tab);
    //! Import image data from a vector
    Image<T>& operator << (vector<T> &vect);
    //! Export image data to a vector
    Image<T>& operator >> (vector<T> &vect);
    
    Image<T>& operator << (const char *s);
    Image<T>& operator >> (const char *s);
protected:
  
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
