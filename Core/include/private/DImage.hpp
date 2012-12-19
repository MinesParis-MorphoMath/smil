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

namespace smil
{
  
    /**
    * \ingroup Core
    * @{
    */

    template <class T> 
    class ImageViewer;


    /**
    * Main Image class.
    * 
    * \tparam T Image data type (UINT8, UINT16, ...)
    */  
    template <class T>
    class Image : public BaseImage
    {
	typedef BaseImage parentClass;
    public:

	//! Default constructor
	Image();
	Image(size_t w, size_t h, size_t d = 1);
	Image(const char *fileName);
      
	virtual ~Image();
	
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
	virtual const char* getTypeAsString()
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
	inline T getPixel(size_t x, size_t y, size_t z=0) const
	{
	    if (x>=width || y>=height || z>=depth)
		return T(NULL);
	    return pixels[z*width*height+y*width+x];
	}
	//! Return the value of the pixel at a given offset
	inline T getPixel(size_t offset) const
	{
	    if (offset >= pixelCount)
		return RES_ERR;
	    return pixels[offset];
	}

	//! Set the value of the pixel at pos x,y,z (for 3D image)
	inline RES_T setPixel(size_t x, size_t y, size_t z, const T &value)
	{
	    if (x>=width || y>=height || z>=depth)
		return RES_ERR;
	    pixels[z*width*height+y*width+x] = value;
	    modified();
	    return RES_OK;
	}
	
	//! Set the value of the pixel at pos x,y
	inline RES_T setPixel(size_t x, size_t y, const T &value)
	{
	    return setPixel(x, y, 0, value);
	}
	
	//! Set the value of the pixel at a given offset
	inline RES_T setPixel(size_t offset, const T &value)
	{
	    if (offset >= pixelCount)
		return RES_ERR;
	    pixels[offset] = value;
	    modified();
	    return RES_OK;
	}

	//! Get the image viewer (create one if needed)
	virtual BaseImageViewer *getViewer();
	
	//! Check if the image is visible
	//! \return \b true if the viewer is visible, \b false otherwise
	virtual bool isVisible();
	
	virtual void init();
	virtual void clone(const Image<T> &rhs);
	template <class T2>
	void clone(const Image<T2> &rhs);
	virtual Image<T> clone(bool cloneData=true)
	{
	    Image<T> im(*this, cloneData);
	    return im;
	}
	virtual RES_T setSize(size_t w, size_t h, size_t d = 1, bool doAllocate = true);
	virtual RES_T setSize(size_t s[3], bool doAllocate = true)
	{
	    return setSize(s[0], s[1], s[2], doAllocate);
	}
	virtual RES_T setSize(const BaseImage &rhs, bool doAllocate = true) 
	{ 
	    return setSize(rhs.getWidth(), rhs.getHeight(), rhs.getDepth(), doAllocate); 
	}
	virtual RES_T setSize(const vector<size_t> s, bool doAllocate = true) 
	{ 
	    if (s.size()==3)
	      return setSize(s[0], s[1], s[2], doAllocate); 
	else return setSize(s[0], s[1], 1, doAllocate);
	}
	virtual RES_T allocate();
	virtual RES_T deallocate();

	void printSelf(ostream &os, bool displayPixVals, string indent="") const;
	virtual void printSelf(ostream &os=std::cout, string indent="") const
	{
	    printSelf(os, false, indent);
	}
	void printSelf(bool displayPixVals)
	{
	    printSelf(std::cout, displayPixVals);
	}
	virtual const char *getInfoString(string indent = "") const 
	{
	    stringstream s;
	    this->printSelf(s, indent);
	    return s.str().c_str();
	}

	virtual void* getVoidPointer(void) 
	{
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
	
	inline int getLineAlignment(size_t l);

	virtual void modified();

	T dataTypeMin;
	T dataTypeMax;

	inline T &operator [] (size_t i) 
	{ 
	    if (i<pixelCount) return this->pixels[i]; 
	}
	
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
	
	operator bool();
	
	//! Import image data from an array
	Image<T>& operator << (const lineType &tab);
	//! Import image data from a vector
	Image<T>& operator << (vector<T> &vect);
	//! Export image data to a vector
	Image<T>& operator >> (vector<T> &vect);
	
	Image<T>& operator << (const char *s);
	inline Image<T>& operator << (const string s) { return this->operator<<(s.c_str()); }
	Image<T>& operator >> (const char *s);
	inline Image<T>& operator >> (const string s) { return this->operator>>(s.c_str()); }
    protected:
      
	lineType pixels;
	sliceType  lines;
	volType slices;

	size_t lineAlignment[SIMD_VEC_SIZE];

	RES_T restruct(void);

	ImageViewer<T> *viewer;
	void createViewer();
	// Specify if the viewer has been created internally
    //     ImageViewerWidget *viewer;
	
    public:
	virtual void setName(const char *_name);
	virtual void show(const char *_name=NULL, bool labelImage=false);
	virtual void showLabel(const char *_name=NULL);
	virtual void showNormal(const char *_name=NULL);
	virtual void hide();

    };
      

    template <class T>
    Image<T> *createImage(const T)
    {
	return new Image<T>();
    }


/** @}*/

} // namespace smil


#endif // _DIMAGE_HPP
