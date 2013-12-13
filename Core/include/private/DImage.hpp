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

#include <string>

#include "DBaseImage.h"
#include "Gui/include/DBaseImageViewer.h"

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
	//! Contruction with a given size (automatic allocation)
	Image(size_t w, size_t h, size_t d = 1);
	//! Contruction from a file
	Image(const char *fileName);
      
	virtual ~Image();
	
	// Provide explicit copy constructor and assignment operator
	//! Copy constructor
	Image(const Image<T> & rhs, bool cloneData=false);
	template <class T2>
	Image(const Image<T2> &rhs, bool cloneData=false);
	// Assignment operator
	Image<T>& operator = (const Image<T> &rhs)
	{
	    this->clone(rhs);
	    return *this;
	}
	
	Image(BaseImage *_im, bool stealIdentity=false)
	{
	    init();
	    
	    Image *im = castBaseImage(_im, T());
	    if (im==NULL)
	      return;
	    
	    if (!stealIdentity)
	    {
	      setSize(*im);
	      return;
	    }
	    
	    if (!im->isAllocated())
	      return;
	    
	    // Steal BaseImage identity
	    // Transfert data from the BaseImage to this
	    
	    width = im->width;
	    height = im->height;
	    depth = im->depth;
	    
	    sliceCount = im->sliceCount;
	    lineCount = im->lineCount;
	    pixelCount = im->pixelCount;
	    
	    pixels = im->pixels;
	    slices = im->slices;
	    lines = im->lines;
	    
	    allocated = true;
	    allocatedSize = im->allocatedSize;
	    
	    im->allocated = false;
	    im->pixels = NULL;
	    im->slices = NULL;
	    im->lines = NULL;
	    
	}
      
	//! Get the image type.
	//! \return The type of the image data as a string ("UINT8", "UINT16", ...)
	virtual const char* getTypeAsString()
	{
	    return getDataTypeAsString<T>();
	}
	typedef typename ImDtTypes<T>::pixelType pixelType;
	typedef typename ImDtTypes<T>::lineType lineType;
	typedef typename ImDtTypes<T>::sliceType sliceType;
	typedef typename ImDtTypes<T>::volType volType;
	
	//! Get the pixels as a 1D array
	lineType getPixels() const {
	    return pixels;
	}
	//! Get an array containing the start offset of each line
	sliceType getLines() const {
	    return lines;
	}
	//! Get an array containing the start offset of each slice
	volType getSlices() const {
	    return slices;
	}
	
	//! Return the value of the pixel at pos x,y(,z)
	inline T getPixel(size_t x, size_t y, size_t z=0) const
	{
	    ASSERT((x<width && y<height && z<depth), "Coords out of image range", T(0));
	    return pixels[z*width*height+y*width+x];
	}
	//! Return the value of the pixel at a given offset
	inline T getPixel(size_t offset) const
	{
	    ASSERT((offset < pixelCount), "Offset out of image range", T(0));
	    return pixels[offset];
	}

	//! Set the value of the pixel at pos x,y,z (for 3D image)
	inline RES_T setPixel(size_t x, size_t y, size_t z, const T &value)
	{
	    ASSERT((x<width && y<height && z<depth), "Coords out of image range", RES_ERR);
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
	    ASSERT((offset < pixelCount), "Offset out of image range", RES_ERR);
	    pixels[offset] = value;
	    modified();
	    return RES_OK;
	}
	
	//! Copy pixel values to a given array
	void toArray(T outArray[]);
	//! Copy pixel values from a given array
	void fromArray(T inArray[]);

	//! Copy pixel values to a given char array
	void toCharArray(signed char outArray[]);
	char *toCharArray() { return (char *)pixels; }
	//! Copy pixel values from a given char array
	void fromCharArray(signed char inArray[]);

	//! Copy pixel values to a given int array
	void toIntArray(int outArray[]);
	//! Copy pixel values from a given int array
	void fromIntArray(int inArray[]);

	//! Get the image viewer (create one if needed)
	virtual ImageViewer<T> *getViewer();
	
	//! Check if the image is visible
	//! \return \b true if the viewer is visible, \b false otherwise
	virtual bool isVisible();
	
	virtual void init();
	//! Clone from a given image (set same size and copy content)
	virtual void clone(const Image<T> &rhs);
	template <class T2>
	void clone(const Image<T2> &rhs);
	//! Create a clone of the image (with same size and content )
	virtual Image<T> clone(bool cloneData=true)
	{
	    Image<T> im(*this, cloneData);
	    return im;
	}
	//! Set the size of image
	virtual RES_T setSize(size_t w, size_t h, size_t d = 1, bool doAllocate = true);
	//! Set the size of image
	virtual RES_T setSize(size_t s[3], bool doAllocate = true)
	{
	    return setSize(s[0], s[1], s[2], doAllocate);
	}
	//! Set the size of image
	virtual RES_T setSize(const BaseImage &rhs, bool doAllocate = true) 
	{ 
	    return setSize(rhs.getWidth(), rhs.getHeight(), rhs.getDepth(), doAllocate); 
	}
	//! Set the size of image
	virtual RES_T setSize(const vector<UINT> s, bool doAllocate = true) 
	{ 
	    if (s.size()==3)
	      return setSize(s[0], s[1], s[2], doAllocate); 
	    else return setSize(s[0], s[1], 1, doAllocate);
	}
	//! Allocate image
	virtual RES_T allocate();
	//! Deallocate image
	virtual RES_T deallocate();

	/**
	 * Print a description of the image
	 * \param displayPixVals If true, display pixel values
	 * \param hexaGrid If true (and displayPixVals is true), display pixel values as an hexahedral grid
	 * \param indent Optional prefix
	 */	
	void printSelf(ostream &os, bool displayPixVals, bool hexaGrid=false, string indent="") const;
	virtual void printSelf(ostream &os=std::cout, string indent="") const
	{
	    printSelf(os, false, false, indent);
	}
	void printSelf(bool displayPixVals, bool hexaGrid=false)
	{
	    printSelf(std::cout, displayPixVals, hexaGrid);
	}
	//! Get the description of the image as a string
	virtual const char *getInfoString(string indent = "") const 
	{
	    stringstream s;
	    this->printSelf(s, indent);
	    return s.str().c_str();
	}

	//! Get pixels as a void pointer
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
	
	//! Trigger modified event (allows to force display update)
	virtual void modified();

	static T getDataTypeMin() { return ImDtTypes<T>::min(); }
	static T getDataTypeMax() { return ImDtTypes<T>::max(); }

	inline T &operator [] (size_t i) 
	{ 
	    if (i<pixelCount) 
	      return this->pixels[i];
	    ERR_MSG("Offset out of range.");
	    return this->dumPixel;
	}
	
	//! Copy image
	Image<T>& operator << (const Image<T> &rhs);
	//! Fill image
	Image<T>& operator << (const T &value);
	//! Negate image
	Image<T> operator ~ () const;
	Image<T> operator - () const;
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
	//! Diff boolean operator (see \ref equ).
	Image<T> operator != (const Image<T> &rhs);
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
	
	//! Boolean operator
	//! \return \b true, if if every pixel has the max type value ( vol(im)==ImDtTypes<T>::max()*pixelCount )
	//! \return \b false, otherwise
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

	RES_T restruct(void);

	ImageViewer<T> *viewer;
	void createViewer();
	
	T dumPixel;
	// Specify if the viewer has been created internally
    //     ImageViewerWidget *viewer;
	
    public:
	//! Set the name of the image
	virtual void setName(const char *_name);
	//! Show the default viewer associated with the image
	virtual void show(const char *_name=NULL, bool labelImage=false);
	//! Show the default viewer associated with the image using a color lookup table
	virtual void showLabel(const char *_name=NULL);
	virtual void showNormal(const char *_name=NULL);
	//! Hide image
	virtual void hide();

    };
      

    template <class T>
    Image<T> *createImage(const T)
    {
	return new Image<T>();
    }

    template <class T>
    Image<T> *castBaseImage(BaseImage *img, const T &)
    {
	ASSERT(strcmp(getDataTypeAsString<T>(), img->getTypeAsString())==0, "Bad type for cast", NULL);
	return static_cast< Image<T>* >(img);
    }


/** @}*/

} // namespace smil


#endif // _DIMAGE_HPP
