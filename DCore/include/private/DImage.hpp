#ifndef _DIMAGE_HPP
#define _DIMAGE_HPP


#include "DCommon.h"

#include "D_Types.h"
#include "D_BaseObject.h"
#include "DBaseImageOperations.hpp"



#ifdef USE_QT
#include <QApplication>
#include "gui/Qt/ImageViewer.h"
#include "gui/Qt/ImageViewerWidget.h"
#include "gui/Qt/QtApp.h"
#endif // USE_QT

#include "DBaseImage.h"


// template <class T>
// const char *getImageDataTypeAsString(Image<T> &im)
// {
//     return "Unknown";
// }



//! Base images class
//
//!
//
template <class T>
class Image : public baseImage
{
public:

    //! Default constructor
    Image();
    Image(UINT w, UINT h, UINT d = 1);
    Image(Image<T> &rhs, bool cloneit=false);
    template <class T2>
    Image(Image<T2> &rhs, bool cloneit=false);

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
    inline Image<T>& clone(Image<T> &rhs);
    template <class T2>
    inline Image<T>& clone(Image<T2> &rhs);
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
    Image<T>& operator < (Image<T> &rhs);
    Image<T>& operator < (T value);
    Image<T>& operator > (Image<T> &rhs);
    Image<T>& operator > (T value);

    Image<T>& operator << (const char *s) { return *this; };
    Image<T>& operator >> (const char *s) { return *this; };
protected:
    pixelType *pixels;
    lineType  *lines;
    sliceType *slices;

    UINT lineAlignment[SIMD_VEC_SIZE];

    RES_T restruct(void);

#ifdef USE_QT
    ImageViewerWidget *viewer;
    inline void updateViewerData();
public:
    inline void setName(const char* name);
    inline void show(const char* name=NULL);
    inline void _show(const char* name=NULL) { return show(name); }
#endif // USE_QT

};

#include "DImage.hxx"





enum DType
{
    dtUINT8, dtUINT16
};

// Image<UINT8> *createImage();
//
template <class T>
Image<T> *createImage(Image<T> &src)
{
    return new Image<T>(src);
}


#endif // _DIMAGE_HPP
