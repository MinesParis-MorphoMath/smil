#ifndef _DIMAGE_H
#define _DIMAGE_H

#include "DBaseImage.h"
#include "DImage.hpp"
#include "DImageArith.hpp"
#include "DImageMorph.hpp"


typedef Image<UINT8> Image_UINT8;
typedef Image<UINT16> Image_UINT16;
typedef Image<UINT32> Image_UINT32;

// class Image_UINT8 : public Image<UINT8> {};

// #ifdef SWIG
// %{
// #define Image<UINT8> Image_UINT8;
// typedef Image<UINT16> Image_UINT16;
// typedef Image<UINT32> Image_UINT32;
// %}
// #endif // SWIG

// inline RES_T invIm(Image_UINT8 &im1, Image_UINT8 &im2) 
// { return invIm<UINT8>(im1, im2); }


// inline RES_T labelIm(Image<UINT8> &imIn, Image<UINT8> &imOut, StrElt se=DEFAULT_SE);

#endif // _DIMAGE_H

