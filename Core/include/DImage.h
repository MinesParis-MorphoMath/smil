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


#ifndef _DIMAGE_H
#define _DIMAGE_H

#include "DBaseImage.h"
#include "DImage.hpp"
#include "DImageArith.hpp"


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

