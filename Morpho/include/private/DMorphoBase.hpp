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


#ifndef _D_MORPHO_BASE_HPP
#define _D_MORPHO_BASE_HPP

#include "DLineArith.hpp"
#include "DMorphImageOperations.hpp"
#include "DImageArith.hpp"

template <class T>
inline RES_T label(Image<T> &imIn, Image<T> &imOut, StrElt se=DEFAULT_SE)
{
    T curLabel = 0;
    for (UINT z=0;z<imIn.getSliceCount();z++)
    {
    }
    unaryMorphImageFunction<T, equLine<T> > iFunc(numeric_limits<T>::min());
    return iFunc(imIn, imOut, se);
}

template <class T>
inline RES_T dilate(Image<T> &imIn, Image<T> &imOut, StrElt se=DEFAULT_SE)
{
    unaryMorphImageFunction<T, supLine<T> > iFunc(numeric_limits<T>::min());
    return iFunc(imIn, imOut, se);
}

template <class T>
inline RES_T erode(Image<T> &imIn, Image<T> &imOut, StrElt se=DEFAULT_SE)
{
    unaryMorphImageFunction<T, infLine<T> > iFunc(numeric_limits<T>::min());
    return iFunc(imIn, imOut, se);
}

template <class T>
inline RES_T close(Image<T> &imIn, Image<T> &imOut, StrElt se=DEFAULT_SE)
{
    RES_T res = dilate(imIn, imOut, se);
    if (res==RES_OK)
      res = erode(imOut, imOut, se);
    return res;
}

template <class T>
inline RES_T open(Image<T> &imIn, Image<T> &imOut, StrElt se=DEFAULT_SE)
{
    RES_T res = erode(imIn, imOut, se);
    if (res==RES_OK)
      res = dilate(imOut, imOut, se);
    return res;
}

template <class T>
inline RES_T gradient(Image<T> &imIn, Image<T> &imOut, StrElt dilSe, StrElt eroSe)
{
    Image<T> dilIm(imIn);
    Image<T> eroIm(imIn);
    
    RES_T res = dilate(imIn, dilIm, dilSe);
    if (res==RES_OK)
      res = erode(imIn, eroIm, eroSe);
    if (res==RES_OK)
      res = sub(dilIm, eroIm, imOut);
    return res;
}

template <class T>
inline RES_T gradient(Image<T> &imIn, Image<T> &imOut, StrElt se=DEFAULT_SE)
{
    return gradient(imIn, imOut, se, se);
}




#endif // _D_MORPHO_BASE_HPP

