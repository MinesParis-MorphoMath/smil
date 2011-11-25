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


#ifndef _D_MORPHO_EXTREMA_HPP
#define _D_MORPHO_EXTREMA_HPP

#include "DMorphoGeodesic.hpp"
#include "DImageArith.hpp"


// Extrema

template <class T>
inline RES_T hMinima(Image<T> &imIn, UINT height, Image<T> &imOut, StrElt se=DEFAULT_SE)
{
    StrElt tmpSe(se);
    tmpSe.size = 1;
    
    Image<T> *tmpIm;
    
    if (&imIn==&imOut)
      tmpIm = new Image<T>(imIn, true); // clone
    else tmpIm = &imOut;

    RES_T res;
    
    res = add(imIn, T(height), *tmpIm);
    if (res!=RES_OK)
      return res;
    
    res = dualBuild(*tmpIm, imIn, *tmpIm, tmpSe);
    if (res!=RES_OK)
      return res;
    
    low(imIn, *tmpIm, imOut);
    
    return res;
}

template <class T>
inline RES_T hMaxima(Image<T> &imIn, UINT height, Image<T> &imOut, StrElt se=DEFAULT_SE)
{
    StrElt tmpSe(se);
    tmpSe.size = 1;
    
    Image<T> *tmpIm;
    
    if (&imIn==&imOut)
      tmpIm = new Image<T>(imIn, true); // clone
    else tmpIm = &imOut;

    RES_T res;
    
    res = sub(imIn, T(height), *tmpIm);
    if (res!=RES_OK)
      return res;
    
    res = build(*tmpIm, imIn, *tmpIm, tmpSe);
    if (res!=RES_OK)
      return res;
    
    grt(imIn, *tmpIm, imOut);
    
    return res;
}

template <class T>
inline RES_T minima(Image<T> &imIn, Image<T> &imOut, StrElt se=DEFAULT_SE)
{
    return hMinima(imIn, 1, imOut, se);
}

template <class T>
inline RES_T maxima(Image<T> &imIn, Image<T> &imOut, StrElt se=DEFAULT_SE)
{
    return hMaxima(imIn, 1, imOut, se);
}

#endif // _D_MORPHO_EXTREMA_HPP

