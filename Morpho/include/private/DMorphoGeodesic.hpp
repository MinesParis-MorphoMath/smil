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


#ifndef _D_MORPHO_GEODESIC_HPP
#define _D_MORPHO_GEODESIC_HPP

#include "DLineArith.hpp"
#include "DMorphImageOperations.hpp"
#include "DImageArith.hpp"


// Geodesy

template <class T>
inline RES_T geoDil(Image<T> &imIn, Image<T> &imMask, Image<T> &imOut, StrElt se=DEFAULT_SE)
{
    StrElt tmpSe(se);
    tmpSe.size = 1;
    
    RES_T res = inf(imIn, imMask, imOut);
    
    for (int i=0;i<se.size;i++)
    {
	res = dilate<T>(imOut, imOut, tmpSe);
	if (res==RES_OK)
	  res = inf(imOut, imMask, imOut);
	if (res!=RES_OK)
	  break;
    }
    return res;
}

template <class T>
inline RES_T geoEro(Image<T> &imIn, Image<T> &imMask, Image<T> &imOut, StrElt se=DEFAULT_SE)
{
    StrElt tmpSe(se);
    tmpSe.size = 1;
    
    RES_T res = sup(imIn, imMask, imOut);
    
    for (int i=0;i<se.size;i++)
    {
	res = erode(imOut, imOut, tmpSe);
	if (res==RES_OK)
	  res = sup(imOut, imMask, imOut);
	if (res!=RES_OK)
	  break;
    }
    return res;
}

template <class T>
inline RES_T build(Image<T> &imIn, Image<T> &imMask, Image<T> &imOut, StrElt se=DEFAULT_SE)
{
    StrElt tmpSe(se);
    tmpSe.size = 1;
    
    RES_T res = inf(imIn, imMask, imOut);
    
    int vol1 = vol(imOut), vol2;
    
    while (true)
    {
	res = dilate<T>(imOut, imOut, tmpSe);
	if (res==RES_OK)
	  res = inf(imOut, imMask, imOut);
	if (res!=RES_OK)
	  break;
	vol2 = vol(imOut);
	if (vol2==vol1)
	  break;
	vol1 = vol2;
    }
    return res;
}

template <class T>
inline RES_T dualBuild(Image<T> &imIn, Image<T> &imMask, Image<T> &imOut, StrElt se=DEFAULT_SE)
{
    StrElt tmpSe(se);
    tmpSe.size = 1;
    
    RES_T res = sup(imIn, imMask, imOut);
    
    int vol1 = vol(imOut), vol2;
    
    while (true)
    {
	res = erode(imOut, imOut, tmpSe);
	if (res==RES_OK)
	  res = sup(imOut, imMask, imOut);
	if (res!=RES_OK)
	  break;
	vol2 = vol(imOut);
	if (vol2==vol1)
	  break;
	vol1 = vol2;
    }
    return res;
}

template <class T>
inline RES_T fillHoles(Image<T> &imIn, Image<T> &imOut, StrElt se=DEFAULT_SE)
{
    StrElt tmpSe(se);
    tmpSe.size = 1;
    
    RES_T res;
    
    Image<T> tmpIm(imIn);
    
    fill(tmpIm, numeric_limits<T>::max());
    dualBuild(tmpIm, imIn, imOut);
    
    return res;
}

template <class T>
inline RES_T levelPics(Image<T> &imIn, Image<T> &imOut, StrElt se=DEFAULT_SE)
{
    Image<T> tmpIm(imIn);
    inv(imIn, tmpIm);
    fillHoles(tmpIm, imOut);
    inv(imOut, imOut);
    
//     return res;
}



#endif // _D_MORPHO_GEODESIC_HPP

