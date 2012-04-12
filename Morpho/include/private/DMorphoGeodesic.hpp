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
 *     * Neither the name of the University of California, Berkeley nor the
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


#ifndef _D_MORPHO_GEODESIC_HPP
#define _D_MORPHO_GEODESIC_HPP

#include "DLineArith.hpp"
#include "DMorphImageOperations.hpp"
#include "DImageArith.hpp"


// Geodesy

template <class T>
RES_T geoDil(Image<T> &imIn, Image<T> &imMask, Image<T> &imOut, StrElt se=DEFAULT_SE())
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
RES_T geoEro(Image<T> &imIn, Image<T> &imMask, Image<T> &imOut, StrElt se=DEFAULT_SE())
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
RES_T build(Image<T> &imIn, Image<T> &imMask, Image<T> &imOut, StrElt se=DEFAULT_SE())
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
RES_T dualBuild(Image<T> &imIn, Image<T> &imMask, Image<T> &imOut, StrElt se=DEFAULT_SE())
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
RES_T fillHoles(Image<T> &imIn, Image<T> &imOut, StrElt se=DEFAULT_SE())
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
RES_T levelPics(Image<T> &imIn, Image<T> &imOut, StrElt se=DEFAULT_SE())
{
    Image<T> tmpIm(imIn);
    inv(imIn, tmpIm);
    fillHoles(tmpIm, imOut);
    inv(imOut, imOut);
    
//     return res;
}



#endif // _D_MORPHO_GEODESIC_HPP

