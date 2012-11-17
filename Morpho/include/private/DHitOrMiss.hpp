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


#ifndef _D_THINNING_HPP
#define _D_THINNING_HPP

#include "DMorphoBase.hpp"
#include "DCompositeSE.h"

/**
 * \ingroup Morpho
 * \defgroup Thinning
 * \{
 */



template <class T>
RES_T hitOrMiss(const Image<T> &imIn, const StrElt &foreSE, const StrElt &backSE, Image<T> &imOut, T borderVal=ImDtTypes<T>::min())
{
    SLEEP(imOut);
    Image<T> tmpIm(imIn);
    inv<T>(imIn, tmpIm);
    erode(tmpIm, imOut, backSE, borderVal);
    erode<T>(imIn, tmpIm, foreSE, borderVal);
    inf(tmpIm, imOut, imOut);
    WAKE_UP(imOut);
    
    imOut.modified();
    
    return RES_OK;
}

template <class T>
RES_T hitOrMiss(const Image<T> &imIn, const CompStrElt &compSE, Image<T> &imOut, T borderVal=ImDtTypes<T>::min())
{
    return hitOrMiss(imIn, compSE.fgSE, compSE.bgSE, imOut, borderVal);
}

template <class T>
RES_T hitOrMiss(const Image<T> &imIn, const CompStrEltList &mhtSE, Image<T> &imOut, T borderVal=ImDtTypes<T>::min())
{
    Image<T> tmpIm(imIn);
    SLEEP(imOut);
    fill(imOut, ImDtTypes<T>::min());
    for (std::vector<CompStrElt>::const_iterator it=mhtSE.compSeList.begin();it!=mhtSE.compSeList.end();it++)
    {
	hitOrMiss<T>(imIn, (*it).fgSE, (*it).bgSE, tmpIm, borderVal);
	sup(imOut, tmpIm, imOut);
    }
    imOut.modified();
    WAKE_UP(imOut);
    
    return RES_OK;
}

template <class T>
RES_T thin(const Image<T> &imIn, const CompStrEltList &mhtSE, Image<T> &imOut)
{
    SLEEP(imOut);
    Image<T> tmpIm(imIn);
    hitOrMiss<T>(imIn, mhtSE, tmpIm);
    inv(tmpIm, tmpIm);
    inf(imIn, tmpIm, imOut);
    WAKE_UP(imOut);
    
    return RES_OK;
}

template <class T>
RES_T thin(const Image<T> &imIn, const CompStrElt &compSE, Image<T> &imOut)
{
    return thin(imIn, CompStrElt(compSE), imOut);
}

template <class T>
RES_T thin(const Image<T> &imIn, const StrElt &foreSE, const StrElt &backSE, Image<T> &imOut)
{
    return thin(imIn, CompStrElt(CompStrElt(foreSE, backSE)), imOut);
}


template <class T>
RES_T thick(const Image<T> &imIn, const CompStrEltList &mhtSE, Image<T> &imOut)
{
    SLEEP(imOut);
    Image<T> tmpIm(imIn);
    hitOrMiss<T>(imIn, mhtSE, tmpIm);
    sup(imIn, tmpIm, imOut);
    WAKE_UP(imOut);
    
    return RES_OK;
}

template <class T>
RES_T thick(const Image<T> &imIn, const CompStrElt &compSE, Image<T> &imOut)
{
    return thick(imIn, CompStrElt(compSE), imOut);
}

template <class T>
RES_T thick(const Image<T> &imIn, const StrElt &foreSE, const StrElt &backSE, Image<T> &imOut)
{
    return thick(imIn, CompStrElt(CompStrElt(foreSE, backSE)), imOut);
}

template <class T>
RES_T fullThin(const Image<T> &imIn, const CompStrEltList &mhtSE, Image<T> &imOut)
{
    SLEEP(imOut);
    double v1, v2;
    thin<T>(imIn, mhtSE, imOut);
    v1 = vol(imOut);
    while(true)
    {
	thin<T>(imOut, mhtSE, imOut);
	v2 = vol(imOut);
	if (v2==v1)
	  break;
	v1 = v2;
    }
    WAKE_UP(imOut);
    
    return RES_OK;
}

template <class T>
RES_T fullThin(const Image<T> &imIn, const CompStrElt &compSE, Image<T> &imOut)
{
    return fullThin(imIn, CompStrElt(compSE), imOut);
}

template <class T>
RES_T fullThin(const Image<T> &imIn, const StrElt &foreSE, const StrElt &backSE, Image<T> &imOut)
{
    return fullThin(imIn, CompStrElt(CompStrElt(foreSE, backSE)), imOut);
}

template <class T>
RES_T fullThick(const Image<T> &imIn, const CompStrEltList &mhtSE, Image<T> &imOut)
{
    SLEEP(imOut);
    double v1, v2;
    thick<T>(imIn, mhtSE, imOut);
    v1 = vol(imOut);
    while(true)
    {
	thick<T>(imOut, mhtSE, imOut);
	v2 = vol(imOut);
	if (v2==v1)
	  break;
	v1 = v2;
    }
    WAKE_UP(imOut);
    
    return RES_OK;
}

template <class T>
RES_T fullThick(const Image<T> &imIn, const CompStrElt &compSE, Image<T> &imOut)
{
    return fullThick(imIn, CompStrElt(compSE), imOut);
}

template <class T>
RES_T fullThick(const Image<T> &imIn, const StrElt &foreSE, const StrElt &backSE, Image<T> &imOut)
{
    return fullThick(imIn, CompStrElt(CompStrElt(foreSE, backSE)), imOut);
}


/** \} */

#endif // _D_THINNING_HPP

