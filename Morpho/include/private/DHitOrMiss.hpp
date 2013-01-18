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

#include "DCompositeSE.h"
#include "DMorphoBase.hpp"


namespace smil
{
   /**
    * \ingroup Morpho
    * \defgroup HitOrMiss
    * \{
    */


    /**
    * Hit Or Miss transform
    */
    template <class T>
    RES_T hitOrMiss(const Image<T> &imIn, const StrElt &foreSE, const StrElt &backSE, Image<T> &imOut, T borderVal=ImDtTypes<T>::min())
    {
	ASSERT_ALLOCATED(&imIn, &imOut);
	ASSERT_SAME_SIZE(&imIn, &imOut);
	
	Image<T> tmpIm(imIn);
	ASSERT_ALLOCATED(&tmpIm);
	ImageFreezer freezer(imOut);
	ASSERT((inv<T>(imIn, tmpIm)==RES_OK));
	ASSERT((erode(tmpIm, imOut, backSE, borderVal)==RES_OK));
	ASSERT((erode(imIn, tmpIm, foreSE, borderVal)==RES_OK));
	ASSERT((inf(tmpIm, imOut, imOut)==RES_OK));
	
	return RES_OK;
    }

    /**
    * Hit Or Miss transform
    */
    template <class T>
    RES_T hitOrMiss(const Image<T> &imIn, const CompStrElt &compSE, Image<T> &imOut, T borderVal=ImDtTypes<T>::min())
    {
	return hitOrMiss(imIn, compSE.fgSE, compSE.bgSE, imOut, borderVal);
    }

    /**
    * Hit Or Miss transform
    */
    template <class T>
    RES_T hitOrMiss(const Image<T> &imIn, const CompStrEltList &mhtSE, Image<T> &imOut, T borderVal=ImDtTypes<T>::min())
    {
	ASSERT_ALLOCATED(&imIn, &imOut);
	ASSERT_SAME_SIZE(&imIn, &imOut);
	
	Image<T> tmpIm(imIn);
	ASSERT_ALLOCATED(&tmpIm);
	
	ImageFreezer freezer(imOut);
	ASSERT((fill(imOut, ImDtTypes<T>::min())==RES_OK));
	for (std::vector<CompStrElt>::const_iterator it=mhtSE.compSeList.begin();it!=mhtSE.compSeList.end();it++)
	{
	    ASSERT((hitOrMiss<T>(imIn, (*it).fgSE, (*it).bgSE, tmpIm, borderVal)==RES_OK));
	    ASSERT((sup(imOut, tmpIm, imOut)==RES_OK));
	}
	
	return RES_OK;
    }

    /**
    * Thinning transform
    */
    template <class T>
    RES_T thin(const Image<T> &imIn, const StrElt &foreSE, const StrElt &backSE, Image<T> &imOut)
    {
	ASSERT_ALLOCATED(&imIn, &imOut);
	ASSERT_SAME_SIZE(&imIn, &imOut);
	
	Image<T> tmpIm(imIn);
	ASSERT_ALLOCATED(&tmpIm);
	ImageFreezer freezer(imOut);

	ASSERT((hitOrMiss(imIn, foreSE, backSE, tmpIm)==RES_OK));
	ASSERT((inv(tmpIm, tmpIm)==RES_OK));
	ASSERT((inf(imIn, tmpIm, imOut)==RES_OK));
	
	return RES_OK;
    }
    
    template <class T>
    RES_T thin(const Image<T> &imIn, const CompStrElt &compSE, Image<T> &imOut)
    {
	return thin(imIn, compSE.fgSE, compSE.bgSE, imOut);
    }

    template <class T>
    RES_T thin(const Image<T> &imIn, const CompStrEltList &mhtSE, Image<T> &imOut)
    {
	ASSERT_ALLOCATED(&imIn, &imOut);
	ASSERT_SAME_SIZE(&imIn, &imOut);
	
	Image<T> tmpIm(imIn, true); // clone
	ASSERT_ALLOCATED(&tmpIm);
	
	ImageFreezer freezer(imOut);
	ASSERT((fill(imOut, ImDtTypes<T>::min())==RES_OK));
	for (std::vector<CompStrElt>::const_iterator it=mhtSE.compSeList.begin();it!=mhtSE.compSeList.end();it++)
	{
	    ASSERT((thin<T>(tmpIm, (*it).fgSE, (*it).bgSE, tmpIm)==RES_OK));
	}
	copy(tmpIm, imOut);
	
	return RES_OK;
    }



    /**
    * Thicking transform
    */
    template <class T>
    RES_T thick(const Image<T> &imIn, const StrElt &foreSE, const StrElt &backSE, Image<T> &imOut)
    {
	ASSERT_ALLOCATED(&imIn, &imOut);
	ASSERT_SAME_SIZE(&imIn, &imOut);
	
	Image<T> tmpIm(imIn);
	ASSERT_ALLOCATED(&tmpIm);
	ImageFreezer freezer(imOut);

	ASSERT((hitOrMiss(imIn, foreSE, backSE, tmpIm)==RES_OK));
	ASSERT((sup(imIn, tmpIm, imOut)==RES_OK));
	
	return RES_OK;
    }
    
    template <class T>
    RES_T thick(const Image<T> &imIn, const CompStrElt &compSE, Image<T> &imOut)
    {
	return thick(imIn, compSE.fgSE, compSE.bgSE, imOut);
    }

    template <class T>
    RES_T thick(const Image<T> &imIn, const CompStrEltList &mhtSE, Image<T> &imOut)
    {
	ASSERT_ALLOCATED(&imIn, &imOut);
	ASSERT_SAME_SIZE(&imIn, &imOut);
	
	Image<T> tmpIm(imIn, true); // clone
	ASSERT_ALLOCATED(&tmpIm);
	
	ImageFreezer freezer(imOut);
	ASSERT((fill(imOut, ImDtTypes<T>::min())==RES_OK));
	for (std::vector<CompStrElt>::const_iterator it=mhtSE.compSeList.begin();it!=mhtSE.compSeList.end();it++)
	{
	    ASSERT((thick<T>(tmpIm, (*it).fgSE, (*it).bgSE, tmpIm)==RES_OK));
	}
	copy(tmpIm, imOut);
	
	return RES_OK;
    }



    /**
    * Thinning transform (full)
    */
    template <class T>
    RES_T fullThin(const Image<T> &imIn, const CompStrEltList &mhtSE, Image<T> &imOut)
    {
	ASSERT_ALLOCATED(&imIn, &imOut);
	ASSERT_SAME_SIZE(&imIn, &imOut);
	
	ImageFreezer freezer(imOut);
	
	double v1, v2;
	ASSERT((thin<T>(imIn, mhtSE, imOut)==RES_OK));
	v1 = vol(imOut);
	while(true)
	{
	    ASSERT((thin<T>(imOut, mhtSE, imOut)==RES_OK));
	    v2 = vol(imOut);
	    if (v2==v1)
	      break;
	    v1 = v2;
	}
	
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

    /**
    * Thicking transform (full)
    */
    template <class T>
    RES_T fullThick(const Image<T> &imIn, const CompStrEltList &mhtSE, Image<T> &imOut)
    {
	ASSERT_ALLOCATED(&imIn, &imOut);
	ASSERT_SAME_SIZE(&imIn, &imOut);
	
	ImageFreezer freezer(imOut);
	double v1, v2;
	ASSERT((thick<T>(imIn, mhtSE, imOut)==RES_OK));
	v1 = vol(imOut);
	while(true)
	{
	    ASSERT((thick<T>(imOut, mhtSE, imOut)==RES_OK));
	    v2 = vol(imOut);
	    if (v2==v1)
	      break;
	    v1 = v2;
	}
	
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

} // namespace smil


#endif // _D_THINNING_HPP

