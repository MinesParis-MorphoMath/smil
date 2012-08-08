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


#ifndef _D_MORPHO_BASE_HPP
#define _D_MORPHO_BASE_HPP

#include "DImage.h"
#include "DImageArith.h"
#include "DMorphImageOperations.hpp"

/**
 * \addtogroup Morpho Mathematical Morphology
 * \{
 */


/**
 * Morphological grayscale dilation
 * 
 * \begintheory{dilation}
 * Denoting an image by \f$ f(x) \f$  and the \ref StrElt "structuring function" by \f$ B(x) \f$, 
 * the grayscale dilation of \f$ f \f$ by \f$ B \f$ is given by \cite serra_image_1982 :
 * \f[ (f\oplus B)(x)=\sup_{y \in \Re^3 }[f(y)+B(x-y)] \f] 
 * \endtheory
 * 
 */
template <class T>
RES_T dilate(const Image<T> &imIn, Image<T> &imOut, const StrElt &se=DEFAULT_SE)
{
    unaryMorphImageFunction<T, supLine<T> > iFunc(numeric_limits<T>::min());
    return iFunc(imIn, imOut, se);
}

template <class T>
RES_T dilate(const Image<T> &imIn, Image<T> &imOut, UINT seSize)
{
    return dilate(imIn, imOut, DEFAULT_SE(seSize));
}

/**
 * Morphological grayscale erosion
 * 
 * \begintheory{erosion}
 * Denoting an image by \f$ f(x) \f$  and the \ref StrElt "structuring function" by \f$ B(x) \f$, 
 * the grayscale dilation of \f$ f \f$ by \f$ B \f$ is given by \cite serra_image_1982 :
 * \f[ (f\ominus B)(x)=\inf_{y \in \Re^3 }[f(y)-B(x-y)] \f]
 * \endtheory
 */
template <class T>
RES_T erode(const Image<T> &imIn, Image<T> &imOut, const StrElt &se=DEFAULT_SE)
{
    unaryMorphImageFunction<T, infLine<T> > iFunc(numeric_limits<T>::max());
    return iFunc(imIn, imOut, se);
}

template <class T>
RES_T erode(const Image<T> &imIn, Image<T> &imOut, UINT seSize)
{
    return erode(imIn, imOut, DEFAULT_SE(seSize));
}

template <class T>
RES_T close(const Image<T> &imIn, Image<T> &imOut, const StrElt &se=DEFAULT_SE)
{
    RES_T res = dilate(imIn, imOut, se);
    if (res==RES_OK)
      res = erode(imOut, imOut, se);
    return res;
}

template <class T>
RES_T close(const Image<T> &imIn, Image<T> &imOut, UINT seSize)
{
    return close(imIn, imOut, DEFAULT_SE(seSize));
}

template <class T>
RES_T open(const Image<T> &imIn, Image<T> &imOut, const StrElt &se=DEFAULT_SE)
{
    RES_T res = erode(imIn, imOut, se);
    if (res==RES_OK)
      res = dilate(imOut, imOut, se);
    return res;
}

template <class T>
RES_T open(const Image<T> &imIn, Image<T> &imOut, UINT seSize)
{
    return open(imIn, imOut, DEFAULT_SE(seSize));
}

template <class T>
RES_T gradient(const Image<T> &imIn, Image<T> &imOut, StrElt &dilSe, StrElt &eroSe)
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
RES_T gradient(const Image<T> &imIn, Image<T> &imOut, const StrElt &se=DEFAULT_SE)
{
    return gradient(imIn, imOut, se, se);
}


/** \} */

#endif // _D_MORPHO_BASE_HPP

