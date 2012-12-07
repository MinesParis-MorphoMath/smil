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


#ifndef _D_MORPHO_FILTER_HPP
#define _D_MORPHO_FILTER_HPP

/**
 * \ingroup Morpho
 * \defgroup Filters
 * @{
 */

#include "DImage.h"
#include "DMorphImageOperations.hpp"

/**
 * Alternate Sequential Filter beginning by a closing
 * 
 * Sequence of closings and openings with increasing size 1,2,...,max_size.
 * The max_size is given by the size of the structuring element (for example 3 for hSE(3)).
 * 
 */ 
template <class T>
RES_T asfClose(const Image<T> &imIn, Image<T> &imOut, const StrElt &se=DEFAULT_SE)
{
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    ImageFreezer freeze(imOut);
    
    Image<T> tmpIm(imIn, true); // clone
    for (int i=1;i<=se.size;i++)
    {
	ASSERT((close(tmpIm, imOut, se(i))==RES_OK));
	ASSERT((open(imOut, tmpIm, se(i))==RES_OK));
    }
    ASSERT((copy(tmpIm, imOut)==RES_OK));
        
    return RES_OK;
}

/**
 * Alternate Sequential Filter beginning by an opening
 * 
 * \see asfClose
 */ 
template <class T>
RES_T asfOpen(const Image<T> &imIn, Image<T> &imOut, const StrElt &se=DEFAULT_SE)
{
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);
    
    ImageFreezer freeze(imOut);
    
    Image<T> tmpIm(imIn, true); // clone
    for (int i=1;i<=se.size;i++)
    {
	ASSERT((open(tmpIm, imOut, se(i))==RES_OK));
	ASSERT((close(imOut, tmpIm, se(i))==RES_OK));
    }
    ASSERT((copy(tmpIm, imOut)==RES_OK));

    return RES_OK;
}

/** @}*/


#endif // _D_MORPHO_FILTER_HPP

