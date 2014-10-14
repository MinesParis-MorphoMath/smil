/*
 * Copyright (c) 2011-2014, Matthieu FAESSEL and ARMINES
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


#ifndef _D_MORPHO_ARROW_HPP
#define _D_MORPHO_ARROW_HPP

#include "DMorphImageOperations.hpp"


namespace smil
{
   /**
    * \ingroup Morpho
    * \defgroup Arrow
    * @{
    */


    template <class T, class lineFunction_T>
    class unaryMorphArrowImageFunction : public unaryMorphImageFunction<T, lineFunction_T>
    {
    public:
	typedef unaryMorphImageFunction<T, lineFunction_T> parentClass;
	typedef Image<T> imageType;
	typedef typename imageType::lineType lineType;
	typedef typename imageType::sliceType sliceType;
	typedef typename imageType::volType volType;
	
	unaryMorphArrowImageFunction(T border=numeric_limits<T>::min()) 
	  : unaryMorphImageFunction<T, lineFunction_T>(border) 
	{
	}
	virtual RES_T _exec_single(const imageType &imIn, imageType &imOut, const StrElt &se);
	virtual RES_T _exec_single_generic(const imageType &imIn, imageType &imOut, const StrElt &se);
    };


    template <class T, class lineFunction_T>
    RES_T unaryMorphArrowImageFunction<T, lineFunction_T>::_exec_single(const imageType &imIn, imageType &imOut, const StrElt &se)
    {
	return _exec_single_generic(imIn, imOut, se);
    }

    template <class T, class lineFunction_T>
    RES_T unaryMorphArrowImageFunction<T, lineFunction_T>::_exec_single_generic(const imageType &imIn, imageType &imOut, const StrElt &se)
    {
	ASSERT_ALLOCATED(&imIn, &imOut);
	ASSERT_SAME_SIZE(&imIn, &imOut);
	
	if (&imIn==&imOut)
	{
	    Image<T> tmpIm = imIn;
	    return _exec_single_generic(tmpIm, imOut, se);
	}
	
	if (!areAllocated(&imIn, &imOut, NULL))
	  return RES_ERR_BAD_ALLOCATION;

	UINT sePtsNumber = se.points.size();
	if (sePtsNumber==0)
	    return RES_OK;
	
	size_t nSlices = imIn.getSliceCount();
	size_t nLines = imIn.getHeight();

	
	volType srcSlices = imIn.getSlices();
	volType destSlices = imOut.getSlices();
	
	lineType *srcLines;
	lineType *destLines;
	
	bool oddSe = se.odd, oddLine = 0;
	
	size_t x, y, z;
	
	for (size_t s=0;s<nSlices;s++)
	{
	    srcLines = srcSlices[s];
	    destLines = destSlices[s];
	    
	    for (size_t l=0;l<nLines;l++)
	    {
		lineType lineIn  = srcLines[l];
		lineType lineOut = destLines[l];

//	    oddLine = oddSe && l%2;
		
		fillLine<T>(lineOut, parentClass::lineLen, 0);
		
		for (UINT p=0;p<sePtsNumber;p++)
		{
		    x = - se.points[p].x;
		    y = l + se.points[p].y;
		    z = s + se.points[p].z;

		    parentClass::lineFunction.trueVal = (1UL << p);
		    
		    this->_exec_line(lineIn, &imIn, x, y, z, lineOut);   
		}
		if (oddSe)
		  oddLine = !oddLine;
	    }
	}

	imOut.modified();

	    return RES_OK;
    }


    template <class T>
    RES_T arrowLow(const Image<T> &imIn, Image<T> &imOut, const StrElt &se=DEFAULT_SE, T borderValue=numeric_limits<T>::min())
    {
	unaryMorphArrowImageFunction<T, lowSupLine<T> > iFunc(borderValue);
	return iFunc(imIn, imOut, se);
    }

    template <class T>
    RES_T arrowLowOrEqu(const Image<T> &imIn, Image<T> &imOut, const StrElt &se=DEFAULT_SE, T borderValue=numeric_limits<T>::min())
    {
	unaryMorphArrowImageFunction<T, lowOrEquSupLine<T> > iFunc(borderValue);
	return iFunc(imIn, imOut, se);
    }

    template <class T>
    RES_T arrowGrt(const Image<T> &imIn, Image<T> &imOut, const StrElt &se=DEFAULT_SE, T borderValue=numeric_limits<T>::min())
    {
	unaryMorphArrowImageFunction<T, grtSupLine<T> > iFunc(borderValue);
	return iFunc(imIn, imOut, se);
    }

    template <class T>
    RES_T arrowGrtOrEqu(const Image<T> &imIn, Image<T> &imOut, const StrElt &se=DEFAULT_SE, T borderValue=numeric_limits<T>::min())
    {
	unaryMorphArrowImageFunction<T, grtOrEquSupLine<T> > iFunc(borderValue);
	return iFunc(imIn, imOut, se);
    }

    template <class T>
    RES_T arrowEqu(const Image<T> &imIn, Image<T> &imOut, const StrElt &se=DEFAULT_SE, T borderValue=numeric_limits<T>::min())
    {
	unaryMorphArrowImageFunction<T, equSupLine<T> > iFunc(borderValue);
	return iFunc(imIn, imOut, se);
    }

    /**
    * Arrow operator
    * 
    * \param imIn
    * \param operation "==", ">", ">=", "<" or "<="
    * \param imOut
    * \param se
    */
    template <class T>
    RES_T arrow(const Image<T> &imIn, const char *operation, Image<T> &imOut, const StrElt &se=DEFAULT_SE, T borderValue=numeric_limits<T>::min())
    {
	if (strcmp(operation, "==")==0)
	  return arrowEqu(imIn, imOut, se, borderValue);
	else if (strcmp(operation, ">")==0)
	  return arrowGrt(imIn, imOut, se, borderValue);
	else if (strcmp(operation, ">=")==0)
	  return arrowGrtOrEqu(imIn, imOut, se, borderValue);
	else if (strcmp(operation, "<")==0)
	  return arrowLow(imIn, imOut, se, borderValue);
	else if (strcmp(operation, "<=")==0)
	  return arrowLowOrEqu(imIn, imOut, se, borderValue);
	  
	else return RES_ERR;
    }

/** \} */

} // namespace smil


#endif // _D_MORPHO_ARROW_HPP

