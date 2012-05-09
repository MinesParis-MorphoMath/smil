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


#ifndef _D_MORPHO_ARROW_HPP
#define _D_MORPHO_ARROW_HPP

#include "DMorphImageOperations.hpp"


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
    virtual RES_T _exec_single(imageType &imIn, imageType &imOut, StrElt &se);
    virtual RES_T _exec_single_generic(imageType &imIn, imageType &imOut, StrElt &se);
};


template <class T, class lineFunction_T>
RES_T unaryMorphArrowImageFunction<T, lineFunction_T>::_exec_single(imageType &imIn, imageType &imOut, StrElt &se)
{
    return _exec_single_generic(imIn, imOut, se);
}

template <class T, class lineFunction_T>
RES_T unaryMorphArrowImageFunction<T, lineFunction_T>::_exec_single_generic(imageType &imIn, imageType &imOut, StrElt &se)
{
    if (!areAllocated(&imIn, &imOut, NULL))
      return RES_ERR_BAD_ALLOCATION;

    int bufSize = parentClass::lineLen * sizeof(T);
    int lineCount = imIn.getLineCount();
    
    int sePtsNumber = se.points.size();
    if (sePtsNumber==0)
	return RES_OK;
    
    int nSlices = imIn.getSliceCount();
    int nLines = imIn.getHeight();

    lineType outBuf = ImDtTypes<T>::createLine(parentClass::lineLen);

    Image<T> *tmpIm;
    
    if (&imIn==&imOut)
      tmpIm = new Image<T>(imIn, true); // clone
    else tmpIm = &imIn;
    
    volType srcSlices = tmpIm->getSlices();
    volType destSlices = imOut.getSlices();
    
    lineType *srcLines;
    lineType *destLines;
    
    bool oddSe = se.odd, oddLine = 0;
    
    int x, y, z;
    
    for (int s=0;s<nSlices;s++)
    {
	srcLines = srcSlices[s];
	destLines = destSlices[s];
	if (oddSe)
	  oddLine = s%2!=0;
	
	for (int l=0;l<nLines;l++)
	{
	    lineType lineIn  = srcLines[l];
	    lineType lineOut = destLines[l];
	    
	    fillLine<T>(lineOut, parentClass::lineLen, 0);
	    
	    for (int p=0;p<sePtsNumber;p++)
	    {
		x = - se.points[p].x + oddLine;
		y = l - se.points[p].y;
		z = s + se.points[p].z;
		
		parentClass::lineFunction.trueVal = (1UL << p);
		
		this->_exec_line(lineIn, tmpIm, x, y, z, lineOut);   
	    }
	    if (oddSe)
	      oddLine = !oddLine;
	}
    }

    ImDtTypes<T>::deleteLine(outBuf);
    
    if (&imIn==&imOut)
      delete tmpIm;
    
    imOut.modified();

	return RES_OK;
}


template <class T>
RES_T arrowLow(Image<T> &imIn, Image<T> &imOut, StrElt se=DEFAULT_SE(), T borderValue=numeric_limits<T>::min())
{
    unaryMorphArrowImageFunction<T, lowSupLine<T> > iFunc(borderValue);
    return iFunc(imIn, imOut, se);
}

template <class T>
RES_T arrowLowOrEqu(Image<T> &imIn, Image<T> &imOut, StrElt se=DEFAULT_SE(), T borderValue=numeric_limits<T>::min())
{
    unaryMorphArrowImageFunction<T, lowOrEquSupLine<T> > iFunc(borderValue);
    return iFunc(imIn, imOut, se);
}

template <class T>
RES_T arrowGrt(Image<T> &imIn, Image<T> &imOut, StrElt se=DEFAULT_SE(), T borderValue=numeric_limits<T>::min())
{
    unaryMorphArrowImageFunction<T, grtSupLine<T> > iFunc(borderValue);
    return iFunc(imIn, imOut, se);
}

template <class T>
RES_T arrowGrtOrEqu(Image<T> &imIn, Image<T> &imOut, StrElt se=DEFAULT_SE(), T borderValue=numeric_limits<T>::min())
{
    unaryMorphArrowImageFunction<T, grtOrEquSupLine<T> > iFunc(borderValue);
    return iFunc(imIn, imOut, se);
}

template <class T>
RES_T arrowEqu(Image<T> &imIn, Image<T> &imOut, StrElt se=DEFAULT_SE(), T borderValue=numeric_limits<T>::min())
{
    unaryMorphArrowImageFunction<T, equSupLine<T> > iFunc(borderValue);
    return iFunc(imIn, imOut, se);
}

template <class T>
RES_T arrow(Image<T> &imIn, const char *operation, Image<T> &imOut, StrElt se=DEFAULT_SE(), T borderValue=numeric_limits<T>::min())
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


#endif // _D_MORPHO_ARROW_HPP

