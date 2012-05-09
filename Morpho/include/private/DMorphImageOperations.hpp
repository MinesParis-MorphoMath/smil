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


#ifndef _MORPH_IMAGE_OPERATIONS_HXX
#define _MORPH_IMAGE_OPERATIONS_HXX

#include "DCore.h"
#include "DStructuringElement.h"



template <class T, class lineFunction_T>
class unaryMorphImageFunction : public imageFunctionBase<T>
{
  public:
    typedef imageFunctionBase<T> parentClass;
    typedef Image<T> imageType;
    typedef typename imageType::lineType lineType;
    typedef typename imageType::sliceType sliceType;
    typedef typename imageType::volType volType;
    
    unaryMorphImageFunction(T border=numeric_limits<T>::min()) 
      : borderValue(border) 
    {
    }
    
    virtual RES_T _exec(imageType &imIn, imageType &imOut, StrElt &se);
    
    virtual RES_T _exec_single(imageType &imIn, imageType &imOut, StrElt &se);
    virtual RES_T _exec_single_generic(imageType &imIn, imageType &imOut, StrElt &se);
    virtual RES_T _exec_single_hexSE(imageType &imIn, imageType &imOut);
    
    inline RES_T operator()(imageType &imIn, imageType &imOut, StrElt se) { return this->_exec(imIn, imOut, se); }

    lineFunction_T lineFunction;
    
  protected:
    T borderValue;
    lineType borderBuf, cpBuf;
    UINT lineLen;
    
    inline void _extract_translated_line(Image<T> *imIn, int &x, int &y, int &z, lineType outBuf);
    inline void _exec_shifted_line(lineType inBuf1, lineType inBuf2, int dx, int lineLen, lineType outBuf);
    inline void _exec_line(lineType inBuf, Image<T> *imIn, int &x, int &y, int &z, lineType outBuf);
};


template <class T, class lineFunction_T>
RES_T unaryMorphImageFunction<T, lineFunction_T>::_exec(imageType &imIn, imageType &imOut, StrElt &se)
{
    lineLen = imIn.getWidth();
    borderBuf = ImDtTypes<T>::createLine(lineLen);
//     ImDtTypes<T>::deleteLine(borderBuf);
//     return RES_OK;
    cpBuf = ImDtTypes<T>::createLine(lineLen);
    fillLine<T> f;
    f(borderBuf, lineLen, borderValue);
//     cout << "bord val " << (int)borderValue << endl;
    int seSize = se.size;
    if (seSize==1) _exec_single(imIn, imOut, se);
    else
    {
	Image<T> tmpIm(imIn, true); // clone
	for (int i=0;i<seSize;i++)
	{
	   _exec_single(tmpIm, imOut, se);
	   if (i<seSize-1)
	     copy(imOut, tmpIm);
	}
    }
    ImDtTypes<T>::deleteLine(borderBuf);
    ImDtTypes<T>::deleteLine(cpBuf);
    return RES_OK;
}

template <class T, class lineFunction_T>
RES_T unaryMorphImageFunction<T, lineFunction_T>::_exec_single(imageType &imIn, imageType &imOut, StrElt &se)
{
    seType st = se.getType();
    
    switch(st)
    {
      case stGeneric:
	return _exec_single_generic(imIn, imOut, se);
      case stHexSE:
	return _exec_single_hexSE(imIn, imOut);
    }
    
    return RES_NOT_IMPLEMENTED;
}

template <class T, class lineFunction_T>
inline void unaryMorphImageFunction<T, lineFunction_T>::_extract_translated_line(Image<T> *imIn, int &x, int &y, int &z, lineType outBuf)
{
    if (z<0 || z>=imIn->getSliceCount() || y<0 || y>=imIn->getLineCount())
      copyLine<T>(borderBuf, lineLen, outBuf);
// 	memcpy(outBuf, borderBuf, lineLen*sizeof(T));
    else
	shiftLine<T>(imIn->getSlices()[z][y], x, lineLen, outBuf, borderValue);
}

template <class T, class lineFunction_T>
inline void unaryMorphImageFunction<T, lineFunction_T>::_exec_shifted_line(lineType inBuf1, lineType inBuf2, int dx, int lineLen, lineType outBuf)
{
    shiftLine<T>(inBuf2, dx, lineLen, cpBuf, borderValue);
    lineFunction(inBuf1, cpBuf, lineLen, outBuf);
}


template <class T, class lineFunction_T>
inline void unaryMorphImageFunction<T, lineFunction_T>::_exec_line(lineType inBuf, Image<T> *imIn, int &x, int &y, int &z, lineType outBuf)
{
    _extract_translated_line(imIn, x, y, z, cpBuf);
    lineFunction(inBuf, cpBuf, lineLen, outBuf);
}



template <class T, class lineFunction_T>
RES_T unaryMorphImageFunction<T, lineFunction_T>::_exec_single_generic(imageType &imIn, imageType &imOut, StrElt &se)
{
    if (!areAllocated(&imIn, &imOut, NULL))
      return RES_ERR_BAD_ALLOCATION;

    int bufSize = lineLen * sizeof(T);
    int lineCount = imIn.getLineCount();
    
    int sePtsNumber = se.points.size();
    if (sePtsNumber==0)
	return RES_OK;
    
    int nSlices = imIn.getSliceCount();
    int nLines = imIn.getHeight();

    lineType outBuf = ImDtTypes<T>::createLine(lineLen);

    Image<T> *tmpIm;
    
    if (&imIn==&imOut)
      tmpIm = new Image<T>(imIn, true); // clone
    else tmpIm = &imIn;
    
    volType srcSlices = tmpIm->getSlices();
    volType destSlices = imOut.getSlices();
    
    //lineType *srcLines;
    lineType *destLines;
    
    bool oddSe = se.odd, oddLine = 0;
    
    int x, y, z;
    
    for (int s=0;s<nSlices;s++)
    {
	destLines = destSlices[s];
	if (oddSe)
	  oddLine = s%2!=0;
	
	for (int l=0;l<nLines;l++)
	{
	    x = se.points[0].x + (oddLine && oddSe);
	    y = l - se.points[0].y;
	    z = s + se.points[0].z;

	    _extract_translated_line(tmpIm, x, y, z, outBuf);
	    
	    lineType lineOut = destLines[l];
	    
	    for (int p=1;p<sePtsNumber;p++)
	    {
		x = -se.points[p].x + (oddLine && oddSe);
		y = l - se.points[p].y;
		z = s + se.points[p].z;
		
		_exec_line(outBuf, tmpIm, x, y, z, outBuf);   
	    }
	    copyLine<T>(outBuf, lineLen, lineOut);
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


template <class T, class lineFunction_T>
RES_T unaryMorphImageFunction<T, lineFunction_T>::_exec_single_hexSE(imageType &imIn, imageType &imOut)
{
    if (!areAllocated(&imIn, &imOut, NULL))
      return RES_ERR_BAD_ALLOCATION;

    int bufSize = lineLen * sizeof(T);
    int lineCount = imIn.getLineCount();
    
    int nSlices = imIn.getSliceCount();
    int nLines = imIn.getHeight();

    lineType inBuf = ImDtTypes<T>::createLine(lineLen);
    lineType outBuf = ImDtTypes<T>::createLine(lineLen);
    lineType tmpBuf1 = ImDtTypes<T>::createLine(lineLen);
    lineType tmpBuf2 = ImDtTypes<T>::createLine(lineLen);
    lineType tmpBuf3 = ImDtTypes<T>::createLine(lineLen);
    lineType tmpBuf4 = ImDtTypes<T>::createLine(lineLen);
    lineType tmpBuf;
        
    Image<T> *tmpIm;
    
    if (&imIn==&imOut)
      tmpIm = new Image<T>(imIn, true); // clone
    else tmpIm = &imIn;
    
    sliceType *srcSlices = tmpIm->getSlices();
    sliceType *destSlices = imOut.getSlices();
    
    lineType *srcLines;
    lineType *destLines;
    
//    bool oddLine;
    
    for (int s=0;s<nSlices;s++)
    {
	srcLines = srcSlices[s];
	destLines = destSlices[s];
//	oddLine = !s%2;
	
	// Process first line
// 	copyLine<T,T>(srcLines[0], lineLen, inBuf);
	_exec_shifted_line(srcLines[0], srcLines[0], -1, lineLen, tmpBuf1);
	_exec_shifted_line(tmpBuf1, tmpBuf1, 1, lineLen, tmpBuf4);
	
// 	copyLine<T,T>(srcLines[1], lineLen, inBuf);
	_exec_shifted_line(srcLines[1], srcLines[1], 1, lineLen, tmpBuf2);
	lineFunction(tmpBuf4, tmpBuf2, lineLen, outBuf);
	lineFunction(borderBuf, outBuf, lineLen, destLines[0]);
// 	copyLine(outBuf, lineLen, destLines[0]);
	
// imOut.modified();
// return RES_OK;
	for (int l=2;l<nLines;l++)
	{
// 	    copyLine<T,T>(srcLines[l], lineLen, inBuf);
	    if((l%2)==0)
	    {
// 		_exec_shifted_line(inBuf, inBuf, -1, lineLen, tmpBuf3);
		_exec_shifted_line(srcLines[l], srcLines[l], -1, lineLen, tmpBuf3);
		_exec_shifted_line(tmpBuf2, tmpBuf2, -1, lineLen, tmpBuf4);
	    }
	    else
	    {
// 		_exec_shifted_line(inBuf, inBuf, 1, lineLen, tmpBuf3);
		_exec_shifted_line(srcLines[l], srcLines[l], 1, lineLen, tmpBuf3);
		_exec_shifted_line(tmpBuf2, tmpBuf2, 1, lineLen, tmpBuf4);
	    }
	    lineFunction(tmpBuf1, tmpBuf3, lineLen, outBuf);
	    lineFunction(tmpBuf4, outBuf, lineLen, destLines[l-1]);
// 	    copyLine(outBuf, lineLen, destLines[l-1]);
	    
	    tmpBuf = tmpBuf1;
	    tmpBuf1 = tmpBuf2;
	    tmpBuf2 = tmpBuf3;
	    tmpBuf3 = tmpBuf;
	}
	
	if (nLines%2 != 0)
	  _exec_shifted_line(tmpBuf2, tmpBuf2, 1, lineLen, tmpBuf4);
	else
	  _exec_shifted_line(tmpBuf2, tmpBuf2, -1, lineLen, tmpBuf4);
	lineFunction(tmpBuf4, tmpBuf1, lineLen, outBuf);
	lineFunction._exec(borderBuf, outBuf, lineLen, destLines[nLines-1]);
// 	copyLine(outBuf, lineLen, destLines[nLines-1]);
	
    }

    ImDtTypes<T>::deleteLine(inBuf);
    ImDtTypes<T>::deleteLine(outBuf);
    ImDtTypes<T>::deleteLine(tmpBuf1);
    ImDtTypes<T>::deleteLine(tmpBuf2);
    ImDtTypes<T>::deleteLine(tmpBuf3);
    ImDtTypes<T>::deleteLine(tmpBuf4);
//     ImDtTypes<T>::deleteLine(lineIn);
    
    if (&imIn==&imOut)
      delete tmpIm;
    
    imOut.modified();

	return RES_OK;
}


#endif // _MORPH_IMAGE_OPERATIONS_HXX
