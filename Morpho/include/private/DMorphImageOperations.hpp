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

#include "DImage.hpp"
#include "DMemory.hpp"
#include "DLineArith.hpp"
#include "DBaseLineOperations.hpp"
#include "DStructuringElement.h"



template <class T, class lineFunction_T>
class unaryMorphImageFunction : public imageFunctionBase<T>
{
  public:
    typedef imageFunctionBase<T> parentClass;
    typedef Image<T> imageType;
    typedef typename imageType::sliceType sliceType;
    typedef typename imageType::lineType lineType;
    
    unaryMorphImageFunction(T border=numeric_limits<T>::min()) 
      : borderValue(border) {}
    
    inline RES_T _exec(imageType &imIn, imageType &imOut, StrElt se);
    
    inline RES_T _exec_single(imageType &imIn, imageType &imOut, StrElt se);
    inline RES_T _exec_single_generic(imageType &imIn, imageType &imOut, StrElt se);
    inline RES_T _exec_single_hexSE(imageType &imIn, imageType &imOut);
    
    inline RES_T operator()(imageType &imIn, imageType &imOut, StrElt se) { return this->_exec(imIn, imOut, se); }

    lineFunction_T lineFunction;
    
  protected:
    T borderValue;
    T *borderBuf, *cpBuf;
    UINT lineLen;
    
    inline void _extract_translated_line(Image<T> *imIn, int &x, int &y, int &z, T *outBuf);
    inline void _exec_shifted_line(T *inBuf1, T *inBuf2, int dx, int lineLen, T *outBuf);
    inline void _exec_line(T *inBuf, Image<T> *imIn, int &x, int &y, int &z, T *outBuf);
};


template <class T, class lineFunction_T>
inline RES_T unaryMorphImageFunction<T, lineFunction_T>::_exec(imageType &imIn, imageType &imOut, StrElt se)
{
    lineLen = imIn.getAllocatedWidth();
    borderBuf = createAlignedBuffer<T>(lineLen);
    cpBuf = createAlignedBuffer<T>(lineLen);
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
    deleteAlignedBuffer<T>(borderBuf);
    deleteAlignedBuffer<T>(cpBuf);
    return RES_OK;
}

template <class T, class lineFunction_T>
inline RES_T unaryMorphImageFunction<T, lineFunction_T>::_exec_single(imageType &imIn, imageType &imOut, StrElt se)
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

void printLine(UINT8 *buf, int size)
{
  for (int i=0;i<size;i++)
    cout << (int)(*(buf+i)) << " ";
  cout << endl;
}

template <class T, class lineFunction_T>
inline void unaryMorphImageFunction<T, lineFunction_T>::_extract_translated_line(Image<T> *imIn, int &x, int &y, int &z, T *outBuf)
{
    if (z<0 || z>=imIn->getSliceCount() || y<0 || y>=imIn->getLineCount())
      copyLine<T,T>(borderBuf, lineLen, outBuf);
// 	memcpy(outBuf, borderBuf, lineLen*sizeof(T));
    else
	shiftLine<T>(imIn->getSlices()[z][y], x, lineLen, outBuf, borderValue);
}

template <class T, class lineFunction_T>
inline void unaryMorphImageFunction<T, lineFunction_T>::_exec_shifted_line(T *inBuf1, T *inBuf2, int dx, int lineLen, T *outBuf)
{
    shiftLine<T>(inBuf2, dx, lineLen, cpBuf, borderValue);
    lineFunction(inBuf1, cpBuf, lineLen, outBuf);
}


template <class T, class lineFunction_T>
inline void unaryMorphImageFunction<T, lineFunction_T>::_exec_line(T *inBuf, Image<T> *imIn, int &x, int &y, int &z, T *outBuf)
{
    _extract_translated_line(imIn, x, y, z, cpBuf);
    lineFunction(inBuf, cpBuf, lineLen, outBuf);
}



template <class T, class lineFunction_T>
inline RES_T unaryMorphImageFunction<T, lineFunction_T>::_exec_single_generic(imageType &imIn, imageType &imOut, StrElt se)
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

    T *outBuf = createAlignedBuffer<T>(lineLen);

    Image<T> *tmpIm;
    
    if (&imIn==&imOut)
      tmpIm = new Image<T>(imIn, true); // clone
    else tmpIm = &imIn;
    
    sliceType *srcSlices = tmpIm->getSlices();
    sliceType *destSlices = imOut.getSlices();
    
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
	    x = se.points[0].x + oddLine;
	    y = l + se.points[0].y;
	    z = s + se.points[0].z;

	    _extract_translated_line(tmpIm, x, y, z, outBuf);
	    
	    T *lineOut = destLines[l];
	    
	    for (int p=1;p<sePtsNumber;p++)
	    {
		bool pass = false;
		
		x = se.points[p].x + oddLine;
		y = l + se.points[p].y;
		z = s + se.points[p].z;
		
		
		_exec_line(outBuf, tmpIm, x, y, z, outBuf);   
	    }
	    copyLine<T,T>(outBuf, lineLen, lineOut);
	    if (oddSe)
	      oddLine = !oddLine;
	}
    }

    deleteAlignedBuffer<T>(outBuf);
    
    if (&imIn==&imOut)
      delete tmpIm;
    
    imOut.modified();

	return RES_OK;
}


template <class T, class lineFunction_T>
inline RES_T unaryMorphImageFunction<T, lineFunction_T>::_exec_single_hexSE(imageType &imIn, imageType &imOut)
{
    if (!areAllocated(&imIn, &imOut, NULL))
      return RES_ERR_BAD_ALLOCATION;

    int bufSize = lineLen * sizeof(T);
    int lineCount = imIn.getLineCount();
    
    int nSlices = imIn.getSliceCount();
    int nLines = imIn.getHeight();

    T *inBuf = createAlignedBuffer<T>(lineLen);
    T *outBuf = createAlignedBuffer<T>(lineLen);
    T *tmpBuf1 = createAlignedBuffer<T>(lineLen);
    T *tmpBuf2 = createAlignedBuffer<T>(lineLen);
    T *tmpBuf3 = createAlignedBuffer<T>(lineLen);
    T *tmpBuf4 = createAlignedBuffer<T>(lineLen);
    T *tmpBuf;
        
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
	copyLine<T,T>(srcLines[0], lineLen, inBuf);
	_exec_shifted_line(inBuf, inBuf, -1, lineLen, tmpBuf1);
	_exec_shifted_line(tmpBuf1, tmpBuf1, 1, lineLen, tmpBuf4);
	
	copyLine<T,T>(srcLines[1], lineLen, inBuf);
	_exec_shifted_line(inBuf, inBuf, 1, lineLen, tmpBuf2);
	lineFunction._exec(tmpBuf4, tmpBuf2, lineLen, outBuf);
	lineFunction._exec(borderBuf, outBuf, lineLen, destLines[0]);
// 	copyLine(outBuf, lineLen, destLines[0]);
	
// imOut.modified();
// return RES_OK;
	for (int l=2;l<nLines;l++)
	{
	    copyLine<T,T>(srcLines[l], lineLen, inBuf);
	    if((l%2)==0)
	    {
		_exec_shifted_line(inBuf, inBuf, -1, lineLen, tmpBuf3);
		_exec_shifted_line(tmpBuf2, tmpBuf2, -1, lineLen, tmpBuf4);
	    }
	    else
	    {
		_exec_shifted_line(inBuf, inBuf, 1, lineLen, tmpBuf3);
		_exec_shifted_line(tmpBuf2, tmpBuf2, 1, lineLen, tmpBuf4);
	    }
	    lineFunction._exec(tmpBuf1, tmpBuf3, lineLen, outBuf);
	    lineFunction._exec(tmpBuf4, outBuf, lineLen, destLines[l-1]);
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
	lineFunction._exec(tmpBuf4, tmpBuf1, lineLen, outBuf);
	lineFunction._exec(borderBuf, outBuf, lineLen, destLines[nLines-1]);
// 	copyLine(outBuf, lineLen, destLines[nLines-1]);
	
    }

    deleteAlignedBuffer<T>(inBuf);
    deleteAlignedBuffer<T>(outBuf);
    deleteAlignedBuffer<T>(tmpBuf1);
    deleteAlignedBuffer<T>(tmpBuf2);
    deleteAlignedBuffer<T>(tmpBuf3);
    deleteAlignedBuffer<T>(tmpBuf4);
//     deleteAlignedBuffer<T>(lineIn);
    
    if (&imIn==&imOut)
      delete tmpIm;
    
    imOut.modified();

	return RES_OK;
}


#endif // _MORPH_IMAGE_OPERATIONS_HXX
