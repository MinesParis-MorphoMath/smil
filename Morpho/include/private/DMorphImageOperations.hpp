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


#ifndef _MORPH_IMAGE_OPERATIONS_HXX
#define _MORPH_IMAGE_OPERATIONS_HXX

#include "DCore.h"
#include "DStructuringElement.h"

#ifdef USE_OPEN_MP
#include <omp.h>
#endif // USE_OPEN_MP

template <class T_in, class T_out=T_in>
class unaryMorphImageFunctionGeneric : public imageFunctionBase<T_in>
{
public:
    unaryMorphImageFunctionGeneric(T_in _borderValue = numeric_limits<T_in>::min())
      : borderValue(_borderValue),
	initialValue(_borderValue)
    {
    }
    
    unaryMorphImageFunctionGeneric(T_in _borderValue, T_out _initialValue = numeric_limits<T_out>::min())
      : borderValue(_borderValue),
	initialValue(_initialValue)
    {
    }
    
    typedef Image<T_in> imageInType;
    typedef typename imageInType::lineType lineInType;
    typedef typename imageInType::sliceType sliceInType;
    typedef typename imageInType::volType volInType;
    
    typedef Image<T_out> imageOutType;
    typedef typename imageOutType::lineType lineOutType;
    typedef typename imageOutType::sliceType sliceOutType;
    typedef typename imageOutType::volType volOutType;
    
    virtual RES_T initialize(const imageInType &imIn, imageOutType &imOut, const StrElt &se)
    {
	imIn.getSize(imSize);
	
	slicesIn = imIn.getSlices();
	slicesOut = imOut.getSlices();
	pixelsIn = imIn.getPixels();
	pixelsOut = imOut.getPixels();
	
	sePoints = se.points;
	IntPoint p0 = sePoints[0];
	if (p0.x==0 && p0.y==0 && p0.z==0)
	{
	    copy(imIn, imOut);
	    sePoints.erase(sePoints.begin());
	}
	else fill(imOut, initialValue);
	
	sePointNbr = sePoints.size();
	relativeOffsets.clear();
	vector<IntPoint>::iterator pt = sePoints.begin();
	se_xmin = numeric_limits<int>::max();
	se_xmax = numeric_limits<int>::min();
	se_ymin = numeric_limits<int>::max();
	se_ymax = numeric_limits<int>::min();
	se_zmin = numeric_limits<int>::max();
	se_zmax = numeric_limits<int>::min();
	while(pt!=sePoints.end())
	{
	    if(pt->x < se_xmin) se_xmin = pt->x;
	    if(pt->x > se_xmax) se_xmax = pt->x;
	    if(pt->y < se_ymin) se_ymin = pt->y;
	    if(pt->y > se_ymax) se_ymax = pt->y;
	    if(pt->z < se_zmin) se_zmin = pt->z;
	    if(pt->z > se_zmax) se_zmax = pt->z;
	    
	    relativeOffsets.push_back(pt->x + pt->y*imSize[0] + pt->z*imSize[0]*imSize[1]);
	    pt++;
	}
	return RES_OK;
    }
    virtual RES_T finalize(const imageInType &imIn, imageOutType &imOut, const StrElt &se)
    {
	return RES_OK;
    }
    
    virtual RES_T _exec(const imageInType &imIn, imageOutType &imOut, const StrElt &se)
    {
	initialize(imIn, imOut, se);
	
	seType st = se.getType();
	RES_T retVal;
	
	switch(st)
	{
	  case SE_Hex:
	    retVal = processImage(imIn, imOut, *static_cast<const HexSE*>(&se));
	    break;
	  case SE_Squ:
	    retVal = processImage(imIn, imOut, *static_cast<const SquSE*>(&se));
	    break;
	  default:
	    retVal = processImage(imIn, imOut, se);
	}
	
	finalize(imIn, imOut, se);
	imOut.modified();
	return retVal;
	
    }
    virtual RES_T processImage(const imageInType &imIn, imageOutType &imOut, const StrElt &se)
    {
	for(curSlice=0;curSlice<imSize[2];curSlice++)
	{
	    curLine = 0;
	    processSlice(*slicesIn, *slicesOut, imSize[1], se);
	    slicesIn++;
	    slicesOut++;
	}
	return RES_OK;
    }
//     virtual RES_T processImage(imageInType &imIn, imageOutType &imOut, hSE &se)
//     {
//     }
    virtual inline void processSlice(sliceInType linesIn, sliceOutType linesOut, UINT &lineNbr, const StrElt &se)
    {
	while(curLine<lineNbr)
	{
	    curPixel = 0;
	    processLine(*linesIn, *linesOut, imSize[0], se);
	    curLine++;
	    linesIn++;
	    linesOut++;
	}
    }
    virtual inline void processLine(lineInType pixIn, lineOutType pixOut, UINT &pixNbr, const StrElt &se)
    {
	int x, y, z;
	IntPoint p;
	UINT offset = pixIn - pixelsIn;
	vector<IntPoint> ptList;
	vector<UINT> relOffsetList;
	vector<UINT> offsetList;
	
	// Remove points wich are outside the image
	for (UINT i=0;i<sePointNbr;i++)
	{
	    p = sePoints[i];
	    y = curLine + p.y;
	    z = curSlice + p.z;
	    if (y>=0 && y<int(imSize[1]) && z>=0 && z<int(imSize[2]))
	    {
	      ptList.push_back(p);
	      relOffsetList.push_back(relativeOffsets[i]);
	    }
	}
	UINT ptNbr = ptList.size();
	
	// Left border
	while((int)curPixel < -se_xmin)
	{
	    offsetList.clear();
	    for (UINT i=0;i<ptNbr;i++)
	    {
		x = curPixel + ptList[i].x;
		
		if (x>=0 && x<(int)imSize[0])
		  offsetList.push_back(relOffsetList[i]);
	    }
	    processPixel(offset, offsetList.begin(), offsetList.end());
	    curPixel++;
	    offset++;
	}
	
	// Middle
	offsetList.clear();
	for (UINT i=0;i<ptNbr;i++)
	  offsetList.push_back(relOffsetList[i]);
	while(curPixel < pixNbr-se_xmax)
	{
	    processPixel(offset, offsetList.begin(), offsetList.end());
	    curPixel++;
	    offset++;
	}
	
	// Right border
	while(curPixel<pixNbr)
	{
	    offsetList.clear();
	    for (UINT i=0;i<ptNbr;i++)
	    {
		x = curPixel + ptList[i].x;
		
		if (x>=0 && x<int(imSize[0]))
		  offsetList.push_back(relOffsetList[i]);
	    }
	    processPixel(offset, offsetList.begin(), offsetList.end());
	    curPixel++;
	    offset++;
	}
    }
    virtual inline void processPixel(UINT &pointOffset, vector<UINT>::iterator dOffset, vector<UINT>::iterator dOffsetEnd)
    {
	// Example: dilation function
	while(dOffset!=dOffsetEnd)
	{
// 	    pixelsOut[pointOffset] = max(pixelsOut[pointOffset], pixelsIn[pointOffset + *dOffset]);
// 	    dOffset++;
	}
    }
protected:
      UINT imSize[3];
      volInType slicesIn;
      volOutType slicesOut;
      lineInType pixelsIn;
      lineOutType pixelsOut;
      
      UINT curSlice;
      UINT curLine;
      UINT curPixel;
      
      vector<IntPoint> sePoints;
      UINT sePointNbr;
      vector<int> relativeOffsets;
      
      int se_xmin;
      int se_xmax;
      int se_ymin;
      int se_ymax;
      int se_zmin;
      int se_zmax;
public:
    T_out initialValue;
    T_in borderValue;
};




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
    
    virtual RES_T _exec(const imageType &imIn, imageType &imOut, const StrElt &se);
    
    virtual RES_T _exec_single(const imageType &imIn, imageType &imOut, const StrElt &se);
    virtual RES_T _exec_single_generic(const imageType &imIn, imageType &imOut, const StrElt &se);
    virtual RES_T _exec_single_hexSE(const imageType &imIn, imageType &imOut);
    virtual RES_T _exec_single_squSE(const imageType &imIn, imageType &imOut);
    
    inline RES_T operator()(const imageType &imIn, imageType &imOut, const StrElt se) { return this->_exec(imIn, imOut, se); }

    lineFunction_T lineFunction;
    
  protected:
    T borderValue;
    lineType borderBuf, cpBuf;
    UINT lineLen;
    
    inline void _extract_translated_line(const Image<T> *imIn, const int &x, const int &y, const int &z, lineType outBuf);
    inline void _exec_shifted_line(const lineType inBuf1, const lineType inBuf2, const int &dx, const int &lineLen, lineType outBuf);
    inline void _exec_line(const lineType inBuf, const Image<T> *imIn, const int &x, const int &y, const int &z, lineType outBuf);
};


template <class T, class lineFunction_T>
RES_T unaryMorphImageFunction<T, lineFunction_T>::_exec(const imageType &imIn, imageType &imOut, const StrElt &se)
{
    if (!areAllocated(&imIn, &imOut, NULL))
      return RES_ERR_BAD_ALLOCATION;
    
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
RES_T unaryMorphImageFunction<T, lineFunction_T>::_exec_single(const imageType &imIn, imageType &imOut, const StrElt &se)
{
    int st = se.getType();
    
    switch(st)
    {
      case SE_Hex:
	return _exec_single_hexSE(imIn, imOut);
      default:
	return _exec_single_generic(imIn, imOut, se);
    }
    
    return RES_ERR_NOT_IMPLEMENTED;
}

template <class T, class lineFunction_T>
inline void unaryMorphImageFunction<T, lineFunction_T>::_extract_translated_line(const Image<T> *imIn, const int &x, const int &y, const int &z, lineType outBuf)
{
    if (z<0 || z>=int(imIn->getDepth()) || y<0 || y>=int(imIn->getHeight()))
      copyLine<T>(borderBuf, lineLen, outBuf);
// 	memcpy(outBuf, borderBuf, lineLen*sizeof(T));
    else
	shiftLine<T>(imIn->getSlices()[z][y], x, lineLen, outBuf, borderValue);
}

template <class T, class lineFunction_T>
inline void unaryMorphImageFunction<T, lineFunction_T>::_exec_shifted_line(const lineType inBuf1, const lineType inBuf2, const int &dx, const int &lineLen, lineType outBuf)
{
    shiftLine<T>(inBuf2, dx, lineLen, cpBuf, borderValue);
    lineFunction(inBuf1, cpBuf, lineLen, outBuf);
}


template <class T, class lineFunction_T>
inline void unaryMorphImageFunction<T, lineFunction_T>::_exec_line(const lineType inBuf, const Image<T> *imIn, const int &x, const int &y, const int &z, lineType outBuf)
{
    _extract_translated_line(imIn, x, y, z, cpBuf);
    lineFunction(inBuf, cpBuf, lineLen, outBuf);
}


template <class T, class lineFunction_T>
RES_T unaryMorphImageFunction<T, lineFunction_T>::_exec_single_generic(const imageType &imIn, imageType &imOut, const StrElt &se)
{
    int bufSize = lineLen * sizeof(T);
    int lineCount = imIn.getLineCount();
    
    int sePtsNumber = se.points.size();
    if (sePtsNumber==0)
	return RES_OK;
    
    int nSlices = imIn.getSliceCount();
    int nLines = imIn.getHeight();

    int nthreads = Core::getInstance()->getNumberOfThreads();
    lineType *_bufs = this->createAlignedBuffers(2*nthreads, lineLen);
    lineType tmpBuf = _bufs[0];
    lineType tmpBuf2 = _bufs[nthreads];

    const Image<T> *tmpIm;
    
    if (&imIn==&imOut)
      tmpIm = new Image<T>(imIn, true); // clone
    else tmpIm = &imIn;
    
    volType srcSlices = tmpIm->getSlices();
    volType destSlices = imOut.getSlices();
    
    //lineType *srcLines;
    lineType *destLines, lineOut;
    
    bool oddSe = se.odd, oddLine = 0;
    


    for (int s=0;s<nSlices;s++)
    {
	destLines = destSlices[s];
	if (oddSe)
	  oddLine = s%2!=0;
	
      int l, p;
      int tid;
      int dx, dy, dz;
      
#pragma omp parallel for private(l, tid) shared(tmpIm) schedule(dynamic, nthreads)
      for (l=0;l<nLines;l++)
      {
#ifdef _OPENMP
	  tid = omp_get_thread_num();
	  tmpBuf = _bufs[tid];
	  tmpBuf2 = _bufs[tid+nthreads];
#endif // _OPENMP
	  int x, y, z;
	  x = se.points[0].x + (oddLine && oddSe);
	  y = l - se.points[0].y;
	  z = s + se.points[0].z;

	  _extract_translated_line(tmpIm, x, y, z, tmpBuf);
	  
	  lineOut = destLines[l];
	  
	  for (p=1;p<sePtsNumber;p++)
	  {
	      dx = -se.points[p].x + (oddLine && oddSe);
	      dy = l - se.points[p].y;
	      dz = s + se.points[p].z;
	      
	      _extract_translated_line(tmpIm, dx, dy, dz, tmpBuf2);
	      lineFunction(tmpBuf, tmpBuf2, lineLen, tmpBuf);
// 	      _exec_line(tmpBuf, tmpIm, dx, dy, dz, tmpBuf);
	  }
	  
	  copyLine<T>(tmpBuf, lineLen, lineOut);
	  if (oddSe)
	    oddLine = !oddLine;
      }
    }

    if (&imIn==&imOut)
      delete tmpIm;
    
    imOut.modified();

	return RES_OK;
}


template <class T, class lineFunction_T>
RES_T unaryMorphImageFunction<T, lineFunction_T>::_exec_single_hexSE(const imageType &imIn, imageType &imOut)
{
    int lineCount = imIn.getLineCount();
    
    int nSlices = imIn.getSliceCount();
    int nLines = imIn.getHeight();

//     int nthreads = Core::getInstance()->getNumberOfThreads();
    lineType *_bufs = this->createAlignedBuffers(5, lineLen);
    lineType buf0 = _bufs[0];
    lineType buf1 = _bufs[1];
    lineType buf2 = _bufs[2];
    lineType buf3 = _bufs[3];
    lineType buf4 = _bufs[4];
    
    lineType tmpBuf;
        
    const Image<T> *tmpIm;
    
    if (&imIn==&imOut)
      tmpIm = new Image<T>(imIn, true); // clone
    else tmpIm = &imIn;
    
    volType srcSlices = tmpIm->getSlices();
    volType destSlices = imOut.getSlices();
    
    sliceType srcLines;
    sliceType destLines;
    
    lineType curSrcLine;
    lineType curDestLine;
    
    for (int s=0;s<nSlices;s++)
    {
	srcLines = srcSlices[s];
	destLines = destSlices[s];
	
	// Process first line
	_exec_shifted_line(srcLines[0], srcLines[0], -1, lineLen, buf0);
	_exec_shifted_line(buf0, buf0, 1, lineLen, buf3);
	
	_exec_shifted_line(srcLines[1], srcLines[1], 1, lineLen, buf1);
	lineFunction(buf3, buf1, lineLen, buf4);
	lineFunction(borderBuf, buf4, lineLen, destLines[0]);
	
// 	int tid;
// #pragma omp parallel
	{
	  int l;
	    
// #pragma omp parallel for private(l) shared(tmpIm) ordered
	    for (l=2;l<nLines;l++)
	    {
	    
		curSrcLine = srcLines[l];
		curDestLine = destLines[l-1];
		
		if(!(l%2==0 ^ s%2==0))
		{
		    _exec_shifted_line(curSrcLine, curSrcLine, -1, lineLen, buf2);
		    _exec_shifted_line(buf1, buf1, -1, lineLen, buf3);
		}
		else
		{
		    _exec_shifted_line(curSrcLine, curSrcLine, 1, lineLen, buf2);
		    _exec_shifted_line(buf1, buf1, 1, lineLen, buf3);
		}

		lineFunction(buf0, buf2, lineLen, buf4);
		lineFunction(buf3, buf4, lineLen, curDestLine);
		tmpBuf = buf0;
		buf0 = buf1;
		buf1 = buf2;
		buf2 = tmpBuf;
	    }
	}
	
	if (!(nLines%2==0 ^ s%2==0))
	  _exec_shifted_line(buf1, buf1, -1, lineLen, buf3);
	else
	  _exec_shifted_line(buf1, buf1, 1, lineLen, buf3);
	lineFunction(buf3, buf0, lineLen, buf4);
	lineFunction._exec(borderBuf, buf4, lineLen, destLines[nLines-1]);
	
    }

//     this->deleteAlignedBuffers();
    
    if (&imIn==&imOut)
      delete tmpIm;
    
    imOut.modified();

	return RES_OK;
}

template <class T, class lineFunction_T>
RES_T unaryMorphImageFunction<T, lineFunction_T>::_exec_single_squSE(const imageType &imIn, imageType &imOut)
{
//     int bufSize = lineLen * sizeof(T);
//     int lineCount = imIn.getLineCount();
//     
//     int nSlices = imIn.getSliceCount();
//     int nLines = imIn.getHeight();
// 
// //     int nthreads = Core::getInstance()->getNumberOfThreads();
//     lineType *_bufs = this->createAlignedBuffers(5, lineLen);
//     lineType buf0 = _bufs[0];
//     lineType buf1 = _bufs[1];
//     lineType buf2 = _bufs[2];
//     lineType buf3 = _bufs[3];
//     lineType buf4 = _bufs[4];
//     
//     lineType tmpBuf;
//         
//     const Image<T> *tmpIm;
//     
//     if (&imIn==&imOut)
//       tmpIm = new Image<T>(imIn, true); // clone
//     else tmpIm = &imIn;
//     
//     volType srcSlices = tmpIm->getSlices();
//     volType destSlices = imOut.getSlices();
//     
//     sliceType srcLines;
//     sliceType destLines;
//     
//     lineType curSrcLine;
//     lineType curDestLine;
//     
//     for (int s=0;s<nSlices;s++)
//     {
// 	srcLines = srcSlices[s];
// 	destLines = destSlices[s];
// 	
// 	// Process first line
// 	_exec_shifted_line(srcLines[0], srcLines[0], -1, lineLen, buf0);
// 	_exec_shifted_line(buf0, buf0, 1, lineLen, buf1);
// 	
// 	_exec_shifted_line(srcLines[1], srcLines[1], -1, lineLen, buf0);
// 	_exec_shifted_line(buf0, buf0, 1, lineLen, buf2);
// 	
// 	lineFunction(buf1, buf2, lineLen, buf3);
// 	lineFunction(borderBuf, buf3, lineLen, destLines[0]);
// 	
// 	for (l=2;l<nLines;l++)
// 	{
// 	
// 	    curSrcLine = srcLines[l];
// 	    curDestLine = destLines[l-1];
// 		
// 	    _exec_shifted_line(curSrcLine, curSrcLine, -1, lineLen, buf0);
// 	    _exec_shifted_line(buf0, buf0, 1, lineLen, buf1);
// 	    
// //       for(i = 2 ; i<wySize ; i++) {
// // 	t_LineCopyFromImage2D(psrc, xSize, wxSize, i, lineIn);
// // 	t_LineErode1D(lineIn, wxSize, lineTmpL, lineTmp3);
// // 
// // 	t_LineArithInf1D(lineTmp1, lineTmp2, wxSize, lineOut);
// // 	t_LineArithInf1D(lineOut, lineTmp3, wxSize, lineOut);
// // 
// // 	t_LineCopyToImage2D(lineOut, xSize, wxSize, i-1, pdest);	
// // 
// // 	lineTmp = lineTmp1;
// // 	lineTmp1 = lineTmp2;
// // 	lineTmp2 = lineTmp3;
// // 	lineTmp3 = lineTmp;	
// //       }
// //     
// //       t_LineArithInf1D(lineTmp1, lineTmp2, wxSize, lineOut);
// //       t_LineCopyToImage2D(lineOut, xSize, wxSize, wySize-1, pdest);	 
// 	}
//     }
// 
// //     this->deleteAlignedBuffers();
//     
//     if (&imIn==&imOut)
//       delete tmpIm;
//     
//     imOut.modified();
// 
	return RES_OK;
}


#endif // _MORPH_IMAGE_OPERATIONS_HXX
