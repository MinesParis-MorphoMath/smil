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
      : borderValue(border), 
	vectorSize(SIMD_VEC_SIZE/sizeof(T)) 
	{}
    
    inline RES_T _exec(imageType &imIn, imageType &imOut, StrElt se);
    
    inline RES_T _exec_single(imageType &imIn, imageType &imOut, StrElt se);
    inline RES_T _exec_single_generic(imageType &imIn, imageType &imOut, StrElt se);
    inline RES_T _exec_single_hexSE(imageType &imIn, imageType &imOut);
    
    inline RES_T operator()(imageType &imIn, imageType &imOut, StrElt se) { return this->_exec(imIn, imOut, se); }

    lineFunction_T lineFunction;
    
  protected:
    T borderValue;
    T *borderBuf, *cpBuf;
    UINT vectorSize;
    
    inline void _extract_translated_line(Image<T> *imIn, int &x, int &y, int &z, T *outBuf);
    inline void _exec_translated_line(T *inBuf1, T *inBuf2, int dx, int lineLen, T *outBuf);
    inline void _exec_line(T *inBuf, Image<T> *imIn, int &x, int &y, int &z, T *outBuf);
};


template <class T, class lineFunction_T>
inline RES_T unaryMorphImageFunction<T, lineFunction_T>::_exec(imageType &imIn, imageType &imOut, StrElt se)
{
    int lineLen = imIn.getWidth();
    borderBuf = createAlignedBuffer<T>(lineLen);
    cpBuf = createAlignedBuffer<T>(lineLen);
    fillLine<T>::_exec(borderBuf, lineLen, borderValue);
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
    int lineLen = imIn->getWidth();
    
    if (z<0 || z>=imIn->getSliceCount() || y<0 || y>=imIn->getLineCount())
	memcpy(outBuf, borderBuf, lineLen*sizeof(T));
    else if (x>0)
    {
	memcpy(outBuf, borderBuf, x*sizeof(T));
	memcpy(outBuf+x, imIn->getSlices()[z][y], (lineLen-x)*sizeof(T));
    }
    else
    {
	memcpy(outBuf, imIn->getSlices()[z][y]-x, (lineLen+x)*sizeof(T));
	memcpy(outBuf+lineLen+x, borderBuf, -x*sizeof(T));
    }
}

template <class T, class lineFunction_T>
inline void unaryMorphImageFunction<T, lineFunction_T>::_exec_translated_line(T *inBuf1, T *inBuf2, int dx, int lineLen, T *outBuf)
{
    if (dx==0)
      lineFunction._exec(inBuf1, inBuf2, lineLen, outBuf);
    else if (dx>0)
    {
      lineFunction._exec(inBuf1, borderBuf, dx, outBuf);
      lineFunction._exec(inBuf1+dx, inBuf2, lineLen-dx, outBuf+dx);
    }
    else
    {
      lineFunction._exec(inBuf1, inBuf2-dx, lineLen+dx, outBuf);
      lineFunction._exec(inBuf1+lineLen+dx, borderBuf, -dx, outBuf+lineLen+dx);
    }
}


template <class T, class lineFunction_T>
inline void unaryMorphImageFunction<T, lineFunction_T>::_exec_line(T *inBuf, Image<T> *imIn, int &x, int &y, int &z, T *outBuf)
{
    int lineLen = imIn->getWidth();
    
    if (z<0 || z>=imIn->getSliceCount() || y<0 || y>=imIn->getLineCount())
	_exec_translated_line(inBuf, borderBuf, 0, lineLen, outBuf);
    else 
	_exec_translated_line(inBuf, imIn->getSlices()[z][y], x, lineLen, outBuf);
}



template <class T, class lineFunction_T>
inline RES_T unaryMorphImageFunction<T, lineFunction_T>::_exec_single_generic(imageType &imIn, imageType &imOut, StrElt se)
{
    if (!areAllocated(&imIn, &imOut, NULL))
      return RES_ERR_BAD_ALLOCATION;

    int lineLen = imIn.getWidth();
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
	    memcpy(lineOut, outBuf, bufSize);
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

    int lineLen = imIn.getWidth();
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
	memcpy(inBuf, srcLines[0], lineLen*sizeof(T));
	_exec_translated_line(inBuf, inBuf, -1, lineLen, tmpBuf1);
	_exec_translated_line(tmpBuf1, tmpBuf1, 1, lineLen, tmpBuf4);
	
	memcpy(inBuf, srcLines[1], lineLen*sizeof(T));
	_exec_translated_line(inBuf, inBuf, 1, lineLen, tmpBuf2);
	lineFunction._exec(tmpBuf4, tmpBuf2, lineLen, outBuf);
	lineFunction._exec(borderBuf, outBuf, lineLen, outBuf);
	memcpy(destLines[0], outBuf, lineLen*sizeof(T));
	
// imOut.modified();
// return RES_OK;
	for (int l=2;l<nLines;l++)
	{
	    memcpy(inBuf, srcLines[l], lineLen*sizeof(T));
	    if((l%2)==0)
	    {
		_exec_translated_line(inBuf, inBuf, -1, lineLen, tmpBuf3);
		_exec_translated_line(tmpBuf2, tmpBuf2, -1, lineLen, tmpBuf4);
	    }
	    else
	    {
		_exec_translated_line(inBuf, inBuf, 1, lineLen, tmpBuf3);
		_exec_translated_line(tmpBuf2, tmpBuf2, 1, lineLen, tmpBuf4);
	    }
	    lineFunction._exec(tmpBuf1, tmpBuf3, lineLen, outBuf);
	    lineFunction._exec(tmpBuf4, outBuf, lineLen, outBuf);
	    memcpy(destLines[l-1], outBuf, lineLen*sizeof(T));
	    
	    tmpBuf = tmpBuf1;
	    tmpBuf1 = tmpBuf2;
	    tmpBuf2 = tmpBuf3;
	    tmpBuf3 = tmpBuf;
	}
	
	if (nLines%2 != 0)
	  _exec_translated_line(tmpBuf2, tmpBuf2, 1, lineLen, tmpBuf4);
	else
	  _exec_translated_line(tmpBuf2, tmpBuf2, -1, lineLen, tmpBuf4);
	lineFunction._exec(tmpBuf4, tmpBuf1, lineLen, outBuf);
	lineFunction._exec(borderBuf, outBuf, lineLen, outBuf);
	memcpy(destLines[nLines-1], outBuf, lineLen*sizeof(T));
	
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
