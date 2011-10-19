#ifndef _BASE_IMAGE_OPERATIONS_HXX
#define _BASE_IMAGE_OPERATIONS_HXX

#include "DImage.hpp"
#include "DMemory.hpp"
#include "DLineArith.hpp"



template <class T>
inline typename Image<T>::lineType *imageFunctionBase<T>::createAlignedBuffers(UINT8 nbr, UINT32 len)
{
    if (alignedBuffers) 
    {
	if (nbr==bufferNumber && len==bufferLength)
	  return alignedBuffers;
	
	deleteAlignedBuffers();
    }
    
    
    bufferNumber = nbr;
    bufferLength = len;
    bufferSize = bufferLength * sizeof(T);
    
    alignedBuffers = new lineType[bufferNumber];
    for (int i=0;i<bufferNumber;i++)
      alignedBuffers[i] = createAlignedBuffer<T>(len);
    
    return alignedBuffers;
}
    

template <class T>
inline void imageFunctionBase<T>::deleteAlignedBuffers()
{
    if (!alignedBuffers) return;
    
    for (UINT i=0;i<bufferNumber;i++)
      deleteAlignedBuffer<T>(alignedBuffers[i]);
}

template <class T>
inline void imageFunctionBase<T>::copyLineToBuffer(T *line, UINT32 bufIndex)
{
    memcpy(alignedBuffers[bufIndex], line, bufferSize);
}

template <class T>
inline void imageFunctionBase<T>::copyBufferToLine(UINT32 bufIndex, T *line)
{
    memcpy(line, alignedBuffers[bufIndex], bufferSize);
}




template <class T, class lineFunction_T>
inline RES_T unaryImageFunction<T, lineFunction_T>::_exec(imageType &imIn, imageType &imOut)
{
    if (!areAllocated(&imIn, &imOut, NULL))
      return RES_ERR_BAD_ALLOCATION;

    int lineLen = imIn.getWidth();
    int bufSize = lineLen * sizeof(T);
    int lineCount = imIn.getLineCount();
    
    lineType *srcLine = imIn.getLines();
    lineType *destLine = imOut.getLines();
    
    T *l1, *l2;
    
    UINT alStart;
    
    for (int i=0;i<lineCount;i++)
    {
	l1 = srcLine[i];
	l2 = destLine[i];
	
	alStart = imIn.getLineAlignment(i);
	if (alStart)
	{
	    lineFunction._exec(l1, alStart, l2);
	    l1 += alStart;
	    l2 += alStart;
	}
	int remLen = lineLen-alStart;
	lineFunction._exec(l1, remLen, l2);
	
    }
    imOut.modified();

	return RES_OK;
}


template <class T, class lineFunction_T>
inline RES_T unaryImageFunction<T, lineFunction_T>::_exec(imageType &imOut, T &value)
{
    if (!areAllocated(&imOut, NULL))
      return RES_ERR_BAD_ALLOCATION;

    int lineLen = imOut.getWidth();
    int lineCount = imOut.getLineCount();

    lineType *destLine = imOut.getLines();
    T *lout;
    T *constBuf = createAlignedBuffer<T>(lineLen);
    
    UINT alStart;
    
    // Fill the first aligned buffer with the constant value
    fillLine<T>::_exec(constBuf, lineLen, value);

    // Use it for operations on lines
    for (int i=0;i<lineCount;i++)
    {
	lout = destLine[i];
	
	alStart = imOut.getLineAlignment(i);
	
	if (alStart)
	{
	    lineFunction._exec(constBuf, alStart, lout);
	    lout += alStart;
	}
	lineFunction._exec(constBuf, lineLen-alStart, lout);
      
    }
    
    deleteAlignedBuffer<T>(constBuf);
    imOut.modified();
}


// Binary image function
template <class T, class lineFunction_T>
inline RES_T binaryImageFunction<T, lineFunction_T>::_exec(imageType &imIn1, imageType &imIn2, imageType &imOut)
{
    if (&imOut==&imIn2) return _exec(imIn1, imIn2);
    else if (&imOut==&imIn1) return _exec(imIn2, imIn1);
    
    if (!areAllocated(&imIn1, &imIn2, &imOut, NULL))
      return RES_ERR_BAD_ALLOCATION;

    int lineLen = imIn1.getWidth();
    int bufSize = lineLen * sizeof(T);
    int lineCount = imIn1.getLineCount();
    
    lineType *srcLine1 = imIn1.getLines();
    lineType *srcLine2 = imIn2.getLines();
    lineType *destLine = imOut.getLines();
    
    T *l1, *l2, *l3;
    
    UINT alStart;
    
    for (int i=0;i<lineCount;i++)
    {
	l1 = srcLine1[i];
	l2 = srcLine2[i];
	l3 = destLine[i];
	
	alStart = imIn1.getLineAlignment(i);
	if (alStart)
	{
	    lineFunction._exec(l1, l2, alStart, l3);
	    l1 += alStart;
	    l2 += alStart;
	    l3 += alStart;
	}
	int remLen = lineLen-alStart;
	lineFunction._exec(l1, l2, remLen, l3);
	
    }
    imOut.modified();

	return RES_OK;
}

// Binary image function
template <class T, class lineFunction_T>
inline RES_T binaryImageFunction<T, lineFunction_T>::_exec(imageType &imIn, imageType &imInOut)
{
    if (!areAllocated(&imIn, &imInOut, NULL))
      return RES_ERR_BAD_ALLOCATION;

    int lineLen = imIn.getWidth();
    int lineCount = imIn.getLineCount();
    
    lineType *srcLine1 = imIn.getLines();
    lineType *srcLine2 = imInOut.getLines();
    
    T *tmpBuf = createAlignedBuffer<T>(lineLen);
    
    T *l1, *l2;
    
    UINT alStart;
    
    for (int i=0;i<lineCount;i++)
    {
	l1 = srcLine1[i];
	l2 = srcLine2[i];
	
	alStart = imIn.getLineAlignment(i);
	if (alStart)
	{
	    lineFunction._exec(l1, l2, alStart, tmpBuf);
	    memcpy(l2, tmpBuf, alStart*sizeof(T));
	    l1 += alStart;
	    l2 += alStart;
	}
	
	lineFunction._exec(l1, l2, lineLen-alStart, tmpBuf);
	memcpy(l2, tmpBuf, (lineLen-alStart)*sizeof(T));
	
    }
    deleteAlignedBuffer<T>(tmpBuf);
    imInOut.modified();

	return RES_OK;
}


// Binary image function
template <class T, class lineFunction_T>
inline RES_T binaryImageFunction<T, lineFunction_T>::_exec(imageType &imIn, T value, imageType &imOut)
{
    if (!areAllocated(&imIn, &imOut, NULL))
      return RES_ERR_BAD_ALLOCATION;

    int lineLen = imIn.getWidth();
    int lineCount = imIn.getLineCount();
    
    lineType *srcLines = imIn.getLines();
    lineType *destLines = imOut.getLines();
    
    T *constBuf = createAlignedBuffer<T>(lineLen);
    T *lin, *lout;
    
    // Fill the const buffer with the value
    fillLine<T>::_exec(constBuf, lineLen, value);
    
    UINT alStart;
    
    for (int i=0;i<lineCount;i++)
    {
	lin = srcLines[i];
	lout = destLines[i];
	
	alStart = imIn.getLineAlignment(i);
	if (alStart)
	{
	    lineFunction._exec(lin, constBuf, alStart, lout);
	    lin += alStart;
	    lout += alStart;
	}
	
	lineFunction._exec(lin, constBuf, lineLen-alStart, lout);
    
    }
    deleteAlignedBuffer<T>(constBuf);
    imOut.modified();

	return RES_OK;
}



#endif
