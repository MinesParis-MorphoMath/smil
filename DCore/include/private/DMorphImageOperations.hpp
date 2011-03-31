#ifndef _MORPH_IMAGE_OPERATIONS_HXX
#define _MORPH_IMAGE_OPERATIONS_HXX

#include "DImage.hpp"
#include "DMemory.hpp"
#include "DLineArith.hpp"



template <class T, class lineFunction_T>
class unaryMorphImageFunction : public imageFunctionBase<T>
{
  public:
    typedef imageFunctionBase<T> parentClass;
    typedef Image<T> imageType;
    typedef typename imageType::sliceType sliceType;
    typedef typename imageType::lineType lineType;
    
    unaryMorphImageFunction(T border=T(0)) 
      : borderValue(border), 
	vectorSize(SIMD_VEC_SIZE/sizeof(T)) 
	{}
    
    inline RES_T _exec(imageType &imIn, imageType &imOut, StrElt se);
    inline RES_T _exec_single(imageType &imIn, imageType &imOut, StrElt se);
    inline RES_T operator()(imageType &imIn, imageType &imOut, StrElt se) { return this->_exec(imIn, imOut, se); }

    lineFunction_T lineFunction;
    
  protected:
    T borderValue;
    UINT vectorSize;
    
    inline void _exec_line(T *inBuf, Image<T> *imIn, int &x, int &y, int &z, T *outBuf);
};

template <class T, class lineFunction_T>
inline RES_T unaryMorphImageFunction<T, lineFunction_T>::_exec(imageType &imIn, imageType &imOut, StrElt se)
{
    int seSize = se.size;
    if (seSize==1) _exec_single(imIn, imOut, se);
    else
    {
	Image<T> tmpIm(imIn, true); // clone
	for (int i=0;i<seSize;i++)
	{
	   _exec_single(tmpIm, imOut, se);
	   if (i<seSize-1)
	     copyIm(imOut, tmpIm);
	}
    }
}


template <class T, class lineFunction_T>
inline void unaryMorphImageFunction<T, lineFunction_T>::_exec_line(T *inBuf, Image<T> *imIn, int &x, int &y, int &z, T *outBuf)
{
    if (z<0 || z>=imIn->getSliceCount()) return;
    else if (y<0 || y>=imIn->getLineCount()) return;

    T *lineIn = imIn->getSlices()[z][y];
    int lineLen = imIn->getWidth();

    if (x==0)
      lineFunction._exec(inBuf, lineIn, lineLen, outBuf);
    else if (x>0)
    {
      int alStart = (x/vectorSize + 1)*vectorSize;
//       lineFunction._exec(outBuf+x, lineIn, lineLen-x, outBuf+x);
      lineFunction._exec(inBuf+x, lineIn, alStart-x, outBuf+x);
      lineFunction._exec(inBuf+alStart, lineIn+alStart-x, lineLen-alStart, outBuf+alStart);
    }
    else
    {
      lineFunction._exec(inBuf, lineIn-x, lineLen+x, outBuf);
    }
	
}



template <class T, class lineFunction_T>
inline RES_T unaryMorphImageFunction<T, lineFunction_T>::_exec_single(imageType &imIn, imageType &imOut, StrElt se)
{
    if (!areAllocated(&imIn, &imOut, NULL))
      return RES_ERR_BAD_ALLOCATION;

    int lineLen = imIn.getWidth();
    int bufSize = lineLen * sizeof(T);
    int lineCount = imIn.getLineCount();
    
    int sePtsNumber = se.points.size();
    int nSlices = imIn.getSliceCount();
    int nLines = imIn.getHeight();

    T *borderBuf = createAlignedBuffer<T>(lineLen);
    T *outBuf = createAlignedBuffer<T>(lineLen);
        
    fillLine<T>(borderBuf, bufSize, borderValue);

    Image<T> *tmpIm;
    
    if (&imIn==&imOut)
      tmpIm = new Image<T>(imIn, true); // clone
    else tmpIm = &imIn;
    
    sliceType *srcSlices = tmpIm->getSlices();
    sliceType *destSlices = imOut.getSlices();
    
    lineType *srcLines;
    lineType *destLines;
    
    bool oddSe = se.odd, oddLine;
    
    int vec_size = SIMD_VEC_SIZE / sizeof(T);
    
    for (int s=0;s<nSlices;s++)
    {
	destLines = destSlices[s];
	oddLine = !s%2;
	
	for (int l=0;l<nLines;l++)
	{
	    memcpy(outBuf, borderBuf, bufSize);
	    T *lineOut = destLines[l];
	    
	    if (oddSe)
	      oddLine = !oddLine;
	    
	    for (int p=0;p<sePtsNumber;p++)
	    {
		int x, y, z;
		bool pass = false;
		
		x = se.points[p].x + !oddLine;
		y = l + se.points[p].y;
		z = s + se.points[p].z;
		
		_exec_line(outBuf, tmpIm, x, y, z, outBuf);   
		/*
		
		if (z<0 || z>=nSlices) pass = true;
		else if (y<0 || y>=nLines) pass = true;

		if(!pass)
		{
		    lineIn = srcSlices[z][y];
// 		    memcpy(lineIn, srcSlices[z][y], lineLen*sizeof(T));
// 		    t_LineCopyFromImage2D(pixIn, lineLen, y, lineIn);
		
		    if (x==0)
		    {
// 		      memcpy(bufs[p], lineIn, lineLen*sizeof(T));
		      lineFunction._exec_aligned(outBuf, lineIn, lineLen, outBuf);
// 		      memcpy(borderBuf, lineIn, lineLen*sizeof(T));
// 		      lineFunction._exec_aligned(outBuf, borderBuf, lineLen, outBuf);
		    }
		    else if (x>0)
		    {
		      t_LineShiftRight1D(lineIn, lineLen, x, borderValue, borderBuf);
		      lineFunction._exec_aligned(outBuf, borderBuf, lineLen, outBuf);
// 		      lineFunction._exec(outBuf+x, lineIn, lineLen-x, outBuf+x);
		    }
		    else
		    {
		      t_LineShiftLeft1D(lineIn, lineLen, -x, borderValue, borderBuf);
		      lineFunction._exec_aligned(outBuf, borderBuf, lineLen, outBuf);
// 		      lineFunction._exec(outBuf, lineIn-x, lineLen+x, outBuf);
		    }
		}*/
	    }
	    memcpy(lineOut, outBuf, lineLen*sizeof(T));
	}
    }

    deleteAlignedBuffer<T>(outBuf);
    deleteAlignedBuffer<T>(borderBuf);
//     deleteAlignedBuffer<T>(lineIn);
    
    if (&imIn==&imOut)
      delete tmpIm;
    
    imOut.modified();
}


#endif // _MORPH_IMAGE_OPERATIONS_HXX
