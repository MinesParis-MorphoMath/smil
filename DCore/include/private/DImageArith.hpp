#ifndef _D_IMAGE_ARITH_HPP
#define _D_IMAGE_ARITH_HPP

#include "DLineArith.hpp"

template <class T>
inline RES_T invIm(Image<T> &imIn, Image<T> &imOut)
{
//     unaryImageFunction<T, invLine<T> > iFunc;
    return unaryImageFunction<T, invLine<T> >::_exec(imIn, imOut);
}


template <class T>
inline RES_T addIm(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    return binaryImageFunction<T, addLine<T> >::_exec(imIn1, imIn2, imOut);
//     return iFunc(imIn1, imIn2, imOut);

/*    T *pix1 = imIn1.getPixels();
    T *pix2 = imIn2.getPixels();
    T *pix3 = imOut.getPixels();
    
    int npix = imIn1.getPixelCount();
    
    for(int i=0;i<npix;i++)
      pix3[i] = pix1[i] > (T)(numeric_limits<T>::max() - pix2[i]) ? numeric_limits<T>::max() : pix1[i] + pix2[i];*/
}

template <class T>
inline double volIm(Image<T> &imIn)
{
    if (!imIn.isAllocated())
      return RES_ERR_BAD_ALLOCATION;
    
    int npix = imIn.getPixelCount();
    T *pixels = imIn.getPixels();
    double vol = 0;
    
    for (int i=0;i<npix;i++)
      vol += pixels[i];
    
    return vol;
}

template <class T>
inline RES_T addIm(Image<T> &imIn1, const T value, Image<T> &imOut)
{
    binaryImageFunction<T, addLine<T> > iFunc;
    return iFunc(imIn1, value, imOut);
}

template <class T>
inline RES_T addNoSatIm(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    binaryImageFunction<T, addNoSatLine<T> > iFunc;
    return iFunc(imIn1, imIn2, imOut);
}

template <class T>
inline RES_T addNoSatIm(Image<T> &imIn1, const T value, Image<T> &imOut)
{
    binaryImageFunction<T, addNoSatLine<T> > iFunc;
    return iFunc(imIn1, value, imOut);
}

template <class T>
inline RES_T subIm(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    binaryImageFunction<T, subLine<T> > iFunc;
    return iFunc(imIn1, imIn2, imOut);
}

template <class T>
inline RES_T subIm(Image<T> &imIn1, T value, Image<T> &imOut)
{
    binaryImageFunction<T, subLine<T> > iFunc;
    return iFunc(imIn1, value, imOut);
}

template <class T>
inline RES_T subNoSatIm(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    binaryImageFunction<T, subNoSatLine<T> > iFunc;
    return iFunc(imIn1, imIn2, imOut);
}

template <class T>
inline RES_T subNoSatIm(Image<T> &imIn1, T value, Image<T> &imOut)
{
    binaryImageFunction<T, subNoSatLine<T> > iFunc;
    return iFunc(imIn1, value, imOut);
}

template <class T>
inline RES_T supIm(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    return binaryImageFunction<T, supLine<T> >::_exec(imIn1, imIn2, imOut);
//     return iFunc(imIn1, imIn2, imOut);
    
/*    T *pix1 = imIn1.getPixels();
    T *pix2 = imIn2.getPixels();
    T *pix3 = imOut.getPixels();
    
    int npix = imIn1.getPixelCount();
    int vec_size = SIMD_VEC_SIZE / sizeof(T);
    int ndivs = npix / vec_size;
    
    for (int n=0;n<ndivs;n++)
    {
	for(int i=0;i<vec_size;i++)
	  pix3[i] = pix1[i] > pix2[i] ? pix1[i] : pix2[i];
	pix1 += vec_size;
	pix2 += vec_size;
	pix3 += vec_size;
    }*/
}

template <class T>
inline RES_T infIm(Image<T> &imIn1, T value, Image<T> &imOut)
{
    binaryImageFunction<T, infLine<T> > iFunc;
    return iFunc(imIn1, value, imOut);
}

template <class T>
inline RES_T infIm(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    binaryImageFunction<T, infLine<T> > iFunc;
    return iFunc(imIn1, imIn2, imOut);
}

template <class T>
inline RES_T supIm(Image<T> &imIn1, T value, Image<T> &imOut)
{
    binaryImageFunction<T, supLine<T> > iFunc;
    return iFunc(imIn1, value, imOut);
}

template <class T>
inline RES_T grtIm(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    binaryImageFunction<T, grtLine<T> > iFunc;
    return iFunc(imIn1, imIn2, imOut);
}

template <class T>
inline RES_T grtIm(Image<T> &imIn1, T value, Image<T> &imOut)
{
    binaryImageFunction<T, grtLine<T> > iFunc;
    return iFunc(imIn1, value, imOut);
}

template <class T>
inline RES_T lowIm(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    binaryImageFunction<T, lowLine<T> > iFunc;
    return iFunc(imIn1, imIn2, imOut);
}

template <class T>
inline RES_T lowIm(Image<T> &imIn1, T value, Image<T> &imOut)
{
    binaryImageFunction<T, lowLine<T> > iFunc;
    return iFunc(imIn1, value, imOut);
}

template <class T>
inline RES_T divIm(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    binaryImageFunction<T, divLine<T> > iFunc;
    return iFunc(imIn1, imIn2, imOut);
}

template <class T>
inline RES_T mulIm(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    binaryImageFunction<T, mulLine<T> > iFunc;
    return iFunc(imIn1, imIn2, imOut);
}

template <class T>
inline RES_T mulIm(Image<T> &imIn1, T value, Image<T> &imOut)
{
    binaryImageFunction<T, mulLine<T> > iFunc;
    return iFunc(imIn1, value, imOut);
}

template <class T>
inline RES_T mulNoSatIm(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    binaryImageFunction<T, mulLine<T> > iFunc;
    return iFunc(imIn1, imIn2, imOut);
}

template <class T>
inline RES_T mulNoSatIm(Image<T> &imIn1, T value, Image<T> &imOut)
{
    binaryImageFunction<T, mulLine<T> > iFunc;
    return iFunc(imIn1, value, imOut);
}

template <class T>
inline RES_T fillIm(Image<T> &imOut, const T value)
{
    if (!areAllocated(&imOut, NULL))
      return RES_ERR_BAD_ALLOCATION;
    
    typedef typename Image<T>::lineType lineType;
    lineType *lineOut = imOut.getLines();
    int lineLen = imOut.getWidth();
    int lineCount = imOut.getLineCount();
    
    // Fill first line
    fillLine<T>::_exec(lineOut[0], lineLen, value);
    
    for (int i=1;i<lineCount;i++)
      memcpy(lineOut[i], lineOut[0], lineLen*sizeof(T));
    
    imOut.modified();
    return RES_OK;
}

// Copy/cast (two images with different types)
template <class T1, class T2>
RES_T copyIm(Image<T1> &imIn, Image<T2> &imOut)
{
    if (!areAllocated(&imIn, &imOut, NULL))
      return RES_ERR_BAD_ALLOCATION;
    
    if (haveSameSize(&imIn, &imOut, NULL))
    {
	T1 *pix1 = imIn.getPixels();
	T2 *pix2 = imOut.getPixels();
	
	int pixCount = imIn.getPixelCount();
	
	for (int i=0;i<pixCount;i++)
	  pix2[i] = static_cast<T2>(pix1[i]);

	imOut.modified();
	return RES_OK;
    }
}

// Copy (two images of same type)
template <class T>
RES_T copyIm(Image<T> &imIn, Image<T> &imOut)
{
    if (!areAllocated(&imIn, &imOut, NULL))
      return RES_ERR_BAD_ALLOCATION;
    
    if (haveSameSize(&imIn, &imOut, NULL))
    {
	memcpy(imOut.getPixels(), imIn.getPixels(), imIn.getPixelCount());

	imOut.modified();
	return RES_OK;
    }
}





#endif // _D_IMAGE_ARITH_HPP

