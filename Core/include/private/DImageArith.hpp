#ifndef _D_IMAGE_ARITH_HPP
#define _D_IMAGE_ARITH_HPP

#include "DLineArith.hpp"

template <class T>
inline RES_T inv(Image<T> &imIn, Image<T> &imOut)
{
    unaryImageFunction<T, invLine<T> > iFunc;
    return iFunc(imIn, imOut);
}

template <class T>
inline RES_T add(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
     return binaryImageFunction<T, addLine<T> >::_exec(imIn1, imIn2, imOut);
}

template <class T>
inline double vol(Image<T> &imIn)
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
inline RES_T add(Image<T> &imIn1, const T value, Image<T> &imOut)
{
    binaryImageFunction<T, addLine<T> > iFunc;
    return iFunc(imIn1, value, imOut);
}

template <class T>
inline RES_T addNoSat(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    binaryImageFunction<T, addNoSatLine<T> > iFunc;
    return iFunc(imIn1, imIn2, imOut);
}

template <class T>
inline RES_T addNoSat(Image<T> &imIn1, const T value, Image<T> &imOut)
{
    binaryImageFunction<T, addNoSatLine<T> > iFunc;
    return iFunc(imIn1, value, imOut);
}

template <class T>
inline RES_T sub(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    binaryImageFunction<T, subLine<T> > iFunc;
    return iFunc(imIn1, imIn2, imOut);
}

template <class T>
inline RES_T sub(Image<T> &imIn1, T value, Image<T> &imOut)
{
    binaryImageFunction<T, subLine<T> > iFunc;
    return iFunc(imIn1, value, imOut);
}

template <class T>
inline RES_T subNoSat(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    binaryImageFunction<T, subNoSatLine<T> > iFunc;
    return iFunc(imIn1, imIn2, imOut);
}

template <class T>
inline RES_T subNoSat(Image<T> &imIn1, T value, Image<T> &imOut)
{
    binaryImageFunction<T, subNoSatLine<T> > iFunc;
    return iFunc(imIn1, value, imOut);
}

template <class T>
inline RES_T sup(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
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
inline RES_T inf(Image<T> &imIn1, T value, Image<T> &imOut)
{
    binaryImageFunction<T, infLine<T> > iFunc;
    return iFunc(imIn1, value, imOut);
}

template <class T>
inline RES_T inf(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    binaryImageFunction<T, infLine<T> > iFunc;
    return iFunc(imIn1, imIn2, imOut);
}

template <class T>
inline RES_T sup(Image<T> &imIn1, T value, Image<T> &imOut)
{
    binaryImageFunction<T, supLine<T> > iFunc;
    return iFunc(imIn1, value, imOut);
}

template <class T>
inline RES_T grt(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    binaryImageFunction<T, grtLine<T> > iFunc;
    return iFunc(imIn1, imIn2, imOut);
}

template <class T>
inline RES_T grt(Image<T> &imIn1, T value, Image<T> &imOut)
{
    binaryImageFunction<T, grtLine<T> > iFunc;
    return iFunc(imIn1, value, imOut);
}

template <class T>
inline RES_T low(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    binaryImageFunction<T, lowLine<T> > iFunc;
    return iFunc(imIn1, imIn2, imOut);
}

template <class T>
inline RES_T low(Image<T> &imIn1, T value, Image<T> &imOut)
{
    binaryImageFunction<T, lowLine<T> > iFunc;
    return iFunc(imIn1, value, imOut);
}

template <class T>
inline RES_T div(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    binaryImageFunction<T, divLine<T> > iFunc;
    return iFunc(imIn1, imIn2, imOut);
}

template <class T>
inline RES_T div(Image<T> &imIn, T value, Image<T> &imOut)
{
    binaryImageFunction<T, divLine<T> > iFunc;
    return iFunc(imIn, value, imOut);
}

template <class T>
inline RES_T mul(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    binaryImageFunction<T, mulLine<T> > iFunc;
    return iFunc(imIn1, imIn2, imOut);
}

template <class T>
inline RES_T mul(Image<T> &imIn1, T value, Image<T> &imOut)
{
    binaryImageFunction<T, mulLine<T> > iFunc;
    return iFunc(imIn1, value, imOut);
}

template <class T>
inline RES_T mulNoSat(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imOut)
{
    binaryImageFunction<T, mulLine<T> > iFunc;
    return iFunc(imIn1, imIn2, imOut);
}

template <class T>
inline RES_T mulNoSat(Image<T> &imIn1, T value, Image<T> &imOut)
{
    binaryImageFunction<T, mulLine<T> > iFunc;
    return iFunc(imIn1, value, imOut);
}

template <class T>
inline RES_T test(Image<T> &imIn1, Image<T> &imIn2, Image<T> &imIn3, Image<T> &imOut)
{
    tertiaryImageFunction<T, testLine<T> > iFunc;
    return iFunc(imIn1, imIn2, imIn3, imOut);
}

template <class T>
inline RES_T test(Image<T> &imIn1, Image<T> &imIn2, T value, Image<T> &imOut)
{
    tertiaryImageFunction<T, testLine<T> > iFunc;
    return iFunc(imIn1, imIn2, value, imOut);
}

template <class T>
inline RES_T test(Image<T> &imIn1, T value, Image<T> &imIn2, Image<T> &imOut)
{
    tertiaryImageFunction<T, testLine<T> > iFunc;
    return iFunc(imIn1, value, imIn2, imOut);
}

template <class T>
inline RES_T test(Image<T> &imIn, T value1, T value2, Image<T> &imOut)
{
    tertiaryImageFunction<T, testLine<T> > iFunc;
    return iFunc(imIn, value1, value2, imOut);
}

template <class T>
inline RES_T fill(Image<T> &imOut, const T value)
{
    if (!areAllocated(&imOut, NULL))
      return RES_ERR_BAD_ALLOCATION;
    
    typedef typename Image<T>::lineType lineType;
    lineType *lineOut = imOut.getLines();
    int lineLen = imOut.getWidth();
    int lineCount = imOut.getLineCount();
    
    // Fill first line
//     fillLine<T>::_exec(lineOut[0], lineLen, value);
    fillLine<T>::_exec(imOut.getPixels(), imOut.getPixelCount(), value);
    
//     for (int i=1;i<lineCount;i++)
//       memcpy(lineOut[i], lineOut[0], lineLen*sizeof(T));
    
    imOut.modified();
    return RES_OK;
}

// Copy/cast (two images with different types)
template <class T1, class T2>
RES_T copy(Image<T1> &imIn, Image<T2> &imOut)
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
RES_T copy(Image<T> &imIn, Image<T> &imOut)
{
    if (!areAllocated(&imIn, &imOut, NULL))
      return RES_ERR_BAD_ALLOCATION;
    
    if (haveSameSize(&imIn, &imOut, NULL))
    {
	memcpy(imOut.getPixels(), imIn.getPixels(), imIn.getPixelCount());

	imOut.modified();
	return RES_OK;
    }
	return RES_OK;
}





#endif // _D_IMAGE_ARITH_HPP

