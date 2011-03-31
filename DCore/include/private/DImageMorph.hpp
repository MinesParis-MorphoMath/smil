#ifndef _D_IMAGE_MORPH_HPP
#define _D_IMAGE_MORPH_HPP

#include "DLineArith.hpp"
#include "DMorphImageOperations.hpp"
#include "DImageArith.hpp"

template <class T>
inline RES_T dilateIm(Image<T> &imIn, Image<T> &imOut, StrElt se=DEFAULT_SE)
{
    unaryMorphImageFunction<T, supLine<T> > iFunc(numeric_limits<T>::min());
    return iFunc(imIn, imOut, se);
}

template <class T>
inline RES_T erodeIm(Image<T> &imIn, Image<T> &imOut, StrElt se=DEFAULT_SE)
{
    unaryMorphImageFunction<T, infLine<T> > iFunc(numeric_limits<T>::max());
    return iFunc(imIn, imOut, se);
}

template <class T>
inline RES_T closeIm(Image<T> &imIn, Image<T> &imOut, StrElt se=DEFAULT_SE)
{
    RES_T res = dilateIm(imIn, imOut, se);
    if (res==RES_OK)
      res = erodeIm(imIn, imOut, se);
    return res;
}

template <class T>
inline RES_T openIm(Image<T> &imIn, Image<T> &imOut, StrElt se=DEFAULT_SE)
{
    RES_T res = erodeIm(imIn, imOut, se);
    if (res==RES_OK)
      res = dilateIm(imIn, imOut, se);
    return res;
}

template <class T>
inline RES_T gradientIm(Image<T> &imIn, Image<T> &imOut, StrElt dilSe=DEFAULT_SE, StrElt eroSe=DEFAULT_SE)
{
    Image<T> dilIm(imIn);
    Image<T> eroIm(imIn);
    
    RES_T res = dilateIm(imIn, dilIm, dilSe);
    if (res==RES_OK)
      res = erodeIm(imIn, eroIm, eroSe);
    if (res==RES_OK)
      res = subIm(dilIm, eroIm, imOut);
    return res;
}


// Geodesy

template <class T>
inline RES_T geoDilIm(Image<T> &imIn, Image<T> &imMask, Image<T> &imOut, StrElt se=DEFAULT_SE)
{
    StrElt tmpSe(se);
    tmpSe.size = 1;
    
    RES_T res = infIm(imIn, imMask, imOut);
    
    for (int i=0;i<se.size,res==RES_OK;i++)
    {
	res = dilateIm(imOut, imOut, tmpSe);
	if (res==RES_OK)
	  res = infIm(imOut, imMask, imOut);	
    }
    return res;
}



#endif // _D_IMAGE_MORPH_HPP

