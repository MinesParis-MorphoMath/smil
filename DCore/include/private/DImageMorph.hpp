#ifndef _D_IMAGE_MORPH_HPP
#define _D_IMAGE_MORPH_HPP

#include "DLineArith.hpp"
#include "DMorphImageOperations.hpp"
#include "DImageArith.hpp"

template <class T>
inline RES_T label(Image<T> &imIn, Image<T> &imOut, StrElt se=DEFAULT_SE)
{
    T curLabel = 0;
    for (UINT z=0;z<imIn.getSliceCount();z++)
    {
    }
    unaryMorphImageFunction<T, equLine<T> > iFunc(numeric_limits<T>::min());
    return iFunc(imIn, imOut, se);
}

template <class T>
inline RES_T dilate(Image<T> &imIn, Image<T> &imOut, StrElt se=DEFAULT_SE)
{
    unaryMorphImageFunction<T, supLine<T> > iFunc(numeric_limits<T>::min());
    return iFunc(imIn, imOut, se);
}

template <class T>
inline RES_T erode(Image<T> &imIn, Image<T> &imOut, StrElt se=DEFAULT_SE)
{
    unaryMorphImageFunction<T, infLine<T> > iFunc(numeric_limits<T>::max());
    return iFunc(imIn, imOut, se);
}

template <class T>
inline RES_T close(Image<T> &imIn, Image<T> &imOut, StrElt se=DEFAULT_SE)
{
    RES_T res = dilate(imIn, imOut, se);
    if (res==RES_OK)
      res = erode(imIn, imOut, se);
    return res;
}

template <class T>
inline RES_T open(Image<T> &imIn, Image<T> &imOut, StrElt se=DEFAULT_SE)
{
    RES_T res = erode(imIn, imOut, se);
    if (res==RES_OK)
      res = dilate(imIn, imOut, se);
    return res;
}

template <class T>
inline RES_T gradient(Image<T> &imIn, Image<T> &imOut, StrElt dilSe, StrElt eroSe)
{
    Image<T> dilIm(imIn);
    Image<T> eroIm(imIn);
    
    RES_T res = dilate(imIn, dilIm, dilSe);
    if (res==RES_OK)
      res = erode(imIn, eroIm, eroSe);
    if (res==RES_OK)
      res = sub(dilIm, eroIm, imOut);
    return res;
}

template <class T>
inline RES_T gradient(Image<T> &imIn, Image<T> &imOut, StrElt se=DEFAULT_SE)
{
    return gradient(imIn, imOut, se, se);
}

// Geodesy

template <class T>
inline RES_T geoDil(Image<T> &imIn, Image<T> &imMask, Image<T> &imOut, StrElt se=DEFAULT_SE)
{
    StrElt tmpSe(se);
    tmpSe.size = 1;
    
    RES_T res = inf(imIn, imMask, imOut);
    
    for (int i=0;i<se.size;i++)
    {
	res = dilate(imOut, imOut, tmpSe);
	if (res==RES_OK)
	  res = inf(imOut, imMask, imOut);
	if (res!=RES_OK)
	  break;
    }
    return res;
}

template <class T>
inline RES_T geoEro(Image<T> &imIn, Image<T> &imMask, Image<T> &imOut, StrElt se=DEFAULT_SE)
{
    StrElt tmpSe(se);
    tmpSe.size = 1;
    
    RES_T res = sup(imIn, imMask, imOut);
    
    for (int i=0;i<se.size;i++)
    {
	res = erode(imOut, imOut, tmpSe);
	if (res==RES_OK)
	  res = sup(imOut, imMask, imOut);
	if (res!=RES_OK)
	  break;
    }
    return res;
}

template <class T>
inline RES_T build(Image<T> &imIn, Image<T> &imMask, Image<T> &imOut, StrElt se=DEFAULT_SE)
{
    StrElt tmpSe(se);
    tmpSe.size = 1;
    
    RES_T res = inf(imIn, imMask, imOut);
    
    int vol1 = vol(imOut), vol2;
    
    while (true)
    {
	res = dilate(imOut, imOut, tmpSe);
	if (res==RES_OK)
	  res = inf(imOut, imMask, imOut);
	if (res!=RES_OK)
	  break;
	vol2 = vol(imOut);
	if (vol2==vol1)
	  break;
	vol1 = vol2;
    }
    return res;
}

template <class T>
inline RES_T dualBuild(Image<T> &imIn, Image<T> &imMask, Image<T> &imOut, StrElt se=DEFAULT_SE)
{
    StrElt tmpSe(se);
    tmpSe.size = 1;
    
    RES_T res = sup(imIn, imMask, imOut);
    
    int vol1 = vol(imOut), vol2;
    
    while (true)
    {
	res = erode(imOut, imOut, tmpSe);
	if (res==RES_OK)
	  res = sup(imOut, imMask, imOut);
	if (res!=RES_OK)
	  break;
	vol2 = vol(imOut);
	if (vol2==vol1)
	  break;
	vol1 = vol2;
    }
    return res;
}



#endif // _D_IMAGE_MORPH_HPP

