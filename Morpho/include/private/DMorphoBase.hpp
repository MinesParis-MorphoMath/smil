#ifndef _D_MORPHO_BASE_HPP
#define _D_MORPHO_BASE_HPP

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
    unaryMorphImageFunction<T, infLine<T> > iFunc(numeric_limits<T>::min());
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




#endif // _D_MORPHO_BASE_HPP

