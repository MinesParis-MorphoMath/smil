#ifndef _D_IMAGE_MORPH_HPP
#define _D_IMAGE_MORPH_HPP

#include "DLineArith.hpp"
#include "DMorphImageOperations.hpp"

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



#endif // _D_IMAGE_MORPH_HPP

