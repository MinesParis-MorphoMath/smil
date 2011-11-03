#ifndef _D_MORPHO_EXTREMA_HPP
#define _D_MORPHO_EXTREMA_HPP

#include "DMorphoGeodesic.hpp"
#include "DImageArith.hpp"


// Extrema

template <class T>
inline RES_T hMinima(Image<T> &imIn, UINT height, Image<T> &imOut, StrElt se=DEFAULT_SE)
{
    StrElt tmpSe(se);
    tmpSe.size = 1;
    
    Image<T> *tmpIm;
    
    if (&imIn==&imOut)
      tmpIm = new Image<T>(imIn, true); // clone
    else tmpIm = &imOut;

    RES_T res;
    
    res = add(imIn, T(height), *tmpIm);
    if (res!=RES_OK)
      return res;
    
    res = dualBuild(*tmpIm, imIn, *tmpIm, tmpSe);
    if (res!=RES_OK)
      return res;
    
    low(*tmpIm, imIn, imOut);
    
    return res;
}

template <class T>
inline RES_T hMaxima(Image<T> &imIn, UINT height, Image<T> &imOut, StrElt se=DEFAULT_SE)
{
    StrElt tmpSe(se);
    tmpSe.size = 1;
    
    Image<T> *tmpIm;
    
    if (&imIn==&imOut)
      tmpIm = new Image<T>(imIn, true); // clone
    else tmpIm = &imOut;

    RES_T res;
    
    res = sub(imIn, T(height), *tmpIm);
    if (res!=RES_OK)
      return res;
    
    res = build(*tmpIm, imIn, *tmpIm, tmpSe);
    if (res!=RES_OK)
      return res;
    
    grt(*tmpIm, imIn, imOut);
    
    return res;
}

template <class T>
inline RES_T minima(Image<T> &imIn, Image<T> &imOut, StrElt se=DEFAULT_SE)
{
    return hMinima(imIn, 1, imOut, se);
}

template <class T>
inline RES_T maxima(Image<T> &imIn, Image<T> &imOut, StrElt se=DEFAULT_SE)
{
    return hMaxima(imIn, 1, imOut, se);
}

#endif // _D_MORPHO_EXTREMA_HPP

