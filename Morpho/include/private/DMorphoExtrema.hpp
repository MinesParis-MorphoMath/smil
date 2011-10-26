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
    else tmpIm = &imIn;

    RES_T res;
    
    res = add(imIn, T(height), *tmpIm);
    if (res!=RES_OK)
      return res;
    
    res = dualBuild(*tmpIm, imIn, *tmpIm, tmpSe);
    if (res!=RES_OK)
      return res;
    
    sub(*tmpIm, imIn, imOut);
    
    return res;
}


#endif // _D_MORPHO_EXTREMA_HPP

