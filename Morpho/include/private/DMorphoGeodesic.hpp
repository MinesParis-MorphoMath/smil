#ifndef _D_MORPHO_GEODESIC_HPP
#define _D_MORPHO_GEODESIC_HPP

#include "DLineArith.hpp"
#include "DMorphImageOperations.hpp"
#include "DImageArith.hpp"


// Geodesy

template <class T>
inline RES_T geoDil(Image<T> &imIn, Image<T> &imMask, Image<T> &imOut, StrElt se=DEFAULT_SE)
{
    StrElt tmpSe(se);
    tmpSe.size = 1;
    
    RES_T res = inf(imIn, imMask, imOut);
    
    for (int i=0;i<se.size;i++)
    {
	res = dilate<T>(imOut, imOut, tmpSe);
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
	res = dilate<T>(imOut, imOut, tmpSe);
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

template <class T>
inline RES_T fillHoles(Image<T> &imIn, Image<T> &imOut, StrElt se=DEFAULT_SE)
{
    StrElt tmpSe(se);
    tmpSe.size = 1;
    
    RES_T res;
    
    Image<T> tmpIm(imIn);
    
    fill(tmpIm, numeric_limits<T>::max());
    dualBuild(tmpIm, imIn, imOut);
    
    return res;
}

template <class T>
inline RES_T levelPics(Image<T> &imIn, Image<T> &imOut, StrElt se=DEFAULT_SE)
{
    Image<T> tmpIm(imIn);
    inv(imIn, tmpIm);
    fillHoles(tmpIm, imOut);
    inv(imOut, imOut);
    
//     return res;
}



#endif // _D_MORPHO_GEODESIC_HPP

