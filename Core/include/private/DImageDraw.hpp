#ifndef _D_IMAGE_DRAW_HPP
#define _D_IMAGE_DRAW_HPP

#include "DLineArith.hpp"

/**
 * \ingroup Core
 * \defgroup Draw
 * @{
 */


/**
 * Draw a rectangle
 * 
 * 
 * \param imOut Output image.
 */
template <class T>
inline RES_T drawRectangle(Image<T> &imOut, UINT centerX, UINT centerY, UINT width, UINT height, T value=numeric_limits<T>::max(), bool fill=false)
{
    if (!imOut.isAllocated())
        return RES_ERR_BAD_ALLOCATION;

    UINT x1 = centerX - width/2;
    UINT x2 = x1+width-1;
    UINT y1 = centerY - height/2;
    UINT y2 = y1+height-1;
    
    typename Image<T>::lineType *lines = imOut.getLines();
    fillLine<T> fillFunc;
    
    if (fill)
    {
	for (int j=y1;j<=y2;j++)
	  fillFunc._exec(lines[j]+x1, width, value);
    }
    else
    {
	fillFunc._exec(lines[y1]+x1, width, value);
	fillFunc._exec(lines[y2]+x1, width, value);
	for (int j=y1+1;j<=y2;j++)
	{
	    lines[j][x1] = value;
	    lines[j][x2] = value;
	}
    }
    
    imOut.modified();
    
    return RES_OK;
}



/** @}*/

#endif // _D_IMAGE_DRAW_HPP

