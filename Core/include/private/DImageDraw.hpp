/*
 * Smil
 * Copyright (c) 2010 Matthieu Faessel
 *
 * This file is part of Smil.
 *
 * Smil is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * Smil is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Smil.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 */


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
	  fillFunc(lines[j]+x1, width, value);
    }
    else
    {
	fillFunc(lines[y1]+x1, width, value);
	fillFunc(lines[y2]+x1, width, value);
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

