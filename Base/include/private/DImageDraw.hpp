/*
 * Copyright (c) 2011, Matthieu FAESSEL and ARMINES
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of Matthieu FAESSEL, or ARMINES nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS AND CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#ifndef _D_IMAGE_DRAW_HPP
#define _D_IMAGE_DRAW_HPP

#include "DLineArith.hpp"
#include "Base/include/DImageDraw.h"

#ifdef USE_FREETYPE
    #include <ft2build.h>
    #include FT_FREETYPE_H
#endif // USE_FREETYPE


namespace smil
{
  
    /**
    * \ingroup Base
    * \defgroup Draw
    * @{
    */
    

    /**
    * Draws a line between two points p1(p1x,p1y) and p2(p2x,p2y).
    * This function is based on the Bresenham's line algorithm.
    * (works only on 2D images)
    */
    template <class T>
    RES_T drawLine(Image<T> &im, int p1x, int p1y, int p2x, int p2y, T value=numeric_limits<T>::max())
    {
	if (!im.isAllocated())
	    return RES_ERR_BAD_ALLOCATION;

	if (p1x<0 || p1x>=int(im.getWidth()) || p1y<0 || p1y>=int(im.getHeight()))
	  return RES_ERR;
	if (p2x<0 || p2x>=int(im.getWidth()) || p2y<0 || p2y>=int(im.getHeight()))
	  return RES_ERR;
	
	typename Image<T>::sliceType lines = im.getLines();
	vector<IntPoint> bPoints = bresenhamPoints(p1x, p1y, p2x, p2y);
	
	for(vector<IntPoint>::iterator it=bPoints.begin();it!=bPoints.end();it++)
	  lines[(*it).y][(*it).x] = value;
	
	im.modified();
	return RES_OK;
    }
    
    template <class T>
    RES_T drawLine(Image<T> &imOut, vector<UINT> coords, T value=numeric_limits<T>::max())
    {
	if (coords.size()!=4)
	  return RES_ERR;
	return drawLine<T>(imOut, coords[0], coords[1], coords[2], coords[3], value);
    }



    /**
    * Draw a rectangle
    * 
    * 
    * \param imOut Output image.
    */
    template <class T>
    RES_T drawRectangle(Image<T> &imOut, size_t x0, size_t y0, size_t width, size_t height, T value=numeric_limits<T>::max(), bool fill=false, size_t zSlice=0)
    {
	ASSERT_ALLOCATED(&imOut);

	ImageFreezer freeze(imOut);
	
	size_t imW = imOut.getWidth();
	size_t imH = imOut.getHeight();
	size_t imD = imOut.getDepth();
	
	ASSERT((zSlice<imD), "zSlice is out of range", RES_ERR);
	
	size_t x1 = x0 + width - 1;
	size_t y1 = y0 + height -1;
	x1 = x1<imW ? x1 : imW-1;
	y1 = y1<imH ? y1 : imH-1;
	
	x0 = x0>=0 ? x0 : 0;
	y0 = y0>=0 ? y0 : 0;
	
	typename Image<T>::volType slices = imOut.getSlices();
	const typename Image<T>::sliceType lines = slices[zSlice];
	fillLine<T> fillFunc;
	
	if (fill)
	{
	    for (size_t j=y0;j<=y1;j++)
	      fillFunc(lines[j]+x0, width, value);
	}
	else
	{
	    fillFunc(lines[y0]+x0, width, value);
	    fillFunc(lines[y1]+x0, width, value);
	    for (size_t j=y0+1;j<=y1;j++)
	    {
		lines[j][x0] = value;
		lines[j][x1] = value;
	    }
	}
	
	return RES_OK;
    }


    template <class T>
    RES_T drawRectangle(Image<T> &imOut, vector<UINT> coords, T value=numeric_limits<T>::max(), bool fill=false)
    {
	if (coords.size()!=4)
	  return RES_ERR;
	return drawRectangle<T>(imOut, coords[0], coords[1], coords[2]-coords[0]+1, coords[3]-coords[1]+1, value, fill);
    }

    template <class T>
    RES_T drawRectangle(Image<T> &imOut, const map<UINT, UintVector> &coordsVect, T value=0, bool fill=false)
    {
	map<UINT, UintVector>::const_iterator it = coordsVect.begin();
	if (it->second.size()!=4)
	  return RES_ERR;
	for (;it!=coordsVect.end();it++)
	{
	    vector<UINT> coords = it->second;
	    T val = value==0 ? it->first : value;
	    if (drawRectangle<T>(imOut, coords[0], coords[1], coords[2]-coords[0]+1, coords[3]-coords[1]+1, val, fill)!=RES_OK)
	      return RES_ERR;
	}
	return RES_OK;
    }


    /**
    * Draw a box
    * 
    * 
    * \param imOut Output image.
    */
    template <class T>
    RES_T drawBox(Image<T> &imOut, size_t x0, size_t y0, size_t z0, size_t width, size_t height, size_t depth, T value=numeric_limits<T>::max(), bool fill=false)
    {
	ASSERT_ALLOCATED(&imOut);
	
	ImageFreezer freeze(imOut);
	
	ASSERT((drawRectangle(imOut, x0, y0, width, height, value, true, z0)==RES_OK));
	for (size_t z=z0+1;z<z0+depth-1;z++)
	  ASSERT((drawRectangle(imOut, x0, y0, width, height, value, fill, z)==RES_OK));
	ASSERT((drawRectangle(imOut, x0, y0, width, height, value, true, z0+depth-1)==RES_OK));
	
	return RES_OK;
	  
    }


    #ifdef USE_FREETYPE

    /**
    * Draw text on image
    * 
    * Requires the FreeType library
    */
    template <class T>
    RES_T drawText(Image<T> &imOut, size_t x, size_t y, size_t z, string txt, string font, UINT size=20, T value=ImDtTypes<T>::max())
    {
	ASSERT_ALLOCATED(&imOut);
	
	size_t imW = imOut.getWidth();
	size_t imH = imOut.getHeight();

	ASSERT((x>=0 && x<imW && y>=0 && y<imH && z>=0 && z<imOut.getDepth()), "Text position out of image range.", RES_ERR);
	
	FT_Library    library;
	FT_Face       face;
	FT_GlyphSlot  slot;
	const char *text = txt.c_str();
	
	ASSERT((!FT_Init_FreeType( &library )), "Problem initializing freetype library.", RES_ERR);
	ASSERT((!FT_New_Face( library, font.c_str(), 0, &face )), "The font file could not be opened.", RES_ERR);
	ASSERT((!FT_Set_Pixel_Sizes( face, 0, size )), "Error defining font size.", RES_ERR);
	
	slot = face->glyph;
	
	for (UINT c=0;c<txt.length();c++)
	{
	    FT_Load_Char( face, text[c], FT_LOAD_NO_BITMAP | FT_LOAD_RENDER | FT_LOAD_TARGET_MONO);
	
	    FT_Bitmap *bitmap = &slot->bitmap;
	
	    FT_Int  i, j, p, q;
	    FT_Int  x_min = x + slot->bitmap_left;
	    FT_Int  y_min = y - slot->bitmap_top;
	    FT_Int  x_max = x_min + bitmap->width;
	    FT_Int  y_max = y_min + bitmap->rows;

	    typename ImDtTypes<T>::sliceType slc = imOut.getSlices()[z];
	    

	    for ( j = y_min, q = 0; j < y_max; j++, q++ )
	    {
	      unsigned char *in = bitmap->buffer + q * bitmap->pitch;
	      typename ImDtTypes<T>::lineType out = slc[j];
	      unsigned char bit = 0x80;
	      for ( i = x_min, p = 0; i < x_max; i++, p++ )
	      {
		if (i>=0 && j>=0 && i<(int)imW && j<(int)imH)
		  if (*in & bit)
		    out[i] = value;
		bit >>= 1;
		if (bit == 0)
		{
		    bit = 0x80;
		    in++;
		}
	      }
	    }
	    x += slot->advance.x >> 6;
	    y += slot->advance.y >> 6;
	}
	
	FT_Done_Face    ( face );
	FT_Done_FreeType( library );
	
	imOut.modified();
	
	return RES_OK;
    }

    template <class T>
    RES_T drawText(Image<T> &imOut, size_t x, size_t y, string txt, string font, UINT size=20, T value=ImDtTypes<T>::max())
    {
	return drawText(imOut, x, y, 0, txt, font, size);
    }

    #endif // USE_FREETYPE

/** @}*/

} // namespace smil


#endif // _D_IMAGE_DRAW_HPP

