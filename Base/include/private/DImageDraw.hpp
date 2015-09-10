/*
 * Copyright (c) 2011-2015, Matthieu FAESSEL and ARMINES
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
    * Draws a line between two points P0(x0,y0) and P1(x1,y1).
    * This function is based on the Bresenham's line algorithm.
    * (works only on 2D images)
    * \param x0,y0 Coordinates of the first point
    * \param x1,y1 Coordinates of the second point
    */
    template <class T>
    RES_T drawLine(Image<T> &im, int x0, int y0, int x1, int y1, T value=ImDtTypes<T>::max())
    {
        if (!im.isAllocated())
            return RES_ERR_BAD_ALLOCATION;

        size_t imW = im.getWidth();
        size_t imH = im.getHeight();
        
        vector<IntPoint> bPoints;
        if ( x0<0 || x0>=int(imW) || y0<0 || y0>=int(imH) || x1<0 || x1>=int(imW) || y1<0 || y1>=int(imH) )
          bPoints = bresenhamPoints(x0, y0, x1, y1, imW, imH);
        else
          bPoints = bresenhamPoints(x0, y0, x1, y1); // no image range check (faster)
        
        typename Image<T>::sliceType lines = im.getLines();
        
        for(vector<IntPoint>::iterator it=bPoints.begin();it!=bPoints.end();it++)
          lines[(*it).y][(*it).x] = value;
        
        im.modified();
        return RES_OK;
    }
    
    /**
     * \overload 
     * \brief Draw line from vector
     * \param coords Vector containing the coordiantes of the two end points (x0, y0, x1, y1)
     */
    template <class T>
    RES_T drawLine(Image<T> &imOut, vector<UINT> coords, T value=ImDtTypes<T>::max())
    {
        if (coords.size()!=4)
          return RES_ERR;
        return drawLine<T>(imOut, coords[0], coords[1], coords[2], coords[3], value);
    }



    /**
    * Draw a rectangle
    * 
    * \param imOut Output image.
    */
    template <class T>
    RES_T drawRectangle(Image<T> &imOut, int x0, int y0, size_t width, size_t height, T value=ImDtTypes<T>::max(), bool fill=false, size_t zSlice=0)
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
    RES_T drawRectangle(Image<T> &imOut, vector<UINT> coords, T value=ImDtTypes<T>::max(), bool fill=false)
    {
        if (coords.size()!=4)
          return RES_ERR;
        return drawRectangle<T>(imOut, coords[0], coords[1], coords[2]-coords[0]+1, coords[3]-coords[1]+1, value, fill);
    }

    /**
    * Draw a list of rectangles
    * 
    */
    template <class T, class MapT>
    RES_T drawRectangles(Image<T> &imOut, const map<MapT, Vector_size_t> &coordsVect, bool fill=false)
    {
        ASSERT_ALLOCATED(&imOut);
        ImageFreezer freeze(imOut);
        
        typename map<MapT, Vector_size_t>::const_iterator it = coordsVect.begin();
        if (it->second.size()!=4)
          return RES_ERR;
        for (;it!=coordsVect.end();it++)
        {
            vector<size_t> coords = it->second;
            T val = T(it->first);
            if (drawRectangle<T>(imOut, coords[0], coords[1], coords[2]-coords[0]+1, coords[3]-coords[1]+1, val, fill)!=RES_OK)
              return RES_ERR;
        }
        return RES_OK;
    }


    /**
    * Draw a circle
    * 
    * Bressenham's Midpoint Circle algoritm
    * \see drawDisc
    * 
    * \param imOut Output image.
    */
    template <class T>
    RES_T drawCircle(Image<T> &imOut, int x0, int y0, int radius, T value=ImDtTypes<T>::max(), size_t zSlice=0)
    {
        ASSERT_ALLOCATED(&imOut);
        ASSERT((zSlice<imOut.getDepth()), "zSlice is out of range", RES_ERR);
        
        ImageFreezer freeze(imOut);
        
        int imW = imOut.getWidth();
        int imH = imOut.getHeight();
        
        typename ImDtTypes<T>::sliceType lines = imOut.getSlices()[zSlice];
        
        int d = (5 - radius * 4) / 4;
        int x = 0;
        int y = radius;
//         int ptX, ptY;

        do
        {
            if (x0+x >= 0 && x0+x <= imW-1 && y0+y >= 0 && y0+y <= imH-1) lines[y0+y][x0+x] = value;
            if (x0+x >= 0 && x0+x <= imW-1 && y0-y >= 0 && y0-y <= imH-1) lines[y0-y][x0+x] = value;
            if (x0-x >= 0 && x0-x <= imW-1 && y0+y >= 0 && y0+y <= imH-1) lines[y0+y][x0-x] = value;
            if (x0-x >= 0 && x0-x <= imW-1 && y0-y >= 0 && y0-y <= imH-1) lines[y0-y][x0-x] = value;
            if (x0+y >= 0 && x0+y <= imW-1 && y0+x >= 0 && y0+x <= imH-1) lines[y0+x][x0+y] = value;
            if (x0+y >= 0 && x0+y <= imW-1 && y0-x >= 0 && y0-x <= imH-1) lines[y0-x][x0+y] = value;
            if (x0-y >= 0 && x0-y <= imW-1 && y0+x >= 0 && y0+x <= imH-1) lines[y0+x][x0-y] = value;
            if (x0-y >= 0 && x0-y <= imW-1 && y0-x >= 0 && y0-x <= imH-1) lines[y0-x][x0-y] = value;
            if (d < 0) 
            {
                  d = d + (4 * x) + 6;
            } 
            else 
            {
                  d = d + 4 * (x - y) + 10;
                  y--;
            }
            x++;
        } while (x <= y);
        
        return RES_OK;
    }
    
    /**
    * Draw a sphere
    * 
    * \param imOut Output image.
    */
    template <class T>
    RES_T drawSphere(Image<T> &imOut, int x0, int y0, int z0, int radius, T value=ImDtTypes<T>::max())
    {
        ASSERT_ALLOCATED(&imOut);

        ImageFreezer freeze(imOut);
        
        int imW = imOut.getWidth();
        int imH = imOut.getHeight();
        int imD = imOut.getDepth();
        
        int x1 = MAX(x0-radius, 0);
        int y1 = MAX(y0-radius, 0);
        int z1 = MAX(z0-radius, 0);
        
        int x2 = MIN(x0+radius, imW-1);
        int y2 = MIN(y0+radius, imH-1);
        int z2 = MIN(z0+radius, imD-1);
        
        int r2 = radius*radius;
        
        
        typename Image<T>::volType slices = imOut.getSlices();
        typename Image<T>::sliceType lines;
        typename Image<T>::lineType curLine;
        
        for (int z=z1;z<=z2;z++)
        {
            lines = slices[z];
            for (int y=y1;y<=y2;y++)
            {
                curLine = lines[y];
                for (int x=x1;x<=x2;x++)
                  if ((x-x0)*(x-x0)+(y-y0)*(y-y0)+(z-z0)*(z-z0)<=r2)
                    curLine[x] = value;
            }
            
        }
        
        return RES_OK;
    }

    /**
    * Draw a disc
    * 
    * \see drawCircle
    * 
    * \param imOut Output image.
    */
    template <class T>
    RES_T drawDisc(Image<T> &imOut, int x0, int y0, size_t zSlice, int radius, T value=ImDtTypes<T>::max())
    {
        ASSERT_ALLOCATED(&imOut);
        ASSERT((zSlice<imOut.getDepth()), "zSlice is out of range", RES_ERR);

        ImageFreezer freeze(imOut);
        
        int imW = imOut.getWidth();
        int imH = imOut.getHeight();
        
        int x1 = MAX(x0-radius, 0);
        int y1 = MAX(y0-radius, 0);
        
        int x2 = MIN(x0+radius, imW-1);
        int y2 = MIN(y0+radius, imH-1);
        
        int r2 = radius*radius;
        
        
        typename Image<T>::sliceType lines = imOut.getSlices()[zSlice];
        typename Image<T>::lineType curLine;
        
        for (int y=y1;y<=y2;y++)
        {
            curLine = lines[y];
            for (int x=x1;x<=x2;x++)
              if ((x-x0)*(x-x0)+(y-y0)*(y-y0)<=r2)
                curLine[x] = value;
        }
            
        return RES_OK;
    }
    
    // 2D Overload
    template <class T>
    RES_T drawDisc(Image<T> &imOut, int x0, int y0, int radius, T value=ImDtTypes<T>::max())
    {
        return drawDisc(imOut, x0, y0, 0, radius, value);
    }
    

    /**
    * Draw a box (3D)
    * 
    * 
    * \param imOut Output image.
    */
    template <class T>
    RES_T drawBox(Image<T> &imOut, size_t x0, size_t y0, size_t z0, size_t width, size_t height, size_t depth, T value=ImDtTypes<T>::max(), bool fill=false)
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
        return drawText(imOut, x, y, 0, txt, font, size, value);
    }

    #endif // USE_FREETYPE

    /**
     * Copy a given pattern (zone of input image) several times in an output image.
     * 
     * Only 2D for now.
     * 
     * If nbr_along_x and int nbr_along_y are not specified, fill completely the output image with a maximum number of parterns.
     * If x0, y0, width and height are not specified, the full image imIn will be copied.
     * 
     * Python example:
     * \code{.py}
     * im1 = Image(256,256)
     * im2 = Image(im1)
     * drawDisc(im1, 100,100, 15)
     * copyPattern(im1, 80, 80, 40, 40, im2)
     * im2.show()
     * \endcode
     */
    // TODO Extend to 3D
    template <class T>
    RES_T copyPattern(const Image<T> &imIn, int x0, int y0, int width, int height, Image<T> &imOut, int nbr_along_x, int nbr_along_y)
    {
        ASSERT_ALLOCATED(&imIn, &imOut)
        
        typename ImDtTypes<T>::sliceType linesIn = imIn.getSlices()[0];
        typename ImDtTypes<T>::sliceType linesOut = imOut.getSlices()[0];
        
        size_t imSize[3];
        imOut.getSize(imSize);
        
        int nX = min( int(ceil(double(imSize[0])/width)), nbr_along_x );
        int xPad = imSize[0] % width;
        int nXfull = xPad==0 ? nX : nX-1;
        
        int nY = min( int(ceil(double(imSize[1])/height)), nbr_along_y );
        int yPad = imSize[1] % height;
        int nYfull = yPad==0 ? nY : nY-1;
        
        size_t cpLen = nXfull*width + xPad;

#ifdef USE_OPEN_MP
        int nthreads = Core::getInstance()->getNumberOfThreads();
        #pragma omp parallel num_threads(nthreads)
#endif // USE_OPEN_MP
        {
            // Copy along X
            
#ifdef USE_OPEN_MP
            #pragma omp for
#endif // USE_OPEN_MP
            for (int j=0;j<height;j++)
            {
                typename ImDtTypes<T>::lineType lineIn = linesIn[y0+j] + x0;
                typename ImDtTypes<T>::lineType lineOut = linesOut[j];
                
                for (int i=0;i<nXfull;i++)
                {
                    copyLine<T>(lineIn, width, lineOut);
                    lineOut += width;
                }
                copyLine<T>(lineIn, xPad, lineOut);
            }
            
#ifdef USE_OPEN_MP
            #pragma omp barrier
#endif // USE_OPEN_MP
            
            // Copy along Y
            
            for (int n=1;n<nYfull;n++)
            {
#ifdef USE_OPEN_MP
                #pragma omp for
#endif // USE_OPEN_MP
                for (int j=0;j<height;j++)
                {
                    typename ImDtTypes<T>::lineType lineIn = linesOut[j];
                    typename ImDtTypes<T>::lineType lineOut = linesOut[n*height + j];
                    
                    copyLine<T>(lineIn, cpLen, lineOut);
                }
            }
            for (int n=nYfull;n<nY;n++)
            {
#ifdef USE_OPEN_MP
                #pragma omp for
#endif // USE_OPEN_MP
                for (int j=0;j<yPad;j++)
                {
                    typename ImDtTypes<T>::lineType lineIn = linesOut[j];
                    typename ImDtTypes<T>::lineType lineOut = linesOut[n*height + j];
                    
                    copyLine<T>(lineIn, cpLen, lineOut);
                }
            }
        }
          
        
        return RES_OK;
        
    }
    
    template <class T>
    RES_T copyPattern(const Image<T> &imIn, int x0, int y0, int width, int height, Image<T> &imOut)
    {
        return copyPattern(imIn, x0, y0, width, height, imOut, numeric_limits<int>::max(), numeric_limits<int>::max());
    }
    
    template <class T>
    RES_T copyPattern(const Image<T> &imIn, Image<T> &imOut, int nbr_along_x, int nbr_along_y)
    {
        return copyPattern(imIn, 0, 0, imIn.getWidth(), imIn.getHeight(), imOut, nbr_along_x, nbr_along_y);
    }

    template <class T>
    RES_T copyPattern(const Image<T> &imIn, Image<T> &imOut)
    {
        return copyPattern(imIn, 0, 0, imIn.getWidth(), imIn.getHeight(), imOut, numeric_limits<int>::max(), numeric_limits<int>::max());
    }
    
    
/** @}*/

} // namespace smil


#endif // _D_IMAGE_DRAW_HPP

