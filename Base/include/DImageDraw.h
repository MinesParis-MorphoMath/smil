/*
 * Copyright (c) 2011-2014, Matthieu FAESSEL and ARMINES
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


#ifndef _D_IMAGE_DRAW_H
#define _D_IMAGE_DRAW_H

#include <vector>

#include "Core/include/DCommon.h"

namespace smil
{
  
    /**
    * \ingroup Base
    * \defgroup Draw
    * @{
    */
    
    static std::vector<IntPoint> bresenhamPoints(int p1x, int p1y, int p2x, int p2y, int xMax=0, int yMax=0)
    {
	vector<IntPoint> points;
	int F, x, y;

	if (p1x > p2x)  // Swap points if p1 is on the right of p2
	{
	    swap(p1x, p2x);
	    swap(p1y, p2y);
	}

	// Handle trivial cases separately for algorithm speed up.
	// Trivial case 1: m = +/-INF (Vertical line)
	if (p1x == p2x)
	{
	    if (p1y > p2y)  // Swap y-coordinates if p1 is above p2
	    {
		swap(p1y, p2y);
	    }

	    x = p1x;
	    y = p1y;
	    if (!xMax || (x>=0 && x<xMax))
	      while (y <= p2y)
	      {
		  points.push_back(IntPoint(x,y,0));
		  y++;
	      }
	    return points;
	}
	// Trivial case 2: m = 0 (Horizontal line)
	else if (p1y == p2y)
	{
	    x = p1x;
	    y = p1y;

	    if (!yMax || (y>=0 && y<yMax))
	      while (x <= p2x)
	      {
		  points.push_back(IntPoint(x,y,0));
		  x++;
	      }
	    return points;
	}


	int dy            = p2y - p1y;  // y-increment from p1 to p2
	int dx            = p2x - p1x;  // x-increment from p1 to p2
	int dy2           = (dy << 1);  // dy << 1 == 2*dy
	int dx2           = (dx << 1);
	int dy2_minus_dx2 = dy2 - dx2;  // precompute constant for speed up
	int dy2_plus_dx2  = dy2 + dx2;


	if (dy >= 0)    // m >= 0
	{
	    // Case 1: 0 <= m <= 1 (Original case)
	    if (dy <= dx)   
	    {
		F = dy2 - dx;    // initial F

		x = p1x;
		y = p1y;
		while (x <= p2x)
		{
		    if (!xMax || (x>=0 && x<xMax && y>=0 && y<yMax))
		      points.push_back(IntPoint(x,y,0));
		    if (F <= 0)
		    {
			F += dy2;
		    }
		    else
		    {
			y++;
			F += dy2_minus_dx2;
		    }
		    x++;
		}
	    }
	    // Case 2: 1 < m < INF (Mirror about y=x line
	    // replace all dy by dx and dx by dy)
	    else
	    {
		F = dx2 - dy;    // initial F

		y = p1y;
		x = p1x;
		while (y <= p2y)
		{
		    if (!xMax || (x>=0 && x<xMax && y>=0 && y<yMax))
		      points.push_back(IntPoint(x,y,0));
		    if (F <= 0)
		    {
			F += dx2;
		    }
		    else
		    {
			x++;
			F -= dy2_minus_dx2;
		    }
		    y++;
		}
	    }
	}
	else    // m < 0
	{
	    // Case 3: -1 <= m < 0 (Mirror about x-axis, replace all dy by -dy)
	    if (dx >= -dy)
	    {
		F = -dy2 - dx;    // initial F

		x = p1x;
		y = p1y;
		while (x <= p2x)
		{
		    if (!xMax || (x>=0 && x<xMax && y>=0 && y<yMax))
		      points.push_back(IntPoint(x,y,0));
		    if (F <= 0)
		    {
			F -= dy2;
		    }
		    else
		    {
			y--;
			F -= dy2_plus_dx2;
		    }
		    x++;
		}
	    }
	    // Case 4: -INF < m < -1 (Mirror about x-axis and mirror 
	    // about y=x line, replace all dx by -dy and dy by dx)
	    else    
	    {
		F = dx2 + dy;    // initial F

		y = p1y;
		x = p1x;
		while (y >= p2y)
		{
		    if (!xMax || (x>=0 && x<xMax && y>=0 && y<yMax))
		      points.push_back(IntPoint(x,y,0));
		    if (F <= 0)
		    {
			F += dx2;
		    }
		    else
		    {
			x++;
			F += dy2_plus_dx2;
		    }
		    y--;
		}
	    }
	}
	return points;
    }

/** @}*/

} // namespace smil


#endif // _D_IMAGE_DRAW_HPP

