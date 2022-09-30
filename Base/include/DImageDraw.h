/*
 * Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS AND CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _D_IMAGE_DRAW_H
#define _D_IMAGE_DRAW_H

#include <vector>
#include <algorithm>

#include "Core/include/DCommon.h"
#include "Core/include/DBaseObject.h"

namespace smil
{
  /**
   * @ingroup Base
   * @defgroup Draw Draw Operations
   * @{
   */

  /**
   * @brief Find intermediate points forming a line between two end points,
   * using the Bresenham Line Draw Algorithm
   *
   * @param[in] (p1x, p1y), (p2x, p2y) : end points
   * @param[in] xMax, yMax : maximum values of X and y
   * @return vector with the intermediate points between extreme points.
   *
   * @note
   * - @I2D only
   *
   * @see
   *  - @UrlWikipedia{Bresenham%27s_line_algorithm, Bresenham's Line algorithm}
   *
   */
  inline vector<IntPoint> bresenhamPoints(int p1x, int p1y, int p2x, int p2y,
                                          int xMax = 0, int yMax = 0)
  {
    vector<IntPoint> points;
    int              F, x, y;

    bool swapped = false;
    if (p1x > p2x) // Swap points if p1 is on the right of p2
    {
      swap(p1x, p2x);
      swap(p1y, p2y);
      swapped = true;
    }

    // Handle trivial cases separately for algorithm speed up.
    // Trivial case 1: m = +/-INF (Vertical line)
    if (p1x == p2x) {
      if (p1y > p2y) // Swap y-coordinates if p1 is above p2
      {
        swap(p1y, p2y);
      }

      x = p1x;
      y = p1y;
      if ((xMax == 0) || (x >= 0 && x < xMax))
        while (y <= p2y) {
          points.push_back(IntPoint(x, y, 0));
          y++;
        }
      return points;
    }
    // Trivial case 2: m = 0 (Horizontal line)
    else if (p1y == p2y) {
      x = p1x;
      y = p1y;

      if ((yMax == 0) || (y >= 0 && y < yMax))
        while (x <= p2x) {
          points.push_back(IntPoint(x, y, 0));
          x++;
        }
      return points;
    }

    int dy            = p2y - p1y; // y-increment from p1 to p2
    int dx            = p2x - p1x; // x-increment from p1 to p2
    int dx2           = (dx << 1); // dx << 1 == 2*dx
    int dy2           = 2 * dy;    // dy can be negative, prefer * over <<
    int dy2_minus_dx2 = dy2 - dx2; // precompute constant for speed up
    int dy2_plus_dx2  = dy2 + dx2;

    if (dy >= 0) // m >= 0
    {
      // Case 1: 0 <= m <= 1 (Original case)
      if (dy <= dx) {
        F = dy2 - dx; // initial F

        x = p1x;
        y = p1y;
        while (x <= p2x) {
          if ((xMax == 0) || (x >= 0 && x < xMax && y >= 0 && y < yMax))
            points.push_back(IntPoint(x, y, 0));
          if (F <= 0) {
            F += dy2;
          } else {
            y++;
            F += dy2_minus_dx2;
          }
          x++;
        }
      }
      // Case 2: 1 < m < INF (Mirror about y=x line
      // replace all dy by dx and dx by dy)
      else {
        F = dx2 - dy; // initial F

        y = p1y;
        x = p1x;
        while (y <= p2y) {
          if ((xMax == 0) || (x >= 0 && x < xMax && y >= 0 && y < yMax))
            points.push_back(IntPoint(x, y, 0));
          if (F <= 0) {
            F += dx2;
          } else {
            x++;
            F -= dy2_minus_dx2;
          }
          y++;
        }
      }
    } else // m < 0
    {
      // Case 3: -1 <= m < 0 (Mirror about x-axis, replace all dy by -dy)
      if (dx >= -dy) {
        F = -dy2 - dx; // initial F

        x = p1x;
        y = p1y;
        while (x <= p2x) {
          if ((xMax == 0) || (x >= 0 && x < xMax && y >= 0 && y < yMax))
            points.push_back(IntPoint(x, y, 0));
          if (F <= 0) {
            F -= dy2;
          } else {
            y--;
            F -= dy2_plus_dx2;
          }
          x++;
        }
      }
      // Case 4: -INF < m < -1 (Mirror about x-axis and mirror
      // about y=x line, replace all dx by -dy and dy by dx)
      else {
        F = dx2 + dy; // initial F

        y = p1y;
        x = p1x;
        while (y >= p2y) {
          if ((xMax == 0) || (x >= 0 && x < xMax && y >= 0 && y < yMax))
            points.push_back(IntPoint(x, y, 0));
          if (F <= 0) {
            F += dx2;
          } else {
            x++;
            F += dy2_plus_dx2;
          }
          y--;
        }
      }
    }
    // If input points have been swapped, reverse the vector
    if (swapped)
      std::reverse(points.begin(), points.end());
    return points;
  }

  /**
   * @brief Find intermediate points forming a line between two end points,
   * using the Bresenham Line Draw Algorithm
   *
   * @param[in] (p1x, p1y), (p2x, p2y) : end points
   * @return vector with intermediate points between extreme points
   *
   * @note
   * - @I2D only
   *
   * @see
   *  - @UrlWikipedia{Bresenham%27s_line_algorithm, Bresenham's Line algorithm}
   */
  vector<IntPoint> bresenhamLine(int p1x, int p1y, int p2x, int p2y);

  /**
   * @brief Bresenham Class
   *
   * Find intermediate points forming a line between two end points,
   * using the Bresenham Line Draw Algorithm - @I2D or @I3D lines
   *
   * @smilexample{example-bresenham.py}
   *
   * @see
   *  - @UrlWikipedia{Bresenham%27s_line_algorithm, Bresenham's Line algorithm}
   */
  class Bresenham : public BaseObject
  {
  public:
    /**
     * Constructor : build a line (@I2D or @I3D) with extremities
     *  @TB{pi} and @TB{pf}
     *
     * @param[in] pi : initial point
     * @param[in] pf : final point
     */
    Bresenham(const IntPoint &pi, const IntPoint &pf)
        : BaseObject("Bresenham"), pi(pi), pf(pf)
    {
      doBresenham3D(pi.x, pi.y, pi.z, pf.x, pf.y, pf.z);
    }

    /**
     * Constructor : build a line (@I2D or @I3D) with extremities
     *  @TB{the origin} and @TB{pf}
     *
     * @param[in] pf : final point
     */
    Bresenham(const IntPoint &pf)
        : BaseObject("Bresenham"), pi(IntPoint(0, 0, 0)), pf(pf)
    {
      doBresenham3D(pi.x, pi.y, pi.z, pf.x, pf.y, pf.z);
    }

    /**
     * Constructor : build a @I3D line defined by the coordinates of
     * extremities
     *
     * @param[in] xi, yi, zi : coordinates of initial point
     * @param[in] xf, yf, zf : coordinates of final point
     */
    Bresenham(int xi, int yi, int zi, int xf, int yf, int zf)
        : BaseObject("Bresenham")
    {
      pi.x = xi;
      pi.y = yi;
      pi.z = zi;
      pf.x = xf;
      pf.y = yf;
      pf.z = zf;

      doBresenham3D(xi, yi, zi, xf, yf, zf);
    }

    /**
     * Constructor : build a @I2D line defined by the coordinates of
     * extremities
     *
     * @param[in] xi, yi : coordinates of initial point
     * @param[in] xf, yf : coordinates of final point
     */
    Bresenham(int xi, int yi, int xf, int yf) : BaseObject("Bresenham Line")
    {
      pi.x = xi;
      pi.y = yi;
      pi.z = 0;
      pf.x = xf;
      pf.y = yf;
      pf.z = 0;

      doBresenham3D(xi, yi, 0, xf, yf, 0);
    }

    /**
     * getPoints() - access a vector with the points of the line
     *
     * @returns a vector with the points of the line
     */
    vector<IntPoint> getPoints() const
    {
      return pts;
    }

    /**
     * getPoint() -
     *
     * @param[in] i : the index of the point to be accessed
     */
    IntPoint getPoint(UINT i)
    {
      if (i < pts.size())
        return pts[i];
      return IntPoint(0, 0, 0);
    }

    /** nbPoints() - the number of pixels in the line
     *
     * @returns - the number of pixels
     */
    size_t nbPoints()
    {
      return pts.size();
    }

    /** length() - length of the line (@TI{Euclidean distance} between
     * extremities)
     *
     * @returns line length
     */
    double length()
    {
      return std::sqrt(std::pow(pf.x - pi.x, 2) + std::pow(pf.y - pi.y, 2) +
                       std::pow(pf.z - pi.z, 2));
    }

    void printSelf(ostream &os = std::cout, string indent = "")
    {
      os << indent << "Bresenham Line" << endl;
      os << indent << "Class     : " << className << endl;
      // os << indent << "Name      : " << name << endl;

      for (UINT i = 0; i < pts.size(); i++)
        os << indent << "#" << i + 1 << "\t: (" << pts[i].x << "," << pts[i].y
           << "," << pts[i].z << ")" << endl;
    }

  private:
    IntPoint pi;
    IntPoint pf;

    vector<IntPoint> pts;

    void addPoint(IntPoint &p)
    {
      pts.push_back(p);
    }

    void doBresenham3D(int x1, int y1, int z1, int x2, int y2, int z2)
    {
      int dx, dy, dz;
      int dx2, dy2, dz2;

      int xInc, yInc, zInc;
      int xLen, yLen, zLen;

      IntPoint pt(x1, y1, z1);

      dx = x2 - x1;
      dy = y2 - y1;
      dz = z2 - z1;

      xInc = (dx < 0) ? -1 : 1;
      xLen = abs(dx);

      yInc = (dy < 0) ? -1 : 1;
      yLen = abs(dy);

      zInc = (dz < 0) ? -1 : 1;
      zLen = abs(dz);

      dx2 = 2 * xLen;
      dy2 = 2 * yLen;
      dz2 = 2 * zLen;

      if ((xLen >= yLen) && (xLen >= zLen)) {
        int err_1 = dy2 - xLen;
        int err_2 = dz2 - xLen;
        for (int i = 0; i < xLen; i++) {
          addPoint(pt);
          if (err_1 > 0) {
            pt.y += yInc;
            err_1 -= dx2;
          }
          if (err_2 > 0) {
            pt.z += zInc;
            err_2 -= dx2;
          }
          err_1 += dy2;
          err_2 += dz2;
          pt.x += xInc;
        }
        addPoint(pt);
        return;
      }

      if ((yLen >= xLen) && (yLen >= zLen)) {
        int err_1 = dx2 - yLen;
        int err_2 = dz2 - yLen;
        for (int i = 0; i < yLen; i++) {
          addPoint(pt);
          if (err_1 > 0) {
            pt.x += xInc;
            err_1 -= dy2;
          }
          if (err_2 > 0) {
            pt.z += zInc;
            err_2 -= dy2;
          }
          err_1 += dx2;
          err_2 += dz2;
          pt.y += yInc;
        }
        addPoint(pt);
        return;
      }

      if ((zLen >= xLen) && (zLen >= yLen)) {
        int err_1 = dy2 - zLen;
        int err_2 = dx2 - zLen;
        for (int i = 0; i < zLen; i++) {
          addPoint(pt);
          if (err_1 > 0) {
            pt.y += yInc;
            err_1 -= dz2;
          }
          if (err_2 > 0) {
            pt.x += xInc;
            err_2 -= dz2;
          }
          err_1 += dy2;
          err_2 += dx2;
          pt.z += zInc;
        }
        addPoint(pt);
        return;
      }
    }
  };

  /** @}*/

} // namespace smil

#endif // _D_IMAGE_DRAW_H
