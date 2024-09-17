/* __HEAD__
 * Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES
 * Copyright (c) 2017-2024, Centre de Morphologie Mathematique
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
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Description :
 *   This file evaluates the Euclidean Distance on a binary image
 *
 * History :
 *   - 04/03/2022 - by Jose-Marcio
 *     Euclidean Distance - based on Beatriz Marcotegui implementation
 *
 * __HEAD__ - Stop here !
 */

#ifndef _D_MORPHOEUCLIDEAN_HPP
#define _D_MORPHOEUCLIDEAN_HPP

namespace smil
{
  class EuclideanFunctor
  {
  public:
    EuclideanFunctor()
    {
    }

    template <class T1, class T2>
    RES_T _exec(const Image<T1> &imIn, Image<T2> &imOut,
                const StrElt &se = DEFAULT_SE)
    {
      return _distance(imIn, imOut, se);
    }

  private:
    double sqr(size_t x)
    {
      return x * x;
    }

    double sqModule(IntPoint &p1, IntPoint &p2)
    {
      return sqr(p1.x - p2.x) + sqr(p1.y - p2.y) + sqr(p1.z - p2.z);
    }

    template <class T1, class T2>
    RES_T _distance(const Image<T1> &imIn, Image<T2> &imOut,
                    const StrElt &se = DEFAULT_SE)
    {
      ASSERT_ALLOCATED(&imIn, &imOut);
      ASSERT_SAME_SIZE(&imIn, &imOut);

      ImageFreezer freeze(imOut);

      typedef Image<T1>                       imageInType;
      typedef typename imageInType::lineType  lineInType;
      typedef Image<T2>                       imageOutType;
      typedef typename imageOutType::lineType lineOutType;

      lineInType  pixelsIn  = imIn.getPixels();
      lineOutType pixelsOut = imOut.getPixels();

      HierarchicalQueue<T2, off_t> hq;

      off_t         pixelCount = imIn.getPixelCount();
      vector<off_t> pixelsOffset(pixelCount, 0);

      StrElt seLoc = se.noCenter();

      off_t maxOffset = pixelCount;

      // INITIALIZE
      // Pixels != 0 but with a 0 neighbor (contour pixels) are:
      //  - pushed in the FAH
      //  - out[p] = 1
      //  - pixelOffset[p]=bckg neighbor

      // pixelOffset: contains the closest bck pixel for each object pixel (for
      // pixels with computed distance value)
      //  maxOffset+1 for pixels not processed yet (or background pixels)
      hq.initialize(imIn);
      for (auto i = 0; i < pixelCount; i++) {
        pixelsOffset[i] = maxOffset + 1;
        if (pixelsIn[i] == T1(0))
          continue;

        IntPoint pt = imIn.getPointFromOffset(i);

        bool oddLine = seLoc.odd && (pt.y % 2 != 0);

        for (auto it = seLoc.points.begin(); it != seLoc.points.end(); it++) {
          IntPoint ptSE = *it;
          IntPoint ptNb = pt + ptSE;
          off_t    ofNb;

          if (oddLine && (ptNb.y + 1) % 2 != 0)
            ptNb.x += 1;

          if (imIn.isPointInImage(ptNb))
            ofNb = imIn.getOffsetFromPoint(ptNb);
          else
            continue;

          if (pixelsIn[ofNb] == 0) {
            hq.push(T2(1), i);
            pixelsOffset[i] = ofNb;
            pixelsOut[i]    = 1;
            break;
          }
        }
      } // for each pixel

      //#####################################

      // Process hierarchical queue. Get a pixel(p0), find ngb without
      // computed distance(p1), p2 is a ngb of p1. If p2 knows its
      // distance, find p3 (the closest bckg pixel to p2) and compute
      // distance (p1,p3). Keep the smallest distance to p1 and assign
      // it. The origin of p1 is set, pixelsOffset[p1] is set to oo
      // (the p3 leading to the smallest distance).

      T2 T2_maxVal = ImDtTypes<T2>::max();

      while (!hq.isEmpty()) {
        off_t pCurr = hq.pop();
        if (pixelsOffset[pCurr] > maxOffset) {
          std::cout << "BAD PLACE, BAD TIME." << int(pCurr)
                    << "goes out without pixelFrom\n";
          continue;
        }

        IntPoint ptCurr = imIn.getPointFromOffset(pCurr);

        bool oddLine = seLoc.odd && (ptCurr.y % 2 != 0);
        for (auto it1 = seLoc.points.begin(); it1 != seLoc.points.end();
             it1++) {
          IntPoint ptSe1 = *it1;
          IntPoint pt1   = ptCurr + ptSe1;

          if (oddLine && ((pt1.y + 1) % 2) != 0)
            pt1.x += 1;

          if (!imIn.isPointInImage(pt1))
            continue;

          off_t p1 = imIn.getOffsetFromPoint(pt1);

          // non NULL input pixel, but without output value
          // (PixelsOffset == maxOffset+1)
          if ((pixelsIn[p1] == 0) || (pixelsOffset[p1] <= maxOffset))
            continue;

          int current_dist = ImDtTypes<int>::max(); // INFINITE!!!

          off_t oo;
          for (auto it2 = seLoc.points.begin(); it2 != seLoc.points.end();
               it2++) {
            IntPoint ptSe2 = *it2;
            IntPoint pt2   = pt1 + ptSe2;

            bool oddLine2 = seLoc.odd && (pt1.y % 2 != 0);
            if (oddLine2 && (((pt2.y + 1) % 2) != 0))
              pt2.x += 1;

            if (!imIn.isPointInImage(pt2))
              continue;

            off_t p2 = imIn.getOffsetFromPoint(pt2);

            if (pixelsOffset[p2] <= maxOffset) {
              /* voisin de distance calculee */
              off_t    p3  = pixelsOffset[p2];
              IntPoint pt3 = imIn.getPointFromOffset(p3);

              int wd = sqModule(pt1, pt3);
              if (wd < current_dist) {
                current_dist = wd;
                oo           = p2;
              }
            } // if ngb knows its distance
          }   // for ngb
          T2 pr1 = T2(std::sqrt(current_dist));

          if (pr1 > T2_maxVal)
            pr1 = T2_maxVal;

          hq.push(pr1, p1);

          pixelsOut[p1]    = pr1;
          pixelsOffset[p1] = pixelsOffset[oo];
        } // for each ngb of p
      }   // while ! EMPTY

      return RES_OK;
    } // END euclidian_distance
  };

  template <class T1, class T2>
  RES_T distanceEuclidean(const Image<T1> &imIn, Image<T2> &imOut,
                          const StrElt &se)
  {
    EuclideanFunctor func;

    return func._exec(imIn, imOut, se);
  }

} // namespace smil

#endif // _D_MORPHOEUCLIDEAN_HPP
