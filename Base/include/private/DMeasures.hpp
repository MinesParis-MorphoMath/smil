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

#ifndef _D_MEASURES_HPP
#define _D_MEASURES_HPP

#include "Core/include/private/DImage.hpp"
#include "DBaseMeasureOperations.hpp"
#include "DImageArith.hpp"
#include "Base/include/DImageDraw.h"

#include <cmath>
#include <map>
#include <set>
#include <iostream>
#include <iterator> // std::back_inserter

namespace smil
{
  /**
   * @ingroup Base
   * @defgroup Measures Base measures
   *
   * Common useful measures on images, not based on Morphological operations
   *
   * @{
   */

  /** @cond */
  template <class T>
  struct measAreaFunc : public MeasureFunctionBase<T, double> {
    typedef typename Image<T>::lineType lineType;

    virtual void processSequence(lineType /*lineIn*/, size_t size)
    {
      this->retVal += size;
    }
  };
  /** @endcond */

  //
  //     ##    #####   ######    ##
  //    #  #   #    #  #        #  #
  //   #    #  #    #  #####   #    #
  //   ######  #####   #       ######
  //   #    #  #   #   #       #    #
  //   #    #  #    #  ######  #    #
  //
  /**
   * area() - Area of an image
   *
   * The area of an image is defined as the number of non-zero pixels
   *
   * @param[in] imIn : Input image.
   * @return the number of non-zero pixels
   *
   * @note The name of this function comes from times where images were
   * mostly 2D only. In those days, the area of an image was said to be the
   * area of @b xy plane where pixel values are non-zero, i.e., the number of
   * non-zero pixels.
   * This may be confusing for 3D images, but the idea remains the same.
   */
  template <class T> size_t area(const Image<T> &imIn)
  {
    measAreaFunc<T> func;
    return static_cast<size_t>(func(imIn, true));
  }

  /** @cond */
  template <class T>
  struct measVolFunc : public MeasureFunctionBase<T, double> {
    typedef typename Image<T>::lineType lineType;

    virtual void processSequence(lineType lineIn, size_t size)
    {
      for (size_t i = 0; i < size; i++)
        this->retVal += double(lineIn[i]);
    }
  };
  /** @endcond */

  //
  //   #    #   ####   #       #    #  #    #  ######
  //   #    #  #    #  #       #    #  ##  ##  #
  //   #    #  #    #  #       #    #  # ## #  #####
  //   #    #  #    #  #       #    #  #    #  #
  //    #  #   #    #  #       #    #  #    #  #
  //     ##     ####   ######   ####   #    #  ######
  //
  /**
   * vol() - Volume of an image
   *
   * @param[in] imIn : Input image.
   * @return the sum of pixel values
   *
   * This is the same than the @b volume function call. Better use the
   * unabridged name. The abridged name remains for back compatibility.
   *
   */
  template <class T> double vol(const Image<T> &imIn)
  {
#if 1
    return volume(imIn);
#else
    measVolFunc<T> func;
    return func(imIn, false);
#endif
  }

  /**
   * colume() - Volume of an image
   *
   * The volume of an image is defined as the sum of the pixel values.
   *
   * @param[in] imIn : Input image.
   * @return the sum of pixel values (as a double)
   *
   * @note The name of this function comes from times where images were
   * mostly 2D only. In those days, the volume of an image was said to be the
   * volume defined by the @b xy plane with the third dimension being the
   * intensity (pixel values).
   * This may be confusing for 3D images, but the idea remains the same.
   */
  template <class T> double volume(const Image<T> &imIn)
  {
    measVolFunc<T> func;
    return func(imIn, false);
  }

  //
  //   #    #  ######    ##    #    #  #    #    ##    #
  //   ##  ##  #        #  #   ##   #  #    #   #  #   #
  //   # ## #  #####   #    #  # #  #  #    #  #    #  #
  //   #    #  #       ######  #  # #  #    #  ######  #
  //   #    #  #       #    #  #   ##   #  #   #    #  #
  //   #    #  ######  #    #  #    #    ##    #    #  ######
  //
  /** @cond */
  template <class T>
  struct measMeanValFunc : public MeasureFunctionBase<T, Vector_double> {
    typedef typename Image<T>::lineType lineType;
    double sum1, sum2;
    double pixNbr;

    virtual void initialize(const Image<T> & /*imIn*/)
    {
      this->retVal.clear();
      sum1 = sum2 = pixNbr = 0.;
    }
    virtual void processSequence(lineType lineIn, size_t size)
    {
      double curV;
      for (size_t i = 0; i < size; i++) {
        pixNbr += 1;
        curV = lineIn[i];
        sum1 += curV;
        sum2 += curV * curV;
      }
    }
    virtual void finalize(const Image<T> & /*imIn*/)
    {
      double mean_val = pixNbr == 0 ? 0 : sum1 / pixNbr;
      double std_dev_val =
          pixNbr == 0 ? 0 : sqrt(sum2 / pixNbr - mean_val * mean_val);

      this->retVal.push_back(mean_val);
      this->retVal.push_back(std_dev_val);
    }
  };
  /** @endcond */

  /**
   * meanVal() - Mean value and standard deviation
   *
   * @param[in] imIn : Input image.
   * @param[in] onlyNonZero : If true, only non-zero pixels are considered.
   * @return a vector with the @b mean and <b>standard deviation</b> of pixel
   * values
   */
  template <class T>
  Vector_double meanVal(const Image<T> &imIn, bool onlyNonZero = false)
  {
    measMeanValFunc<T> func;
    return func(imIn, onlyNonZero);
  }

  //
  //   #    #     #    #    #  #    #    ##    #
  //   ##  ##     #    ##   #  #    #   #  #   #
  //   # ## #     #    # #  #  #    #  #    #  #
  //   #    #     #    #  # #  #    #  ######  #
  //   #    #     #    #   ##   #  #   #    #  #
  //   #    #     #    #    #    ##    #    #  ######
  //
  /** @cond */
  template <class T> struct measMinValFunc : public MeasureFunctionBase<T, T> {
    typedef typename Image<T>::lineType lineType;
    virtual void initialize(const Image<T> & /*imIn*/)
    {
      this->retVal = numeric_limits<T>::max();
    }
    virtual void processSequence(lineType lineIn, size_t size)
    {
      for (size_t i = 0; i < size; i++)
        if (lineIn[i] < this->retVal)
          this->retVal = lineIn[i];
    }
  };

  template <class T>
  struct measMinValPosFunc : public MeasureFunctionWithPos<T, T> {
    typedef typename Image<T>::lineType lineType;
    Point<UINT> pt;
    virtual void initialize(const Image<T> & /*imIn*/)
    {
      this->retVal = numeric_limits<T>::max();
    }
    virtual void processSequence(lineType lineIn, size_t size, size_t x,
                                 size_t y, size_t z)
    {
      for (size_t i = 0; i < size; i++, x++)
        if (lineIn[i] < this->retVal) {
          this->retVal = lineIn[i];
          pt.x         = x;
          pt.y         = y;
          pt.z         = z;
        }
    }
  };
  /** @endcond */

  /**
   * minVal() - Min value of an image
   *
   * @param[in] imIn : Input image.
   * @param[in] onlyNonZero : If true, only non-zero pixels are considered.
   * @return the min of the pixel values.
   */
  template <class T> T minVal(const Image<T> &imIn, bool onlyNonZero = false)
  {
    measMinValFunc<T> func;
    return func(imIn, onlyNonZero);
  }

  /**
   * minVal() - Min value of an image
   *
   * @param[in] imIn : Input image.
   * @param[out] pt : point coordinates of the minimum value in the image.
   * @param[in] onlyNonZero : If true, only non-zero pixels are considered.
   * @return the min of the pixel values.
   */
  template <class T>
  T minVal(const Image<T> &imIn, Point<UINT> &pt, bool onlyNonZero = false)
  {
    measMinValPosFunc<T> func;
    func(imIn, onlyNonZero);
    pt = func.pt;
    return func.retVal;
  }

  //
  //   #    #    ##    #    #  #    #    ##    #
  //   ##  ##   #  #    #  #   #    #   #  #   #
  //   # ## #  #    #    ##    #    #  #    #  #
  //   #    #  ######    ##    #    #  ######  #
  //   #    #  #    #   #  #    #  #   #    #  #
  //   #    #  #    #  #    #    ##    #    #  ######
  //
  /** @cond */
  template <class T> struct measMaxValFunc : public MeasureFunctionBase<T, T> {
    typedef typename Image<T>::lineType lineType;
    virtual void initialize(const Image<T> & /*imIn*/)
    {
      this->retVal = numeric_limits<T>::min();
    }
    virtual void processSequence(lineType lineIn, size_t size)
    {
      for (size_t i = 0; i < size; i++)
        if (lineIn[i] > this->retVal)
          this->retVal = lineIn[i];
    }
  };

  template <class T>
  struct measMaxValPosFunc : public MeasureFunctionWithPos<T, T> {
    typedef typename Image<T>::lineType lineType;
    Point<UINT> pt;
    virtual void initialize(const Image<T> & /*imIn*/)
    {
      this->retVal = numeric_limits<T>::min();
    }
    virtual void processSequence(lineType lineIn, size_t size, size_t x,
                                 size_t y, size_t z)
    {
      for (size_t i = 0; i < size; i++, x++)
        if (lineIn[i] > this->retVal) {
          this->retVal = lineIn[i];
          pt.x         = x;
          pt.y         = y;
          pt.z         = z;
        }
    }
  };
  /** @endcond */

  /**
   * maxVal() - Max value of an image
   *
   * @param[in] imIn : Input image.
   * @param[in] onlyNonZero : If true, only non-zero pixels are considered.
   * @return the max of the pixel values.
   */
  template <class T> T maxVal(const Image<T> &imIn, bool onlyNonZero = false)
  {
    measMaxValFunc<T> func;
    return func(imIn, onlyNonZero);
  }

  /**
   * maxVal() - Max value of an image
   *
   * @param[in] imIn : Input image.
   * @param[out] pt : point coordinates of the maximum value in the image.
   * @param[in] onlyNonZero : If true, only non-zero pixels are considered.
   * @return the max of the pixel values.
   */
  template <class T>
  T maxVal(const Image<T> &imIn, Point<UINT> &pt, bool onlyNonZero = false)
  {
    measMaxValPosFunc<T> func;
    func(imIn, onlyNonZero);
    pt = func.pt;
    return func.retVal;
  }

  //
  //   #    #     #    #    #  #    #    ##    #    #  #    #    ##    #
  //   ##  ##     #    ##   #  ##  ##   #  #    #  #   #    #   #  #   #
  //   # ## #     #    # #  #  # ## #  #    #    ##    #    #  #    #  #
  //   #    #     #    #  # #  #    #  ######    ##    #    #  ######  #
  //   #    #     #    #   ##  #    #  #    #   #  #    #  #   #    #  #
  //   #    #     #    #    #  #    #  #    #  #    #    ##    #    #  ######
  //
  /** @cond */
  template <class T>
  struct measMinMaxValFunc : public MeasureFunctionBase<T, vector<T>> {
    typedef typename Image<T>::lineType lineType;
    T minVal, maxVal;
    virtual void initialize(const Image<T> & /*imIn*/)
    {
      this->retVal.clear();
      maxVal = numeric_limits<T>::min();
      minVal = numeric_limits<T>::max();
    }
    virtual void processSequence(lineType lineIn, size_t size)
    {
      for (size_t i = 0; i < size; i++) {
        T val = lineIn[i];
        if (val > maxVal)
          maxVal = val;
        if (val < minVal)
          minVal = val;
      }
    }
    virtual void finalize(const Image<T> & /*imIn*/)
    {
      this->retVal.push_back(minVal);
      this->retVal.push_back(maxVal);
    }
  };
  /** @endcond */

  /**
   * rangeVal() - Min and Max values of an image
   *
   * @param[in] imIn : Input image.
   * @param[in] onlyNonZero : If true, only non-zero pixels are considered.
   * @returns a vector with the min and the max of the pixel values.
   */
  template <class T>
  vector<T> rangeVal(const Image<T> &imIn, bool onlyNonZero = false)
  {
    measMinMaxValFunc<T> func;
    return func(imIn, onlyNonZero);
  }

  //
  //   #    #    ##    #       #    #  ######  #          #     ####    #####
  //   #    #   #  #   #       #    #  #       #          #    #          #
  //   #    #  #    #  #       #    #  #####   #          #     ####      #
  //   #    #  ######  #       #    #  #       #          #         #     #
  //    #  #   #    #  #       #    #  #       #          #    #    #     #
  //     ##    #    #  ######   ####   ######  ######     #     ####      #
  //
  /** @cond */
  template <class T>
  struct valueListFunc : public MeasureFunctionBase<T, vector<T>> {
    typedef typename Image<T>::lineType lineType;
    set<T> valList;

    virtual void initialize(const Image<T> & /*imIn*/)
    {
      this->retVal.clear();
      valList.clear();
    }

    virtual void processSequence(lineType lineIn, size_t size)
    {
      for (size_t i = 0; i < size; i++)
        valList.insert(lineIn[i]);
    }
    virtual void finalize(const Image<T> & /*imIn*/)
    {
      // Copy the content of the set into the ret vector
      std::copy(valList.begin(), valList.end(),
                std::back_inserter(this->retVal));
    }
  };
  /** @endcond */

  /**
   * valueList() - Get the list of the pixel values present in the image
   *
   * @param[in] imIn : Input image.
   * @param[in] onlyNonZero : If true, only non-zero pixels are considered.
   * @return a vector with the values found in the image.
   *
   * @see histogram
   * @warning In huge images of type @b UINT32, this function may return a
   * huge vector.
   */
  template <class T>
  vector<T> valueList(const Image<T> &imIn, bool onlyNonZero = true)
  {
    valueListFunc<T> func;
    return func(imIn, onlyNonZero);
  }

  //
  //   #    #   ####   #####   ######  #    #    ##    #
  //   ##  ##  #    #  #    #  #       #    #   #  #   #
  //   # ## #  #    #  #    #  #####   #    #  #    #  #
  //   #    #  #    #  #    #  #       #    #  ######  #
  //   #    #  #    #  #    #  #        #  #   #    #  #
  //   #    #   ####   #####   ######    ##    #    #  ######
  //
  /** @cond */
  template <class T> struct measModeValFunc : public MeasureFunctionBase<T, T> {
    typedef typename Image<T>::lineType lineType;

    map<int, int> nbList;
    int maxNb;
    T mode;
    virtual void initialize(const Image<T> & /*imIn*/)
    {
      // BMI            this->retVal.clear();
      nbList.clear();
      maxNb = 0;
      mode  = 0;
    }

    virtual void processSequence(lineType lineIn, size_t size)
    {
      for (size_t i = 0; i < size; i++) {
        T val = lineIn[i];
        if (val > 0) {
          if (nbList.find(val) == nbList.end()) {
            nbList.insert(std::pair<int, int>(val, 1));
          } else
            nbList[val]++;
          if (nbList[val] > maxNb) {
            mode  = val;
            maxNb = nbList[val];
          }
        } // if (val>0)
      }   // for i= 0; i < size
      this->retVal = mode;

    } // virtual
  };  // END measModeValFunc
  /** @endcond */

  /**
   * modeVal() - Get the mode of the histogram present in the image, i.e. the
   * value that appears most often.
   *
   * @param[in] imIn : input image
   * @param[in] onlyNonZero : consider only non zero values
   * @return the value that appears more often
   *
   * @note
   * As this function returns only one value :
   * - in a distribution with the same maximum for many values, it returns the
   * first one;
   * - in a multimodal distribution, it returns the first biggest one;
   */
  template <class T> T modeVal(const Image<T> &imIn, bool onlyNonZero = true)
  {
    measModeValFunc<T> func;
    return func(imIn, onlyNonZero);
  }

  //
  //   #    #  ######  #####      #      ##    #    #  #    #    ##    #
  //   ##  ##  #       #    #     #     #  #   ##   #  #    #   #  #   #
  //   # ## #  #####   #    #     #    #    #  # #  #  #    #  #    #  #
  //   #    #  #       #    #     #    ######  #  # #  #    #  ######  #
  //   #    #  #       #    #     #    #    #  #   ##   #  #   #    #  #
  //   #    #  ######  #####      #    #    #  #    #    ##    #    #  ######
  //
  /** @cond */
  template <class T>
  struct measMedianValFunc : public MeasureFunctionBase<T, T> {
    typedef typename Image<T>::lineType lineType;

    map<int, int> nbList;
    size_t acc_elem, total_elems;
    T medianval;
    virtual void initialize(const Image<T> & /*imIn*/)
    {
      // BMI            this->retVal.clear();
      nbList.clear();
      medianval   = 0;
      acc_elem    = 0;
      total_elems = 0;
    }

    virtual void processSequence(lineType lineIn, size_t size)
    {
      for (size_t i = 0; i < size; i++) {
        T val = lineIn[i];
        if (val > 0) {
          total_elems++;
          if (nbList.find(val) == nbList.end()) {
            nbList.insert(std::pair<int, int>(val, 1));
          } else
            nbList[val]++;

        } // if (val>0)
      }   // for i= 0; i < size
          //            this->retVal = medianval;
    }     // virtual processSequence

    virtual void finalize(const Image<T> & /*imIn*/)
    {
      typedef std::map<int, int>::iterator it_type;

      for (it_type my_iterator = nbList.begin(); my_iterator != nbList.end();
           my_iterator++) {
        acc_elem = acc_elem + my_iterator->second; //  nbList;
        if (acc_elem > total_elems / 2.0) {
          medianval = my_iterator->first;
          break;
        }
        // iterator->first = key
        // iterator->second = value
      } // iterator

      this->retVal = medianval;
      // this->retVal.push_back(xSum/tSum);
    }

  }; // END measMedianValFunc
  /** @endcond */

  /**
   * medianVal() - Get the median of the image histogram.
   * @param[in] imIn : Input image.
   * @param[in] onlyNonZero : If true, only non-zero pixels are considered.
   * @return the median of the image histogram
   */
  template <class T> T medianVal(const Image<T> &imIn, bool onlyNonZero = true)
  {
    measMedianValFunc<T> func;
    return func(imIn, onlyNonZero);
  }

  //
  //   #####   #####    ####   ######     #    #       ######
  //   #    #  #    #  #    #  #          #    #       #
  //   #    #  #    #  #    #  #####      #    #       #####
  //   #####   #####   #    #  #          #    #       #
  //   #       #   #   #    #  #          #    #       #
  //   #       #    #   ####   #          #    ######  ######
  //
  /**
   * profile() - Get image values along a line defined by the points
   * @f$(x_0, y_0)@f$ and @f$(x_1, y_1)@f$ in the slice @f$z@f$.
   *
   * @param[in] im : input image
   * @param[in] x0, y0 : start point
   * @param[in] x1, y1 : end point
   * @param[in] z : slice
   * @return vector with pixel values
   */
  template <class T>
  vector<T> profile(const Image<T> &im, size_t x0, size_t y0, size_t x1,
                    size_t y1, size_t z = 0)
  {
    vector<T> vec;
    ASSERT(im.isAllocated(), vec);

    size_t imW = im.getWidth();
    size_t imH = im.getHeight();

    vector<IntPoint> bPoints;
    if (x0 >= imW || y0 >= imH || x1 >= imW || y1 >= imH)
      bPoints = bresenhamPoints(x0, y0, x1, y1, imW, imH);
    else
      // no image range check (faster)
      bPoints = bresenhamPoints(x0, y0, x1, y1);

    typename Image<T>::sliceType lines = im.getSlices()[z];

    for (vector<IntPoint>::iterator it = bPoints.begin(); it != bPoints.end();
         it++)
      vec.push_back(lines[(*it).y][(*it).x]);

    return vec;
  }

  //
  // #####     ##    #####  #   #  ####   ######  #    #  #####  ######  #####
  // #    #   #  #   #    #  # #  #    #  #       ##   #    #    #       #    #
  // #####   #    #  #    #   #   #       #####   # #  #    #    #####   #    #
  // #    #  ######  #####    #   #       #       #  # #    #    #       #####
  // #    #  #    #  #   #    #   #    #  #       #   ##    #    #       #   #
  // #####   #    #  #    #   #    ####   ######  #    #    #    ######  #    #
  //
  /** @cond */
  template <class T>
  struct measBarycenterFunc : public MeasureFunctionWithPos<T, Vector_double> {
    typedef typename Image<T>::lineType lineType;
    double xSum, ySum, zSum, tSum;
    virtual void initialize(const Image<T> & /*imIn*/)
    {
      this->retVal.clear();
      xSum = ySum = zSum = tSum = 0.;
    }
    virtual void processSequence(lineType lineIn, size_t size, size_t x,
                                 size_t y, size_t z)
    {
      for (size_t i = 0; i < size; i++, x++) {
        T pixVal = lineIn[i];
        xSum += double(pixVal) * x;
        ySum += double(pixVal) * y;
        zSum += double(pixVal) * z;
        tSum += double(pixVal);
      }
    }
    virtual void finalize(const Image<T> &imIn)
    {
      this->retVal.push_back(xSum / tSum);
      this->retVal.push_back(ySum / tSum);
      if (imIn.getDimension() == 3)
        this->retVal.push_back(zSum / tSum);
    }
  };
  /** @endcond */

  /**
   * measBarycenter() - Gets the barycenter coordinates of an image
   * @param[in] im : input image
   * @return vector with the coordinates of barycenter
   */
  template <class T> Vector_double measBarycenter(Image<T> &im)
  {
    measBarycenterFunc<T> func;
    return func(im, false);
  }

  //
  //   #####    ####   #    #  #    #  #####   #####    ####   #    #
  //   #    #  #    #  #    #  ##   #  #    #  #    #  #    #   #  #
  //   #####   #    #  #    #  # #  #  #    #  #####   #    #    ##
  //   #    #  #    #  #    #  #  # #  #    #  #    #  #    #    ##
  //   #    #  #    #  #    #  #   ##  #    #  #    #  #    #   #  #
  //   #####    ####    ####   #    #  #####   #####    ####   #    #
  //
  /** @cond */
  template <class T>
  struct measBoundBoxFunc : public MeasureFunctionWithPos<T, vector<size_t>> {
    typedef typename Image<T>::lineType lineType;
    double xMin, xMax, yMin, yMax, zMin, zMax;
    bool im3d;
    virtual void initialize(const Image<T> &imIn)
    {
      this->retVal.clear();
      size_t imSize[3];
      imIn.getSize(imSize);
      im3d = (imSize[2] > 1);

      xMin = imSize[0];
      xMax = 0;
      yMin = imSize[1];
      yMax = 0;
      zMin = imSize[2];
      zMax = 0;
    }
    virtual void processSequence(lineType /*lineIn*/, size_t size, size_t x,
                                 size_t y, size_t z)
    {
      if (x < xMin)
        xMin = x;
      if (x + size - 1 > xMax)
        xMax = x + size - 1;
      if (y < yMin)
        yMin = y;
      if (y > yMax)
        yMax = y;
      if (im3d) {
        if (z < zMin)
          zMin = z;
        if (z > zMax)
          zMax = z;
      }
    }
    virtual void finalize(const Image<T> & /*imIn*/)
    {
      this->retVal.push_back(UINT(xMin));
      this->retVal.push_back(UINT(yMin));
      if (im3d)
        this->retVal.push_back(UINT(zMin));
      this->retVal.push_back(UINT(xMax));
      this->retVal.push_back(UINT(yMax));
      if (im3d)
        this->retVal.push_back(UINT(zMax));
    }
  };
  /** @endcond */

  /**
   * measBoundBox() - Bounding Box measure - gets the coordinates of the
   * bounding box
   *
   * @param[in] im : input image
   * @return a vector with <b> xMin, yMin (,zMin), xMax, yMax (,zMax) </b>
   */
  template <class T> vector<size_t> measBoundBox(Image<T> &im)
  {
    measBoundBoxFunc<T> func;
    return func(im, true);
  }

  //
  //   #    #   ####   #    #  ######  #    #   #####   ####
  //   ##  ##  #    #  ##  ##  #       ##   #     #    #
  //   # ## #  #    #  # ## #  #####   # #  #     #     ####
  //   #    #  #    #  #    #  #       #  # #     #         #
  //   #    #  #    #  #    #  #       #   ##     #    #    #
  //   #    #   ####   #    #  ######  #    #     #     ####
  //
  /** @cond */
  template <class T>
  struct measMomentsFunc : public MeasureFunctionWithPos<T, Vector_double> {
    typedef typename Image<T>::lineType lineType;
    double m000, m100, m010, m110, m200, m020, m001, m101, m011, m002;
    bool im3d;
    virtual void initialize(const Image<T> &imIn)
    {
      im3d = (imIn.getDimension() == 3);
      this->retVal.clear();
      m000 = m100 = m010 = m110 = m200 = m020 = m001 = m101 = m011 = m002 = 0.;
    }
    virtual void processSequence(lineType lineIn, size_t size, size_t x,
                                 size_t y, size_t z)
    {
      for (size_t i = 0; i < size; i++, x++) {
        double pxVal = double(lineIn[i]);
        m000 += pxVal;
        m100 += pxVal * x;
        m010 += pxVal * y;
        m110 += pxVal * x * y;
        m200 += pxVal * x * x;
        m020 += pxVal * y * y;
        if (im3d) {
          m001 += pxVal * z;
          m101 += pxVal * x * z;
          m011 += pxVal * y * z;
          m002 += pxVal * z * z;
        }
      }
    }
    virtual void finalize(const Image<T> & /*imIn*/)
    {
      this->retVal.push_back(m000);
      this->retVal.push_back(m100);
      this->retVal.push_back(m010);
      if (im3d)
        this->retVal.push_back(m001);
      this->retVal.push_back(m110);
      if (im3d) {
        this->retVal.push_back(m101);
        this->retVal.push_back(m011);
      }
      this->retVal.push_back(m200);
      this->retVal.push_back(m020);
      if (im3d)
        this->retVal.push_back(m002);
    }
  };
  /** @endcond */

  /** @cond */
  inline Vector_double centerMoments(Vector_double &moments)
  {
    double EPSILON = 1e-8;

    Vector_double m(moments.size(), 0);
    if (moments[0] < EPSILON)
      return m;

    bool im3d = (moments.size() == 10);

    /* Moments order in moment vector :
     * 2D - m00   m10   m01   m11   m20   m02
     * 3D - m000  m100  m010  m001  m110  m101  m011  m200  m020  m002
     */
    m = moments;
    if (im3d) {
      // m000
      m[0] = moments[0];
      // m100 m010 m001
      m[1] = 0.;
      m[2] = 0.;
      m[3] = 0.;
      // m110 m101 m011
      m[4] -= moments[2] * moments[1] / moments[0];
      m[5] -= moments[3] * moments[1] / moments[0];
      m[6] -= moments[3] * moments[2] / moments[0];
      // m200 m020 m002
      m[7] -= moments[1] * moments[1] / moments[0];
      m[8] -= moments[2] * moments[2] / moments[0];
      m[9] -= moments[3] * moments[3] / moments[0];
    } else {
      // m00
      m[0] = moments[0];
      // m10 m01
      m[1] = 0.;
      m[2] = 0.;
      // m11
      m[3] -= moments[2] * moments[1] / moments[0];
      // m20 m02
      m[4] -= moments[1] * moments[1] / moments[0];
      m[5] -= moments[2] * moments[2] / moments[0];
    }

    return m;
  }
  /** @endcond */

  /**
   * measMoments() - Measure image moments
   *
   * @param[in] im : input image
   * @param[in] onlyNonZero : use only non zero values
   * @param[in] centered : returns centered moments
   *
   * @return For 2D images: vector(m00, m10, m01, m11, m20, m02)
   * @return For 3D images: vector(m000, m100, m010, m001, m110, m101, m011,
   * m200, m020, m002)
   *
   * @see blobsMoments() call if you want to evaluate moments for each
   * blob.
   *
   * @see <a href="http://en.wikipedia.org/wiki/Image_moment">Image moment on
   * Wikipedia</a>
   *
   * @par Inertia matrix can be evaluated :
   *
   *  @arg For @b 3D images :
   *    @f[
   *    M =
   *      \begin{bmatrix}
   *        m020 + m002  & -m110 & -m101 \\
   *       -m110 & m200 + m002   & -m011 \\
   *       -m101 & -m011 &  m200 + m020
   *      \end{bmatrix}
   *    @f]
   *    @arg For @b 2D images :
   *    @f[
   *    M =
   *      \begin{bmatrix}
   *        m20 & -m11 \\
   *       -m11 &  m02
   *      \end{bmatrix}
   *    @f]
   *
   */
  template <class T>
  Vector_double measMoments(Image<T> &im, const bool onlyNonZero = true,
                            const bool centered = false)
  {
    Vector_double m;

    measMomentsFunc<T> func;
    m = func(im, onlyNonZero);

    if (centered)
      m = centerMoments(m);

    return m;
  }

  //
  //    ####    ####   #    #    ##    #####   #    ##    #    #   ####   ######
  //   #    #  #    #  #    #   #  #   #    #  #   #  #   ##   #  #    #  #
  //   #       #    #  #    #  #    #  #    #  #  #    #  # #  #  #       #####
  //   #       #    #  #    #  ######  #####   #  ######  #  # #  #       #
  //   #    #  #    #   #  #   #    #  #   #   #  #    #  #   ##  #    #  #
  //    ####    ####     ##    #    #  #    #  #  #    #  #    #   ####   ######
  //
  template <class T>
  inline vector<double>
  genericCovariance(const Image<T> &imIn1, const Image<T> &imIn2, size_t dx,
                    size_t dy, size_t dz, size_t maxSteps = 0,
                    bool normalize = false)
  {
    vector<double> vec;
    ASSERT(areAllocated(&imIn1, &imIn2, NULL), vec);
    ASSERT(haveSameSize(&imIn1, &imIn2, NULL),
           "Input images must have the same size", vec);
    ASSERT((dx + dy + dz > 0),
           "dx, dy and dz can't be all zero at the same time", vec);

    size_t s[3];
    imIn1.getSize(s);

    size_t maxH = max(max(s[0], s[1]), s[2]);
    if (dx > 0)
      maxH = min(maxH, s[0]);
    if (dy > 0)
      maxH = min(maxH, s[1]);
    if (dz > 0)
      maxH = min(maxH, s[2]);
    if (maxH > 0)
      maxH--;

    if (maxSteps == 0)
      maxSteps = maxH;

    maxSteps = min(maxSteps, maxH);
    if (maxSteps == 0) {
      ERR_MSG("Too small");
      return vec;
    }

    vec.clear();

    typename ImDtTypes<T>::volType slicesIn1 = imIn1.getSlices();
    typename ImDtTypes<T>::volType slicesIn2 = imIn2.getSlices();
    typename ImDtTypes<T>::sliceType curSliceIn1;
    typename ImDtTypes<T>::sliceType curSliceIn2;
    typename ImDtTypes<T>::lineType lineIn1;
    typename ImDtTypes<T>::lineType lineIn2;
    typename ImDtTypes<T>::lineType bufLine = ImDtTypes<T>::createLine(s[0]);

    for (size_t len = 0; len <= maxSteps; len++) {
      double prod = 0;

      size_t mdx = min(dx * len, s[0] - 1);
      size_t mdy = min(dy * len, s[1] - 1);
      size_t mdz = min(dz * len, s[2] - 1);

      size_t xLen = s[0] - mdx;
      size_t yLen = s[1] - mdy;
      size_t zLen = s[2] - mdz;

      for (size_t z = 0; z < zLen; z++) {
        curSliceIn1 = slicesIn1[z];
        curSliceIn2 = slicesIn2[z + mdz];
        for (size_t y = 0; y < yLen; y++) {
          lineIn1 = curSliceIn1[y];
          lineIn2 = curSliceIn2[y + mdy];
          copyLine<T>(lineIn2 + mdx, xLen, bufLine);
          // Vectorized loop
          for (size_t x = 0; x < xLen; x++) {
            prod += lineIn1[x] * bufLine[x];
          }
        }
      }
      if (xLen * yLen * zLen != 0)
        prod /= (xLen * yLen * zLen);
      vec.push_back(prod);
    }

    if (normalize) {
      double orig = vec[0];
      for (vector<double>::iterator it = vec.begin(); it != vec.end(); it++)
        *it /= orig;
    }

    ImDtTypes<T>::deleteLine(bufLine);

    return vec;
  }
#if 0
  /*
   * measCovariance() - Covariance of two images in the direction defined by
   * @b dx, @b dy and @b dz.
   *
   * The direction is given by @b dx, @b dy and @b dz.
   *
   * The lenght corresponds to the max number of steps @b maxSteps. When @b 0,
   * the length is limited by the dimensions of the image.
   *
   * @f[
   *    vec[h] = \sum_{p \:\in\: imIn1} \frac{imIn1(p) \;.\; imIn2(p + h)}{N_p}
   * @f]
   *
   * @f[
   *    vec[h] = \sum_{p \:\in\: imIn1} \frac{(imIn1(p) - meanVal(imIn1))
   *                     \;.\; (imIn2(p + h) - meanVal(imIn2))}{N_p}
   * @f]
   * where @b h are displacements in the direction defined by @b dx, @b dy and
   * @b dz.
   *
   * @f$N_p@f$ is the number of pixels used in each term of the sum, which may
   * different for each term in the sum.
   *
   * @param[in] imIn1, imIn2 : Input Images
   * @param[in] dx, dy, dz : direction
   * @param[in] maxSteps : number maximum of displacements to evaluate
   * @param[in] normalize : normalize result with respect to @b vec[0]
   * @return vec[h]
   */
  template <class T>
  vector<double> measCovariance(const Image<T> &imIn1, const Image<T> &imIn2,
                                size_t dx, size_t dy, size_t dz,
                                size_t maxSteps = 0, bool normalize = false)
  {
    vector<double> vec;
    ASSERT(areAllocated(&imIn1, &imIn2, NULL), vec);
    ASSERT(haveSameSize(&imIn1, &imIn2, NULL),
           "Input images must have the same size", vec);
    ASSERT((dx + dy + dz > 0),
           "dx, dy and dz can't be all zero at the same time", vec);

    size_t s[3];
    imIn1.getSize(s);

    size_t maxH = max(max(s[0], s[1]), s[2]);
    if (dx > 0)
      maxH = min(maxH, s[0]);
    if (dy > 0)
      maxH = min(maxH, s[1]);
    if (dz > 0)
      maxH = min(maxH, s[2]);
    if (maxH > 0)
      maxH--;

    if (maxSteps == 0)
      maxSteps = maxH;

    maxSteps = min(maxSteps, maxH);
    if (maxSteps == 0) {
      ERR_MSG("Too small");
      return vec;
    }

    vec.clear();

    typename ImDtTypes<T>::volType slicesIn1 = imIn1.getSlices();
    typename ImDtTypes<T>::volType slicesIn2 = imIn2.getSlices();
    typename ImDtTypes<T>::sliceType curSliceIn1;
    typename ImDtTypes<T>::sliceType curSliceIn2;
    typename ImDtTypes<T>::lineType lineIn1;
    typename ImDtTypes<T>::lineType lineIn2;
    typename ImDtTypes<T>::lineType bufLine = ImDtTypes<T>::createLine(s[0]);

    for (size_t len = 0; len <= maxSteps; len++) {
      double prod = 0;

      size_t mdx = min(dx * len, s[0] - 1);
      size_t mdy = min(dy * len, s[1] - 1);
      size_t mdz = min(dz * len, s[2] - 1);

      size_t xLen = s[0] - mdx;
      size_t yLen = s[1] - mdy;
      size_t zLen = s[2] - mdz;

      for (size_t z = 0; z < zLen; z++) {
        curSliceIn1 = slicesIn1[z];
        curSliceIn2 = slicesIn2[z + mdz];
        for (size_t y = 0; y < yLen; y++) {
          lineIn1 = curSliceIn1[y];
          lineIn2 = curSliceIn2[y + mdy];
          copyLine<T>(lineIn2 + mdx, xLen, bufLine);
          // Vectorized loop
          for (size_t x = 0; x < xLen; x++) {
            prod += lineIn1[x] * bufLine[x];
          }
        }
      }
      if (xLen * yLen * zLen != 0)
        prod /= (xLen * yLen * zLen);
      vec.push_back(prod);
    }

    if (normalize) {
      double orig = vec[0];
      for (vector<double>::iterator it = vec.begin(); it != vec.end(); it++)
        *it /= orig;
    }

    ImDtTypes<T>::deleteLine(bufLine);

    return vec;
  }
#endif

  /**
   * measCovariance() - Centered covariance of two images in the
   * direction defined by @b dx, @b dy and @b dz.
   *

   * The direction is given by @b dx, @b dy and @b dz.
   *
   * The lenght corresponds to the max number of steps @b maxSteps. When @b 0,
   * the length is limited by the dimensions of the image.
   *
   * @f[
   *    vec[h] = \sum_{p \:\in\: imIn1} \frac{imIn1(p) \;.\; imIn2(p + h)}{N_p}
   * @f]
   *
   * where @b h are displacements in the direction defined by @b dx, @b dy and
   * @b dz.
   *
   * @f$N_p@f$ is the number of pixels used in each term of the sum, which may
   * different for each term in the sum.
   *
   * @param[in] imIn1, imIn2 : Input Images
   * @param[in] dx, dy, dz : direction
   * @param[in] maxSteps : number maximum of displacements to evaluate
   * @param[in] centered : if this parameter is set to @b true, the mean value
   *  (meanVal()) will be subtracted from each input image
   * @param[in] normalize : normalize result with respect to @b vec[0]

   * @return vec[h]
   *
   */
  template <class T>
  vector<double> measCovariance(const Image<T> &imIn1, const Image<T> &imIn2,
                                size_t dx, size_t dy, size_t dz,
                                size_t maxSteps = 0, bool centered = false,
                                bool normalize = false)
  {
    if (centered) {
      Image<float> imMean1(imIn1, true);
      float meanV1 = meanVal(imMean1)[0];
      sub(imMean1, meanV1, imMean1);

      Image<float> imMean2(imIn2, true);
      float meanV2 = meanVal(imMean2)[0];
      sub(imMean2, meanV2, imMean2);

      return genericCovariance(imMean1, imMean2, dx, dy, dz, maxSteps,
                               normalize);
    } else {
      return genericCovariance(imIn1, imIn2, dx, dy, dz, maxSteps, normalize);
    }
  }

  /**
   * measAutoCovariance() - Auto-covariance
   *
   * The direction is given by @b dx, @b dy and @b dz.
   * The lenght corresponds to the max number of steps @b maxSteps
   *
   * @param[in] imIn : Input Image
   * @param[in] dx, dy, dz : direction
   * @param[in] maxSteps : number maximum of displacements to evaluate
   * @param[in] centered : if this parameter is set to @b true, the mean value
   *  (meanVal()) will be subtracted from the input image
   * @param[in] normalize : normalize result with respect to @b vec[0]
   * @return vec[h]
   */
  template <class T>
  vector<double> measAutoCovariance(const Image<T> &imIn, size_t dx, size_t dy,
                                    size_t dz, size_t maxSteps = 0,
                                    bool centered  = false,
                                    bool normalize = false)
  {
    if (centered) {
      Image<float> imMean(imIn, true);
      float meanV = meanVal(imMean)[0];
      sub(imMean, meanV, imMean);
      return genericCovariance(imMean, imMean, dx, dy, dz, maxSteps, normalize);
    } else {
      return genericCovariance(imIn, imIn, dx, dy, dz, maxSteps, normalize);
    }
  }

  //
  //   ######  #    #   #####  #####    ####   #####    #   #
  //   #       ##   #     #    #    #  #    #  #    #    # #
  //   #####   # #  #     #    #    #  #    #  #    #     #
  //   #       #  # #     #    #####   #    #  #####      #
  //   #       #   ##     #    #   #   #    #  #          #
  //   ######  #    #     #    #    #   ####   #          #
  //
  /** @cond */
  template <class T>
  struct measEntropyFunc : public MeasureFunctionBase<T, double> {
    typedef typename Image<T>::lineType lineType;

    map<T, UINT> histo;

    virtual void initialize(const Image<T> & /*imIn*/)
    {
      histo.clear();
    }

    virtual void processSequence(lineType lineIn, size_t size)
    {
      for (size_t i = 0; i < size; i++) {
        T val = lineIn[i];

        UINT nb    = histo[val];
        histo[val] = ++nb;
      }
    }

    virtual void finalize(const Image<T> & /*imIn*/)
    {
      double entropy = 0.;
      double sumP    = 0.;
      double sumN    = 0.;

      typename map<T, UINT>::iterator it;
      for (it = histo.begin(); it != histo.end(); it++) {
        if (it->second > 0) {
          sumN += it->second;
          sumP += it->second * log2(it->second);
        }
      }
      if (sumN > 0)
        entropy = log2(sumN) - sumP / sumN;

      this->retVal = entropy;
    }
  }; // END measEntropyFunc

  /** @endcond */

  /**
   * measEntropy() - Image entropy
   *
   * @details Evaluate Shannon entropy of the image (in bits)
   *
   * @param[in] imIn : input image
   * @return image entropy
   *
   * @see blobsEntropy() to compute entropy inside each label.
   */
  template <class T> double measEntropy(const Image<T> &imIn)
  {
    ASSERT_ALLOCATED(&imIn);

#if 0
    double entropy = 0.;
    double sumP    = 0.;
    double sumN    = 0.;

    map<T, UINT> hist = histogram(imIn, false);
    typename map<T, UINT>::iterator it;
    for (it = hist.begin(); it != hist.end(); it++) {
      if (it->second > 0) {
        sumP += it->second * log2(it->second);
        sumN += it->second;
      }
    }

    if (sumN > 0)
      entropy = log2(sumN) - sumP / sumN;

    return entropy;
#else
    measEntropyFunc<T> func;
    return func(imIn, false);
#endif
  }

  /**
   * measEntropy() - Image entropy
   *
   * Evaluate Shannon entropy of the image in a region defined by a mask.
   *
   * @param[in] imIn : input image
   * @param[in] imMask : mask defining where the entropy shall be evaluated
   * @return image entropy
   *
   * @see blobsEntropy() to compute entropy inside each label.
   *
   */
  template <class T>
  double measEntropy(const Image<T> &imIn, const Image<T> &imMask)
  {
    ASSERT_ALLOCATED(&imIn, &imMask);
    ASSERT_SAME_SIZE(&imIn, &imMask);

    double entropy = 0.;
    double sumP    = 0.;
    double sumN    = 0.;

    map<T, UINT> hist = histogram(imIn, imMask, false);
    typename map<T, UINT>::iterator it;
    for (it = hist.begin(); it != hist.end(); it++) {
      if (it->second > 0) {
        sumP += it->second * log2(it->second);
        sumN += it->second;
      }
    }

    if (sumN > 0)
      entropy = log2(sumN) - sumP / sumN;

    return entropy;
  }

  //
  //    ####    #####  #    #  ######  #####    ####
  //   #    #     #    #    #  #       #    #  #
  //   #    #     #    ######  #####   #    #   ####
  //   #    #     #    #    #  #       #####        #
  //   #    #     #    #    #  #       #   #   #    #
  //    ####      #    #    #  ######  #    #   ####
  //
  /**
   * nonZeroOffsets() - Returns the offsets of pixels having non nul values.
   *
   * @param[in] imIn : input image
   * @return a vector containing the offset of all non-zero points in image.
   * @warning In huge images this can return a very big vector, in the same
   * order than the image size.
   */
  template <class T> Vector_size_t nonZeroOffsets(Image<T> &imIn)
  {
    Vector_size_t offsets;

    ASSERT(CHECK_ALLOCATED(&imIn), RES_ERR_BAD_ALLOCATION, offsets);

    typename Image<T>::lineType pixels = imIn.getPixels();

    for (size_t i = 0; i < imIn.getPixelCount(); i++)
      if (pixels[i] != 0)
        offsets.push_back(i);

    return offsets;
  }

  /**
   * isBinary() - Test if an image is binary.
   *
   * @param[in] imIn : image
   * @return @b true if the only pixel values are @b 0 and any other positive
   * value.
   */
  template <class T> bool isBinary(const Image<T> &imIn)
  {
    CHECK_ALLOCATED(&imIn);

    map<T, size_t> h;
    typename Image<T>::lineType pixels = imIn.getPixels();

    for (size_t i = 0; i < imIn.getPixelCount(); i++) {
      h[pixels[i]]++;
      if (h.size() > 2)
        return false;
    }

    return h[0] > 0 && h.size() == 2;
  }

  /** @}*/

} // namespace smil

#endif // _D_MEASURES_HPP
