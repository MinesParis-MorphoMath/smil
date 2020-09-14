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

#ifndef _D_IMAGE_TRANSFORM_HPP
#define _D_IMAGE_TRANSFORM_HPP

#include "DBaseImageOperations.hpp"
#include "DLineArith.hpp"

namespace smil
{
  /**
   * @ingroup Base
   * @defgroup Transform Image Transformations
   * 
   */

  /**
    * @defgroup TransformCut Cut and paste
    * @ingroup Transform
    *
    * @addtogroup TransformCut
    *
    * @{
    */

  /* 
   *  ####    ####   #####    #   #
   * #    #  #    #  #    #    # #
   * #       #    #  #    #     #
   * #       #    #  #####      #
   * #    #  #    #  #          #
   *  ####    ####   #          #
   */

  /**
   * copy() - Copy image (or a zone) into an output image
   *
   * This is the most complete version of @b smil::copy().
   *
   * It copies a region of @b imIn
   * defined by its start point <b>(startX, startY, startZ)</b> and size
   * <b>(sizeX, sizeY, sizeZ)</b> into a region on @b imOut beginning at
   * position <b>(outStartX, outStartY, outStartZ)</b>.
   *
   * @param[in] imIn : input image
   * @param[in] startX, startY, [startZ] : (optional) start position of the zone
   * in the input image
   * @param[in] sizeX,  sizeY,  [sizeZ] : (optional) size of the zone in the
   * input image
   * @param[out] imOut : output image
   * @param[in] outStartX, outStartY, [outStartZ] : (optional) position to copy
   * the selected zone in the output image (default is the origin (0,0,0))
   *
   * @smilexample{copy_crop.py}
   */
  template <class T1, class T2>
  RES_T copy(const Image<T1> &imIn, size_t startX, size_t startY, size_t startZ,
             size_t sizeX, size_t sizeY, size_t sizeZ, Image<T2> &imOut,
             size_t outStartX = 0, size_t outStartY = 0, size_t outStartZ = 0)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);

    size_t inW = imIn.getWidth();
    size_t inH = imIn.getHeight();
    size_t inD = imIn.getDepth();

    size_t outW = imOut.getWidth();
    size_t outH = imOut.getHeight();
    size_t outD = imOut.getDepth();

    ASSERT(startX < inW && startY < inH && startZ < inD);
    ASSERT(outStartX < outW && outStartY < outH && outStartZ < outD);

    size_t realSx = min(min(sizeX, inW - startX), outW - outStartX);
    size_t realSy = min(min(sizeY, inH - startY), outH - outStartY);
    size_t realSz = min(min(sizeZ, inD - startZ), outD - outStartZ);

    typename Image<T1>::volType slIn  = imIn.getSlices() + startZ;
    typename Image<T2>::volType slOut = imOut.getSlices() + outStartZ;

    size_t y;

    for (size_t z = 0; z < realSz; z++) {
      typename Image<T1>::sliceType lnIn  = *slIn + startY;
      typename Image<T2>::sliceType lnOut = *slOut + outStartY;

#ifdef USE_OPEN_MP
      int nthreads = Core::getInstance()->getNumberOfThreads();
#pragma omp parallel private(y)
#endif // USE_OPEN_MP
      {
#ifdef USE_OPEN_MP
#pragma omp for schedule(dynamic, nthreads) nowait
#endif // USE_OPEN_MP
        for (y = 0; y < realSy; y++)
          copyLine<T1, T2>(lnIn[y] + startX, realSx, lnOut[y] + outStartX);
      }

      slIn++;
      slOut++;
    }

    imOut.modified();
    return RES_OK;
  }

  /**
   * copy() - Copy Image
   *
   * @b 2D version of the general @b smil::copy function.
   *
   * @overload
   */
  template <class T1, class T2>
  RES_T copy(const Image<T1> &imIn, size_t startX, size_t startY, size_t sizeX,
             size_t sizeY, Image<T2> &imOut, size_t outStartX = 0,
             size_t outStartY = 0, size_t outStartZ = 0)
  {
    return copy(imIn, startX, startY, 0, sizeX, sizeY, 1, imOut, outStartX,
                outStartY, outStartZ);
  }

  /**
   * copy() - Copy Image
   *
   * Copies the entire content of @b imIn beginning at <b>(startX, startY,
   * startZ)</b> into @b imOut beginning at <b>(outStartX, outStartY,
   * outStartZ)</b>.
   *
   * @overload
   */
  template <class T1, class T2>
  RES_T copy(const Image<T1> &imIn, size_t startX, size_t startY, size_t startZ,
             Image<T2> &imOut, size_t outStartX = 0, size_t outStartY = 0,
             size_t outStartZ = 0)
  {
    return copy(imIn, startX, startY, startZ, imIn.getWidth(), imIn.getHeight(),
                imIn.getDepth(), imOut, outStartX, outStartY, outStartZ);
  }

  /**
   * copy() - Copy Image (2D overload)
   *
   * Copies the entire content of @b imIn beginning at <b>(startX, startY)</b>
   * into @b imOut beginning at <b>(outStartX, outStartY)</b>.
   *
   * @overload
   */
  template <class T1, class T2>
  RES_T copy(const Image<T1> &imIn, size_t startX, size_t startY,
             Image<T2> &imOut, size_t outStartX = 0, size_t outStartY = 0,
             size_t outStartZ = 0)
  {
    return copy(imIn, startX, startY, 0, imIn.getWidth(), imIn.getHeight(), 1,
                imOut, outStartX, outStartY, outStartZ);
  }

  /**
   * copy() - Copy Image
   *
   * Copies the entire content of @b imIn into @b imOut beginning at
   <b>(outStartX, outStartY, outStartZ)</b>

   * @overload
   */
  template <class T1, class T2>
  RES_T copy(const Image<T1> &imIn, Image<T2> &imOut, size_t outStartX,
             size_t outStartY, size_t outStartZ = 0)
  {
    return copy(imIn, 0, 0, 0, imIn.getWidth(), imIn.getHeight(),
                imIn.getDepth(), imOut, outStartX, outStartY, outStartZ);
  }

  // Copy/cast two images with different types but same size (quick way)
  /**
   * copy() - Copy / cast two images, convert their types.
   *
   * If @b imIn and @b imOut types may be different, the contents of @b imIn
   * beginning at <b> (0,0,0)</b> will copied to @b imOut, at the same place but
   * the range of @b imOut won't be adjusted.
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   *
   */
  template <class T1, class T2>
  RES_T copy(const Image<T1> &imIn, Image<T2> &imOut)
  {
    // Swig is (surprisingly;)) lost with overloads of template functions, so we
    // try to reorient him
    if (typeid(imIn) == typeid(imOut))
      return copy(imIn, imOut);

    ASSERT_ALLOCATED(&imIn, &imOut);

    if (!haveSameSize(&imIn, &imOut, NULL))
      return copy<T1, T2>(imIn, 0, 0, 0, imOut, 0, 0, 0);

    copyLine<T1, T2>(imIn.getPixels(), imIn.getPixelCount(), imOut.getPixels());

    imOut.modified();
    return RES_OK;
  }

  /**
   * copy() - Copy Image
   *
   * Copy one image to the other. @b imOut shall be of the same kind of @b imIn.
   * Its size will be set to the same of @b imIn
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   */
  template <class T>
  RES_T copy(const Image<T> &imIn, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);

    imOut.setSize(imIn);

    return unaryImageFunction<T, fillLine<T>>(imIn, imOut).retVal;
  }

  /**
   * clone() - Clone an image
   *
   * Make @b imOut a clone @b imIn, with the same size and content.
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   *
   * @note
   * @b imOut must be previously declared as an image with the same data type of
   * @b imIn. Its initial size doesn't matter. It will be set to the same size
   * of @b imIn.
   */
  template <class T>
  RES_T clone(const Image<T> &imIn, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn);

    ASSERT((imOut.setSize(imIn) == RES_OK));
    return copy<T>(imIn, imOut);
  }
    
  /*
   *  ####   #####    ####   #####
   * #    #  #    #  #    #  #    #
   * #       #    #  #    #  #    #
   * #       #####   #    #  #####
   * #    #  #   #   #    #  #
   *  ####   #    #   ####   #
   */
  /**
   * crop() - Crop image
   *
   * Crop an image into an output image
   * @param[in] imIn : input image
   * @param[in] startX, startY, startZ : start position of the zone in the
   * input image
   * @param[in] sizeX, sizeY, sizeZ : size of the zone in the input image
   * @param[out] imOut : output image
   *
   * @see
   * Use copy() if you want to place the cropped image into a particular place
   * of the output image
   *
   * @smilexample{copy_crop.py}
   */
  template <class T>
  RES_T crop(const Image<T> &imIn, size_t startX, size_t startY, size_t startZ,
             size_t sizeX, size_t sizeY, size_t sizeZ, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn);

    size_t inW = imIn.getWidth();
    size_t inH = imIn.getHeight();
    size_t inD = imIn.getDepth();

    size_t realSx = min(sizeX, inW - startX);
    size_t realSy = min(sizeY, inH - startY);
    size_t realSz = min(sizeZ, inD - startZ);

    imOut.setSize(realSx, realSy, realSz);
    return copy(imIn, startX, startY, startZ, realSx, realSy, realSz, imOut, 0,
                0, 0);
  }

  /**
   * crop() - Crop image
   *
   * Crop an image in the same image.
   *
   * @param[in,out] imInOut : input image
   * @param[in] startX, startY, startZ : start position of the zone in the 
   * input image
   * @param[in] sizeX, sizeY, sizeZ : size of the zone in the input image
   *
   */
  template <class T>
  RES_T crop(Image<T> &imInOut, size_t startX, size_t startY, size_t startZ,
             size_t sizeX, size_t sizeY, size_t sizeZ)
  {
    Image<T> tmpIm(imInOut, true); // clone
    return crop(tmpIm, startX, startY, startZ, sizeX, sizeY, sizeZ, imInOut);
  }

  // 2D overload
  /**
   * crop() - @b 2D Crop image
   *
   * Crop an image (this is just an overload)
   *
   * @param[in] imIn : input image
   * @param[in] startX, startY : start position of the zone in the input image
   * @param[in] sizeX, sizeY : size of the zone in the input image
   * @param[out] imOut : output image
   *
   * @overload
   */
  template <class T>
  RES_T crop(const Image<T> &imIn, size_t startX, size_t startY, size_t sizeX,
             size_t sizeY, Image<T> &imOut)
  {
    return crop(imIn, startX, startY, 0, sizeX, sizeY, 1, imOut);
  }

  /**
   * crop() - @b 2D Crop image
   *
   * Crop an image in itself (this is just an overload)
   *
   * @param[in,out] imInOut : input image
   * @param[in] startX, startY : start position of the zone in the input
   * image
   * @param[in] sizeX, sizeY : size of the zone in the input image
   *
   * @overload
   */
  template <class T>
  RES_T crop(Image<T> &imInOut, size_t startX, size_t startY, size_t sizeX,
             size_t sizeY)
  {
    return crop(imInOut, startX, startY, 0, sizeX, sizeY, 1);
  }

  /*
   *   ##    #####   #####       #####    ####   #####   #####   ######  #####
   *  #  #   #    #  #    #      #    #  #    #  #    #  #    #  #       #    #
   * #    #  #    #  #    #      #####   #    #  #    #  #    #  #####   #    #
   * ######  #    #  #    #      #    #  #    #  #####   #    #  #       #####
   * #    #  #    #  #    #      #    #  #    #  #   #   #    #  #       #   #
   * #    #  #####   #####       #####    ####   #    #  #####   ######  #    #
   */
  /**
   * addBorder() - Add a border of size @b bSize around the original image
   *
   * @param[in] imIn : input image
   * @param[in] bSize : border size
   * @param[out] imOut : output image
   * @param[in] borderValue : value to assign to each pixel in the border
   *
   * @note
   *  Image size is increased by @txtbold{2 * bSize} pixels in each direction
   *
   */
  template <class T>
  RES_T addBorder(const Image<T> &imIn, const size_t &bSize, Image<T> &imOut,
                  const T &borderValue = ImDtTypes<T>::max())
  {
    ASSERT_ALLOCATED(&imIn)

    if (&imIn == &imOut) {
      Image<T> tmpIm(imIn, true); // clone
      return addBorder(tmpIm, bSize, imOut, borderValue);
    }

    ImageFreezer freeze(imOut);

    size_t s[3];
    imIn.getSize(s);

    if (imIn.getDimension() == 3) {
      imOut.setSize(s[0] + 2 * bSize, s[1] + 2 * bSize, s[2] + 2 * bSize);
      ASSERT_ALLOCATED(&imOut)
      fill(imOut, borderValue);
      copy(imIn, 0, 0, 0, s[0], s[1], s[2], imOut, bSize, bSize, bSize);
    } else {
      imOut.setSize(s[0] + 2 * bSize, s[1] + 2 * bSize, 1);
      ASSERT_ALLOCATED(&imOut)
      fill(imOut, borderValue);
      copy(imIn, 0, 0, s[0], s[1], imOut, bSize, bSize);
    }
    return RES_OK;
  }
  /** @} */

  /**
    * @defgroup TransformView Image View
    * @ingroup Transform
    *
    * @addtogroup TransformView
    *
    * @{
    */
  /*
   * ######  #          #    #####
   * #       #          #    #    #
   * #####   #          #    #    #
   * #       #          #    #####
   * #       #          #    #
   * #       ######     #    #
   */
  /**
   * @cond
   * FlipClassFunc
   */
  template <class T> class FlipClassFunc
  {
  private:
    void copyReverse(T *in, size_t width, T *out)
    {
      for (size_t i = 0; i < width; i++)
        out[i] = in[width - 1 - i];
    }

  public:
    RES_T flipIt(Image<T> &imIn, Image<T> &imOut, string direction)
    {
      ASSERT_ALLOCATED(&imIn, &imOut);
      ASSERT_SAME_SIZE(&imIn, &imOut);

      if (&imIn == &imOut) {
        Image<T> imTmp = Image<T>(imIn, true);
        return flipIt(imTmp, imOut, direction);
      }

      bool vflip = (direction == "vertical");

      typename Image<T>::sliceType *slicesIn  = imIn.getSlices();
      typename Image<T>::sliceType *slicesOut = imOut.getSlices();

      size_t width  = imIn.getWidth();
      size_t height = imIn.getHeight();
      size_t depth  = imIn.getDepth();

      ImageFreezer freeze(imOut);
#ifdef USE_OPEN_MP
      int nthreads = Core::getInstance()->getNumberOfThreads();
#pragma omp parallel for num_threads(nthreads)
#endif // USE_OPEN_MP
      for (size_t k = 0; k < depth; k++) {
        typename Image<T>::sliceType linesIn;
        typename Image<T>::sliceType linesOut;
        linesIn  = slicesIn[k];
        linesOut = slicesOut[k];

        if (vflip)
          for (size_t j = 0; j < height; j++)
            copyLine<T>(linesIn[j], width, linesOut[height - 1 - j]);
        else
          for (size_t j = 0; j < height; j++)
            copyReverse(linesIn[j], width, linesOut[j]);
      }

      return RES_OK;
    }
  };
  /** @endcond */

  /**
   * vertFlip() : Vertical Flip
   *
   * Mirror an image using an horizontal line (or plan for 3D images) in the
   * center of the image. In 3D images, each slice is flipped vertically.
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   */
  template <class T> RES_T vertFlip(Image<T> &imIn, Image<T> &imOut)
  {
    string direction = "vertical";
    FlipClassFunc<T> flip;
    return flip.flipIt(imIn, imOut, "vertical");
  }

  /**
   * vertFlip() : Vertical Flip
   *
   * @param[in,out]  im : input/output Image
   *
   * @overload
   */
  template <class T> RES_T vertFlip(Image<T> &im)
  {
    return vertFlip(im, im);
  }

  /**
   * horizFlip() : Horizontal Flip
   *
   * Mirror an image using a vertical line (or plan for 3D images) in the
   * center of the image. In 3D images, each slice is flipped horizontally.
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   */
  template <class T> RES_T horizFlip(Image<T> &imIn, Image<T> &imOut)
  {
    string direction = "horizontal";
    FlipClassFunc<T> flip;
    return flip.flipIt(imIn, imOut, direction);
  }

  /**
   * horizFlip() : Horizontal Flip
   *
   * @param[in,out]  im : input/output Image
   *
   * @overload
   */
  template <class T> RES_T horizFlip(Image<T> &im)
  {
    return horizFlip(im, im);
  }

  /*
   * #####    ####    #####    ##     #####  ######
   * #    #  #    #     #     #  #      #    #
   * #    #  #    #     #    #    #     #    #####
   * #####   #    #     #    ######     #    #
   * #   #   #    #     #    #    #     #    #
   * #    #   ####      #    #    #     #    ######
   */
  /** @cond */
  template <class T> class ImageRotateFunct
  {
  public:
    ImageRotateFunct()
    {
    }

    ~ImageRotateFunct()
    {
    }

    RES_T Rotate(Image<T> &imIn, int count, Image<T> &imOut)
    {
      count = count % 4;

      return RotateX90(imIn, count, imOut);
    }

  private:
    RES_T RotateX90(Image<T> &imIn, int count, Image<T> &imOut)
    {
      ASSERT_ALLOCATED(&imIn, &imOut);

      count     = count % 4;
      int angle = count * 90;

      if (angle == 0) {
        ImageFreezer freeze(imOut);
        return copy(imIn, imOut);
      }

      off_t w = imIn.getWidth();
      off_t h = imIn.getHeight();
      off_t d = imIn.getDepth();

      /* 90 and 270 degres */
      if (angle == 90 || angle == 270) {
        imOut.setSize(h, w, d);
      } else {
        imOut.setSize(w, h, d);
      }

      ImageFreezer freeze(imOut);

      typedef typename ImDtTypes<T>::lineType lineType;
      lineType pixIn  = imIn.getPixels();
      lineType pixOut = imOut.getPixels();

      switch (angle) {
      case 90:
#ifdef USE_OPEN_MP
#pragma omp parallel for
#endif // USE_OPEN_MP
        for (off_t k = 0; k < d; k++) {
          off_t offset = k * w * h;
          T *sIn       = (T *) (pixIn + offset);
          T *sOut      = (T *) (pixOut + offset);
          for (off_t j = 0; j < h; j++) {
            for (off_t i = 0; i < w; i++) {
              sOut[i * h + (w - 1 - j)] = sIn[j * w + i];
            }
          }
        }
        break;
      case 180:
#ifdef USE_OPEN_MP
#pragma omp parallel for
#endif // USE_OPEN_MP
        for (off_t k = 0; k < d; k++) {
          off_t offset = k * w * h;
          T *sIn       = (T *) (pixIn + offset);
          T *sOut      = (T *) (pixOut + offset);
          for (off_t j = 0; j < h; j++) {
            for (off_t i = 0; i < w; i++) {
              sOut[(h - 1 - j) * w + (w - 1 - i)] = sIn[j * w + i];
            }
          }
        }
        break;
      case 270:
#ifdef USE_OPEN_MP
#pragma omp parallel for
#endif // USE_OPEN_MP
        for (off_t k = 0; k < d; k++) {
          off_t offset = k * w * h;
          T *sIn       = (T *) (pixIn + offset);
          T *sOut      = (T *) (pixOut + offset);
          for (off_t j = 0; j < h; j++) {
            for (off_t i = 0; i < w; i++) {
              sOut[(w - 1 - i) * h + j] = sIn[j * w + i];
            }
          }
        }
        break;
      default:
        break;
      }

      imOut.modified();
      return RES_OK;
    }
  };
  /** @endcond */

  /**
   * rotateX90() - Rotate an image by an angle multiple of 90 degres
   *
   * @param[in]  imIn : input Image
   * @param[in]  count : number of 90 degres steps to rotate
   * @param[out] imOut : output Image
   *
   * @note
   * - If @b count equals @b 0, just copy the input image into the output image.
   * @note
   * - When calling this function on @b 3D images, rotation is done around @b z
   * axis.
   * - Rotation around @b x or @b y axis is possible thanks to matTranspose()
   * call. See the example below on how to rotate around @b y axis.
   *
   * @smilexample{Rotation around axis y, example-3D-image-rotate.py}
   */
  template <class T> RES_T rotateX90(Image<T> &imIn, int count, Image<T> &imOut)
  {
    ImageRotateFunct<T> imr;

    if (&imIn == &imOut) {
      Image<T> imTmp(imIn, true);
      return imr.Rotate(imTmp, count, imOut);
    }

    return imr.Rotate(imIn, count, imOut);
  }

  /**
   * rotateX90() - Rotate an image by an angle multiple of 90 degres
   *
   * @param[in,out]  im : input/output Image
   * @param[in]  count : number of 90 degres steps to rotate
   *
   * @overload
   */
  template <class T> RES_T rotateX90(Image<T> &im, int count)
  {
    return rotateX90(im, count, im);
  }

  /*
   * #####  #####     ##    #    #   ####   #         ##     #####  ######
   *   #    #    #   #  #   ##   #  #       #        #  #      #    #
   *   #    #    #  #    #  # #  #   ####   #       #    #     #    #####
   *   #    #####   ######  #  # #       #  #       ######     #    #
   *   #    #   #   #    #  #   ##  #    #  #       #    #     #    #
   *   #    #    #  #    #  #    #   ####   ######  #    #     #    ######
   */
  /**
   * translate() - %Image translation.
   *
   * @param[in] imIn : input image
   * @param[in] dx, dy, dz : shift to be applied
   * @param[out] imOut : output image
   * @param[in] borderValue : value to be assigned to moved pixels
   */
  template <class T>
  RES_T translate(const Image<T> &imIn, int dx, int dy, int dz, Image<T> &imOut,
                  T borderValue = ImDtTypes<T>::min())
  {
    ASSERT_ALLOCATED(&imIn)
    ASSERT_SAME_SIZE(&imIn, &imOut)

    size_t lineLen = imIn.getWidth();
    typename ImDtTypes<T>::lineType borderBuf =
        ImDtTypes<T>::createLine(lineLen);
    fillLine<T>(borderBuf, lineLen, borderValue);

    size_t height = imIn.getHeight();
    size_t depth  = imIn.getDepth();

    for (size_t k = 0; k < depth; k++) {
      typename Image<T>::sliceType lOut = imOut.getSlices()[k];

      int z = k - dz;
      for (size_t j = 0; j < height; j++, lOut++) {
        int y = j - dy;

        if (z < 0 || z >= (int) depth || y < 0 || y >= (int) height)
          copyLine<T>(borderBuf, lineLen, *lOut);
        else
          shiftLine<T>(imIn.getSlices()[z][y], dx, lineLen, *lOut, borderValue);
      }
    }

    ImDtTypes<T>::deleteLine(borderBuf);

    imOut.modified();

    return RES_OK;
  }

  /**
   * translate() - %Image translation.
   *
   *
   * @param[in] imIn : input image
   * @param[in] dx, dy, dz : shift to be applied
   * @returns translated image
   */
  template <class T>
  ResImage<T> translate(const Image<T> &imIn, int dx, int dy, int dz)
  {
    ResImage<T> imOut(imIn);
    translate<T>(imIn, dx, dy, dz, imOut);
    return imOut;
  }

  /**
   * translate() - @txtbold{2D Image} translation.
   *
   * The same translation is applied to each slice
   *
   * @param[in] imIn : input image
   * @param[in] dx, dy : shift to be applied
   * @param[out] imOut : output image
   * @param[in] borderValue : value to be assigned to moved pixels
   */
  template <class T>
  RES_T translate(const Image<T> &imIn, int dx, int dy, Image<T> &imOut,
                  T borderValue = ImDtTypes<T>::min())
  {
    return translate<T>(imIn, dx, dy, 0, imOut, borderValue);
  }

  /**
   * translate() - @txtbold{2D Image} translation.
   *
   * @param[in] imIn : input image
   * @param[in] dx, dy : shift to be applied
   * @returns translated image
   */
  template <class T> ResImage<T> translate(const Image<T> &imIn, int dx, int dy)
  {
    ResImage<T> imOut(imIn);
    translate<T>(imIn, dx, dy, 0, imOut);
    return imOut;
  }
  
  /** @} */

  /**
    * @defgroup TransformSize Image size
    * @ingroup Transform
    *
    * @addtogroup TransformSize
    *
    * @{
    */
  /*
   * #####   ######   ####      #    ######  ######
   * #    #  #       #          #        #   #
   * #    #  #####    ####      #       #    #####
   * #####   #            #     #      #     #
   * #   #   #       #    #     #     #      #
   * #    #  ######   ####      #    ######  ######
   */
  /** @cond
   *
   */
  template <class T> class ImageResizeFunc
  {
  public:
    ImageResizeFunc(string method)
    {
      if (method != "linear" && method != "trilinear" && method != "closest")
        this->method = "trilinear";
      else
        this->method = method;
    }

    ImageResizeFunc()
    {
      method = "trilinear";
    }

    ~ImageResizeFunc()
    {
    }

    /*
     * Public resize members
     */
    RES_T resize(Image<T> &imIn, size_t width, size_t height, size_t depth,
                 Image<T> &imOut, string algorithm = "trilinear")
    {
      ASSERT_ALLOCATED(&imIn, &imOut)
      if (&imIn == &imOut) {
        Image<T> tmpIm(imIn, true);
        return resize(tmpIm, width, height, depth, imIn);
      }

      size_t dz = imIn.getDepth();
      if (algorithm == "auto") {
        if (isBinary(imIn))
          algorithm = "closest";
        else
          algorithm = dz > 1 ? "trilinear" : "bilinear";
      }

      if (algorithm == "closest") {
        if (dz > 1)
          return resize3DClosest(imIn, width, height, depth, imOut);
        else
          return resize2DClosest(imIn, width, height, imOut);
      }

      if (algorithm == "bilinear" || algorithm == "trilinear") {
        if (dz > 1)
          return resizeTrilinear(imIn, width, height, depth, imOut);
        else
          return resizeBilinear(imIn, width, height, imOut);
      }
      ERR_MSG("* Unknown algorithm " + algorithm);
      return RES_ERR;
    }

    RES_T resize(Image<T> &imIn, size_t width, size_t height, Image<T> &imOut,
                 string algorithm = "trilinear")
    {
      ASSERT_ALLOCATED(&imIn, &imOut)

      size_t depth = imIn.getDepth();
      return resize(imIn, width, height, depth, imOut, algorithm);
    }

    /*
     * Public scale member
     */
    RES_T scale(Image<T> &imIn, double kx, double ky, double kz,
                Image<T> &imOut, string algorithm = "trilinear")
    {
      ASSERT_ALLOCATED(&imIn, &imOut)

      if (&imIn == &imOut) {
        Image<T> tmpIm(imIn, true);
        return scale(tmpIm, kx, ky, kz, imIn);
      }

      size_t width  = imIn.getWidth();
      size_t height = imIn.getHeight();
      size_t depth  = imIn.getDepth();

      if (width > 1)
        width = max(1L, lround(kx * width));
      if (height > 1)
        height = max(1L, lround(ky * height));
      if (depth > 1)
        depth = max(1L, lround(kz * depth));

      return resize(imIn, width, height, depth, imOut, algorithm);
    }

    RES_T scale(Image<T> &imIn, size_t kx, size_t ky, Image<T> &imOut,
                string algorithm = "trilinear")
    {
      return scale(imIn, kx, ky, 1., imOut, algorithm);
    }

  private:
    /*
     *  ####   #        ####    ####   ######   ####    #####
     * #    #  #       #    #  #       #       #          #
     * #       #       #    #   ####   #####    ####      #
     * #       #       #    #       #  #            #     #
     * #    #  #       #    #  #    #  #       #    #     #
     *  ####   ######   ####    ####   ######   ####      #
     */
    /*
     * 2D - closest interpolation algorithm - naive loop implementation
     */
    RES_T resize2DClosest(Image<T> &imIn, size_t sx, size_t sy, Image<T> &imOut)
    {
      size_t width  = imIn.getWidth();
      size_t height = imIn.getHeight();
      size_t depth  = imIn.getDepth();

      if (depth > 1) {
        return resize3DClosest(imIn, sx, sy, depth, imOut);
      }

      imOut.setSize(sx, sy, 1);
      ImageFreezer freeze(imOut);

      double cx = ((double) (width - 1)) / (sx - 1);
      double cy = ((double) (height - 1)) / (sy - 1);

      T *pixIn  = imIn.getPixels();
      T *pixOut = imOut.getPixels();

      for (size_t j = 0; j < sy; j++) {
#ifdef USE_OPEN_MP
        int nthreads = Core::getInstance()->getNumberOfThreads();
#pragma omp parallel for num_threads(nthreads)
#endif // USE_OPEN_MP
        for (size_t i = 0; i < sx; i++) {
          size_t xo = round(cx * i);
          size_t yo = round(cy * j);

          xo = min(xo, width - 1);
          yo = min(yo, height - 1);

          T v                = pixIn[yo * width + xo];
          pixOut[j * sx + i] = v;
        }
      }

      return RES_OK;
    }

    /*
     * 3D - closest interpolation algorithm - naive loop implementation
     */
    RES_T resize3DClosest(Image<T> &imIn, size_t sx, size_t sy, size_t sz,
                          Image<T> &imOut)
    {
      size_t width  = imIn.getWidth();
      size_t height = imIn.getHeight();
      size_t depth  = imIn.getDepth();

      if (depth > 1) {
        return resize2DClosest(imIn, sx, sy, imOut);
      }
      imOut.setSize(sx, sy, sz);

      ImageFreezer freeze(imOut);

      double cx = ((double) (width - 1)) / (sx - 1);
      double cy = ((double) (height - 1)) / (sy - 1);
      double cz = ((double) (depth - 1)) / (sz - 1);

      T *pixIn  = imIn.getPixels();
      T *pixOut = imOut.getPixels();

      for (size_t k = 0; k < sz; k++) {
        for (size_t j = 0; j < sy; j++) {
#ifdef USE_OPEN_MP
          int nthreads = Core::getInstance()->getNumberOfThreads();
#pragma omp parallel for num_threads(nthreads)
#endif // USE_OPEN_MP
          for (size_t i = 0; i < sx; i++) {
            size_t xo = round(cx * i);
            size_t yo = round(cy * j);
            size_t zo = round(cz * k);

            xo = min(xo, width - 1);
            yo = min(yo, height - 1);
            zo = min(zo, depth - 1);

            T v = pixIn[(zo * depth + yo) * width + xo];
            pixOut[(k * sy + j) * sx + i] = v;
          }
        }
      }
      return RES_OK;
    }

    /*
     * #          #    #    #  ######    ##       #    #####   ######
     * #          #    ##   #  #        #  #      #    #    #  #
     * #          #    # #  #  #####   #    #     #    #    #  #####
     * #          #    #  # #  #       ######     #    #####   #
     * #          #    #   ##  #       #    #     #    #   #   #
     * ######     #    #    #  ######  #    #     #    #    #  ######
     */
    /*
     * 2D - Bilinear interpolation algorithm - naive loop implementation
     */
    RES_T resizeBilinear(Image<T> &imIn, size_t sx, size_t sy, Image<T> &imOut)
    {
      off_t width  = imIn.getWidth();
      off_t height = imIn.getHeight();
      off_t depth  = imIn.getDepth();

      if (depth > 1) {
        return resizeTrilinear(imIn, sx, sy, depth, imOut);
      }

      imOut.setSize(sx, sy, 1);
      ImageFreezer freeze(imOut);

      double cx = ((double) (width - 1)) / (sx - 1);
      double cy = ((double) (height - 1)) / (sy - 1);

      T *pixIn  = imIn.getPixels();
      T *pixOut = imOut.getPixels();

      for (size_t j = 0; j < sy; j++) {
        off_t dy = width;
#ifdef USE_OPEN_MP
        int nthreads = Core::getInstance()->getNumberOfThreads();
#pragma omp parallel for num_threads(nthreads)
#endif
        for (size_t i = 0; i < sx; i++) {
          Point<double> P(cx * i, cy * j, 0);

          off_t xl = (off_t) floor(P.x);
          off_t xh = (xl + 1) < width ? xl + 1 : xl;
          off_t yl = (off_t) floor(P.y);
          off_t yh = (yl + 1) < height ? yl + 1 : yl;

          double cxl = xh > xl ? (P.x - xl) : 1.;
          double cxh = 1. - cxl;
          double cyl = yh > yl ? (P.y - yl) : 1.;
          double cyh = 1. - cyl;

          T vp = pixIn[yl * dy + xl] * cxh * cyh +
                 pixIn[yl * dy + xh] * cxl * cyh +
                 pixIn[yh * dy + xl] * cxh * cyl +
                 pixIn[yh * dy + xh] * cxl * cyl;

          pixOut[j * sx + i] = vp;
        }
      }
      return RES_OK;
    }

    /*
     * 3D - Trilinear interpolation algorithm - naive loop implementation
     */
    RES_T resizeTrilinear(Image<T> &imIn, size_t sx, size_t sy, size_t sz,
                          Image<T> &imOut)
    {
      off_t width  = imIn.getWidth();
      off_t height = imIn.getHeight();
      off_t depth  = imIn.getDepth();

      if (depth == 1) {
        return resizeBilinear(imIn, sx, sy, imOut);
      }

      imOut.setSize(sx, sy, sz);
      ImageFreezer freeze(imOut);

      double cx = ((double) (width - 1)) / (sx - 1);
      double cy = ((double) (height - 1)) / (sy - 1);
      double cz = ((double) (depth - 1)) / (sz - 1);

      T *pixIn  = imIn.getPixels();
      T *pixOut = imOut.getPixels();

      for (size_t k = 0; k < sz; k++) {
        off_t dz = width * height;
        for (size_t j = 0; j < sy; j++) {
          off_t dy = width;
#ifdef USE_OPEN_MP
          int nthreads = Core::getInstance()->getNumberOfThreads();
#pragma omp parallel for num_threads(nthreads)
#endif
          for (size_t i = 0; i < sx; i++) {
            Point<double> P(cx * i, cy * j, cz * k);

            off_t xl = floor(P.x);
            off_t xh = (xl + 1) < width ? xl + 1 : xl;
            off_t yl = floor(P.y);
            off_t yh = (yl + 1) < height ? yl + 1 : yl;
            off_t zl = floor(P.z);
            off_t zh = (zl + 1) < depth ? zl + 1 : zl;

            double cxl = xh > xl ? (P.x - xl) : 1.;
            double cxh = 1. - cxl;
            double cyl = yh > yl ? (P.y - yl) : 1.;
            double cyh = 1. - cyl;
            double czl = zh > zl ? (P.z - zl) : 1.;
            double czh = 1. - czl;

            T vp = pixIn[zl * dz + yl * dy + xl] * cxh * cyh * czh +
                   pixIn[zl * dz + yl * dy + xh] * cxl * cyh * czh +
                   pixIn[zl * dz + yh * dy + xl] * cxh * cyl * czh +
                   pixIn[zl * dz + yh * dy + xh] * cxl * cyl * czh +
                   pixIn[zh * dz + yl * dy + xl] * cxh * cyh * czl +
                   pixIn[zh * dz + yl * dy + xh] * cxl * cyh * czl +
                   pixIn[zh * dz + yh * dy + xl] * cxh * cyl * czl +
                   pixIn[zh * dz + yh * dy + xh] * cxl * cyl * czl;

            pixOut[(k * sy + j) * sx + i] = vp;
          }
        }
      }
      return RES_OK;
    }

    string method;
  };
  /** @endcond */

  /** resize() - 3D image resize
   *
   * @details Resize a 3D image - the value of each pixel in the output image is
   * calculated from the input image after an interpolation algorithm.
   *
   * There are two available algorithms :
   * - @b closest - this is the simpler algorithm. Pixel values in the output
   * image are taken from the nearest corresponding pixel in the input image.
   * This algorithm doesn't increases the number of possible values. So, it must
   * be used when resizing @b binary images or images whose possible values
   * shall be preserved in the output image.
   * - @b trilinear (extension of @b bilinear algorithm for @b 3D images) - this
   * is the algorithm to use on @txtbold{gray level} images.
   *
   * @note
   * When algorithm is set to @b auto, the applied algorithm will be @b closest
   * for binary images and @b trilinear or @b bilinear for gray level images
   *
   * @param[in] imIn : input image
   * @param[in] sx, sy, sz : dimensions to be set on output image
   * @param[out] imOut : output image
   * @param[in] algorithm : the interpolation algorithm to use. Can be @b
   * trilinear (default), @b bilinear, @b closest or @b auto.
   */
  template <typename T>
  RES_T resize(Image<T> &imIn, size_t sx, size_t sy, size_t sz,
                    Image<T> &imOut, string algorithm = "trilinear")
  {
    ASSERT_ALLOCATED(&imIn, &imOut)
    ImageResizeFunc<T> func(algorithm);
    return func.resize(imIn, sx, sy, sz, imOut, algorithm);
  }

  /** resize() - 2D image resize
   *
   * @details Resize a 2D image - the value of each pixel in the output image is
   * calculated from the input image after an interpolation algorithm.
   *
   * @param[in] imIn : input image
   * @param[in] sx, sy : dimensions to be set on output image
   * @param[out] imOut : output image
   * @param[in] algorithm : the interpolation algorithm to use. Can be @b
   * trilinear (default), @b bilinear ou @b closest.
   *
   * @overload
   */
  template <typename T>
  RES_T resize(Image<T> &imIn, size_t sx, size_t sy, Image<T> &imOut,
                    string algorithm = "trilinear")
  {
    ASSERT_ALLOCATED(&imIn, &imOut)
    size_t depth = imOut.getDepth();

    ImageResizeFunc<T> func(algorithm);
    return func.resize(imIn, sx, sy, depth, imOut, algorithm);
  }

  /** resize() - 3D image resize
   *
   * @details Resize a 3D image - the value of each pixel in the output image is
   * calculated from the input image after an interpolation algorithm.
   *
   * The size of the output image is already set to what it should be.
   *
   * @param[in] imIn : input image
   * @param[out] imOut : output image
   * @param[in] algorithm : the interpolation algorithm to use. Can be @b
   * trilinear (default), @b bilinear ou @b closest.
   *
   * @overload
   */
  template <typename T>
  RES_T resize(Image<T> &imIn, Image<T> &imOut,
                    string algorithm = "trilinear")
  {
    ASSERT_ALLOCATED(&imIn, &imOut)
    size_t width  = imOut.getWidth();
    size_t height = imOut.getHeight();
    size_t depth  = imOut.getDepth();

    ImageResizeFunc<T> func(algorithm);
    return func.resize(imIn, width, height, depth, imOut, algorithm);
  }

  /** scale() - 3D image scale (resize by a factor)
   *
   * @details 3D image scale - Scaling images is almost the same than resizing.
   Input parameters are the factors to multiply each dimension of the input
   image instead of the dimensions of output image.
   *
   * There are two available algorithms :
   * - @b closest - this is the simpler algorithm. Pixel values in the output
   * image are taken from the nearest corresponding pixel in the input image.
   * This algorithm doesn't increases the number of possible values. So, it must
   * be used when resizing @b binary images or images whose possible values
   * shall be preserved in the output image.
   * - @b trilinear (extension of @b bilinear algorithm for @b 3D images) - this
   * is the algorithm to use on @txtbold{gray level} images.
   *
   * @note
   * When algorithm is set to @b auto, the applied algorithm will be @b closest
   * for binary images and @b trilinear or @b bilinear for gray level images
   *
   * @param[in] imIn : input image
   * @param[in] kx, ky, kz : scale factors
   * @param[out] imOut : output image
   * @param[in] algorithm : the interpolation algorithm to use. Can be @b
   * trilinear (default), @b bilinear, @b closest or @b auto.
   */
  template <typename T>
  RES_T scale(Image<T> &imIn, double kx, double ky, double kz,
                   Image<T> &imOut, string algorithm = "trilinear")
  {
    ASSERT_ALLOCATED(&imIn, &imOut)
    ImageResizeFunc<T> func(algorithm);
    return func.scale(imIn, kx, ky, kz, imOut, algorithm);
  }

  /** scale() - 2D image scale
   *
   * @details 3D image scale - Scaling images is almost the same than resizing.
   * Input parameters are the factors to multiply each dimension of the input
   * image instead of the dimensions of output image.
   *
   * @param[in] imIn : input image
   * @param[in] kx, ky : scale factors
   * @param[out] imOut : output image
   * @param[in] algorithm : the interpolation algorithm to use. Can be @b
   * trilinear (default), @b bilinear ou @b closest.
   */
  template <typename T>
  RES_T scale(Image<T> &imIn, double kx, double ky, Image<T> &imOut,
                   string algorithm = "trilinear")
  {
    ASSERT_ALLOCATED(&imIn, &imOut)
    ImageResizeFunc<T> func(algorithm);
    return func.scale(imIn, kx, ky, 1., imOut, algorithm);
  }

  /** scale() - image scale (resize by a factor)
   *
   * @details 3D image scale - Scaling images is almost the same than resizing.   
   * Input parameters are the factors to multiply each dimension of the input
   * image instead of the dimensions of output image.
   *
   *
   * @param[in] imIn : input image
   * @param[in] k : scale factor applied to each axis.
   * @param[out] imOut : output image
   * @param[in] algorithm : the interpolation algorithm to use. Can be @b
   trilinear (default), @b bilinear ou @b closest.
   */
  template <typename T>
  RES_T scale(Image<T> &imIn, double k, Image<T> &imOut,
                   string algorithm = "trilinear")
  {
    ASSERT_ALLOCATED(&imIn, &imOut)
    ImageResizeFunc<T> func(algorithm);
    return func.scale(imIn, k, k, k, imOut, algorithm);
  }


#if 0
  /**
   * BMI: 08.08.2018
   *
   * resize() - Resize imIn to sx,sy -> imOut. No bilinear interpolation.
   * Closest value
   *
   * @warning Quick implementation (needs better integration and optimization).
   */
  template <class T>
  RES_T resizeClosest(Image<T> &imIn, size_t sx, size_t sy, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn)

    if (&imIn == &imOut) {
      Image<T> tmpIm(imIn, true); // clone
      return resizeClosest(tmpIm, sx, sy, imIn);
    }

    ImageFreezer freeze(imOut);

    imOut.setSize(sx, sy);

    ASSERT_ALLOCATED(&imOut)

    size_t w = imIn.getWidth();
    size_t h = imIn.getHeight();

    typedef typename Image<T>::lineType lineType;

    lineType pixIn  = imIn.getPixels();
    lineType pixOut = imOut.getPixels();

    size_t A, B, C, D, maxVal = numeric_limits<T>::max();
    size_t x, y, index;

    double x_ratio = ((double) (w)) / sx; // BMI -1 removed
    double y_ratio = ((double) (h)) / sy; // BMI -1 removed
    double x_diff, y_diff;
    int offset = 0;

    for (size_t i = 0; i < sy; i++) {
      for (size_t j = 0; j < sx; j++) {
        x      = (int) (x_ratio * j);
        y      = (int) (y_ratio * i);
        x_diff = (x_ratio * j) - x;
        y_diff = (y_ratio * i) - y;
        index  = y * w + x;

        A = size_t(pixIn[index]) & maxVal;
        if (x == w - 1) {
          B = size_t(pixIn[index]) & maxVal;
        } else {
          B = size_t(pixIn[index + 1]) & maxVal;
        }
        if (y < h - 1) {
          C = size_t(pixIn[index + w]) & maxVal;
        } else {
          C = size_t(pixIn[index]) & maxVal;
        }
        size_t myindex;
        myindex = index;
        if (y < h - 1)
          myindex += w;
        if (x < w - 1)
          myindex += 1;
        D = size_t(pixIn[myindex]) & maxVal;

        // Y = A(1-w)(1-h) + B(w)(1-h) + C(h)(1-w) + Dwh
        if ((x_diff < 0.5) && (y_diff < 0.5)) {
          pixOut[offset++] = A;
        } else if (x_diff < 0.5) {
          pixOut[offset++] = C;
        } else if (y_diff < 0.5) {
          pixOut[offset++] = B;
        } else {
          pixOut[offset++] = D;
        }
      }
    }

    return RES_OK;
  }

  /**
   * resizeClosest() - Resize imIn with the dimensions of imOut and put the
   * result in imOut.
   */
  template <class T> RES_T resizeClosest(Image<T> &imIn, Image<T> &imOut)
  {
    return resizeClosest(imIn, imOut.getWidth(), imOut.getHeight(), imOut);
  }

  /**
   * resizeClosest() -
   */
  template <class T> RES_T resizeClosest(Image<T> &imIn, UINT sx, UINT sy)
  {
    ASSERT_ALLOCATED(&imIn)
    Image<T> tmpIm(imIn, true); // clone
    imIn.setSize(sx, sy);
    return resizeClosest<T>(tmpIm, imIn);
  }

  /**
   * resize() - 2D bilinear resize algorithm.
   *
   * Resize imIn to sx,sy -> imOut.
   *
   * Quick implementation (needs better integration and optimization).
   */
  template <class T>
  RES_T resize(Image<T> &imIn, size_t sx, size_t sy, Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn)

    if (&imIn == &imOut) {
      Image<T> tmpIm(imIn, true); // clone
      return resize(tmpIm, sx, sy, imIn);
    }

    ImageFreezer freeze(imOut);

    imOut.setSize(sx, sy);

    ASSERT_ALLOCATED(&imOut)

    size_t w = imIn.getWidth();
    size_t h = imIn.getHeight();

    typedef typename Image<T>::lineType lineType;

    lineType pixIn  = imIn.getPixels();
    lineType pixOut = imOut.getPixels();

    size_t A, B, C, D, maxVal = numeric_limits<T>::max();
    size_t x, y, index;

    double x_ratio = ((double) (w - 1)) / sx;
    double y_ratio = ((double) (h - 1)) / sy;
    double x_diff, y_diff;
    int offset = 0;

    for (size_t i = 0; i < sy; i++) {
      for (size_t j = 0; j < sx; j++) {
        x      = (int) (x_ratio * j);
        y      = (int) (y_ratio * i);
        x_diff = (x_ratio * j) - x;
        y_diff = (y_ratio * i) - y;
        index  = y * w + x;

        A = size_t(pixIn[index]) & maxVal;
        B = size_t(pixIn[index + 1]) & maxVal;
        C = size_t(pixIn[index + w]) & maxVal;
        D = size_t(pixIn[index + w + 1]) & maxVal;

        // Y = A(1-w)(1-h) + B(w)(1-h) + C(h)(1-w) + Dwh
        pixOut[offset++] =
            T(A * (1. - x_diff) * (1. - y_diff) + B * (x_diff) * (1. - y_diff) +
              C * (y_diff) * (1. - x_diff) + D * (x_diff * y_diff));
      }
    }

    return RES_OK;
  }

  /**
   * resize() - Resize imIn with the dimensions of imOut and put the result in
   * imOut.
   */
  template <class T> RES_T resize(Image<T> &imIn, Image<T> &imOut)
  {
    return resize(imIn, imOut.getWidth(), imOut.getHeight(), imOut);
  }

  /**
   * resize() -
   */
  template <class T> RES_T resize(Image<T> &imIn, UINT sx, UINT sy)

  {
    ASSERT_ALLOCATED(&imIn)
    Image<T> tmpIm(imIn, true); // clone

    imIn.setSize(sx, sy);
    return resize<T>(tmpIm, imIn);
  }

  /**
   * Scale image
   * If imIn has the size (W,H), the size of imOut will be (W*cx, H*cy).
   */
  template <class T>
  RES_T scale(Image<T> &imIn, double cx, double cy, Image<T> &imOut)
  {
    return resize<T>(imIn, size_t(imIn.getWidth() * cx),
                     size_t(imIn.getHeight() * cy), imOut);
  }

  /**
   * scale() -
   */
  template <class T> RES_T scale(Image<T> &imIn, double cx, double cy)
  {
    ASSERT_ALLOCATED(&imIn)
    Image<T> tmpIm(imIn, true); // clone
    return resize(tmpIm, cx, cy, imIn);
  }

  /**
   * scaleClosest() - Scale image
   * If imIn has the size (W,H), the size of imOut will be (W*cx, H*cy).
   */
  template <class T>
  RES_T scaleClosest(Image<T> &imIn, double cx, double cy, Image<T> &imOut)
  {
    return resizeClosest<T>(imIn, size_t(imIn.getWidth() * cx),
                            size_t(imIn.getHeight() * cy), imOut);
  }

  /**
   * scaleClosest() - Scale image
   * If imIn has the size (W,H), the size of imOut will be (W*cx, H*cy).
   */
  template <class T> RES_T scaleClosest(Image<T> &imIn, double cx, double cy)
  {
    ASSERT_ALLOCATED(&imIn)
    Image<T> tmpIm(imIn, true); // clone
    return resizeClosest(tmpIm, cx, cy, imIn);
  }
#endif

  /** @}*/

} // namespace smil

#endif // _D_IMAGE_TRANSFORM_HPP
