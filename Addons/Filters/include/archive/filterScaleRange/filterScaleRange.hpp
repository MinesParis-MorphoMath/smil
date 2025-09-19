#ifndef _D_NORMALIZE_FILTER_HPP_
#define _D_NORMALIZE_FILTER_HPP_

namespace smil
{


  /**
   * @brief expand() - Linear conversion of pixels values to the range
   * [Min, Max]
   *
   * Values in the input image are linearly mapped into the output image with
   * the following rules :
   * - if <b>imIn(x) <= inMin</b>, imOut(x) will be mapped in the range <b>[0,
   * outMin]</b>
   * - if <b>inMin < imIn(x) <= inMax</b>, imOut(x) will be mapped in the range
   * <b>[outMin, outMax]</b>
   * - if <b>imIn(x) > inMax</b>, imOut(x) will be mapped in the range
   * <b>[outMax, max(T2)]</b>
   *
   * @param[in] imIn : input Image
   * @param[in] inMin, inMax : control range in the input image
   * @param[in] outMin, outMax : control range in the output image
   * @param[out] imOut : output Image
   *
   */
  template <class T1, class T2>
  RES_T expand(const Image<T1> &imIn, const T1 inMin, const T1 inMax,
                        const T2 outMin, const T2 outMax, Image<T2> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    ImageFreezer freeze(imOut);

    size_t S[3];
    imIn.getSize(S);

    typename ImDtTypes<T1>::lineType bufferIn  = imIn.getPixels();
    typename ImDtTypes<T2>::lineType bufferOut = imOut.getPixels();

    size_t W, H, D;
    W = S[0];
    H = S[1];
    D = S[2];

    if ((inMax - inMin) == 0) {
      // a flat image - can generate division by 0
      return RES_ERR;
    }

    T1 inTop  = imIn.getDataTypeMax();
    T2 outTop = imOut.getDataTypeMax();

    double k1, k2, k3;
    k1 = k2 = k3 = 0;

    if (inMin > 0)
      k1 = ((double) (outMin - 0)) / ((double) (inMin - 0));
    if (inMax > inMin)
      k2 = ((double) (outMax - outMin)) / ((double) (inMax - inMin));
    if (inTop > inMax)
      k3 = ((double) (outTop - outMax)) / ((double) (inTop - inMax));

    size_t iMax = W * H * D;
    for (size_t i = 0; i < iMax; i++) {
      if (bufferIn[i] < inMin) {
        bufferOut[i] = (T2)(k1 * bufferIn[i]);
        continue;
      }
      if (bufferIn[i] >= inMin && bufferIn[i] < inMax) {
        bufferOut[i] = (T2)(outMin + k2 * (bufferIn[i] - inMin));
        continue;
      }
      if (bufferIn[i] >= inMax) {
        bufferOut[i] = (T2)(outMax + k3 * (bufferIn[i] - inMax));
      }
    }
    return RES_OK;
  }

  /**
   * @brief expand() - Linear conversion of pixel values to the range
   * [Min, Max]
   *
   * Maps a range of values in the input image into the range <b>[Min, Max]</b>
   * in the output image.
   *
   * If @b onlyNonZero is @b true uses <b>[minVal(), maxVal()]</b> as the range
   * of values in the input image, otherwise, uses the full range of values.
   *
   * @param[in] imIn : input image
   * @param[in] Min : Minimum value in the output image
   * @param[in] Max : Maximum value in the output image
   * @param[out] imOut : output Image
   * @param[in] onlyNonZero : defines how to find input image range of values
   */
  template <class T1, class T2>
  RES_T expand(const Image<T1> &imIn, const T2 Min, const T2 Max,
                        Image<T2> &imOut, bool onlyNonZero)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    ImageFreezer freeze(imOut);

    size_t S[3];
    imIn.getSize(S);

    typename ImDtTypes<T1>::lineType bufferIn  = imIn.getPixels();
    typename ImDtTypes<T2>::lineType bufferOut = imOut.getPixels();

    size_t W, H, D;
    W = S[0];
    H = S[1];
    D = S[2];

    T1 vMin, vMax;
    if (onlyNonZero) {
      vMin = minVal(imIn);
      vMax = maxVal(imIn);
    } else {
      vMin = imIn.getDataTypeMin();
      vMax = imIn.getDataTypeMax();
    }
    if ((vMax - vMin) == 0) {
      // a flat image - can generate division by 0
      return RES_ERR;
    }

    double k = ((double) (Max - Min)) / ((double) (vMax - vMin));

    size_t iMax = W * H * D;
    for (size_t i = 0; i < iMax; i++)
      bufferOut[i] = (T2)(Min + k * (bufferIn[i] - vMin));

    return RES_OK;
  }

  /**
   * @brief expand() - Linear conversion of pixels values to the
   * domain range
   *
   * Maps a range in the input image into the  range <b>[min(T2), max(T2)]</b>
   * in the output image.
   *
   * If @b onlyNonZero is @b true uses <b>[minVal(), maxVal()]</b> as the range
   * of values in the input image, otherwise, uses the full range of values.
   *
   * @param[in] imIn : input Image
   * @param[out] imOut : output Image
   * @param[in] onlyNonZero : defines how to find input image range of values
   */
  template <class T1, class T2>
  RES_T expand(const Image<T1> &imIn, Image<T2> &imOut,
                        bool onlyNonZero)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

#if 1
    return expand(imIn, imOut.getDataTypeMin(), imOut.getDataTypeMax(),
                           imOut, onlyNonZero);
#else

    ImageFreezer freeze(imOut);

    size_t S[3];
    imIn.getSize(S);

    typename ImDtTypes<T1>::lineType bufferIn  = imIn.getPixels();
    typename ImDtTypes<T2>::lineType bufferOut = imOut.getPixels();

    size_t W, H, D;
    W = S[0];
    H = S[1];
    D = S[2];

    T2 Max = imOut.getDataTypeMax();
    T2 Min = imOut.getDataTypeMin();

    T1 vMin, vMax;
    if (onlyNonZero) {
      vMin = minVal(imIn);
      vMax = maxVal(imIn);
    } else {
      vMin = imIn.getDataTypeMin();
      vMax = imIn.getDataTypeMax();
    }

    if ((vMax - vMin) == 0) {
      // a flat image - can generate division by 0
      return RES_ERR;
    }

    double k = ((double) (Max - Min)) / ((double) (vMax - vMin));

    size_t iMax = W * H * D;
    for (size_t i = 0; i < iMax; i++)
      bufferOut[i] = (T2)(Min + k * (bufferIn[i] - vMin));

    return RES_OK;
#endif
  }

  /**
   * @brief sCurve() - S Curve transform
   *
   * This function emulates the <b>S Curve</b> caracteristic of film
   * photography.
   *
   * Use a sigmoid function centered at @b pivot with derivative @b ratio.
   *
   * One use of this filter is to set (increase or decrease) the contrast in the
   * neighborhood of the @b pivot.
   *
   * @param[in] imIn : input Image
   * @param[in] pivot :
   * * if 0, takes the median of the histogram of input image as pivot
   * * otherwise, use this value
   * @param[in] ratio : derivative of output image at pivot value
   * @param[out] imOut : output Image
   */
  template <class T1, class T2>
  RES_T sCurve(const Image<T1> &imIn, const T1 pivot,
                              const double ratio, Image<T2> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    ImageFreezer freeze(imOut);

    size_t S[3];
    imIn.getSize(S);

    typename ImDtTypes<T1>::lineType bufferIn  = imIn.getPixels();
    typename ImDtTypes<T2>::lineType bufferOut = imOut.getPixels();

    size_t W, H, D;
    W = S[0];
    H = S[1];
    D = S[2];

    T1 vMin = minVal(imIn);
    T1 vMax = maxVal(imIn);
    if ((vMax - vMin) == 0) {
      // a flat image - can generate division by 0
      return RES_ERR;
    }

    T1 ctr = pivot;
    if (pivot == 0 || pivot > vMax)
      ctr = (vMax - vMin) / 2;

    double k = 4. * ratio / (vMax - vMin);

    T2 Max      = imOut.getDataTypeMax();
    size_t iMax = W * H * D;

    for (size_t i = 0; i < iMax; i++)
      bufferOut[i] = (T2)(Max / (1. + exp(-k * (bufferIn[i] - ctr))));

    return RES_OK;
  }

} // namespace smil

#endif // _D_NORMALIZE_FILTER_HPP_
