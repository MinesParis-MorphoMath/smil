#ifndef _D_NORMALIZE_FILTER_HPP_
#define _D_NORMALIZE_FILTER_HPP_

namespace smil
{
  /*
   *
   */
  template <class T1, class T2>
  RES_T ImNormalize(const Image<T1> &imIn, const T2 Min, const T2 Max,
                    Image<T2> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    ImageFreezer freeze(imOut);

    size_t S[3];
    imIn.getSize(S);

    typename ImDtTypes<T1>::lineType bufferIn  = imIn.getPixels();
    typename ImDtTypes<T2>::lineType bufferOut = imOut.getPixels();

    int W, H, D;
    W = S[0];
    H = S[1];
    D = S[2];

    T1 vMin = minVal(imIn);
    T1 vMax = maxVal(imIn);
    if ((vMax - vMin) == 0) {
      // a flat image - can generate division by 0
      return RES_ERR;
    }

    double k = ((double) (Max - Min)) / ((double) (vMax - vMin));

    int iMax = W * H * D;
    for (int i = 0; i < iMax; i++)
      bufferOut[i] = (T2)(Min + k * (bufferIn[i] - vMin));

    return RES_OK;
  }

  /*
   *
   */
  template <class T1, class T2>
  RES_T ImNormalizeAuto(const Image<T1> &imIn, Image<T2> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    ImageFreezer freeze(imOut);

    size_t S[3];
    imIn.getSize(S);

    typename ImDtTypes<T1>::lineType bufferIn  = imIn.getPixels();
    typename ImDtTypes<T2>::lineType bufferOut = imOut.getPixels();

    int W, H, D;
    W = S[0];
    H = S[1];
    D = S[2];

    T2 Max  = imOut.getDataTypeMax();
    T2 Min  = imOut.getDataTypeMin();
    T1 vMin = minVal(imIn);
    T1 vMax = maxVal(imIn);
    if ((vMax - vMin) == 0) {
      // a flat image - can generate division by 0
      return RES_ERR;
    }

    double k = ((double) (Max - Min)) / ((double) (vMax - vMin));

    int iMax = W * H * D;
    for (int i = 0; i < iMax; i++)
      bufferOut[i] = (T2)(Min + k * (bufferIn[i] - vMin));

    return RES_OK;
  }

  /*
   *  
   */
  template <class T1, class T2>
  RES_T ImNormalizeSCurve(const Image<T1> &imIn, const T1 pivot,
                          const double ratio, Image<T2> &imOut)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    ImageFreezer freeze(imOut);

    size_t S[3];
    imIn.getSize(S);

    typename ImDtTypes<T1>::lineType bufferIn  = imIn.getPixels();
    typename ImDtTypes<T2>::lineType bufferOut = imOut.getPixels();

    int W, H, D;
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

    T2 Max = imOut.getDataTypeMax();
    int iMax = W * H * D;

    for (int i = 0; i < iMax; i++)
      bufferOut[i] = (T2) (Max / (1. + exp(-k * (bufferIn[i] - ctr))));

    return RES_OK;
  }

} // namespace smil

#endif // _D_NORMALIZE_FILTER_HPP_
