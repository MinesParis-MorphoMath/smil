#ifndef _DFAST_AREA_OPENING_H_
#define _DFAST_AREA_OPENING_H_

#include "Core/include/DCore.h"

namespace smil
{
  /**
   * @ingroup   Advanced 
   * @defgroup  AdvFastArea    Fast Area Opening/Closing
   * @{ */

  // AREA OPENING

  /** @brief Area closing with pixel queue algorithm (V4)
   * This algorithm needs an INT32 input and an INT32 output. It can be an
   * inplace transform However, we are working with UINT8 and UINT8 input and
   * output buffer. Hence, we have already converted the input image in INT32.
   * (not const) First, the result of the area op will be store in imIn. Then we
   * convert it into imOut
   * @param[in]  imIn : the initial image
   * @param[in]  size :
   * @param[out] imOut : Area Closing of imIn
   */
  template <class T1, class T2>
  RES_T ImAreaClosing_PixelQueue(const Image<T1> &imIn, int size,
                                 Image<T2> &imOut);

  /** @brief Area Opening with pixel queue algorithm (V4)
   * @param[in]  imIn : the initial image
   * @param[in]  size :
   * @param[out] imOut : Area Opening of imIn
   */
  template <class T1, class T2>
  RES_T ImAreaOpening_PixelQueue(const Image<T1> &imIn, int size,
                                 Image<T2> &imOut)
  {
    ASSERT_ALLOCATED(&imIn)
    ASSERT_SAME_SIZE(&imIn, &imOut)
    Image<T1> imNeg(imIn);
    RES_T res = inv(imIn, imNeg);
    if (res != RES_OK)
      return res;
    res = ImAreaClosing_PixelQueue(imNeg, size, imOut);
    if (res != RES_OK)
      return res;
    return inv(imOut, imOut);
  }

  /** @brief Area opening with a max tree algorithm (V4)
   * @param[in]  imIn : the initial image
   * @param[in]  size :
   * @param[out] imOut : Area Opening of imIn
   */
  template <class T1, class T2>
  RES_T ImAreaOpening_MaxTree(const Image<T1> &imIn, int size,
                              Image<T2> &imOut);

  /** @brief Area closing with a max tree algorithm (V4)
   * @param[in]  imIn : the initial image
   * @param[in]  size :
   * @param[out] imOut : Area Closing of imIn
   */
  template <class T1, class T2>
  RES_T ImAreaClosing_MaxTree(const Image<T1> &imIn, int size, Image<T2> &imOut)
  {
    ASSERT_ALLOCATED(&imIn)
    ASSERT_SAME_SIZE(&imIn, &imOut)
    Image<T1> imNeg(imIn);
    RES_T res = inv(imIn, imNeg);
    if (res != RES_OK)
      return res;
    res = ImAreaOpening_MaxTree(imNeg, size, imOut);
    if (res != RES_OK)
      return res;
    return inv(imOut, imOut);
  }

  /** @brief Area opening with an union find algorithm (V4)
   * @param[in]  imIn : the initial image
   * @param[in]  size :
   * @param[out] imOut : Area opening of imIn
   */
  template <class T1, class T2>
  RES_T ImAreaOpening_UnionFind(const Image<T1> &imIn, int size,
                                Image<T2> &imOut);

  /** @brief Area closing with an union find algorithm (V4)
   * @param[in]  imIn : the initial image
   * @param[in]  size :
   * @param[out] imOut : Area Closing of imIn
   */
  template <class T1, class T2>
  RES_T ImAreaClosing_UnionFind(const Image<T1> &imIn, int size,
                                Image<T2> &imOut)
  {
    ASSERT_ALLOCATED(&imIn)
    ASSERT_SAME_SIZE(&imIn, &imOut)
    Image<T1> imNeg(imIn);
    RES_T res = inv(imIn, imNeg);
    if (res != RES_OK)
      return res;
    res = ImAreaOpening_UnionFind(imNeg, size, imOut);
    if (res != RES_OK)
      return res;
    return inv(imOut, imOut);
  }

  /** @brief Non exact implementation of the Area opening with an 1D line (V4)
   * @param[in]  imIn : the initial image
   * @param[in]  size :
   * @param[out] imOut : Area Opening of imIn
   */
  template <class T1, class T2>
  RES_T ImAreaOpening_Line(const Image<T1> &imIn, int size, Image<T2> &imOut);

  /** @brief Non exact implementation of the Area closing with an 1D line (V4)
   * @param[in]  imIn : the initial image
   * @param[in]  size :
   * @param[out] imOut : Area Closing of imIn
   */
  template <class T1, class T2>
  RES_T ImAreaClosing_Line(const Image<T1> &imIn, int size, Image<T2> &imOut)
  {
    ASSERT_ALLOCATED(&imIn)
    ASSERT_SAME_SIZE(&imIn, &imOut)
    Image<T1> imNeg(imIn);
    RES_T res = inv(imIn, imNeg);
    if (res != RES_OK)
      return res;
    res = ImAreaOpening_Line(imNeg, size, imOut);
    if (res != RES_OK)
      return res;
    return inv(imOut, imOut);
  }

  /** @brief Non exact implementation of the Area opening with an 1D line (V4)
   * @param[in]  imIn : the initial image
   * @param[in]  size :
   * @param[out] imOut : Area Opening of imIn
   */
  template <class T1, class T2>
  RES_T ImAreaOpening_LineSupEqu(const Image<T1> &imIn, int size,
                                 Image<T2> &imOut);

  // INERTIA THINNINGS

  /** @brief Inertia thinning with a max tree algorithm (V4)
   * @param[in]  imIn : the initial image
   * @param[in]  size :
   * @param[out] imOut : inertia thinning of imIn
   */
  template <class T1, class T2>
  RES_T ImInertiaThinning_MaxTree(const Image<T1> &imIn, double size,
                                  Image<T2> &imOut);

  /** @brief Inertia thickening with a max tree algorithm (V4)
   * @param[in]  imIn : the initial image
   * @param[in]  size :
   * @param[out] imOut : Area inertia thickening of imIn
   */
  template <class T1, class T2>
  RES_T ImInertiaThickening_MaxTree(const Image<T1> &imIn, double size,
                                    Image<T2> &imOut)
  {
    ASSERT_ALLOCATED(&imIn)
    ASSERT_SAME_SIZE(&imIn, &imOut)
    Image<T1> imNeg(imIn);
    RES_T res = inv(imIn, imNeg);
    if (res != RES_OK)
      return res;
    res = ImInertiaThinning_MaxTree(imNeg, size, imOut);
    if (res != RES_OK)
      return res;
    return inv(imOut, imOut);
  }

  /** @} */
} // namespace smil

// FastAreaOpening Module header
#include "FastAreaOpening/AreaOpeningPixelQueue_T.hpp"
#include "FastAreaOpening/AreaOpeningMaxTree_T.hpp"
#include "FastAreaOpening/AreaOpeningUnionFind_T.hpp"
#include "FastAreaOpening/AreaOpeningLine_T.hpp"

#endif // _DFAST_AREA_OPENING_H_
