#ifndef _DFAST_LINE_H__
#define _DFAST_LINE_H__

#include "Core/include/DCore.h"

namespace smil
{
  /**
   * @ingroup   Advanced
   * @defgroup  AdvMorardFastLine     Line Mathematical Morphology
   *
   * @{ */

  /* author Vincent Morard, Jose-Marcio Martins da Cruz
   * date   1 septembre 2010
   * Update  21 octobre 2010
   */

  /**
   * @brief ImLineErode the SE is a segment of radius "radius" pixels and an
   * orientation H or V
   * @param[in]  imIn the initial image
   * @param[in]  radius the size of the segment  will be radius*2+1
   * @param[in]  horizontal orientation of the line horizontale or vertical
   * @param[out] imOut Result
   */
  template <class T>
  RES_T ImLineErode(const Image<T> &imIn, const int radius, bool horizontal,
                    Image<T> &imOut);

  /**
   * @brief ImLineDilate the SE is a segment of radius "radius" pixels and an
   * orientation H or V
   * @param[in]  imIn the initial image
   * @param[in]  radius the size of the segment  will be radius*2+1
   * @param[in]  horizontal orientation of the line horizontale or vertical
   * @param[out] imOut Result
   */
  template <class T>
  RES_T ImLineDilate(const Image<T> &imIn, const int radius, bool horizontal,
                     Image<T> &imOut);

  /**
   * @brief ImLineOpen the SE is a segment of radius "radius" pixels and an
   * orientation H or V
   * @param[in]  imIn the initial image
   * @param[in]  radius the size of the segment  will be radius*2+1
   * @param[in]  horizontal orientation of the line horizontale or vertical
   * @param[out] imOut Result
   */
  template <class T>
  RES_T ImLineOpen(const Image<T> &imIn, const int radius, bool horizontal,
                   Image<T> &imOut);

  /**
   * @brief ImLineClose the SE is a segment of radius "radius" pixels and an
   * orientation H or V
   * @param[in]  imIn the initial image
   * @param[in]  radius the size of the segment  will be radius*2+1
   * @param[in]  horizontal orientation of the line horizontale or vertical
   * @param[out] imOut Result
   */
  template <class T>
  RES_T ImLineClose(const Image<T> &imIn, const int radius, bool horizontal,
                    Image<T> &imOut);

  /**
   * @brief ImSquareErode the SE is a segment of radius "radius" pixels
   * @param[in]  imIn the initial image
   * @param[in]  radius the size of the segment  will be radius*2+1
   * @param[out] imOut Result
   */
  template <class T>
  RES_T ImSquareErode(const Image<T> &imIn, const int radius, Image<T> &imOut)
  {
    Image<T> imTmp(imIn);

    RES_T res = ImLineErode(imIn, radius, true, imTmp);
    if (res == RES_OK)
      res = ImLineErode(imTmp, radius, false, imOut);
    return res;
  }

  /**
   * @brief ImSquareDilate the SE is a segment of radius "radius" pixels
   * @param[in]  imIn the initial image
   * @param[in]  radius the size of the segment  will be radius*2+1
   * @param[out] imOut Result
   */
  template <class T>
  RES_T ImSquareDilate(const Image<T> &imIn, const int radius, Image<T> &imOut)
  {
    Image<T> imTmp(imIn);

    RES_T res = ImLineDilate(imIn, radius, true, imTmp);
    if (res == RES_OK)
      res = ImLineDilate(imTmp, radius, false, imOut);
    return res;
  }

  /**
   * @brief ImSquareOpen the SE is a segment of radius "radius" pixels
   * @param[in]  imIn the initial image
   * @param[in]  radius the size of the segment  will be radius*2+1
   * @param[out] imOut Result
   */
  template <class T>
  RES_T ImSquareOpen(const Image<T> &imIn, const int radius, Image<T> &imOut)
  {
    Image<T> imTmp(imIn);

    RES_T res = ImLineErode(imIn, radius, true, imTmp);
    if (res == RES_OK)
      res = ImLineErode(imTmp, radius, false, imOut);
    if (res == RES_OK)
      res = ImLineDilate(imOut, radius, true, imTmp);
    if (res == RES_OK)
      res = ImLineDilate(imTmp, radius, false, imOut);
    return res;
  }  

  /** 
   * @brief ImSquareClose the SE is a segment of radius "radius" pixels
   * @param[in]  imIn the initial image
   * @param[in]  radius the size of the segment  will be radius*2+1
   * @param[out] imOut Result
   */
  template <class T>
  RES_T ImSquareClose(const Image<T> &imIn, const int radius, Image<T> &imOut)
  {
    Image<T> imTmp(imIn);

    RES_T res = ImLineDilate(imIn, radius, true, imTmp);
    if (res == RES_OK)
      res = ImLineDilate(imTmp, radius, false, imOut);
    if (res == RES_OK)
      res = ImLineErode(imOut, radius, true, imTmp);
    if (res == RES_OK)
      res = ImLineErode(imTmp, radius, false, imOut);
    return res;
  }

  /** @} */
} // namespace smil

#include "FastLine/LineNaive.hpp"

#endif
