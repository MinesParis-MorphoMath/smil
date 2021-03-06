#ifndef __FAST_LINE_HPP__
#define __FAST_LINE_HPP__

#include "Core/include/DCore.h"

namespace smil
{
  /**
   * @ingroup     Advanced
   * @addtogroup  AdvLine   Line Based Operators
   *
   * @brief Morphological operators using @TB{line segments} as structuring
   * elements.
   *
   * @details Implementation of morphological operations by @b line, @b square
   * and @b circle structuring elements using repeately operations with
   * @b linear structuring elements and arbitrary angles.
   *
   * These algorithms are described in @SoilleBook{Section 3.9, p. 89} and
   * @cite SoilleBJ96
   *
   * Line Structuring Element using Bresenham's Line Drawing Algorithm
   * @cite Hearn_1986. Based on Erik R Urbach (2006) implementation in C.
   *
   *
   * @remark This module works only on 2D images.
   *
   * @author Vincent Morard - Port to Morph-M - September 2010
   * @author Jose-Marcio Martins da Cruz - Port to Smil - July 2019
   *
   * @{ */

  //*************************************************
  // SOILLE 'S ALGORITHM
  //*************************************************

  /** @brief lineDilate() : the Structuring Element is a segment of length
   * <b>(2 * hLen + 1)</b> pixels and an orientation of <b>angle</b> degrees
   * @param[in]  imIn : input image
   * @param[in]  angle : (in degres)
   * @param[in]  hLen : Half Length of the Structuring Element
   * @param[out] imOut : output image
   */
  template <class T>
  RES_T lineDilate(const Image<T> &imIn, const int angle, const int hLen,
                   Image<T> &imOut)
  {
    return lineDilate_Soille(imIn, angle, hLen, imOut);
  }

  /** @brief lineErode() : the Structuring Element is a segment of length
   * <b>(2 * hLen + 1)</b> pixels and an orientation <b>angle</b> degrees
   * @param[in]  imIn : input image
   * @param[in]  angle : (in degres)
   * @param[in]  hLen : Half Length of the Structuring Element
   * @param[out] imOut : Result
   */
  template <class T>
  RES_T lineErode(const Image<T> &imIn, const int angle, const int hLen,
                  Image<T> &imOut)
  {
    return lineErode_Soille(imIn, angle, hLen, imOut);
  }

  /** @brief lineOpen() : the Structuring Element is a segment of length
   * <b>(2 * hLen + 1)</b> pixels and an orientation of <b>angle</b> degrees
   * @param[in]  imIn : input image
   * @param[in]  angle : (in degres)
   * @param[in]  hLen : Half Length of the Structuring Element
   * @param[out] imOut : output image
   */
  template <class T>
  RES_T lineOpen(const Image<T> &imIn, const int angle, const int hLen,
                 Image<T> &imOut)
  {
    return lineOpen_Soille(imIn, angle, hLen, imOut);
  }

  /** @brief lineClose() : the Structuring Element is a segment of length
   * <b>(2 * hLen + 1)</b> pixels and an orientation of <b>angle</b> degrees
   * @param[in]  imIn : input image
   * @param[in]  angle : (in degres)
   * @param[in]  hLen : Half Length of the Structuring Element
   * @param[out] imOut : output image
   */
  template <class T>
  RES_T lineClose(const Image<T> &imIn, const int angle, const int hLen,
                  Image<T> &imOut)
  {
    return lineClose_Soille(imIn, angle, hLen, imOut);
  }

  /**
   * @brief squareErode() : the SE is a square which side is <b>(2 * hSide +
   * 1)</b>
   * @param[in]  imIn : input image
   * @param[in]  hSide : Half side of the Structuring Element
   * @param[out] imOut : output image
   */
  template <class T>
  RES_T squareErode(const Image<T> &imIn, const int hSide, Image<T> &imOut)
  {
    return squareErode_Soille(imIn, hSide, imOut);
  }

  /**
   * @brief squareDilate() : the SE is a square which side is <b>(2 * hSide +
   * 1)</b>
   * @param[in]  imIn : input image
   * @param[in]  hSide : Half side of the Structuring Element
   * @param[out] imOut : output image
   */
  template <class T>
  RES_T squareDilate(const Image<T> &imIn, const int hSide, Image<T> &imOut)
  {
    return squareDilate_Soille(imIn, hSide, imOut);
  }

  /**
   * @brief squareOpen() : the SE is a square which side is <b>(2 * hSide +
   * 1)</b>
   * @param[in]  imIn : input image
   * @param[in]  hSide : Half side of the Structuring Element
   * @param[out] imOut : output image
   */
  template <class T>
  RES_T squareOpen(const Image<T> &imIn, const int hSide, Image<T> &imOut)
  {
    return squareOpen_Soille(imIn, hSide, imOut);
  }

  /**
   * @brief squareClose() : the SE is a square which side is <b>(2 * hSide +
   * 1)</b>
   * @param[in]  imIn : input image
   * @param[in]  hSide : Half side of the Structuring Element
   * @param[out] imOut : output image
   */
  template <class T>
  RES_T squareClose(const Image<T> &imIn, const int hSide, Image<T> &imOut)
  {
    return squareClose_Soille(imIn, hSide, imOut);
  }

  /** @brief circleDilate() : the SE is a <b>disk</b> of radius <b>radius</b>
   * pixels
   *
   * @param[in]  imIn : input image
   * @param[in]  radius : the diameter of the disk will be <b>2 * radius + 1</b>
   * @param[out] imOut : output image
   */
  template <class T>
  RES_T circleDilate(const Image<T> &imIn, const int radius, Image<T> &imOut)
  {
    return circleDilate_Soille(imIn, radius, imOut);
  }

  /** @brief circleErode() : the SE is a <b>disk</b> of radius <b>radius</b>
   * pixels
   *
   * @param[in]  imIn : input image
   * @param[in]  radius : the diameter of the disk will be <b>2 * radius + 1</b>
   * @param[out] imOut : output image
   */
  template <class T>
  RES_T circleErode(const Image<T> &imIn, const int radius, Image<T> &imOut)
  {
    return circleErode_Soille(imIn, radius, imOut);
  }

  /** @brief circleOpen() : the SE is a <b>disk</b> of radius <b>radius</b>
   * pixels
   *
   * @param[in]  imIn : input image
   * @param[in]  radius : the diameter of the disk will be <b>2 * radius + 1</b>
   * @param[out] imOut : output image
   */
  template <class T>
  RES_T circleOpen(const Image<T> &imIn, const int radius, Image<T> &imOut)
  {
    return circleOpen_Soille(imIn, radius, imOut);
  }

  /** @brief circleClose() : the SE is a <b>disk</b> of radius <b>radius</b>
   * pixels
   *
   * @param[in]  imIn : input image
   * @param[in]  radius : the diameter of the disk will be <b>2 * radius + 1</b>
   * @param[out] imOut : output image
   */
  template <class T>
  RES_T circleClose(const Image<T> &imIn, const int radius, Image<T> &imOut)
  {
    return circleClose_Soille(imIn, radius, imOut);
  }

  //*************************************************
  // MORARD 'S ALGORITHM
  //*************************************************
  /** @brief imFastLineOpen() : the Structuring Element is a segment of
   * length <b>(2 * hLen + 1)</b> pixels and an orientation <b>angle</b>
   * @param[in]  imIn : input image
   * @param[in]  angle : (in degres)
   * @param[in]  hLen : Half Length of the Structuring Element
   * @param[out] imOut : output image
   */
  template <class T>
  RES_T imFastLineOpen(const Image<T> &imIn, const int angle, const int hLen,
                       Image<T> &imOut);

  /** @brief imFastLineClose() : the Structuring Element is a segment of
   * length <b>(2 * hLen + 1)</b> pixels and an orientation <b>angle</b>
   * @param[in]  imIn : input image
   * @param[in]  angle : (in degres)
   * @param[in]  hLen : Half Length of the Structuring Element
   * @param[out] imOut : output image
   */
  template <class T>
  RES_T imFastLineClose(const Image<T> &imIn, const int angle, const int hLen,
                        Image<T> &imOut)
  {
    RES_T    r;
    Image<T> imTmp(imIn);

    inv(imIn, imTmp);

    r = imFastLineOpen(imTmp, angle, hLen, imOut);
    if (r == RES_OK)
      return inv(imOut, imOut);
    return RES_ERR;
  }

#if 0
   template <class T>
   RES_T ImFastLineClose_Morard(const Image<T> &imIn, const int angle,
                                    const int radius, Image<T> &imOut);
#endif

#if 0
  /** @brief ImFastLineMaxXXX_Morard() : the SE is a segment of a radius
   * <b>radius</b> pixels and an orientation of angle. We take the supremum of the
   * openings
   * @param[in]  imIn : input image
   * @param[in]  nbAngle : nomber of opening (if nbAngle == 90 => every 2
   * degrees)
   * @param[in]  hLen : Half Length of the Structuring Element
   * @param[out] imOut : output image
   */
   template <class T>
   RES_T ImFastLineMaxOpen_Morard(const Image<T> &imIn,
                                      const int nbAngle, const int radius,
                                      Image<T> &imOut);
   template <class T>
   RES_T ImFastLineMaxClose_Morard(const Image<T> &imIn,
                                       const int nbAngle, const int radius,
                                       Image<T> &imOut);
   template <class T>
   RES_T ImFastLineMaxOpenOrientation_Morard(const Image<T> &imIn,
                                                 const int nbAngle,
                                                 const int radius,
                                                 Image<T> &imOut);

  /** @brief ImFastGranulo() : the SE is a <b>disk</b> of radius <b>radius</b> pixels
   *
   * @param[in]  imIn : input image
   * @param[in]  radius : the diameter of the disk will be <b>2 * radius + 1</b>
   * @param[out] imOut : output image
   */
   template <class T>
   RES_T ImFastGranulo(const Image<T> &imIn, const int angle,
                           UINT32 **granulo, int *sizeGranulo);
   template <class T>
   RES_T i_ImFastGranulo(const Image<T> &imIn, const int angle,
                             char *szFileName);
   template <class T>
   RES_T ii_ImFastGranuloBench(const Image<T> &imIn, const int angle);
   template <class T>
   RES_T ImFastRadialGranulo(const Image<T> &imIn, const int nbAngle,
                                 const int sizeMax, Image<T> &imOut);
   template <class T>
   RES_T i_ImFastGranuloAllDir(const Image<T> &imIn,
                                   const int nbAngle, char *szFileName);

#endif

#if 0
  /** @brief ImFastLineMaxXXX() : the SE is a segment of a radius <b>radius</b>
   * pixels and an orientation of angle. We take the supremum of the openings
   *
   * @param[in]  imIn : input image
   * @param[in]  nbAngle : nomber of opening (if nbAngle == 90 => every 2
   * degrees)
   * @param[in]  radius : the size of the segment  will be radius*2+1
   * @param[out] imOut : output image
   */
   template <class T>
   RES_T ImFastLineMaxOpen_Soille(const Image<T> &imIn,
                                      const int nbAngle, const int radius,
                                      Image<T> &imOut);
   template <class T>
   RES_T ImFastLineMaxClose_Soille(const Image<T> &imIn,
                                       const int nbAngle, const int radius,
                                       Image<T> &imOut);

#endif

  /** @} */
} // namespace smil

//#include "FastLine/LineNaive.hpp"
#include "private/FastLine/FastLineMorard.hpp"
#include "private/FastLine/FastLineSoille.hpp"

#endif
