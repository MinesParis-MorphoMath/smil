#ifndef __FAST_LINE_MORPHO_HPP__
#define __FAST_LINE_MORPHO_HPP__

#include "Core/include/DCore.h"
#include "Morpho/include/DMorpho.h"

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
   * 3D Line Structuring Element using Bresenham's Line Drawing Algorithm.
   *
   *
   *
   * @{ */

  /*
   *   #          #    #    #  ######
   *   #          #    ##   #  #
   *   #          #    # #  #  #####
   *   #          #    #  # #  #
   *   #          #    #   ##  #
   *   ######     #    #    #  ######
   */
  /** @brief lineDilate()
   *
   * The @TB{3D Structuring Element} is a segment of length
   * @Math{(2 \: . \: hLen + 1)} and an orientation given by
   * angles @Math{\theta} and @Math{\zeta} (in radians).
   *
   * @param[in]  imIn : input image
   * @param[in]  hLen : Half Length of the Structuring Element
   * @param[in]  theta : angle (in radians) of the line in the @TB{h x v} plane
   * @param[in]  zeta : elevation angle (in radians)
   * @param[out] imOut : output image
   */
  template <class T>
  RES_T lineDilate(const Image<T> &imIn, Image<T> &imOut, int hLen,
                    double theta, double zeta = 0)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    StrElt se1 = Line3DSE(hLen, theta, zeta);
    StrElt se2 = Line3DSE(hLen, theta + PI, zeta + PI);
    StrElt se  = merge(se1, se2);

    return dilate(imIn, imOut, se);
  }

  /** @brief lineErode()
   *
   * The @TB{3D Structuring Element} is a segment of length
   * @Math{(2 \: . \: hLen + 1)} and an orientation given by
   * angles @Math{\theta} and @Math{\zeta} (in radians).
   *
   * @param[in]  imIn : input image
   * @param[in]  hLen : Half Length of the Structuring Element
   * @param[in]  theta : angle (in radians) of the line in the @TB{h x v} plane
   * @param[in]  zeta : elevation angle (in radians)
   * @param[out] imOut : output image
   */
  template <class T>
  RES_T lineErode(const Image<T> &imIn, Image<T> &imOut, int hLen,
                   double theta, double zeta = 0)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    StrElt se1 = Line3DSE(hLen, theta, zeta);
    StrElt se2 = Line3DSE(hLen, theta + PI, zeta + PI);
    StrElt se  = merge(se1, se2);

    return dilate(imIn, imOut, se);
  }

  /** @brief lineOpen()
   *
   * The @TB{3D Structuring Element} is a segment of length
   * @Math{(2 \: . \: hLen + 1)} and an orientation given by
   * angles @Math{\theta} and @Math{\zeta} (in radians).
   *
   * @param[in]  imIn : input image
   * @param[in]  hLen : Half Length of the Structuring Element
   * @param[in]  theta : angle (in radians) of the line in the @TB{h x v} plane
   * @param[in]  zeta : elevation angle (in radians)
   * @param[out] imOut : output image
   */
  template <class T>
  RES_T lineOpen(const Image<T> &imIn, Image<T> &imOut, int hLen,
                  double theta, double zeta = 0)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    StrElt se1 = Line3DSE(hLen, theta, zeta);
    StrElt se2 = Line3DSE(hLen, theta + PI, zeta + PI);
    StrElt se  = merge(se1, se2);

    RES_T r = erode(imIn, imOut, se);
    if (r == RES_OK)
      return dilate(imOut, imOut, se);
    return r;
  }

  /** @brief lineClose()
   *
   * The @TB{3D Structuring Element} is a segment of length
   * @Math{(2 \: . \: hLen + 1)} and an orientation given by
   * angles @Math{\theta} and @Math{\zeta} (in radians).
   *
   * @param[in]  imIn : input image
   * @param[in]  hLen : Half Length of the Structuring Element
   * @param[in]  theta : angle (in radians) of the line in the @TB{h x v} plane
   * @param[in]  zeta : elevation angle (in radians)
   * @param[out] imOut : output image
   */
  template <class T>
  RES_T lineClose(const Image<T> &imIn, Image<T> &imOut, int hLen,
                   double theta, double zeta = 0)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    StrElt se1 = Line3DSE(hLen, theta, zeta);
    StrElt se2 = Line3DSE(hLen, theta + PI, zeta + PI);
    StrElt se  = merge(se1, se2);

    RES_T r = dilate(imIn, imOut, se);
    if (r == RES_OK)
      return erode(imOut, imOut, se);
    return r;
  }

  /*
   *    ####    ####   #    #    ##    #####   ######
   *   #       #    #  #    #   #  #   #    #  #
   *    ####   #    #  #    #  #    #  #    #  #####
   *        #  #  # #  #    #  ######  #####   #
   *   #    #  #   #   #    #  #    #  #   #   #
   *    ####    ### #   ####   #    #  #    #  ######
   */
  /**
   * @brief squareDilate() : the SE is a square defined by its side
   *
   * @param[in]  imIn : input image
   * @param[out] imOut : output image
   * @param[in]  side : side of the Square Structuring Element
   *
   * @note
   * - in @TB{3D} images, the same square S.E. is applied to each slice.
   */
  template <class T>
  RES_T squareDilate(const Image<T> &imIn, Image<T> &imOut, int side)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    int hLen = (side + side % 2) / 2;
    StrElt se;
    se  = LineSE(hLen, 0);
    se  = merge(se, se.transpose());

    RES_T r = dilate(imIn, imOut, se);
    if (r == RES_OK)
    {
      se  = LineSE(hLen, 90);
      se  = merge(se, se.transpose());

      return dilate(imOut, imOut, se);
    }
    return r;
  }

  /**
   * @brief squareErode() : the SE is a square defined by its side
   *
   * @param[in]  imIn : input image
   * @param[out] imOut : output image
   * @param[in]  side : side of the Square Structuring Element
   *
   * @note
   * - in @TB{3D} images, the same square S.E. is applied to each slice.
   */
  template <class T>
  RES_T squareErode(const Image<T> &imIn, Image<T> &imOut, int side)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    int hLen = (side + side % 2) / 2;
    StrElt se;
    se  = LineSE(hLen, 0);
    se  = merge(se, se.transpose());

    RES_T r = erode(imIn, imOut, se);
    if (r == RES_OK)
    {
      se  = LineSE(hLen, 90);
      se  = merge(se, se.transpose());

      return erode(imOut, imOut, se);
    }
    return r;
  }

  /**
   * @brief squareOpen() : the SE is a square defined by its side
   *
   * @param[in]  imIn : input image
   * @param[out] imOut : output image
   * @param[in]  side : side of the Square Structuring Element
   *
   * @note
   * - in @TB{3D} images, the same square S.E. is applied to each slice.
   */
  template <class T>
  RES_T squareOpen(const Image<T> &imIn, Image<T> &imOut, int side)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    RES_T r = squareErode(imIn, imOut, side);
    if (r == RES_OK)
      return squareDilate(imOut, imOut, side);
    return r;
  }

  /**
   * @brief squareClose() : the SE is a square defined by its side
   *
   * @param[in]  imIn : input image
   * @param[out] imOut : output image
   * @param[in]  side : side of the Square Structuring Element
   *
   * @note
   * - in @TB{3D} images, the same square S.E. is applied to each slice.
   */
  template <class T>
  RES_T squareClose(const Image<T> &imIn, Image<T> &imOut, int side)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    RES_T r = squareDilate(imIn, imOut, side);
    if (r == RES_OK)
      return squareErode(imOut, imOut, side);
    return r;
  }

  /*
   *    ####      #    #####    ####   #       ######
   *   #    #     #    #    #  #    #  #       #
   *   #          #    #    #  #       #       #####
   *   #          #    #####   #       #       #
   *   #    #     #    #   #   #    #  #       #
   *    ####      #    #    #   ####   ######  ######
   */
  int getAngleSteps(int radius)
  {
    if (radius < 10)
      return 4;
    if (radius < 24)
      return 8;
    if (radius < 50)
      return 12;
    if (radius < 100)
      return 16;
    if (radius < 224)
      return 32;
    return 48;
  }

  /** @brief circleDilate() : the SE is a @TB{disk} of radius @TB{radius}
   * pixels
   *
   * @param[in]  imIn : input image
   * @param[out] imOut : output image
   * @param[in]  radius : the radius of the disk
   *
   * @note
   * - in @TB{3D} images, the same disk S.E. is applied to each slice.
   */
  template <class T>
  RES_T circleDilate(const Image<T> &imIn, Image<T> &imOut, int radius)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    copy(imIn, imOut);

    int _NB_STEPS = getAngleSteps(radius);
    double _D_ANGLE = (PI / _NB_STEPS);

    RES_T r = RES_OK;
    double k0 = (radius * PI / _NB_STEPS * 0.5);
    for (double angle = 0; angle < PI && r == RES_OK; angle += _D_ANGLE) {
      double rd = angle;
      int kradius = k0 * max(fabs(cos(rd)), fabs(sin(rd))) + 1;

      StrElt se  = LineSE(kradius, angle);
      se  = merge(se, se.transpose());

      r = dilate(imOut, imOut, se);
    }
    return r;
  }

  /** @brief circleErode() : the SE is a @TB{disk} of radius @TB{radius}
   * pixels
   *
   * @param[in]  imIn : input image
   * @param[out] imOut : output image
   * @param[in]  radius : the radius of the disk
   *
   * @note
   * - in @TB{3D} images, the same disk S.E. is applied to each slice.
   */
  template <class T>
  RES_T circleErode(const Image<T> &imIn, Image<T> &imOut, int radius)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    copy(imIn, imOut);

    int _NB_STEPS = getAngleSteps(radius);
    double _D_ANGLE = (PI / _NB_STEPS);

    RES_T r = RES_OK;
    double k0 = (radius * PI / _NB_STEPS * 0.5);
    for (double angle = 0; angle < PI && r == RES_OK; angle += _D_ANGLE) {
      double rd = angle;
      int kradius = k0 * max(fabs(cos(rd)), fabs(sin(rd))) + 1;

      StrElt se  = LineSE(kradius, angle);
      se  = merge(se, se.transpose());

      r = erode(imOut, imOut, se);
    }
    return r;
  }

  /** @brief circleOpen() : the SE is a @TB{disk} of radius @TB{radius}
   * pixels
   *
   * @param[in]  imIn : input image
   * @param[out] imOut : output image
   * @param[in]  radius : the radius of the disk
   *
   * @note
   * - in @TB{3D} images, the same disk S.E. is applied to each slice.
   */
  template <class T>
  RES_T circleOpen(const Image<T> &imIn, Image<T> &imOut, int radius)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    RES_T r = circleErode(imIn, imOut, radius);
    if (r == RES_OK)
      return circleDilate(imOut, imOut, radius);
    return r;
  }

  /** @brief circleClose() : the SE is a @TB{disk} of radius @TB{radius}
   * pixels
   *
   * @param[in]  imIn : input image
   * @param[out] imOut : output image
   * @param[in]  radius : the radius of the disk
   *
   * @note
   * - in @TB{3D} images, the same disk S.E. is applied to each slice.
   */
  template <class T>
  RES_T circleClose(const Image<T> &imIn, Image<T> &imOut, int radius)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    RES_T r = circleDilate(imIn, imOut, radius);
    if (r == RES_OK)
      return circleErode(imOut, imOut, radius);
    return r;
  }

  #undef _D_ANGLE

  /*
   *    ####   #    #  #####   ######
   *   #    #  #    #  #    #  #
   *   #       #    #  #####   #####
   *   #       #    #  #    #  #
   *   #    #  #    #  #    #  #
   *    ####    ####   #####   ######
   */


  /*
   *    ####   #####   #    #  ######  #####   ######
   *   #       #    #  #    #  #       #    #  #
   *    ####   #    #  ######  #####   #    #  #####
   *        #  #####   #    #  #       #####   #
   *   #    #  #       #    #  #       #   #   #
   *    ####   #       #    #  ######  #    #  ######
   */

  /** @} */
} // namespace smil

#endif
