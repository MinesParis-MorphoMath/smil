#ifndef __FAST_LINE_MORPHO_HPP__
#define __FAST_LINE_MORPHO_HPP__

#include "Core/include/DCore.h"
#include "Morpho/include/DMorpho.h"

namespace smil

// February 23, 2006  Erik R. Urbach
// Email: erik@cs.rug.nl
// Implementation of algorithm by Soille et al. [1] for erosions and
// dilations with linear structuring elements (S.E.) at arbitrary angles.
// S.E. line drawing using Bresenham's Line Algorithm [2].
// Compilation: gcc -ansi -pedantic -Wall -O3 -o polygonsoille polygonsoille.c
// -lm
//
// Related papers:
// [1] P. Soille and E. Breen and R. Jones.
//     Recursive implementation of erosions and dilations along discrete
//     lines at arbitrary angles.
//     IEEE Transactions on Pattern Analysis and Machine Intelligence,
//     Vol. 18, Number 5, Pages 562-567, May 1996.
// [2] Donald Hearn and M. Pauline Baker
//     Computer Graphics, second edition
//     Prentice Hall

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
   * @warning Circle morphological operations based on line segments use
   * an iteractive algorithm. The circle diameter may not be exactly as
   * expected. Most of the time it's always inside a 10 % interval.
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
                   double theta = 0, double zeta = 0)
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
                  double theta = 0, double zeta = 0)
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
                 double theta = 0, double zeta = 0)
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
                  double theta = 0, double zeta = 0)
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

    int    hLen = (side + side % 2) / 2;
    StrElt se;
    se = LineSE(hLen, 0);
    se = merge(se, se.transpose());

    RES_T r = dilate(imIn, imOut, se);
    if (r == RES_OK) {
      se = LineSE(hLen, PI / 2);
      se = merge(se, se.transpose());

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

    int    hLen = (side + side % 2) / 2;
    StrElt se;
    se = LineSE(hLen, 0);
    se = merge(se, se.transpose());

    RES_T r = erode(imIn, imOut, se);
    if (r == RES_OK) {
      se = LineSE(hLen, PI / 2);
      se = merge(se, se.transpose());

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
  /** @cond */
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
  /** @endcond */

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

    int    _NB_STEPS = getAngleSteps(radius);
    double _D_ANGLE  = (PI / _NB_STEPS);

    RES_T  r  = RES_OK;
    double k0 = (radius * PI / _NB_STEPS * 0.5);
    for (double angle = 0; angle < PI && r == RES_OK; angle += _D_ANGLE) {
      double rd      = angle;
      int    kradius = k0 * std::max(fabs(cos(rd)), fabs(sin(rd))) + 1;

      StrElt se = LineSE(kradius, angle);
      se        = merge(se, se.transpose());

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

    int    _NB_STEPS = getAngleSteps(radius);
    double _D_ANGLE  = (PI / _NB_STEPS);

    RES_T  r  = RES_OK;
    double k0 = (radius * PI / _NB_STEPS * 0.5);
    for (double angle = 0; angle < PI && r == RES_OK; angle += _D_ANGLE) {
      double rd      = angle;
      int    kradius = k0 * std::max(fabs(cos(rd)), fabs(sin(rd))) + 1;

      StrElt se = LineSE(kradius, angle);
      se        = merge(se, se.transpose());

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
   * #####   ######   ####    #####    ##    #    #   ####   #       ######
   * #    #  #       #    #     #     #  #   ##   #  #    #  #       #
   * #    #  #####   #          #    #    #  # #  #  #       #       #####
   * #####   #       #          #    ######  #  # #  #  ###  #       #
   * #   #   #       #    #     #    #    #  #   ##  #    #  #       #
   * #    #  ######   ####      #    #    #  #    #   ####   ######  ######
   */
  /** @brief rectangleDilate() : generic dilation of imIn by a rectangle of
   * sides side1 and side2, rotated by an angle theta.
   *
   * @param[in]  imIn : input image
   * @param[out] imOut : output image
   * @param[in]  side1 : rectangle side
   * @param[in]  side2 : rectangle side
   * @param[in]  theta : angle between side1 and horizontal axis
   */
  template <class T>
  RES_T rectangleDilate(const Image<T> &imIn, Image<T> &imOut, int side1,
                        int side2, double theta = 0)
  {
    StrElt se1 = CenteredLineSE(side1, theta);
    StrElt se2 = CenteredLineSE(side2, theta + PI / 2);

    RES_T r = dilate(imIn, imOut, se1);
    if (r == RES_OK)
      r = dilate(imOut, imOut, se2);
    if (r == RES_OK)
      r = close(imOut, imOut, CrossSE(1));
    return r;
  }

  /** @brief rectangleErode() : generic erosion of imIn by a rectangle of
   * sides side1 and side2, rotated by an angle theta.
   *
   * @param[in]  imIn : input image
   * @param[out] imOut : output image
   * @param[in]  side1 : rectangle side
   * @param[in]  side2 : rectangle side
   * @param[in]  theta : angle between side1 and horizontal axis
   */
  template <class T>
  RES_T rectangleErode(const Image<T> &imIn, Image<T> &imOut, int side1,
                       int side2, double theta = 0)
  {
    StrElt se1 = CenteredLineSE(side1, theta);
    StrElt se2 = CenteredLineSE(side2, theta + PI / 2);

    RES_T r = erode(imIn, imOut, se1);
    if (r == RES_OK)
      r = erode(imOut, imOut, se2);

    return r;
  }

  /** @brief rectangleOpen() : generic opening of imIn by a rectangle of
   * sides side1 and side2, rotated by an angle theta.
   *
   * @param[in]  imIn : input image
   * @param[out] imOut : output image
   * @param[in]  side1 : rectangle side
   * @param[in]  side2 : rectangle side
   * @param[in]  theta : angle between side1 and horizontal axis
   */
  template <class T>
  RES_T rectangleOpen(const Image<T> &imIn, Image<T> &imOut, int side1,
                      int side2, double theta = 0)
  {
    RES_T r = rectangleErode(imIn, imOut, side1, side2, theta);
    if (r == RES_OK)
      r = rectangleDilate(imOut, imOut, side1, side2, theta);

    return r;
  }

  /** @brief rectangleClose() : generic closing of imIn by a rectangle of
   * sides side1 and side2, rotated by an angle theta.
   *
   * @param[in]  imIn : input image
   * @param[out] imOut : output image
   * @param[in]  side1 : rectangle side
   * @param[in]  side2 : rectangle side
   * @param[in]  theta : angle between side1 and horizontal axis
   */
  template <class T>
  RES_T rectangleClose(const Image<T> &imIn, Image<T> &imOut, int side1,
                       int side2, double theta = 0)
  {
    RES_T r = rectangleDilate(imIn, imOut, side1, side2, theta);
    if (r == RES_OK)
      r = rectangleErode(imOut, imOut, side1, side2, theta);

    return r;
  }

  /** @cond */
  template <class T>
  RES_T XsquareDilate(const Image<T> &imIn, Image<T> &imOut, int side)
  {
    return rectangleDilate(imIn, imOut, side, side, 0);
  }
  template <class T>
  RES_T XsquareErode(const Image<T> &imIn, Image<T> &imOut, int side)
  {
    return rectangleErode(imIn, imOut, side, side, 0);
  }
  template <class T>
  RES_T XsquareOpen(const Image<T> &imIn, Image<T> &imOut, int side)
  {
    return rectangleOpen(imIn, imOut, side, side, 0);
  }
  template <class T>
  RES_T XsquareClose(const Image<T> &imIn, Image<T> &imOut, int side)
  {
    return rectangleClose(imIn, imOut, side, side, 0);
  }
  /** @endcond */

  /** @} */
} // namespace smil

#endif
