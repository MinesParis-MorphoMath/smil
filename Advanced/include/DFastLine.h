#ifndef __FAST_LINE_HPP__
#define __FAST_LINE_HPP__

#include "Core/include/DCore.h"

namespace smil
{
  /**
   * @ingroup   Advanced
   * @defgroup  AdvLine   Line Based Operators
   *
   * 
   * @brief Based on Erik R Urbach (2006)
   * implementation of algorithm by Soille et al [1] for erosions and
   * dilations with linear structuring elements (S.E.) at arbitrary angles.
   * S.E. line drawing using Bresenham's Line Algorithm [2].
   *
   * @par Related papers:
   *
   * [1] P. Soille and E. Breen and R. Jones.
   *     Recursive implementation of erosions and dilations along discrete
   *     lines at arbitrary angles.
   *     IEEE Transactions on Pattern Analysis and Machine Intelligence,
   *     Vol. 18, Number 5, Pages 562-567, May 1996.
   *
   * [2] Donald Hearn and M. Pauline Baker
   *     Computer Graphics, second edition
   *     Prentice Hall
   *
   * @author Vincent Morard, Jose-Marcio Martins da Cruz
   * @date 1st September 2010, 8 July 2019 
   *
   * @{ */

  // author Vincent Morard, Jose-Marcio Martins da Cruz
  // date   1 septembre 2010
  // Update  21 octobre 2010


  //*************************************************
  // MORARD 'S ALGORITHM
  //*************************************************
  /*! @brief ImFastLineXXX_Morard : the SE is a segment of radius "radius"
   * pixels and an orientation of angle
   * @param[in]  imIn the initial image
   * @param[in]  angle (in degres)
   * @param[in]  radius the size of the segment  will be radius*2+1
   * @param[out] imOut Result
   */
#if 1
   template <class T>
   RES_T ImFastLineOpen_Morard(const Image<T> &imIn, const int angle,
                                   const int radius, Image<T> &imOut);
   template <class T>
   RES_T ImFastLineClose_Morard(const Image<T> &imIn, const int angle,
                                    const int radius, Image<T> &imOut);
#endif
#if 0
   template <class T>
   RES_T ImFastLineOpeningH_v2(const Image<T> &imIn, const int radius,
                                   Image<T> &imOut);
#endif

#if 0
  /*! @brief ImFastLineMaxXXX_Morard : the SE is a segment of a radius
   * "radius" pixels and an orientation of angle. We take the supremum of the
   * openings
   * @param[in]  imIn the initial image
   * @param[in]  nbAngle : nomber of opening (if nbAngle == 90 => every 2
   * degrees)
   * @param[in]  radius, the size of the segment  will be radius*2+1
   * @param[out] imOut Result
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

  /*! @brief ImFastGranulo : the SE is a "disk" of radius "radius" pixels
   * (SOILLE 'S ALGORITHM)
   * @param[in]  imIn the initial image
   * @param[in]  radius, the diameter of the ball will be radius*2+1
   * @param[out] imOut Result
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


#if 1
  //*************************************************
  // SOILLE 'S ALGORITHM
  //*************************************************

  /** @brief ImLineDilate : the SE is a segment of radius "radius" pixels
   * and an orientation of angle (SOILLE 'S ALGORITHM)
   * @param[in]  imIn the initial image
   * @param[in]  angle (in degres)
   * @param[in]  radius the size of the segment  will be radius*2+1
   * @param[out] imOut Result
   */
   template <class T>
   RES_T ImLineDilate(const Image<T> &imIn,
                                     const int angle, const int radius,
                                     Image<T> &imOut)
   {
     return ImLineDilate_Soille(imIn, angle, radius, imOut);
   }                                     

  /** @brief ImLineErode : the SE is a segment of radius "radius" pixels
   * and an orientation of angle (SOILLE 'S ALGORITHM)
   * @param[in]  imIn the initial image
   * @param[in]  angle (in degres)
   * @param[in]  radius the size of the segment  will be radius*2+1
   * @param[out] imOut Result
   */
   template <class T>
   RES_T ImLineErode(const Image<T> &imIn, const int angle,
                                    const int radius, Image<T> &imOut)
   {
     return ImLineErode_Soille(imIn, angle, radius, imOut);
   }                                     

  /** @brief ImLineOpen : the SE is a segment of radius "radius" pixels
   * and an orientation of angle (SOILLE 'S ALGORITHM)
   * @param[in]  imIn the initial image
   * @param[in]  angle (in degres)
   * @param[in]  radius  the size of the segment  will be radius*2+1
   * @param[out] imOut Result
   */
   template <class T>
   RES_T ImLineOpen(const Image<T> &imIn, const int angle,
                                   const int radius, Image<T> &imOut)
   {
     return ImLineOpen_Soille(imIn, angle, radius, imOut);
   }                                     

  /** @brief ImLineClose : the SE is a segment of radius "radius" pixels
   * and an orientation of angle (SOILLE 'S ALGORITHM)
   * @param[in]  imIn the initial image
   * @param[in]  angle (in degres)
   * @param[in]  radius  the size of the segment  will be radius*2+1
   * @param[out] imOut Result
   */
   template <class T>
   RES_T ImLineClose(const Image<T> &imIn, const int angle,
                                    const int radius, Image<T> &imOut)
   {
     return ImLineClose_Soille(imIn, angle, radius, imOut);
   }                                     

#endif

#if 1
  /**
   * @brief ImSquareErode : the SE is a segment of radius "radius" pixels
   * @param[in]  imIn the initial image
   * @param[in]  radius the size of the square side will be radius*2+1
   * @param[out] imOut Result
   */
  template <class T>
  RES_T ImSquareErode(const Image<T> &imIn, const int radius, Image<T> &imOut)
  {
    return ImSquareErode_Soille(imIn, radius, imOut);
  }

  /**
   * @brief ImSquareDilate : the SE is a segment of radius "radius" pixels
   * @param[in]  imIn the initial image
   * @param[in]  radius the size of the square side  will be radius*2+1
   * @param[out] imOut Result
   */
  template <class T>
  RES_T ImSquareDilate(const Image<T> &imIn, const int radius, Image<T> &imOut)
  {
    return ImSquareDilate_Soille(imIn, radius, imOut);
  }

  /**
   * @brief ImSquareOpen : the SE is a segment of radius "radius" pixels
   * @param[in]  imIn the initial image
   * @param[in]  radius the size of the square side  will be radius*2+1
   * @param[out] imOut Result
   */
  template <class T>
  RES_T ImSquareOpen(const Image<T> &imIn, const int radius, Image<T> &imOut)
  {
    return ImSquareOpen_Soille(imIn, radius, imOut);
  }  

  /** 
   * @brief ImSquareClose : the SE is a segment of radius "radius" pixels
   * @param[in]  imIn the initial image
   * @param[in]  radius the size of the square side  will be radius*2+1
   * @param[out] imOut Result
   */
  template <class T>
  RES_T ImSquareClose(const Image<T> &imIn, const int radius, Image<T> &imOut)
  {
    return ImSquareClose_Soille(imIn, radius, imOut);
  }

#endif

#if 0
  /*! @brief ImFastLineMaxXXX : the SE is a segment of a radius "radius"
   * pixels and an orientation of angle. We take the supremum of the openings
   * (SOILLE 'S ALGORITHM)
   * @param[in]  imIn the initial image
   * @param[in]  nbAngle : nomber of opening (if nbAngle == 90 => every 2
   * degrees)
   * @param[in]  radius, the size of the segment  will be radius*2+1
   * @param[out] imOut Result
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
#if 1
  /*! @brief ImCircleDilate : the SE is a "disk" of radius "radius" pixels
   * (SOILLE 'S ALGORITHM)
   * @param[in]  imIn the initial image
   * @param[in]  radius the diameter of the ball will be radius*2+1
   * @param[out] imOut Result
   */
   template <class T>
   RES_T ImCircleDilate(const Image<T> &imIn,
                                       const int radius, Image<T> &imOut)
   {
     return ImCircleDilate_Soille(imIn, radius, imOut);
   }                                       

  /*! @brief ImCircleErode : the SE is a "disk" of radius "radius" pixels
   * (SOILLE 'S ALGORITHM)
   * @param[in]  imIn the initial image
   * @param[in]  radius the diameter of the ball will be radius*2+1
   * @param[out] imOut Result
   */
   template <class T>
   RES_T ImCircleErode(const Image<T> &imIn,
                                      const int radius, Image<T> &imOut)
   {
     return ImCircleErode_Soille(imIn, radius, imOut);
   }                                       

  /*! @brief ImCircleOpen : the SE is a "disk" of radius "radius" pixels
   * (SOILLE 'S ALGORITHM)
   * @param[in]  imIn the initial image
   * @param[in]  radius the diameter of the ball will be radius*2+1
   * @param[out] imOut Result
   */
   template <class T>
   RES_T ImCircleOpen(const Image<T> &imIn,
                                     const int radius, Image<T> &imOut)
   {
     return ImCircleOpen_Soille(imIn, radius, imOut);
   }                                       

  /*! @brief ImCircleClose : the SE is a "disk" of radius "radius" pixels
   * (SOILLE 'S ALGORITHM)
   * @param[in]  imIn the initial image
   * @param[in]  radius the diameter of the ball will be radius*2+1
   * @param[out] imOut Result
   */
   template <class T>
   RES_T ImCircleClose(const Image<T> &imIn,
                                      const int radius, Image<T> &imOut)
   {
     return ImCircleClose_Soille(imIn, radius, imOut);
   }                                       

#endif

  /** @} */
} // namespace smil

//#include "FastLine/LineNaive.hpp"
#include "FastLine/FastLineMorard.hpp"
#include "FastLine/FastLineSoille.hpp"

#endif
