#ifndef __FAST_LINE_HPP__
#define __FAST_LINE_HPP__

#include "Core/include/DCore.h"

namespace smil
{
  /**
   * @ingroup   Advanced
   * @defgroup  AdvLine   Mathematical Morphology with Line
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
   * @param[in]  trigonometric angle
   * @param[in]  radius, the size of the segment  will be radius*2+1
   * @param[out] imOut Result
   */
#if 1
   template <class T>
   RES_T ImFastLineOpen_Morard(const Image<T> &ImIn, const int angle,
                                   const int radius, Image<T> &ImOut);
   template <class T>
   RES_T ImFastLineClose_Morard(const Image<T> &ImIn, const int angle,
                                    const int radius, Image<T> &ImOut);
#endif
#if 0
   template <class T>
   RES_T ImFastLineOpeningH_v2(const Image<T> &ImIn, const int radius,
                                   Image<T> &ImOut);
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
   RES_T ImFastLineMaxOpen_Morard(const Image<T> &ImIn,
                                      const int nbAngle, const int radius,
                                      Image<T> &ImOut);
   template <class T>
   RES_T ImFastLineMaxClose_Morard(const Image<T> &ImIn,
                                       const int nbAngle, const int radius,
                                       Image<T> &ImOut);
   template <class T>
   RES_T ImFastLineMaxOpenOrientation_Morard(const Image<T> &ImIn,
                                                 const int nbAngle,
                                                 const int radius,
                                                 Image<T> &ImOut);

  /*! @brief ImFastGranulo : the SE is a "disk" of radius "radius" pixels
   * (SOILLE 'S ALGORITHM)
   * @param[in]  imIn the initial image
   * @param[in]  radius, the diameter of the ball will be radius*2+1
   * @param[out] imOut Result
   */
   template <class T>
   RES_T ImFastGranulo(const Image<T> &ImIn, const int angle,
                           UINT32 **granulo, int *sizeGranulo);
   template <class T>
   RES_T i_ImFastGranulo(const Image<T> &ImIn, const int angle,
                             char *szFileName);
   template <class T>
   RES_T ii_ImFastGranuloBench(const Image<T> &ImIn, const int angle);
   template <class T>
   RES_T ImFastRadialGranulo(const Image<T> &ImIn, const int nbAngle,
                                 const int sizeMax, Image<T> &ImOut);
   template <class T>
   RES_T i_ImFastGranuloAllDir(const Image<T> &ImIn,
                                   const int nbAngle, char *szFileName);

#endif

  /** @} */
} // namespace smil

// #include "FastLine/LineNaive.hpp"
#include "FastLine/FastLineMorard.hpp"

#endif
