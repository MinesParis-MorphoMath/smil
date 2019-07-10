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
   * @param[in]  radius the size of the segment  will be radius*2+1
   * @param[out] imOut Result
   */
  template <class T>
  RES_T ImSquareErode(const Image<T> &imIn, const int radius, Image<T> &imOut)
  {
    return ImSquareErode_Soille(imIn, radius, imOut);
#if 0   
    Image<T> imTmp(imIn);

    RES_T res = ImLineErode(imIn, 0, radius, imTmp);
    if (res == RES_OK)
      res = ImLineErode(imTmp, 90, radius, imOut);
    return res;
#endif
  }

  /**
   * @brief ImSquareDilate : the SE is a segment of radius "radius" pixels
   * @param[in]  imIn the initial image
   * @param[in]  radius the size of the segment  will be radius*2+1
   * @param[out] imOut Result
   */
  template <class T>
  RES_T ImSquareDilate(const Image<T> &imIn, const int radius, Image<T> &imOut)
  {
    return ImSquareDilate_Soille(imIn, radius, imOut);
#if 0
    Image<T> imTmp(imIn);

    RES_T res = ImLineDilate(imIn, 0, radius, imTmp);
    if (res == RES_OK)
      res = ImLineDilate(imTmp, 90, radius, imOut);
    return res;
#endif
  }

  /**
   * @brief ImSquareOpen : the SE is a segment of radius "radius" pixels
   * @param[in]  imIn the initial image
   * @param[in]  radius the size of the segment  will be radius*2+1
   * @param[out] imOut Result
   */
  template <class T>
  RES_T ImSquareOpen(const Image<T> &imIn, const int radius, Image<T> &imOut)
  {
    return ImSquareOpen_Soille(imIn, radius, imOut);
#if 0
    Image<T> imTmp(imIn);

    RES_T res = ImLineErode(imIn, 0, radius, imTmp);
    if (res == RES_OK)
      res = ImLineErode(imTmp, 90, radius, imOut);
    if (res == RES_OK)
      res = ImLineDilate(imOut, 0, radius, imTmp);
    if (res == RES_OK)
      res = ImLineDilate(imTmp, 90, radius, imOut);
    return res;
#endif
  }  

  /** 
   * @brief ImSquareClose : the SE is a segment of radius "radius" pixels
   * @param[in]  imIn the initial image
   * @param[in]  radius the size of the segment  will be radius*2+1
   * @param[out] imOut Result
   */
  template <class T>
  RES_T ImSquareClose(const Image<T> &imIn, const int radius, Image<T> &imOut)
  {
    return ImSquareClose_Soille(imIn, radius, imOut);
#if 0
    Image<T> imTmp(imIn);

    RES_T res = ImLineDilate(imIn, 0, radius, imTmp);
    if (res == RES_OK)
      res = ImLineDilate(imTmp, 90, radius, imOut);
    if (res == RES_OK)
      res = ImLineErode(imOut, 0, radius, imTmp);
    if (res == RES_OK)
      res = ImLineErode(imTmp, 90, radius, imOut);
    return res;
#endif
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
