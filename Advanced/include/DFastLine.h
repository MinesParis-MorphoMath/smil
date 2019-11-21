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
   * @brief Implementation of the algorithm by Soille et al
   * @cite SoilleBJ96 @cite Soille_2003 for erosions and
   * dilations with linear Structuring Elements (S.E.) and arbitrary angles.
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
  // MORARD 'S ALGORITHM
  //*************************************************
  /** @brief ImFastLineXXX_Morard : the Structuring Element is a segment of 
   * length <b>(2 * hLen + 1)</b> pixels and an orientation <b>angle</b>
   * @param[in]  imIn the initial image
   * @param[in]  angle (in degres)
   * @param[in]  hLen Half Length of the Structuring Element
   * @param[out] imOut Result
   */

   template <class T>
   RES_T ImFastLineOpen_Morard(const Image<T> &imIn, const int angle,
                                   const int hLen, Image<T> &imOut);

#if 0
   template <class T>
   RES_T ImFastLineClose_Morard(const Image<T> &imIn, const int angle,
                                    const int radius, Image<T> &imOut);
#endif                                    

#if 0
  /** @brief ImFastLineMaxXXX_Morard : the SE is a segment of a radius
   * <b>radius</b> pixels and an orientation of angle. We take the supremum of the
   * openings
   * @param[in]  imIn the initial image
   * @param[in]  nbAngle : nomber of opening (if nbAngle == 90 => every 2
   * degrees)
   * @param[in]  hLen Half Length of the Structuring Element
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

  /** @brief ImFastGranulo : the SE is a <b>disk</b> of radius <b>radius</b> pixels
   *
   * @param[in]  imIn the initial image
   * @param[in]  radius, the diameter of the disk will be <b>2 * radius + 1</b>
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


  //*************************************************
  // SOILLE 'S ALGORITHM
  //*************************************************

  /** @brief ImLineDilate : the Structuring Element is a segment of length 
   * <b>(2 * hLen + 1)</b> pixels and an orientation of <b>angle</b> degrees
   * @param[in]  imIn the initial image
   * @param[in]  angle (in degres)
   * @param[in]  hLen Half Length of the Structuring Element
   * @param[out] imOut Result
   */
   template <class T>
   RES_T ImLineDilate(const Image<T> &imIn,
                                     const int angle, const int hLen,
                                     Image<T> &imOut)
   {
     return ImLineDilate_Soille(imIn, angle, hLen, imOut);
   }                                     

  /** @brief ImLineErode : the Structuring Element is a segment of length 
   * <b>(2 * hLen + 1)</b> pixels and an orientation <b>angle</b> degrees
   * @param[in]  imIn the initial image
   * @param[in]  angle (in degres)
   * @param[in]  hLen Half Length of the Structuring Element
   * @param[out] imOut Result
   */
   template <class T>
   RES_T ImLineErode(const Image<T> &imIn, const int angle,
                                    const int hLen, Image<T> &imOut)
   {
     return ImLineErode_Soille(imIn, angle, hLen, imOut);
   }                                     

  /** @brief ImLineOpen : the Structuring Element is a segment of length 
   * <b>(2 * hLen + 1)</b> pixels and an orientation of <b>angle</b> degrees
   * @param[in]  imIn the initial image
   * @param[in]  angle (in degres)
   * @param[in]  hLen Half Length of the Structuring Element
   * @param[out] imOut Result
   */
   template <class T>
   RES_T ImLineOpen(const Image<T> &imIn, const int angle,
                                   const int hLen, Image<T> &imOut)
   {
     return ImLineOpen_Soille(imIn, angle, hLen, imOut);
   }                                     

  /** @brief ImLineClose : the Structuring Element is a segment of length 
   * <b>(2 * hLen + 1)</b> pixels and an orientation of <b>angle</b> degrees
   * @param[in]  imIn the initial image
   * @param[in]  angle (in degres)
   * @param[in]  hLen Half Length of the Structuring Element
   * @param[out] imOut Result
   */
   template <class T>
   RES_T ImLineClose(const Image<T> &imIn, const int angle,
                                    const int hLen, Image<T> &imOut)
   {
     return ImLineClose_Soille(imIn, angle, hLen, imOut);
   }                                     

  /**
   * @brief ImSquareErode : the SE is a square which side is <b>(2 * hSide + 1)</b>
   * @param[in]  imIn the initial image
   * @param[in]  hSide Half side of the Structuring Element
   * @param[out] imOut Result
   */
  template <class T>
  RES_T ImSquareErode(const Image<T> &imIn, const int hSide, Image<T> &imOut)
  {
    return ImSquareErode_Soille(imIn, hSide, imOut);
  }

  /**
   * @brief ImSquareDilate : the SE is a square which side is <b>(2 * hSide + 1)</b>
   * @param[in]  imIn the initial image
   * @param[in]  hSide Half side of the Structuring Element
   * @param[out] imOut Result
   */
  template <class T>
  RES_T ImSquareDilate(const Image<T> &imIn, const int hSide, Image<T> &imOut)
  {
    return ImSquareDilate_Soille(imIn, hSide, imOut);
  }

  /**
   * @brief ImSquareOpen : the SE is a square which side is <b>(2 * hSide + 1)</b>
   * @param[in]  imIn the initial image
   * @param[in]  hSide Half side of the Structuring Element
   * @param[out] imOut Result
   */
  template <class T>
  RES_T ImSquareOpen(const Image<T> &imIn, const int hSide, Image<T> &imOut)
  {
    return ImSquareOpen_Soille(imIn, hSide, imOut);
  }  

  /** 
   * @brief ImSquareClose : the SE is a square which side is <b>(2 * hSide + 1)</b>
   * @param[in]  imIn the initial image
   * @param[in]  hSide Half side of the Structuring Element
   * @param[out] imOut Result
   */
  template <class T>
  RES_T ImSquareClose(const Image<T> &imIn, const int hSide, Image<T> &imOut)
  {
    return ImSquareClose_Soille(imIn, hSide, imOut);
  }

#if 0
  /** @brief ImFastLineMaxXXX : the SE is a segment of a radius <b>radius</b>
   * pixels and an orientation of angle. We take the supremum of the openings
   *
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

  /** @brief ImCircleDilate : the SE is a <b>disk</b> of radius <b>radius</b> pixels
   *
   * @param[in]  imIn the initial image
   * @param[in]  radius the diameter of the disk will be <b>2 * radius + 1</b>
   * @param[out] imOut Result
   */
   template <class T>
   RES_T ImCircleDilate(const Image<T> &imIn,
                                       const int radius, Image<T> &imOut)
   {
     return ImCircleDilate_Soille(imIn, radius, imOut);
   }                                       

  /** @brief ImCircleErode : the SE is a <b>disk</b> of radius <b>radius</b> pixels
   *
   * @param[in]  imIn the initial image
   * @param[in]  radius the diameter of the disk will be <b>2 * radius + 1</b>
   * @param[out] imOut Result
   */
   template <class T>
   RES_T ImCircleErode(const Image<T> &imIn,
                                      const int radius, Image<T> &imOut)
   {
     return ImCircleErode_Soille(imIn, radius, imOut);
   }                                       

  /** @brief ImCircleOpen : the SE is a <b>disk</b> of radius <b>radius</b> pixels
   *
   * @param[in]  imIn the initial image
   * @param[in]  radius the diameter of the disk will be <b>2 * radius + 1</b>
   * @param[out] imOut Result
   */
   template <class T>
   RES_T ImCircleOpen(const Image<T> &imIn,
                                     const int radius, Image<T> &imOut)
   {
     return ImCircleOpen_Soille(imIn, radius, imOut);
   }                                       

  /** @brief ImCircleClose : the SE is a <b>disk</b> of radius <b>radius</b> pixels
   *
   * @param[in]  imIn the initial image
   * @param[in]  radius the diameter of the disk will be <b>2 * radius + 1</b>
   * @param[out] imOut Result
   */
   template <class T>
   RES_T ImCircleClose(const Image<T> &imIn,
                                      const int radius, Image<T> &imOut)
   {
     return ImCircleClose_Soille(imIn, radius, imOut);
   }                                       

  /** @} */
} // namespace smil

//#include "FastLine/LineNaive.hpp"
#include "FastLine/FastLineMorard.hpp"
#include "FastLine/FastLineSoille.hpp"

#endif
