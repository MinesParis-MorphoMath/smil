#ifndef _DGRAY_LEVEL_DISTANCE_H_
#define _DGRAY_LEVEL_DISTANCE_H_

#include "Core/include/DCore.h"

namespace smil
{
  /**
   * @addtogroup   AddonZoneMatrix
   *
   * * <a href="http://thibault.biz/Research/ThibaultMatrices/GLDZM/GLDZM.html">
   *     Distance (to border) Zone Matrix</a>
   *
   * @see
   * - @cite DBLP:journals/corr/ZwanenburgLVL16 - @txtitalic{%Image biomarker 
   *   standardisation initiative}
   *
   * - @cite thibault_angulo_meyer:2014 - @txtitalic{%Advanced Statistical Matrices for 
   *  Texture Characterization: Application to Cell Classification}
   *
   * - @cite thibault:hal-00833529 - @txtitalic{%Advanced Statistical Matrices for Texture
   *   Characterization: Application to DNA Chromatin and Microtubule Network}
   *   Classification
   *
   * @author 
   * - Guillaume Thibault - original code
   * - José-Marcio Martins da Cruz - port to Smil
   *
   * @par ToDo
   * - convert data types to usual @Smil data types;
   * - can't handle images bigger than 2 GBytes;
   * - loop optimisations to do;
   * - convert @b C to @b C++ code.
   * @{ */

  /**
   * @brief grayLevelDistanceZM() -
   * @param[in]  imIn : the initial image
   * @param[in]  NbNDG : NbNDG
   * @param[in]  Method : Geodesic Method
   * - 0 -> Size
   * - 1 -> Diameter
   * - 2 -> Elongation
   * - 3 -> Tortuosity
   * @returns a vector with features
   * - 0 - SZE - Small Zone Emphasis
   * - 1 - LZE - Large Zone Emphasis
   * - 2 - LGZE - Low Gray level Zone Emphasis
   * - 3 - HGZE - High Gray level Zone Emphasis
   * - 4 - SZLGE - Small Zone Low Gray level Emphasis
   * - 5 - SZHGE - Small Zone High Gray level Emphasis
   * - 6 - LZLGE - Large Zone Low Gray level Emphasis
   * - 7 - LZHGE - Large Zone High Gray level Emphasis
   * - 8 - GLNU - Gray Level Non Uniform
   * - 9 - SZNU - Size Zone Non Uniform
   * - 10 - BARYGL - Barycenter on Gray Levels
   * - 11 - BARYS - Barycenter on Zone sizes
   */
  template <class T>
  vector<double> grayLevelZMDistance(const Image<T> &imIn, int NbNDG,
                                     int Method);

  /**
   * @brief grayLevelSizeZM() -
   * @param[in]  imIn : the initial image
   * @param[in]  NbNDG : NbNDG
   * @returns a vector with features (see grayLevelZMDistance() for description)
   *
   * @overload
   */
  template <class T>
  vector<double> grayLevelZMSize(const Image<T> &imIn, int NbNDG);

  /** @} */
} // namespace smil

// Gray Level Distannce Module header
#include "private/GLSZM.hpp"

#endif // _DGRAY_LEVEL_DISTANCE_H_
