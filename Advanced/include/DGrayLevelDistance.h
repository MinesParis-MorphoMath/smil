#ifndef _DGRAY_LEVEL_DISTANCE_H_
#define _DGRAY_LEVEL_DISTANCE_H_

#include "Core/include/DCore.h"

namespace smil
{
  /**
   * @ingroup   Advanced
   * @defgroup  AdvGrayLevelDistance   Gray Level Distance
   * @{ */

  /** @brief GLSZM
   * @param[in]  imIn : the initial image
   * @param[in]  NbNDG : NbNDG
   * @param[out] szFileName : szFileName
   */
  template <class T>
  RES_T GLSZM(const Image<T> &imIn, int NbNDG, char *szFileName);

  template <class T>
  RES_T grayLevelDistanceZM(const Image<T> &imIn, int NbNDG, int GeodesicMethod,
                       char *szFileName);
  template <class T>
  RES_T grayLevelDistanceZM_Diameter(const Image<T> &imIn, int NbNDG,
                                char *szFileName)
  {
    return grayLevelDistanceZM(imIn, NbNDG, 1, szFileName);
  }

  template <class T>
  RES_T grayLevelDistanceZM_Elongation(const Image<T> &imIn, int NbNDG,
                                  char *szFileName)
  {
    return grayLevelDistanceZM(imIn, NbNDG, 2, szFileName);
  }

  template <class T>
  RES_T grayLevelDistanceZM_Tortuosity(const Image<T> &imIn, int NbNDG,
                                  char *szFileName)
  {
    return grayLevelDistanceZM(imIn, NbNDG, 3, szFileName);
  }

  /** @} */
} // namespace smil

// Gray Level Distannce Module header
#include "GrayLevelDistance/GLSZM.hpp"

#endif // _DGRAY_LEVEL_DISTANCE_H_
