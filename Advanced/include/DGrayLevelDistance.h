#ifndef _DGRAY_LEVEL_DISTANCE_H_
#define _DGRAY_LEVEL_DISTANCE_H_

#include "Core/include/DCore.h"

namespace smil
{
  /**
   * @ingroup   Advanced
   * @defgroup  AdvGrayLevelDistance   Gray Level ... Zone Matrix
   * by Guillaume Thibault
   *    
   * * <a href="http://thibault.biz/Research/ThibaultMatrices/GLDZM/GLDZM.html">
   *     Distance (to border) Zone Matrix</a>
   *
   *   References :
   *   * <a href="http://thibault.biz/Doc/Publications/AdvancedStatisticalMatrices_ICIP_2011.pdf">
   Advanced Statistical Matrices for Texture Characterization: Application to DNA Chromatin and Microtubule Network Classification </a> - IEEE ICIP 2012
   *   * <a href="http://thibault.biz/Doc/Publications/AdvancedStatisticalMatrices_IEEEbme_2014.pdf">
   Advanced Statistical Matrices for Texture Characterization: Application to Cell Classification</a> - IEEE Transaction on BioMedical Engineering 2014.
   *
   * * <a href="http://thibault.biz/Research/ThibaultMatrices/GLSZM/GLSZM.html">
   *     Size Zone Matrix</a>
   *
   *   References :
   *   * <a href="https://arxiv.org/abs/1612.07003">Texture Indexes and Gray Level Size Zone Matrix. Application to Cell Nuclei Classification</a> - PRIP 2009.
   *   * <a href="http://thibault.biz/Doc/Publications/ShapeAndTextureIndexesIJPRAI2013">Shape and Texture Indexes: Application to Cell Nuclei Classification</a> - IJPRAI 2013.
   *   * <a href="http://thibault.biz/Doc/Publications/AdvancedStatisticalMatrices_IEEEbme_2014.pdf">Advanced Statistical Matrices for Texture Characterization: Application to Cell Classification</a> - IEEE Transaction on BioMedical Engineering 2014.
   *   * <a href="https://arxiv.org/abs/1611.06009">Fuzzy Statistical Matrices for Cell Classification</a> - ArXiv 2016.
   *
   * * Problems to handle :
   *
   *   * Shall change int to long to support images bigger than 2 Gpixels (mainly 3D)
   * @{ */

  /** 
   * @brief grayLevelSizeZM
   * @param[in]  imIn : the initial image
   * @param[in]  NbNDG : NbNDG
   * @param[in] szFileName : output file with results
   */
  template <class T>
  RES_T grayLevelSizeZM(const Image<T> &imIn, int NbNDG, char *szFileName);

  /** 
   * @brief grayLevelDistanceZM
   * @param[in]  imIn : the initial image
   * @param[in]  NbNDG : NbNDG
   * @param[in]  GeodesicMethod : GeodesicMethod
   * @param[in] szFileName : output file with results
   */
  template <class T>
  RES_T grayLevelDistanceZM(const Image<T> &imIn, int NbNDG, int GeodesicMethod,
                            char *szFileName);

  /** 
   * @brief grayLevelDistanceZM_Diameter
   * @param[in]  imIn : the initial image
   * @param[in]  NbNDG : NbNDG
   * @param[in] szFileName : output file with results
   */
  template <class T>
  RES_T grayLevelDistanceZM_Diameter(const Image<T> &imIn, int NbNDG,
                                     char *szFileName)
  {
    return grayLevelDistanceZM(imIn, NbNDG, 1, szFileName);
  }

  /** 
   * @brief grayLevelDistanceZM_Elongation
   * @param[in]  imIn : the initial image
   * @param[in]  NbNDG : NbNDG
   * @param[in] szFileName : output file with results
   */
  template <class T>
  RES_T grayLevelDistanceZM_Elongation(const Image<T> &imIn, int NbNDG,
                                       char *szFileName)
  {
    return grayLevelDistanceZM(imIn, NbNDG, 2, szFileName);
  }

  /** 
   * @brief grayLevelDistanceZM_Tortuosity
   * @param[in]  imIn : the initial image
   * @param[in]  NbNDG : NbNDG
   * @param[in] szFileName : output file with results
   */
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
