#ifndef __MOSAIC_GEOCUTSALGO_H__
#define __MOSAIC_GEOCUTSALGO_H__

typedef float F_SIMPLE;
typedef double CVariant;


#include "Morpho/include/DMorpho.h"

// using namespace morphee::graph;
namespace smil
{
  /**
   * @ingroup AddonGraphCut
   * @defgroup MosaicGraphCut Mosaic Graph Cut
   */

  /**
   *
   * @ingroup MosaicGraphCut
   * @defgroup MosaicGeoCutsAlgo_SegMinSurf     Segmentation by Minimum Surfaces
   *
   * @brief Segmentation by minimum surfaces (object label =2, background
   * label = 3)
   *
   * @author Jean Stawiaski
   * @{
   */

  /** @brief Geo Cuts algorithm on a region adjacency graph, imMosaic is a
   * mosaic, ImMarker is composed of three values 0 for unmarked pixels, 2 and
   * 3 for object and background markers
   *
   * @param[in] imIn : Image<T> in
   * @param[in] imGrad : Image<T> with original values
   * @param[in] imMarker : Image<T> marker
   * @param[in] nl : StrElt nl
   * @param[out] imOut : Image<T> out
   */
  template <class T>
  RES_T geoCutsMinSurfaces(const Image<T> &imIn, const Image<T> &imGrad,
                            const Image<T> &imMarker, const StrElt &nl,
                            Image<T> &imOut);

  /** 
   * @cond
   */

  /** @brief Geo Cuts algorithm on a region adjacency graph, imMosaic is a
   * mosaic, ImMarker is composed of three values 0 for unmarked pixels, 2 and
   * 3 for object and background markers
   *
   * @param[in] imIn : Image<T> in
   * @param[in] imGrad : Image<T> with original values
   * @param[in] imMarker : Image<T> marker
   * @param[in] nl : StrElt nl
   * @param[out] imOut : Image<T> out
   */
  template <class T>
  RES_T geoCutsMinSurfaces_With_Line(const Image<T> &imIn,
                                      const Image<T> &imGrad,
                                      const Image<T> &imMarker,
                                      const StrElt &nl, Image<T> &imOut);

  /** @brief Geo Cuts algorithm on a region adjacency graph, imMosaic is a
   * mosaic, ImMarker is composed of three values 0 for unmarked pixels, 2 and
   * 3 for object and background markers. This version takes into account the
   * sampling step along the 3 dimensions
   *
   * @param[in] imIn : Image<T> in
   * @param[in] imGrad : Image<T> gradient
   * @param[in] imMarker : Image<T> marker
   * @param[in] nl : StrElt nl
   * @param[in] step_x : Sampling step X
   * @param[in] step_y : Sampling step Y
   * @param[in] step_z : Sampling step Z
   * @param[out] imOut : Image<T> out
   */

  template <class T>
  RES_T
  geoCutsMinSurfaces_with_steps(const Image<T> &imIn, const Image<T> &imGrad,
                                 const Image<T> &imMarker, const StrElt &nl,
                                 F_SIMPLE step_x, F_SIMPLE step_y,
                                 F_SIMPLE step_z, Image<T> &imOut);

  /** @brief Geo Cuts algorithm on a region adjacency graph, imMosaic is a
   * mosaic, ImMarker is composed of three values 0 for unmarked pixels, 2 and
   * 3 for object and background markers
   *
   * @param[in] imIn : Image<T> in
   * @param[in] imGrad : Image<T> with original values
   * @param[in] imMarker : Image<T> marker
   * @param[in] nl : StrElt nl
   * @param[in] step_x : Sampling step X
   * @param[in] step_y : Sampling step Y
   * @param[in] step_z : Sampling step Z
   * @param[out] imOut : Image<T> out
   */
  template <class T>
  RES_T geoCutsMinSurfaces_with_steps_vGradient(
      const Image<T> &imIn, const Image<T> &imGrad, const Image<T> &imMarker,
      const StrElt &nl, F_SIMPLE step_x, F_SIMPLE step_y, F_SIMPLE step_z,
      Image<T> &imOut);

  /** @brief Multiple object segmentation, Geo Cuts algorithm on a region
   * adjacency graph, imMosaic is a mosaic, ImMarker is composed of three
   * values 0 for unmarked pixels, >0 for objects markers
   * @param[in] imIn : Image<T> in
   * @param[in] imGrad : Image<T> gradient
   * @param[in] imMarker : Image<T> marker
   * @param[in] nl : StrElt nl
   * @param[out] imOut : Image<T> out
   */
  template <class T>
  RES_T geoCutsMultiWay_MinSurfaces(const Image<T> &imIn,
                                     const Image<T> &imGrad,
                                     const Image<T> &imMarker, const StrElt &nl,
                                     Image<T> &imOut);

  /** @brief Geo Cuts algorithm with curvature term on a region adjacency
   * graph, imMosaic is a mosaic, ImMarker is composed of three values 0 for
   * unmarked pixels, 2 and 3 for object and background markers
   *
   * @param[in] imIn        : Image<T> in
   * @param[in] imGrad      : Image<T> gradient
   * @param[in] imCurvature : Image<T> curvature
   * @param[in] imMarker    : Image<T> marker
   * @param[in] Beta        :
   * @param[in] nl          : StrElt nl
   * @param[out] imOut      : Image<T> out
   */
  template <class T>
  RES_T geoCutsRegularized_MinSurfaces(const Image<T> &imIn,
                                        const Image<T> &imGrad,
                                        const Image<T> &imCurvature,
                                        const Image<T> &imMarker,
                                        const CVariant &Beta, const StrElt &nl,
                                        Image<T> &imOut);

  /** @} */ 

  /**
   * @ingroup MosaicGraphCut
   * @defgroup MosaicGeoCutsAlgo_AlphaSeg    Mosaic GeoCutsAlgo AlphaSeg
   *
   * @brief Segmentation by minimum surfaces (object label =2, background
   * label = 3)
   * @warning  some annoted functions are tests functions, no guarentee on the
   * results !!!
   *
   * @author Jean Stawiaski
   * @{
   */

  /** @brief Returns alpha extensions of the segmentation, ImIn is a segmented
   * images.
   *
   * @param[in] imIn : Image<T>
   * @param[in] imGrad : Image<T> gradient
   * @param[in] imMosaic : Image<T> mosait
   * @param[in] imMarker : Image<T> marker
   * @param[in] nl : StrElt nl
   * @param[out] imOut : Image<T> out
   */
  template <class T>
  RES_T geoCutsOptimize_Mosaic(const Image<T> &imIn, const Image<T> &imGrad,
                                const Image<T> &imMosaic,
                                const Image<T> &imMarker, const StrElt &nl,
                                Image<T> &imOut);

  /** @brief Returns Geo Cuts algorithm
   * @warning geoCutsSegment_Graph : These function are tests functions, no
   * guarentee on the results !!!
   *
   * @param[in]  imIn : Image<T>
   * @param[in]  imMosaic : Image<T>
   * @param[in]  imMarker : Image<T>
   * @param[in]  nl : StrElt
   * @param[out] imOut : Image<T>
   */
  template <class T>
  RES_T geoCutsSegment_Graph(const Image<T> &imIn, const Image<T> &imMosaic,
                              const Image<T> &imMarker, const StrElt &nl,
                              Image<T> &imOut);

  /** 
   * @endcond 
   */
  /** @} */

  template <class T>
  void testHandleSE(const Image<T> &img, StrElt se = DEFAULT_SE);

} // namespace smil

#include "private/MosaicGeoCuts/Mosaic_GeoCuts.hpp"

#endif //__MOSAIC_GEOCUTSALGO_H__
