#ifndef __MOSAIC_GEOCUTSALGO_HPP__
#define __MOSAIC_GEOCUTSALGO_HPP__

#define __MGCA

typedef RES_T RES_C;
typedef float F_SIMPLE;
typedef double CVariant;

// #include <morphee/image/include/imageInterface.hpp>
// #include <morphee/image/include/imageInterface.hpp>
// #include <morphee/selement/include/selementXXX.hpp>
// #include <morphee/graph/include/private/graph_T.hpp>

// #include "Mosaic_GeoCutsAlgo_DLLEXPORT.hpp"

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
   * @defgroup GeoCutsAlgo_SegMinSurf_group 1. GeoCutsAlgo SegMinSurf
   *
   * @brief Segmentation by minimum surfaces (object label =2, background
   * label = 3)
   *
   * @author Jean Stawiaski
   * @{
   */

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
  GeoCuts_MinSurfaces_with_steps(const Image<T> &imIn, const Image<T> &imGrad,
                                 const Image<T> &imMarker, const StrElt &nl,
                                 F_SIMPLE step_x, F_SIMPLE step_y,
                                 F_SIMPLE step_z, Image<T> &imOut);

  /*! @brief Geo Cuts algorithm on a region adjacency graph, imMosaic is a
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
  RES_T GeoCuts_MinSurfaces_with_steps_vGradient(
      const Image<T> &imIn, const Image<T> &imGrad, const Image<T> &imMarker,
      const StrElt &nl, F_SIMPLE step_x, F_SIMPLE step_y, F_SIMPLE step_z,
      Image<T> &imOut);

  /*! @brief Geo Cuts algorithm on a region adjacency graph, imMosaic is a
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
  RES_T GeoCuts_MinSurfaces(const Image<T> &imIn, const Image<T> &imGrad,
                            const Image<T> &imMarker, const StrElt &nl,
                            Image<T> &imOut);

  /*! @brief Geo Cuts algorithm on a region adjacency graph, imMosaic is a
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
  RES_T GeoCuts_MinSurfaces_With_Line(const Image<T> &imIn,
                                      const Image<T> &imGrad,
                                      const Image<T> &imMarker,
                                      const StrElt &nl, Image<T> &imOut);
#if 1
  /*! @brief Geo Cuts algorithm with curvature term on a region adjacency
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
  RES_T GeoCuts_Regularized_MinSurfaces(const Image<T> &imIn,
                                        const Image<T> &imGrad,
                                        const Image<T> &imCurvature,
                                        const Image<T> &imMarker,
                                        const CVariant &Beta, const StrElt &nl,
                                        Image<T> &imOut);
#endif
  /*! @brief Multiple object segmentation, Geo Cuts algorithm on a region
   * adjacency graph, imMosaic is a mosaic, ImMarker is composed of three
   * values 0 for unmarked pixels, >0 for objects markers
   *
   * @param[in] imIn : Image<T> in
   * @param[in] imGrad : Image<T> gradient
   * @param[in] imMarker : Image<T> marker
   * @param[in] nl : StrElt nl
   * @param[out] imOut : Image<T> out
   */
  template <class T>
  RES_T GeoCuts_MultiWay_MinSurfaces(const Image<T> &imIn,
                                     const Image<T> &imGrad,
                                     const Image<T> &imMarker, const StrElt &nl,
                                     Image<T> &imOut);

  //! @} defgroup GeoCutsAlgo_SegMinSurf_group

  /**
   * @ingroup MosaicGraphCut
   * @defgroup GeoCutsAlgo_AlphaSeg_group 2. GeoCutsAlgo_AlphaSeg_group
   *
   * @brief Segmentation by minimum surfaces (object label =2, background
   * label = 3)
   * @warning  some annoted functions are tests functions, no guarentee on the
   * results !!!
   *
   * @author Jean Stawiaski
   * @{
   */

  /*! @brief Returns alpha extensions of the segmentation, ImIn is a segmented
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
  RES_T GeoCuts_Optimize_Mosaic(const Image<T> &imIn, const Image<T> &imGrad,
                                const Image<T> &imMosaic,
                                const Image<T> &imMarker, const StrElt &nl,
                                Image<T> &imOut);

  /*! @brief Returns Geo Cuts algorithm
   * @warning GeoCuts_Segment_Graph : These function are tests functions, no
   * guarentee on the results !!!
   *
   * @param[in]  imIn : Image<T>
   * @param[in]  imMosaic : Image<T>
   * @param[in]  imMarker : Image<T>
   * @param[in]  nl : StrElt
   * @param[out] imOut : Image<T>
   */
  template <class T>
  RES_T GeoCuts_Segment_Graph(const Image<T> &imIn, const Image<T> &imMosaic,
                              const Image<T> &imMarker, const StrElt &nl,
                              Image<T> &imOut);

  //! @} defgroup GeoCutsAlgo_AlphaSeg_group

  /*!
   * @ingroup MosaicGraphCut
   * @defgroup GeoCutsAlgo_group 3. GeoCutsAlgo_group
   * @brief Segmentation by minimum surfaces (object label =2, background
   * label = 3)
   * @warning  some annoted functions are tests functions, no guarentee on the
   * results !!!
   *
   * @author Jean Stawiaski
   * @{
   */
#if 1
  /*! @brief Markov Random Fields segmentation with two labels (2 and 3) with
   * The Ising Model
   * @note: For Beta and sigma parameters, read Markov Random Fields section
   * on jean Stawiaski thesis
   *
   * @param[in] imIn     : Image<T> imIn
   * @param[in] imMosaic : Image<T> imMosaic
   * @param[in] imMarker : Image<T> imMarker
   * @param[in] Beta     :
   * @param[in] Sigma    :
   * @param[in] nl       : StrElt nl
   * @param[out] imOut   : Image<T> out
   */
  template <class T>
  RES_T MAP_MRF_Ising(const Image<T> &imIn, const Image<T> &imMosaic,
                      const Image<T> &imMarker, const CVariant &Beta,
                      const CVariant &Sigma, const StrElt &nl, Image<T> &imOut);

  /*! @brief Markov Random Fields segmentation with two labels (2 and 3) with
   * edge preserving prior
   * @note: For Beta and sigma parameters, read Markov Random Fields section
   * on jean Stawiaski thesis
   *
   * @param[in] imIn     : Image<T> imIn
   * @param[in] imMosaic : Image<T> imMosaic
   * @param[in] imMarker : Image<T> imMarker
   * @param[in] Beta     :
   * @param[in] Sigma    :
   * @param[in] nl       : StrElt nl
   * @param[out] imOut   : Image<T> out
   */
  template <class T>
  RES_T MAP_MRF_edge_preserving(const Image<T> &imIn, const Image<T> &imMosaic,
                                const Image<T> &imMarker, const CVariant &Beta,
                                const CVariant &Sigma, const StrElt &nl,
                                Image<T> &imOut);

  /*! @brief Multi-Label MAP Markov Random Field with Ising prior (Potts
   * Model)
   * @note: For Beta and sigma parameters, read Markov Random Fields section
   * on jean Stawiaski thesis
   * @warning MAP_MRF_Potts : These function are tests functions, no guarentee
   * on the results !!!
   *
   * @param[in] imIn     : Image<T> imIn
   * @param[in] imMosaic : Image<T> imMosaic
   * @param[in] imMarker : Image<T> imMarker
   * @param[in] Beta     :
   * @param[in] Sigma    :
   * @param[in] nl       : StrElt nl
   * @param[out] imOut   : Image<T> out
   */
  template <class T>
  RES_T MAP_MRF_Potts(const Image<T> &imIn, const Image<T> &imMosaic,
                      const Image<T> &imMarker, const CVariant &Beta,
                      const CVariant &Sigma, const StrElt &nl, Image<T> &imOut);
#endif
  /**  @} defgroup GeoCutsAlgo_Markov_group */

} // namespace smil

#endif //__MOSAIC_GEOCUTSALGO_HPP__
