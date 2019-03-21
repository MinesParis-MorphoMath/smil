#ifndef __D_GEOCUTS_MINSURFACE_H__
#define __D_GEOCUTS_MINSURFACE_H__

typedef float F_SIMPLE;
typedef double CVariant;

namespace smil
{
  /**
   * @ingroup GeoCutGroup
   * @defgroup GeoCuts_MinSurface     Minimum surfaces segmentation
   * @brief Segmentation by minimum surfaces (object label = 2, background
   * label = 3)
   *
   * @author
   * * Jean Stawiaski
   * * José-Marcio Martins da Cruz
   * @{
   */
  // ##################################################
  // BEGIN FROM STAWIASKI JAN 2012
  // ##################################################
#if 0
  // line begin 1
  // line no 4148
  template <class T>
    RES_T NeighborhoodGraphFromMosaic_WithMeanGradientValue_AndQuadError(
        const Image<T> &imIn, const Image<T> &imGrad, const Image<T> &imVal,
        const CVariant &alpha, const StrElt &nl, CommonGraph32 &Gout);

  // line no 88 (old), 668 (2), 1275
  template <class T>
    RES_T TreeReweighting(
        const Image<T> &imWs, const Image<T> &imIn, const Image<T> &imGrad,
        CommonGraph32 &Treein, const CVariant &alpha1, const CVariant &alpha2,
        const CVariant &alpha3, const StrElt &nl, CommonGraph32 &Gout);

  // line no 3546, 3226 (minimean)
  template <class T>
    RES_T AverageLinkageTree(const Image<T> &imIn, const Image<T> &imGrad,
                             const StrElt &nl, CommonGraph32 &Tout);

  // line no 3899
  template <class T> 
    RES_T AverageLinkageTree_MS(const Image<T> &imWs, const Image<T> &imIn,
                                const Image<T> &imGrad, const StrElt &nl,
                                CommonGraph32 &Tout);

  // line no 1959
  template <class T>
    RES_T ScaleSetHierarchyReweighting(
        const Image<T> &imWs, const Image<T> &imIn, const Image<T> &imGrad,
        const CommonGraph32 &Treein, const StrElt &nl, CommonGraph32 &Tree_out);

  // line no 2638
  template <class T>
    RES_T MSMinCutInHierarchy(const Image<T> &imWs, const Image<T> &imIn,
                              const Image<T> &imGrad, const CVariant &alpha1,
                              const CommonGraph32 &Treein, const StrElt &nl,
                              CommonGraph32 &Tree_out);

  // line no 5461
  template <class T>
    RES_T GetUltrametricContourMap(const Image<T> &imIn,
                                   const CommonGraph32 &Tree, const StrElt &nl,
                                   Image<T> &imOut);

  // line no 5618
  template <class T>
    RES_T GetScaleSetUltrametricContourMap(const Image<T> &imIn,
                                           const CommonGraph32 &Tree,
                                           const StrElt &nl, Image<T> &imOut);
#endif

  // line no 4434
  template <class T>
  RES_T ParametricGeoCuts(const Image<T> &imIn, const Image<T> &ImGradx,
                          const Image<T> &ImGrady, const Image<T> &imMarker,
                          const StrElt &nl, Image<T> &imOut);

  // line no 5816
  template <class T>
  RES_T GeoCuts_Stochastic_Watershed_Variance(
      const Image<T> &imIn, const Image<T> &imIn2, const Image<T> &imVal,
      const CVariant &nbmarkers, const CVariant &alpha, const StrElt &nl,
      Image<T> &imOut);

#if 0
  template <class T>
    // lineno 4857
    RES_T GeoCuts_Stochastic_Watershed_Graph(
        const Image<T> &imIn, const Image<T> &imVal, CommonGraph32 &GIn,
        const CVariant &nbmarkers, const StrElt &nl, Image<T> &imOut);

  // line no 5141
  template <class T>
    RES_T GeoCuts_Stochastic_Watershed_Graph_NP(
        const Image<T> &imIn, const Image<T> &imVal, CommonGraph32 &GIn,
        const CVariant &nbmarkers, const StrElt &nl, Image<T> &imOut);

  // line no 5426
  template <class T>
    RES_T UpdateSpanningTreeFromForest(const CommonGraph32 &ForestIn,
                                       const CommonGraph32 &Tin,
                                       CommonGraph32 &Tout);
#endif

  // line no 4668
  template <class T>
  RES_T GeoCuts_Boundary_Constrained_MinSurfaces(const Image<T> &imIn,
                                                 const Image<T> &imMarker,
                                                 const Image<T> &imMarker2,
                                                 const StrElt &nl,
                                                 Image<T> &imOut);

#if 0
  // line no 3550
  template <class T>
    RES_T AverageLinkageTree(const Image<T> &imIn, const Image<T> &imGrad,
                             const StrElt &nl, CommonGraph32 &Tout);

  // line no 3903
  template <class T>
    RES_T AverageLinkageTree_MS(const Image<T> &imWs, const Image<T> &imIn,
                                const Image<T> &imGrad, const StrElt &nl,
                                CommonGraph32 &Tout);
#endif

  // ##################################################
  // END FROM STAWIASKI JAN 2012
  // ##################################################

  /** @brief Returns Geo Cuts algorithm
   *
   * @param[in] imIn : Image<T> in
   * @param[in] imGradx : Image<T> gradient X
   * @param[in] imGrady : Image<T> gradient Y
   * @param[in] imMarker : Image<T> marker
   * @param[in] nl : Neighborlist
   * @param[out] imOut : Image<T> out
   */
  // line no 6266
  template <class T>
  RES_T GeoCuts(const Image<T> &imIn, const Image<T> &imGradx,
                const Image<T> &imGrady, const Image<T> &imMarker,
                const StrElt &nl, Image<T> &imOut);

  /** @brief Geo Cuts algorithm on a pixel adjacency graph, ImMarker is
   * composed of three values 0 for unmarked pixels, 2 and 3 for object and
   * background markers
   *
   * @param[in] imIn : Image<T> in
   * @param[in] imMarker : Image<T> marker
   * @param[in] nl : Neighborlist
   * @param[out] imOut : Image<T> out
   */
  // line no 6468
  template <class T>
  RES_T GeoCuts_MinSurfaces(const Image<T> &imIn, const Image<T> &imMarker,
                            const StrElt &nl, Image<T> &imOut);

  /** @brief Geo Cuts algorithm on a pixel adjacency graph, ImMarker is
   * composed of three values 0 for unmarked pixels, 2 and 3 for object and
   * background markers
   *
   * @param[in] imIn : Image<T> in
   * @param[in] imMarker : Image<T> marker
   * @param[in] nl : Neighborlist
   * @param[out] imOut : Image<T> out
   */
  // line no 9789
  template <class T>
  RES_T GeoCuts_MinSurfaces_With_Line(const Image<T> &imIn,
                                      const Image<T> &imMarker,
                                      const StrElt &nl, Image<T> &imOut);

  /** @brief Multiple object segmentation, Geo Cuts algorithm on a pixel
   * adjacency graph, ImMarker is composed of three values 0 for unmarked
   * pixels, >0 for objects markers
   *
   * @param[in] imIn : Image<T> in
   * @param[in] imMarker : Image<T> marker
   * @param[in] nl : Neighborlist
   * @param[out] imOut : Image<T> out
   */
  // line no 9964
  template <class T1, class T2>
  RES_T GeoCuts_Multiway_MinSurfaces(const Image<T1> &imIn,
                                     const Image<T2> &imMarker,
                                     const StrElt &nl, Image<T2> &imOut);

  /** @} */
} // namespace smil

#include "private/GeoCuts/geo-cuts-tools.hpp"
#include "private/GeoCuts/MinSurfaces.hpp"

#endif // __D_GEOCUTS_MINSURFACE_H__
