#ifndef __D_GEOCUTS_H__
#define __D_GEOCUTS_H__

typedef float F_SIMPLE;
typedef double CVariant;
  
// #include <morphee/image/include/imageInterface.hpp>
// #include <morphee/image/include/imageInterface.hpp>
// #include <morphee/selement/include/selementNeighborList.hpp>
// #include <morphee/graph/include/private/graph_T.hpp>

namespace smil
{
  /**
   * @ingroup AddonGraphCut
   * @defgroup GeoCutGroup Geo Cuts
   *
   * Most of these algorithms were developped in the PhD thesis of Jean Stawiaski.
   *
   * @see 
   * <i>Jean Stawiaski</i>, 
   * <a href="http://smil.cmm.mines-paristech.fr/publis/These_Jean_Stawiaski.pdf">
   *  Mathematical Morphology and Graphs :  Applications 
   * to Interactive Medical Image Segmentation</a>, PhD Thesis - 2008
   *
   * @author
   * * Jean Stawiaski
   * * José-Marcio Martins da Cruz (porting and rewriting) 
   */

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
  // line no 4148
  template <class T>
    RES_T
    NeighborhoodGraphFromMosaic_WithMeanGradientValue_AndQuadError(
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
  template <class T>
  RES_T GeoCuts_Multiway_MinSurfaces(const Image<T> &imIn,
                                     const Image<T> &imMarker, const StrElt &nl,
                                     Image<T> &imOut);

  /** @} */

  // line no end 1
  /**
   * @ingroup GeoCutGroup
   * @defgroup GeoCuts_Watershed    Watershed Algorithms
   * @brief Watershed algorithms
   * @warning  some annoted functions are tests functions, no guarentee on the
   * results !!!
   *
   * @author Jean Stawiaski
   * @{
   */

  /** @brief Stochastic Watershed (first example of a deterministic approach)
   *
   * @param[in] imIn : Image<T> in
   * @param[in] imVal : Image<T> val
   * @param[in] nbmarkers : markers count
   * @param[in] nl : Neighborlist
   * @param[out] imOut : Image<T> out
   */
  // line no 6633
  template <class T>
  RES_T GeoCuts_Stochastic_Watershed(const Image<T> &imIn,
                                     const Image<T> &imVal,
                                     const CVariant &nbmarkers,
                                     const StrElt &nl, Image<T> &imOut);

  /** @brief Stochastic Watershed with deterministic and probabilistic markers
   * (first example of a deterministic approach)
   *
   * @param[in] imIn : Image<T> in
   * @param[in] imVal : Image<T> val
   * @param[in] imMarker : Image<T> marker
   * @param[in] nbmarkers : markers count
   * @param[in] nl : Neighborlist
   * @param[out] imOut : Image<T> out
   */
  // line no 6922
  template <class T>
  RES_T GeoCuts_Stochastic_Watershed_2(const Image<T> &imIn,
                                       const Image<T> &imVal,
                                       const Image<T> &imMarker,
                                       const CVariant &nbmarkers,
                                       const StrElt &nl, Image<T> &imOut);

  /** @brief Watershed as a Minimum Cut (2 labels)
   * @note: See Jean Stawiaski Thesis to understand Power parameter effect
   *
   * @param[in] imIn : Image<T> in
   * @param[in] imMarker : Image<T> marker
   * @param[in] Power
   * @param[in] nl : Neighborlist
   * @param[out] imOut : Image<T> out
   */
  // line no 7236
  template <class T>
  RES_T GeoCuts_Watershed_MinCut(const Image<T> &imIn, const Image<T> &imMarker,
                                 const CVariant &Power, const StrElt &nl,
                                 Image<T> &imOut);

  /** @brief Watershed as a Minimum Cut (2 labels) with progressive power map
   *
   * @param[in] imIn : Image<T> in
   * @param[in] imMarker : Image<T> marker
   * @param[in] nl : Neighborlist
   * @param[out] imOut : Image<T> out
   */
  // line no 7401
  template <class T>
  RES_T GeoCuts_Watershed_Prog_MinCut(const Image<T> &imIn,
                                      const Image<T> &imMarker,
                                      const StrElt &nl, Image<T> &imOut);

  /** @brief Watershed as a Shortest Path Forest
   * @note: See Jean Stawiaski Thesis to understand Power parameter effect
   *
   * @param[in] imIn : Image<T> in
   * @param[in] imMarker : Image<T> marker
   * @param[in] Power
   * @param[in] nl : Neighborlist
   * @param[out] imOut : Image<T> out
   */
  // line no 7603
  template <class T>
  RES_T GeoCuts_Watershed_SPF(const Image<T> &imIn, const Image<T> &imMarker,
                              const CVariant &Power, const StrElt &nl,
                              Image<T> &imOut);

  /** @brief Watershed as a minimum spanning forest
   *
   * @param[in] imIn : Image<T> in
   * @param[in] imMarker : Image<T> marker
   * @param[in] nl : Neighborlist
   * @param[out] imOut : Image<T> out
   */
  // line no 9002
  template <class T>
  RES_T GeoCuts_Watershed_SpanningForest(const Image<T> &imIn,
                                         const Image<T> &imMarker,
                                         const StrElt &nl, Image<T> &imOut);

  /** @brief Watershed as a minimum spanning forest (min and 1/2 gradient)
   *
   * @param[in] imIn : Image<T> in
   * @param[in] imMarker : Image<T> marker
   * @param[in] nl : Neighborlist
   * @param[out] imOut : Image<T> out
   */
  // line no 9199
  template <class T>
  RES_T GeoCuts_Watershed_SpanningForest_v2(const Image<T> &imIn,
                                            const Image<T> &imMarker,
                                            const StrElt &nl, Image<T> &imOut);

  /** @brief Watershed as a mutli_terminal cut (multi label)
   * @note: See Jean Stawiaski Thesis to understand Power parameter effect
   *
   * @param[in] imIn : Image<T> in
   * @param[in] imMarker : Image<T> marker
   * @param[in] Power
   * @param[in] nl : Neighborlist
   * @param[out] imOut : Image<T> out
   */
  // line no 10160
  template <class T>
  RES_T GeoCuts_Multiway_Watershed(const Image<T> &imIn,
                                   const Image<T> &imMarker,
                                   const CVariant &Power, const StrElt &nl,
                                   Image<T> &imOut);

  /** @brief Shortest path forest with length of a path = reliability of the
   * path
   * @warning GeoCuts_Max_Fiability_Forest : These function are tests
   * functions, no guarentee on the results !!!
   *
   * @param[in] imIn : Image<T> in
   * @param[in] imMarker : Image<T> marker
   * @param[in] nl : Neighborlist
   * @param[out] imOut : Image<T> out
   */
  // line no 7846
  template <class T>
  RES_T GeoCuts_Max_Fiability_Forest(const Image<T> &imIn,
                                     const Image<T> &imMarker, const StrElt &nl,
                                     Image<T> &imOut);

  /** @brief Watershed as a shortest path forest with lexicographical ordering
   * of (max path and shortest path)
   *
   * @param[in] imIn : Image<T> in
   * @param[in] imMarker : Image<T> marker
   * @param[in] nl : Neighborlist
   * @param[out] imOut : Image<T> out
   */
  // line no 8073
  template <class T>
  RES_T GeoCuts_BiCriteria_Shortest_Forest(const Image<T> &imIn,
                                           const Image<T> &imMarker,
                                           const StrElt &nl, Image<T> &imOut);

  /** @brief Watershed as a shortest path forest with lexicographical ordering
   *
   * @param[in] imIn : Image<T> in
   * @param[in] imMarker : Image<T> marker
   * @param[in] nl : Neighborlist
   * @param[out] imOut : Image<T> out
   */
  // line no 8271
  template <class T>
  RES_T GeoCuts_Lexicographical_Shortest_Forest(const Image<T> &imIn,
                                                const Image<T> &imMarker,
                                                const StrElt &nl,
                                                Image<T> &imOut);

  /** @brief Color Watershed on color images as a shortest path forest with
   * lexicographical ordering of vector attributes
   * @note: COLOR IMAGES
   * WATERSHED
   *
   * @param[in] imIn : Image<T> in
   * @param[in] imMarker : Image<T> marker
   * @param[in] nl : Neighborlist
   * @param[out] imOut : Image<T> out
   */
  // line no 8576
  template <class T>
  RES_T GeoCuts_Vectorial_Shortest_Forest(const Image<T> &imIn,
                                          const Image<T> &imMarker,
                                          const StrElt &nl, Image<T> &imOut);

  /** @brief Color Watershed on color images as a shortest path forest with
   * lexicographical ordering of vector attributes and lexicographical
   * distances
   * @note: COLOR IMAGES WATERSHED
   *
   * @param[in] imIn : Image<T> in
   * @param[in] imMarker : Image<T> marker
   * @param[in] nl : Neighborlist
   * @param[out] imOut : Image<T> out
   */
  // line no 8482
  template <class T>
  RES_T GeoCuts_Vectorial_Lexicographical_Shortest_Forest(
      const Image<T> &imIn, const Image<T> &imMarker, const StrElt &nl,
      Image<T> &imOut);

  /** @brief Not working yet
   * @warning GeoCuts_Reg_SpanningForest : Not working yet
   *
   * @param[in] imIn : Image<T> in
   * @param[in] imMarker : Image<T> marker
   * @param[in] nl : Neighborlist
   * @param[out] imOut : Image<T> out
   */
  // line no 9473
  template <class T>
  RES_T GeoCuts_Reg_SpanningForest(const Image<T> &imIn,
                                   const Image<T> &imMarker, const StrElt &nl,
                                   Image<T> &imOut);

  /** @} */
  // line no end 2
  /**
   * @ingroup GeoCutGroup
   * @defgroup GeoCuts_Markov    GeoCuts Markov Random Fields Group
   * @brief Segmentation by minimum surfaces (object label = 2, background
   * label = 3)
   * @warning  some annoted functions are tests functions, no guarentee on the
   * results !!!
   *
   * @author Jean Stawiaski
   * @{
   */

  /** @brief Markov Random Fields segmentation with two labels (2 labels,
   * object = 2, background = 3) with The Ising Model
   * @note: For Beta and sigma parameters, read Markov Random Fields section
   * on jean Stawiaski thesis
   *
   * @param[in] imIn : Image<T> in
   * @param[in] imMarker : Image<T> marker
   * @param[in] Beta
   * @param[in] Sigma
   * @param[in] nl : Neighborlist
   * @param[out] imOut : Image<T> out
   */
  // line no 10360
  template <class T>
  RES_T MAP_MRF_Ising(const Image<T> &imIn, const Image<T> &imMarker,
                      const CVariant &Beta, const CVariant &Sigma,
                      const StrElt &nl, Image<T> &imOut);

  /** @brief Markov Random Fields segmentation with two labels (2 labels,
   * object = 2, background = 3) with edge preserving prior
   * @note: For Beta and sigma parameters, read Markov Random Fields section
   * on jean Stawiaski thesis
   *
   * @param[in] imIn : Image<T> in
   * @param[in] imMarker : Image<T> marker
   * @param[in] Beta
   * @param[in] Sigma
   * @param[in] nl : Neighborlist
   * @param[out] imOut : Image<T> out
   */
  // line no 10568
  template <class T>
  RES_T MAP_MRF_edge_preserving(const Image<T> &imIn, const Image<T> &imMarker,
                                const CVariant &Beta, const CVariant &Sigma,
                                const StrElt &nl, Image<T> &imOut);

  /** @brief Multi-Label MAP Markov Random Field with Ising prior (Potts
   * Model)
   * @note: For Beta and sigma parameters, read Markov Random Fields section
   * on jean Stawiaski thesis
   *
   * @param[in] imIn : Image<T> in
   * @param[in] imMarker : Image<T> marker
   * @param[in] Beta
   * @param[in] Sigma
   * @param[in] nl : Neighborlist
   * @param[out] imOut : Image<T> out
   */
  // line no 10989
  template <class T>
  RES_T MAP_MRF_Potts(const Image<T> &imIn, const Image<T> &imMarker,
                      const CVariant &Beta, const CVariant &Sigma,
                      const StrElt &nl, Image<T> &imOut);

  /** @} */

} // namespace smil

#endif // __D_GEOCUTS_H__
