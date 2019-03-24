#ifndef __D_GEOCUTS_WATERSHED_H__
#define __D_GEOCUTS_WATERSHED_H__

typedef float F_SIMPLE;
typedef double CVariant;

// #include <morphee/image/include/imageInterface.hpp>
// #include <morphee/image/include/imageInterface.hpp>
// #include <morphee/selement/include/selementNeighborList.hpp>
// #include <morphee/graph/include/private/graph_T.hpp>

namespace smil
{
  /**
   * @ingroup GeoCutGroup
   * @defgroup geoCutsWatershed    Watershed Algorithms
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
  RES_T geoCutsStochastic_Watershed(const Image<T> &imIn,
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
  RES_T geoCutsStochastic_Watershed_2(const Image<T> &imIn,
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
  RES_T geoCutsWatershed_MinCut(const Image<T> &imIn, const Image<T> &imMarker,
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
  RES_T geoCutsWatershed_Prog_MinCut(const Image<T> &imIn,
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
  RES_T geoCutsWatershed_SPF(const Image<T> &imIn, const Image<T> &imMarker,
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
  RES_T geoCutsWatershed_SpanningForest(const Image<T> &imIn,
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
  RES_T geoCutsWatershed_SpanningForest_v2(const Image<T> &imIn,
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
  template <class T1, class T2>
  RES_T geoCutsMultiway_Watershed(const Image<T1> &imIn,
                                   const Image<T2> &imMarker,
                                   const double Power, const StrElt &nl,
                                   Image<T2> &imOut);

  /** @brief Shortest path forest with length of a path = reliability of the
   * path
   * @warning geoCutsMax_Fiability_Forest : These function are tests
   * functions, no guarentee on the results !!!
   *
   * @param[in] imIn : Image<T> in
   * @param[in] imMarker : Image<T> marker
   * @param[in] nl : Neighborlist
   * @param[out] imOut : Image<T> out
   */
  // line no 7846
  template <class T>
  RES_T geoCutsMax_Fiability_Forest(const Image<T> &imIn,
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
  RES_T geoCutsBiCriteria_Shortest_Forest(const Image<T> &imIn,
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
  RES_T geoCutsLexicographical_Shortest_Forest(const Image<T> &imIn,
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
  RES_T geoCutsVectorial_Shortest_Forest(const Image<T> &imIn,
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
  RES_T geoCutsVectorial_Lexicographical_Shortest_Forest(
      const Image<T> &imIn, const Image<T> &imMarker, const StrElt &nl,
      Image<T> &imOut);

  /** @brief Not working yet
   * @warning geoCutsReg_SpanningForest : Not working yet
   *
   * @param[in] imIn : Image<T> in
   * @param[in] imMarker : Image<T> marker
   * @param[in] nl : Neighborlist
   * @param[out] imOut : Image<T> out
   */
  // line no 9473
  template <class T>
  RES_T geoCutsReg_SpanningForest(const Image<T> &imIn,
                                   const Image<T> &imMarker, const StrElt &nl,
                                   Image<T> &imOut);

  /** @} */
} // namespace smil

#include "private/GeoCuts/geo-cuts-tools.hpp"
#include "private/GeoCuts/Watershed.hpp"

#endif // __D_GEOCUTS_WATERSHED_H__
