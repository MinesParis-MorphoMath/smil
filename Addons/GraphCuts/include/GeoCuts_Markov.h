#ifndef __D_GEOCUTS_MARKOV_H__
#define __D_GEOCUTS_MARKOV_H__

typedef float F_SIMPLE;
typedef double CVariant;

namespace smil
{

  /**
   * @ingroup GeoCutGroup
   * @defgroup geoCutsMarkov    GeoCuts Markov Random Fields
   * @brief Segmentation by minimum surfaces (object label = 2, background
   * label = 3)
   * @warning  some annotated functions are tests functions, no guarantee on the
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

#include "private/GeoCuts/geo-cuts-tools.hpp"

#endif // __D_GEOCUTS_MARKOV_H__
