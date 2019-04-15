#ifndef __D_GEOCUTS_MARKOV_H__
#define __D_GEOCUTS_MARKOV_H__

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
  template <class T1, class T2>
  RES_T geoCutsMRF_Ising(const Image<T1> &imIn, const Image<T2> &imMarker,
                         double Beta, double Sigma, const StrElt &nl,
                         Image<T2> &imOut);

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
  template <class T1, class T2>
  RES_T geoCutsMRF_EdgePreserving(const Image<T1> &imIn,
                                  const Image<T2> &imMarker, double Beta,
                                  double Sigma, const StrElt &nl,
                                  Image<T2> &imOut);

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
  template <class T1, class T2>
  RES_T geoCutsMRF_Potts(const Image<T1> &imIn, const Image<T2> &imMarker,
                         double Beta, double Sigma, const StrElt &nl,
                         Image<T2> &imOut);

  /** @} */
} // namespace smil

#include "private/GeoCuts/geo-cuts-tools.hpp"
#include "private/GeoCuts/MarkovRandomFields.hpp"

#endif // __D_GEOCUTS_MARKOV_H__
