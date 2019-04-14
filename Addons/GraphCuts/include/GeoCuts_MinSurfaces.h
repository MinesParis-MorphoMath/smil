#ifndef __D_GEOCUTS_MINSURFACE_H__
#define __D_GEOCUTS_MINSURFACE_H__

typedef float F_SIMPLE;
typedef double CVariant;

namespace smil
{
  /**
   * @ingroup GeoCutGroup
   * @defgroup geoCutsMinSurface     Minimum surfaces segmentation
   * @brief Segmentation by minimum surfaces (object label = 2, background
   * label = 3)
   *
   * @author
   * * Jean Stawiaski
   * * José-Marcio Martins da Cruz
   * @{
   */

  // line no 4434
  /** @cond */
  /** @brief geoCutsParametric
   *
   * @param[in] imIn  Image<T> in
   * @param[in] imGradx  Image<T> gradient X
   * @param[in] imGrady  Image<T> gradient Y
   * @param[in] imMarker  Image<T> marker
   * @param[in] nl  Neighborlist
   * @param[out] imOut  Image<T> out
   */
  template <class T>
  RES_T geoCutsParametric(const Image<T> &imIn, const Image<T> &imGradx,
                          const Image<T> &imGrady, const Image<T> &imMarker,
                          const StrElt &nl, Image<T> &imOut);
  /** @endcond */

  // line no 5816
  /** @cond */
  /** @brief geoCutsStochastic_Watershed_Variance
   *
   * @param[in] imIn1  Image<T> in1
   * @param[in] imIn2  Image<T> in2
   * @param[in] imVal  Image<T> imVal
   * @param[in] nbmarkers  
   * @param[in] alpha  
   * @param[in] nl  Neighborlist
   * @param[out] imOut  Image<T> out
   */
  template <class T>
  RES_T geoCutsStochastic_Watershed_Variance(
      const Image<T> &imIn1, const Image<T> &imIn2, const Image<T> &imVal,
      const CVariant &nbmarkers, const CVariant &alpha, const StrElt &nl,
      Image<T> &imOut);
  /** @endcond */

  // line no 4668
  /** @cond */  
  /** @brief geoCutsStochastic_Watershed_Variance
   *
   * @param[in] imIn  Image<T> in1
   * @param[in] imMarker1  Image<T> markers 1
   * @param[in] imMarker2  Image<T> markers 2
   * @param[in] nl  Neighborlist
   * @param[out] imOut  Image<T> out
   */
  template <class T>
  RES_T geoCutsBoundary_Constrained_MinSurfaces(const Image<T> &imIn,
                                                 const Image<T> &imMarker1,
                                                 const Image<T> &imMarker2,
                                                 const StrElt &nl,
                                                 Image<T> &imOut);
  /** @endcond */


  /** @brief Returns Geo Cuts algorithm
   *
   * @param[in] imIn  Image<T1> in
   * @param[in] imGradx  Image<T1> gradient X
   * @param[in] imGrady  Image<T1> gradient Y
   * @param[in] imMarker  Image<T2> marker
   * @param[in] nl  Neighborlist
   * @param[out] imOut  Image<T2> out
   */
  // line no 6266
  template <class T1, class T2>
  RES_T geoCuts(const Image<T1> &imIn, const Image<T1> &imGradx,
                const Image<T1> &imGrady, const Image<T2> &imMarker,
                const StrElt &nl, Image<T2> &imOut);

  /** @brief Geo Cuts algorithm on a pixel adjacency graph, ImMarker is
   * composed of three values 0 for unmarked pixels, 2 and 3 for object and
   * background markers
   *
   * @param[in] imIn  Image<T> in
   * @param[in] imMarker  Image<T> marker
   * @param[in] nl  Neighborlist
   * @param[out] imOut  Image<T> out
   */
  // line no 6468
  template <class T1, class T2>
  RES_T geoCutsMinSurfaces(const Image<T1> &imIn, const Image<T2> &imMarker,
                            const StrElt &nl, Image<T2> &imOut);

  /** @cond */
  /** @brief Geo Cuts algorithm on a pixel adjacency graph, ImMarker is
   * composed of three values 0 for unmarked pixels, 2 and 3 for object and
   * background markers
   *
   * @param[in] imIn  Image<T> in
   * @param[in] imMarker  Image<T> marker
   * @param[in] nl  Neighborlist
   * @param[out] imOut  Image<T> out
   */
  // line no 9789
  template <class T>
  RES_T geoCutsMinSurfaces_With_Line(const Image<T> &imIn,
                                      const Image<T> &imMarker,
                                      const StrElt &nl, Image<T> &imOut);
  /** @endcond */

  /** @brief Multiple object segmentation, Geo Cuts algorithm on a pixel
   * adjacency graph, ImMarker is composed of three values 0 for unmarked
   * pixels, >0 for objects markers
   *
   * @param[in] imIn  Image<T> in
   * @param[in] imMarker  Image<T> marker
   * @param[in] nl  Neighborlist
   * @param[out] imOut  Image<T> out
   */
  // line no 9964
  template <class T1, class T2>
  RES_T geoCutsMultiway_MinSurfaces(const Image<T1> &imIn,
                                     const Image<T2> &imMarker,
                                     const StrElt &nl, Image<T2> &imOut);

  /** @} */
} // namespace smil

#include "private/GeoCuts/geo-cuts-tools.hpp"
#include "private/GeoCuts/MinSurfaces.hpp"

#endif // __D_GEOCUTS_MINSURFACE_H__
