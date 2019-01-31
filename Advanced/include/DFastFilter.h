#ifndef _DFAST_FILTER_H_
#define _DFAST_FILTER_H_

#include "Core/include/DCore.h"

namespace smil
{
  /**
   * @ingroup   Advanced
   * @defgroup  AdvFastFilter        Fast Bilateral Filter
   * @{ */

  /** @brief FastBilateral filter : Smooth the picture while conserving the
   * edges
   * @param[in]  imIn the initial image
   * @param[in]	Method : 1 for a gaussian window. Otherwize it is a tukey
   * window.
   * @param[in]  nS : size of the neigbourhood (Outside this window, the gaussian
   * is null) (common value 5)
   * @param[in]  EctS : standard deviation (std) for the spatial filtering
   * (common value : 3 or 5)
   * @param[in]  EctG : standard deviation (std) for the gray level filtering
   * (common value :20 or 40)
   * @param[out] imOut : Result of the bilateral filter of size Lenght
   */
  template <class T1, class T2>
  RES_T ImFastBilateralFilter(const Image<T1> &imIn, const UINT8 Method,
                              const UINT8 nS, const UINT32 EctS,
                              const UINT32 EctG, Image<T2> &imOut);

  /** @} */
} // namespace smil

#include "FastBilateralFilter/FastBilateralFilter_T.hpp"

#endif // _DFAST_FILTER_H_
