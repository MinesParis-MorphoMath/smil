/* __HEAD__
 * Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES
 * Copyright (c) 2017-2022, Centre de Morphologie Mathematique
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of Matthieu FAESSEL, or ARMINES nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS AND CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Description :
 *   Main include file for filters
 *
 * History :
 *   - 05/03/2019 - by Jose-Marcio Martins da Cruz
 *     Just created this file
 *
 * __HEAD__ - Stop here !
 */

#ifndef _D_FILTERS_H_
#define _D_FILTERS_H_

#include "Core/include/DCore.h"

namespace smil
{
  /**
   * @ingroup  AddonFilters    Filters (Non Morphological)
   *
   * A set of non "Mathematical Morphology" based image filters.
   *
   * @note
   * Some of these filters were ported from @b Morph-M (antecessor of @b Smil) :
   * Canny, Deriche, Gabor, Kuwahara, Mean Shift and Sigma and are only 2D 
   * filters. Not optimized for Smil.
   */

} // namespace smil

#include "DfilterCanny.h"
#include "DfilterDeriche.h"
// obsoleted by Dfilter3DBilateral.h
// #include "DfilterFastBilateral.h"
#include "DfilterGabor.h"
#include "DfilterKuwahara.h"
#include "DfilterMeanShift.h"
#include "DfilterSigma.h"

// #include "DfilterScale.h"

#include "Dfilter3DBilateral.h"

#endif // _D_FILTERS_H_
