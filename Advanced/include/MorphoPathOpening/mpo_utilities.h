/* __HEAD__
 * Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES
 * Copyright (c) 2017-2019, Centre de Morphologie Mathematique
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
 *   Portage de MorphM vers Smil
 *
 * History :
 *   - XX/XX/XXXX - by Andres Serna
 *     Ported from MorphM
 *   - 10/02/2019 - by Jose-Marcio
 *     Integrated into Smil Advanced Source Tree with some cosmetics
 *
 * __HEAD__ - Stop here !
 */


#ifndef __MPO_UTILITIES_HPP__
#define __MPO_UTILITIES_HPP__

namespace smil
{
//! Defines for comparing two float types
#define EPSILON 0.000000001

  typedef struct tagMyPOINTS {
    int Seuil, Dist;
  } MyPOINTS;

} // namespace smil

#endif // __MPO_UTILITIES_HPP__
