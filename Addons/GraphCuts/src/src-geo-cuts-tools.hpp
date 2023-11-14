/* __HEAD__
 * Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES
 * Copyright (c) 2017-2023, Centre de Morphologie Mathematique
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
 *   Common Geo-Cuts Tools
 *
 * History :
 *   - 20/03/2019 - by Jose-Marcio Martins da Cruz
 *     Just created it...
 *   - XX/XX/XXXX -
 *
 *
 * __HEAD__ - Stop here !
 */

#ifndef _D_GEOCUTSTOOLS_HPP_
#define _D_GEOCUTSTOOLS_HPP_


  /*
   * Some useful tools
   */

/*
 *
 *
 *
 */

namespace smil {

static int debugOn = false;

#ifndef SMIL_ENTER_FUNCTION
#define SMIL_ENTER_FUNCTION(a)                                                 \
  {                                                                            \
    if (debugOn)                                                               \
      std::cout << "Entering function " << __func__ << " " << (a)              \
                << std::endl;                                                  \
  }

#define SMIL_REGISTER_ERROR(a)
#endif

// #include <math.h>
#define M_PI 3.14159265358979323846
#define INFINI_POSITIF std::numeric_limits<double>::max)()
#define _SECURE_SCL 0
  // #include <stdio.h>

  /* This function selects points from an Structuring element
   * generating a positive offset
   */
  inline vector<IntPoint> filterStrElt(StrElt se)
  {
    vector<IntPoint> pts(0);

    vector<IntPoint>::iterator it, itStart, itEnd;
    itStart = se.points.begin();
    itEnd   = se.points.end();

    for (it = itStart; it != itEnd; it++) {
      bool ok = (4 * it->z + 2 * it->y + it->x) > 0;
      if (ok)
        pts.push_back(*it);

      if (debugOn) {
        std::cout << (ok ? "GOOD " : "BAD  ") << std::right << " "
                  << std::setw(6) << it->x << " " << std::setw(6) << it->y
                  << " " << std::setw(6) << it->z << endl;
      }
    }
    return pts;
  }


}
#endif // _D_GEOCUTSTOOLS_HPP_
