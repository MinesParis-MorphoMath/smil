/* __HEAD__
 * Copyright (c) 2017-2020, Centre de Morphologie Mathematique
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
 *     * Names of the author of the contributors can't be used to endorse
 *       or promote products derived from this software without specific
 *       prior written permission.
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
 *   This file does... some very complex morphological operation...
 *
 * History :
 *   - XX/XX/XXXX - by Joe.Denver
 *     Just created it...
 *
 * __HEAD__ - Stop here !
 */

#include "Core/include/DCore.h"
#include "DMorpho.h"

using namespace smil;

int main()
{
  Image<UINT8> im1(5562, 7949);
  Image<UINT8> im2(im1);

  UINT BENCH_NRUNS = 1E2;

  BENCH_IMG_STR(dilate, "hSE", im1, im2, hSE());
  BENCH_IMG_STR(dilate, "sSE", im1, im2, sSE());
  BENCH_IMG_STR(dilate, "CrossSE", im1, im2, CrossSE());
  BENCH_IMG_STR(open, "hSE", im1, im2, hSE());
  BENCH_IMG_STR(open, "sSE", im1, im2, sSE());
  BENCH_IMG_STR(open, "CrossSE", im1, im2, CrossSE());
}
