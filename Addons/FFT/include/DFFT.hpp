/*
 * Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES
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
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _D_FFT_HPP
#define _D_FFT_HPP

#include "Core/include/private/DImage.hxx"

#include <fftw3.h>

namespace smil
{
  /**
   * @ingroup Addons
   * @addtogroup AddonFFT FFT(W) based functions
   * @{
   */

  /**
   * 2D image (normalized) cross correlation using FFT.
   *
   * Input images must have same size.
   *
   * Can be used to find a template image within a larger one:
   *
   * \code
   * correlation(sourceImg, templateImg, corrImg)
   * pt = IntPoint()
   * maxVal(corrImg, pt)
   * # gives the position of the template image within the source image
   * print pt.x, pt.y
   * \endcode
   */
  template <class T1, class T2>
  RES_T correlation(Image<T1> &imIn1, Image<T1> &imIn2, Image<T2> &imOut)
  {
    ASSERT_SAME_SIZE(&imIn1, &imIn2);

    size_t ncols  = imIn1.getWidth();
    // complex data column nbr
    size_t nccols = ncols;
    size_t nrows  = imIn2.getHeight();

    imOut.setSize(ncols, nrows);

    size_t pixNbr = ncols * nrows;

    // Allocate arrays for FFT of src and tpl
    double *src1_real        = fftw_alloc_real(pixNbr);
    fftw_complex *src1_compl = fftw_alloc_complex(nccols * nrows);
    double *src2_real        = fftw_alloc_real(pixNbr);
    fftw_complex *src2_compl = fftw_alloc_complex(nccols * nrows);

    double *res_real        = fftw_alloc_real(pixNbr);
    fftw_complex *res_compl = fftw_alloc_complex(nccols * nrows);

    T1 *src1_data = imIn1.getPixels();
    T1 *src2_data = imIn2.getPixels();
    T2 *out_data  = imOut.getPixels();

    // Copy image pixel values
    for (size_t i = 0; i < pixNbr; i++) {
      src1_real[i] = (double) src1_data[i];
      src2_real[i] = (double) src2_data[i];
    }

    // Create FFTW plans
    fftw_plan forward1 = fftw_plan_dft_r2c_2d(nrows, ncols, src1_real,
                                              src1_compl, FFTW_ESTIMATE);
    fftw_plan forward2 = fftw_plan_dft_r2c_2d(nrows, ncols, src2_real,
                                              src2_compl, FFTW_ESTIMATE);
    fftw_plan backward = fftw_plan_dft_c2r_2d(nrows, ncols, res_compl, res_real,
                                              FFTW_BACKWARD | FFTW_ESTIMATE);

    // Compute the FFT of the images
    fftw_execute(forward1);
    fftw_execute(forward2);

    // Compute the cross-correlation
    for (size_t i = 0; i < pixNbr; i++) {
      res_compl[i][0] = (src1_compl[i][0] * src2_compl[i][0] +
                         src1_compl[i][1] * src2_compl[i][1]);
      res_compl[i][1] = (src1_compl[i][1] * src2_compl[i][0] -
                         src1_compl[i][0] * src2_compl[i][1]);

      double norm = sqrt(pow(res_compl[i][0], 2) + pow(res_compl[i][1], 2));
      res_compl[i][0] /= norm;
      res_compl[i][1] /= norm;
    }

    // Compute result inverse fft
    fftw_execute(backward);

    // Copy results to imOut
    // Base results are between -1 and 1. We stretch/translate values to the
    // output type value range.
    for (size_t i = 0; i < pixNbr; i++) {
      out_data[i] = ImDtTypes<T2>::min() +
                    (res_real[i] / pixNbr + 1) * ImDtTypes<T2>::cardinal() / 2;
    }

    // Clear memory
    fftw_destroy_plan(forward1);
    fftw_destroy_plan(forward2);
    fftw_destroy_plan(backward);
    fftw_free(src1_real);
    fftw_free(src1_compl);
    fftw_free(src2_real);
    fftw_free(src2_compl);
    fftw_free(res_real);
    fftw_free(res_compl);

    imOut.modified();

    return RES_OK;
  }

  /**@}*/

} // namespace smil

#endif // _D_FFT_HPP
