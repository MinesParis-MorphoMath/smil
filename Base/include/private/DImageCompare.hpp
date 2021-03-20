/*
 * __HEAD__
 * Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES
 * Copyright (c) 2017-2021 Centre de Morphologie Mathematique
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
 *   This file does... some very complex morphological operation...
 *
 * History :
 *   - 08/06/2020 - by Jose-Marcio Martins da Cruz
 *     Porting from xxx
 *
 * __HEAD__ - Stop here !
 */

#ifndef _D_IMAGE_COMPARE_HPP
#define _D_IMAGE_COMPARE_HPP

namespace smil
{
// Estimator epsilon
#define DXM 0.5

  /**
   * @ingroup Base
   * @addtogroup Similarity
   *
   * @details This module provides functions to evaluate the similarity between
   * two images. The main usage may be to validate algorithms when comparing
   * results of some work with some @TI{Ground Truth}
   *
   * The following conventions are used in the definition of some indices in
   * @TB{binary images} :
   *
   * - @TB{Positives} :
   *    @f[ P(imGt) = |\{p \in imGt \;|\; imGt(p) = True  \}| = TP(imGt, imIn)
   *                                                     + FN(imGt, imIn) @f]
   * - @TB{Negatives} :
   *    @f[ N(imGt) = |\{p \in imGt \;|\; imGt(p) = False \}| = TN(imGt, imIn)
   *                                                     + FP(imGt, imIn) @f]
   * - @TB{True positives} :
   *    @f[ TP(imGt, imIn) =
   *    |\{p \in imIn \:|\: imIn(p) = True \land  imGt(p) = True  \}| @f]
   * - @TB{False positives} :
   *    @f[ FP(imGt, imIn) =
   *    |\{p \in imIn \:|\: imIn(p) = True \land  imGt(p) = False  \}| @f]
   * - @TB{True negatives} :
   *    @f[ TN(imGt, imIn) =
   *    |\{p \in imIn \:|\: imIn(p) = False \land  imGt(p) = False  \}| @f]
   * - @TB{False negatives} :
   *    @f[ FN(imGt, imIn) =
   *    |\{p \in imIn \:|\: imIn(p) = False \land  imGt(p) = True  \}| @f]
   *
   * @see
   * - isBinary()
   *
   *
   * @{ */

  /**
   * indexJaccard()
   *
   * - for binary images, this function returns the
   *    @TB{Jaccard similarity coefficient} defined by :
   * @f[
   *  Jaccard(imGt, imIn) = \dfrac{|imGt \cap imIn|}{\vert imGt \cup imIn \vert}
   * = \dfrac{area(logicAnd(imGt, \; imIn))}{area(logicOr(imGt, \; imIn))}
   * @f]
   * - for non binary images, this function returns the
   *    @TB{Weighted Jaccard similarity coefficient}, also known as
   *    @TB{Ruzicka coefficient} - see indexRuzicka()
   *
   * @see
   * - Wikipedia : @UrlWikipedia{Jaccard_index, Jaccard Index}
   *
   * @param[in] imGt : @TI{Ground Truth} image
   * @param[in] imIn : image to verify
   * @returns Jaccard similarity coefficient between the two images
   */
  template <typename T>
  double indexJaccard(const Image<T> &imGt, const Image<T> &imIn)
  {
    ASSERT_ALLOCATED(&imGt, &imIn);
    ASSERT_SAME_SIZE(&imGt, &imIn);
    if (isBinary(imGt) && isBinary(imIn)) {
      Image<T> imOr(imGt);
      Image<T> imAnd(imGt);

      logicOr(imGt, imIn, imOr);
      logicAnd(imGt, imIn, imAnd);

      return double(area(imAnd) + DXM) / double(area(imOr) + DXM);
    }
    return indexRuzicka(imGt, imIn);
  }

  /**
   * indexRuzicka()
   *
   * Returns the @TB{Weighted Jaccard similarity coefficient}, also known as
   *    @TB{Ruzicka coefficient} - see indexJaccard()
   *
   * @f[
   *  Ruzicka(imGt, imIn) = \dfrac{\sum_{p} min(imGt(p), \; imIn(p))}
   *                              {\sum_{p} max(imGt(p), \; imIn(p))}
   *                      = \dfrac{volume(inf(imGt, \; imIn))}
   *                              {volume(sup(imGt, \; imIn))}
   * @f]
   *
   * @see
   * - Wikipedia : @UrlWikipedia{Jaccard_index, Jaccard Index}
   *
   * @param[in] imGt : @TI{Ground Truth} image
   * @param[in] imIn : image to verify
   * @returns Ruzicka similarity coefficient between the two images
   */
  template <typename T>
  double indexRuzicka(const Image<T> &imGt, const Image<T> &imIn)
  {
    ASSERT_ALLOCATED(&imGt, &imIn);
    ASSERT_SAME_SIZE(&imGt, &imIn);

    Image<T> imMax(imGt);
    Image<T> imMin(imGt);

    sup(imGt, imIn, imMax);
    inf(imGt, imIn, imMin);

    return double(volume(imMin) + DXM) / double(volume(imMax) + DXM);
  }

  /**
   * distanceHamming()
   *
   * Returns the number of pixels with different values in the two images.
   *
   * @see
   * - Wikipedia : @UrlWikipedia{Hamming_distance, Hamming distance}
   *
   * @param[in] imGt : @TI{Ground Truth} image
   * @param[in] imIn : image to verify
   * @returns Hamming distance between two images
   */
  template <typename T>
  size_t distanceHamming(const Image<T> &imGt, const Image<T> &imIn)
  {
    ASSERT_ALLOCATED(&imGt, &imIn);
    ASSERT_SAME_SIZE(&imGt, &imIn);

    Image<T> imOut(imGt);

    diff(imGt, imIn, imOut);

    return area(imOut);
  }

  /** @cond */
  /**
   * Not yet integrated
   */
  template <typename T>
  class ContingencyTable {
    size_t FP, FN, TP, TN;
    size_t P, N;
    size_t aGT, aIM;

    ContingencyTable(Image<T> &imGt, Image<T> imIn)
    {
      size_t nPixels = imGt.getPixelCount();

      aGT = area(imGt);
      aIM = area(imIn);

      P = aGT;
      N = nPixels - P;

      Image<T> imTmp(imGt);
      inf(imGt, imIn, imTmp);
      TP = area(imTmp);
      FP = aIM - TP;
      FN = aGT - TP;
      TN = N - TP - FP - FN;
    }

    double Accuracy()
    {
      return double(TP + TN) / double(P + N);
    }

    double Precision()
    {
      return double(TP) / double(TP + FP);
    }

    double Recall()
    {
      return double(TP) / double (TP + FN);
    }

    double FScore(double beta)
    {
      double b2 = b2 * b2;
      double p = Precision();
      double r = Recall();

      return (1 + b2) * (p * r) / (b2 * p + r);
    }

    double Sensitivity()
    {
      return double(TP) / double(TP + FN);
    }

    double Specificity()
    {
      return double(TN) / double(TN + FP);
    }

    double FallOut()
    {
      return double(FP) / double(TN + FP);
    }

    double MissRate()
    {
      return double(FN) / double(FP + FN);
    }

    double Overlap()
    {
      return double(TP) / double(min(aGT, aIM));
    }
  };
  /** @endcond */

  /**
   * indexAccuracy()
   *
   * Returns the @TB{Rand Index}, also called @TB{Accuracy} or
   * @TB{Simple matching coefficient}
   *
   * - For binary images (see isBinary()) this value is evaluated as :
   *  @f[
   *    Accuracy(imGt, imIn) = \dfrac{TP + TN}{TP + FP + TN + FN}
   *  @f]
   * - for non binary images, pixels values are considered to be equal if their
   *   difference is not greater than @TT{threshold}
   *
   * @see
   * - Wikipedia : @UrlWikipedia{Accuracy_and_precision, Accuracy and precision}
   * - Wikipedia : @UrlWikipedia{Rand_index, Rand Index}
   * - Wikipedia : @UrlWikipedia{Simple_matching_coefficient,
   *                             Simple matching coefficient}
   *
   * @param[in] imGt : @TI{Ground Truth} image
   * @param[in] imIn : image to verify
   * @param[in] threshold : difference between pixels accepted as equality.
   *
   * @returns the @TB{indexAccuracy}
   */
  template <typename T>
  double indexAccuracy(const Image<T> &imGt, const Image<T> &imIn,
                       const T threshold = 0)
  {
    ASSERT_ALLOCATED(&imGt, &imIn);
    ASSERT_SAME_SIZE(&imGt, &imIn);

    size_t nbPixels = imGt.getPixelCount();
    if (isBinary(imGt) && isBinary(imIn)) {
      Image<T> imOut(imGt);

      equ(imGt, imIn, imOut);
      size_t dp = area(imOut);
      return double(dp) / nbPixels;
    } else {
      Image<T> imOut(imGt);
      absDiff(imGt, imIn, imOut);
      T tVal = 0;
      T fVal = 1;
      compare(imOut, ">", threshold, tVal, fVal, imOut);
      size_t dp = area(imOut);
      return double(dp) / nbPixels;
    }
  }

  /**
   * indexPrecision()
   *
   * Returns, for binary images (see isBinary()), the @TB{Precision} index, also
   * called  @TB{Positive prediction value}
   *
   * @f[
   *  Precision(imGt, imIn) = \dfrac{TP}{TP + FP}
   * @f]
   *
   * @see
   * - Wikipedia : @UrlWikipedia{Precision_and_recall, Precision and Recall}
   *
   * @param[in] imGt : @TI{Ground Truth} image
   * @param[in] imIn : image to verify
   * @returns indexPrecision
   */
  template <typename T>
  double indexPrecision(const Image<T> &imGt, const Image<T> &imIn)
  {
    ASSERT_ALLOCATED(&imGt, &imIn);
    ASSERT_SAME_SIZE(&imGt, &imIn);

    if (!isBinary(imGt) || !isBinary(imIn)) {
      ERR_MSG("This function is defined only for binary images");
      return 0.;
    }

    Image<T> imTmp(imGt);
    inf(imGt, imIn, imTmp);

    return double(area(imTmp) + DXM) / double(area(imIn) + DXM);
  }

  /**
   * indexRecall()
   *
   * Returns, for binary images (see isBinary()), the @TB{Recall} index, also
   * called @TB{Sensitivity} or @TB{Hit rate} or @TB{True Positive Rate}
   *
   * @f[
   *  Recall(imGt, imIn) = \dfrac{TP}{TP + FN}
   * @f]
   *
   * @see
   * - Wikipedia : @UrlWikipedia{Precision_and_recall, Precision and Recall}
   *
   * @param[in] imGt : @TI{Ground Truth} image
   * @param[in] imIn : image to verify
   * @returns indexRecall
   */
  template <typename T>
  double indexRecall(const Image<T> &imGt, const Image<T> &imIn)
  {
    ASSERT_ALLOCATED(&imGt, &imIn);
    ASSERT_SAME_SIZE(&imGt, &imIn);

    return indexSensitivity(imGt, imIn);
  }

  /**
   * indexFScore()
   *
   * Returns, for binary images (see isBinary()), the @TT{F-Score} between
   * two images.
   *
   * @f[
   *  F_\beta(imGt, imIn) = (1 + \beta^2) . \dfrac{Precision \; . \; Recall}
   *            {\beta^2 \; . \; Precision + Recall}
   * @f]
   *
   * @see
   * - Wikipedia : @UrlWikipedia{F-score, F-Score}
   * - indexPrecision() and indexRecall()
   *
   * @param[in] imGt : @TI{Ground Truth} image
   * @param[in] imIn : image to verify
   * @param[in] beta : @f$ \beta \f$ coefficient (default value : @f$1.@f$)
   * @returns indexFScore
   */
  template <typename T>
  double indexFScore(const Image<T> &imGt, const Image<T> &imIn,
                     const double beta = 1.)
  {
    ASSERT_ALLOCATED(&imGt, &imIn);
    ASSERT_SAME_SIZE(&imGt, &imIn);

    if (!isBinary(imGt) || !isBinary(imIn)) {
      ERR_MSG("This function is defined only for binary images");
      return 0.;
    }

    double precision = indexPrecision(imGt, imIn);
    double recall    = indexRecall(imGt, imIn);

    double b2 = beta * beta;
    return (1. + b2) * (precision * recall) / (b2 * precision + recall);
  }

  /**
   * indexSensitivity()
   *
   * Returns, for binary images (see isBinary()), the @TB{Sensitivity}, also
   * called @TB{Recall}, @TB{Hit rate} or
   * @TB{True Positive Rate}
   *
   * @f[
   *  Sensitivity(imGt, imIn) = \dfrac{TP}{TP+FN}
   * @f]
   *
   * @see
   * - Wikipedia : @UrlWikipedia{Sensitivity_and_specificity,
   *                             Sensitivity and Specificity}
   *
   * @param[in] imGt : @TI{Ground Truth} image
   * @param[in] imIn : image to verify
   * @returns indexSensitivity
   */
  template <typename T>
  double indexSensitivity(const Image<T> &imGt, const Image<T> &imIn)
  {
    ASSERT_ALLOCATED(&imGt, &imIn);
    ASSERT_SAME_SIZE(&imGt, &imIn);

    if (!isBinary(imGt) || !isBinary(imIn)) {
      ERR_MSG("This function is defined only for binary images");
      return 0.;
    }

    Image<T> imTmp(imGt);
    inf(imGt, imIn, imTmp);

    return double(area(imTmp) + DXM) / double(area(imGt) + DXM);
  }

  /**
   * indexSpecificity()
   *
   * Returns, for binary images (see isBinary()), the @TB{Specificity} index,
   * also called @TB{Selectivity} or
   * @TB{True negative rate}
   *
   * @f[
   *  Specificity(imGt, imIn) = \dfrac{TN}{TN+FP}
   * @f]
   *
   * @see
   * - Wikipedia : @UrlWikipedia{Sensitivity_and_specificity,
   *                             Sensitivity and Specificity}
   *
   * @param[in] imGt : @TI{Ground Truth} image
   * @param[in] imIn : image to verify
   * @returns indexSpecificity
   */
  template <typename T>
  double indexSpecificity(const Image<T> &imGt, const Image<T> &imIn)
  {
    ASSERT_ALLOCATED(&imGt, &imIn);
    ASSERT_SAME_SIZE(&imGt, &imIn);

    if (!isBinary(imGt) || !isBinary(imIn)) {
      ERR_MSG("This function is defined only for binary images");
      return 0.;
    }

    Image<T> imTmp(imGt);

    size_t nbPixels = imGt.getPixelCount();

    sup(imGt, imIn, imTmp);

    return double(nbPixels - area(imTmp) + DXM) /
           double(nbPixels - area(imGt) + DXM);
  }


  /**
   * indexFallOut()
   *
   * Returns, for binary images (see isBinary()), the @TB{Fall Out} index,
   * also called @TB{False positive rate} or @TB{False alarm rate}
   *
   * @f[
   *  Fallout(imGt, imIn) = \dfrac{FP}{TN+FP}
   * @f]
   *
   * @see
   * - Wikipedia : @UrlWikipedia{F-score, F-Score}
   *
   * @param[in] imGt : @TI{Ground Truth} image
   * @param[in] imIn : image to verify
   * @returns indexFallOut
   */
  template <typename T>
  double indexFallOut(const Image<T> &imGt, const Image<T> &imIn)
  {
    ASSERT_ALLOCATED(&imGt, &imIn);
    ASSERT_SAME_SIZE(&imGt, &imIn);

    if (!isBinary(imGt) || !isBinary(imIn)) {
      ERR_MSG("This function is defined only for binary images");
      return 0.;
    }

    size_t nbPixels = imGt.getPixelCount();

    Image<T> imTmp(imGt);
    sup(imGt, imIn, imTmp);
    size_t TN = nbPixels - area(imTmp);

    inf(imGt, imIn, imTmp);
    size_t FP = area(imIn) - area(imTmp);

    return double(FP + DXM) / double(TN + FP + DXM);
  }

  /**
   * indexMissRate()
   *
   * Returns, for binary images (see isBinary()), the @TB{Miss Rate} index,
   * also called @TB{False negative rate}
   *
   * @f[
   *  MissRate(imGt, imIn) = \dfrac{FN}{TP+FN}
   * @f]
   *
   * @see
   * - Wikipedia : @UrlWikipedia{Sensitivity_and_specificity,
   *                             Sensitivity and Specificity}
   *
   * @param[in] imGt : @TI{Ground Truth} image
   * @param[in] imIn : image to verify
   * @returns indexSpecificity
   */
  template <typename T>
  double indexMissRate(const Image<T> &imGt, const Image<T> &imIn)
  {
    ASSERT_ALLOCATED(&imGt, &imIn);
    ASSERT_SAME_SIZE(&imGt, &imIn);

    if (!isBinary(imGt) || !isBinary(imIn)) {
      ERR_MSG("This function is defined only for binary images");
      return 0.;
    }

    Image<T> imTmp(imGt);
    inf(imGt, imIn, imTmp);
    size_t TP = area(imTmp);
    size_t FN = area(imGt) - TP;

    return double(FN + DXM) /
           double(TP + FN + DXM);
  }

  /**
   * indexOverlap()
   *
   * Returns, for binary images (see isBinary()), the @TB{Overlap} coefficient
   *
   * @f[
   *  Overlap(imGt, imIn) = \dfrac{|imGt \cap imIn|}{min(|imGt|, |imIn|)} =
   *                        \dfrac{area(inf(imGt, imIn))}
   *                              {min(area(imGt), area(imIn))}
   * @f]
   *
   * @see
   * - Wikipedia : @UrlWikipedia{Overlap_coefficient, Overlap coefficient}
   *
   * @param[in] imGt : @TI{Ground Truth} image
   * @param[in] imIn : image to verify
   * @returns indexSpecificity   */
  template <typename T>
  double indexOverlap(const Image<T> &imGt, const Image<T> &imIn)
  {
    ASSERT_ALLOCATED(&imGt, &imIn);
    ASSERT_SAME_SIZE(&imGt, &imIn);

    if (!isBinary(imGt) || !isBinary(imIn)) {
      ERR_MSG("This function is defined only for binary images");
      return 0.;
    }

    Image<T> imTmp(imGt);
    inf(imGt, imIn, imTmp);

    return double(area(imTmp) + DXM) /
           double(min(area(imGt), area(imIn)) + DXM);
  }

  /** @} */

#undef DXM
} // namespace smil

#endif // _D_IMAGE_COMPARE_HPP
