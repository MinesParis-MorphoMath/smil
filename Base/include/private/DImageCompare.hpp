/*
 * __HEAD__
 * Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES
 * Copyright (c) 2017-2024 Centre de Morphologie Mathematique
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

#include <algorithm>

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
   * results of some algorithm against what is expected (the @TI{Ground Truth})
   *
   * The following conventions are used in the definition of some indices for
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
   * - Wikipedia : @UrlWikipedia{Confusion_matrix, Confusion matrix}
   * - isBinary()
   *
   *
   * @{ */

  /** @cond */
  /**
   * Not yet integrated
   *
   * @see
   * - Wikipedia : @UrlWikipedia{Confusion_matrix, Confusion matrix}
   */
  template <typename T>
  class ConfusionMatrix {
  private:
    size_t FP, FN, TP, TN;
    size_t P, N;
    size_t aGT, aIM;
    bool   okTable;
  public:
    ConfusionMatrix(const Image<T> &imGt, const Image<T> &imIn)
    {
      okTable = false;
      if (!imGt.isAllocated() || !imIn.isAllocated())
      {
        ERR_MSG("imGt or imIn aren't allocated");
        return;
      }
      if (!haveSameSize(&imGt, &imIn, NULL))
      {
        ERR_MSG("imGt and imIn don't have the same size");
        return;
      }
      if (!isBinary(imGt) || !isBinary(imIn)) {
        ERR_MSG("This function is defined only for binary images");
        return;
      }

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
      TN = N - FP;

      okTable = true;
    }

    double Accuracy()
    {
      if (!okTable)
        return 0.;

      return double(TP + TN + DXM) / double(P + N + DXM);
    }

    double Precision()
    {
      if (!okTable)
        return 0.;

      return double(TP + DXM) / double(TP + FP + DXM);
    }

    double Recall()
    {
      if (!okTable)
        return 0.;

      return double(TP + DXM) / double (TP + FN + DXM);
    }

    double FScore(double beta = 1.)
    {
      if (!okTable)
        return 0.;

      double b2 = beta * beta;
      double p = Precision();
      double r = Recall();

      return (1 + b2) * (p * r) / (b2 * p + r);
    }

    double Sensitivity()
    {
      if (!okTable)
        return 0.;

      return double(TP + DXM) / double(TP + FN + DXM);
    }

    double Specificity()
    {
      if (!okTable)
        return 0.;

      return double(TN + DXM) / double(TN + FP + DXM);
    }

    double FallOut()
    {
      if (!okTable)
        return 0.;

      return double(FP + DXM) / double(TN + FP + DXM);
    }

    double MissRate()
    {
      if (!okTable)
        return 0.;

      return double(FN + DXM) / double(TP + FN + DXM);
    }

    double Overlap()
    {
      if (!okTable)
        return 0.;

      return double(TP + DXM) / double(std::min(aGT, aIM) + DXM);
    }

    double Jaccard()
    {
      if (!okTable)
        return 0.;

      return double(TP + DXM) / double(FN + TP + FP + DXM);
    }

    size_t Hamming()
    {
      if (!okTable)
        return 0.;

      return aGT + aIM - TP;
    }

    std::vector<size_t> getTable()
    {
      std::vector<size_t> table(0);

      if (!okTable)
        return table;

      table.push_back(TN);
      table.push_back(FN);
      table.push_back(FP);
      table.push_back(TP);
      return table;
    }

    void printSelf()
    {
      if (!okTable)
        return;

      std::cout << "Predicted Negative \t" << TN << "\t" << FN << std::endl;
      std::cout << "Predicted Positive \t" << FP << "\t" << TP << std::endl;
      std::cout << std::endl;
      std::cout << "Negative \t" << N << std::endl;
      std::cout << "Positive \t" << P << std::endl;
    }
  };

  /** @endcond */



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
   * @returns returns @TB{Jaccard} similarity index between two images
   */
  template <typename T>
  double indexJaccard(const Image<T> &imGt, const Image<T> &imIn)
  {
    ASSERT_ALLOCATED(&imGt, &imIn);
    ASSERT_SAME_SIZE(&imGt, &imIn);
    if (isBinary(imGt) && isBinary(imIn)) {
      ConfusionMatrix<T> cTable(imGt, imIn);
      return cTable.Jaccard();
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
   * @returns returns @TB{Ruzicka} similarity index between two images
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
   * - Wikipedia : @UrlWikipedia{Confusion_matrix, Confusion matrix}
   * - Wikipedia : @UrlWikipedia{Accuracy_and_precision, Accuracy and precision}
   * - Wikipedia : @UrlWikipedia{Rand_index, Rand Index}
   * - Wikipedia : @UrlWikipedia{Simple_matching_coefficient,
   *                             Simple matching coefficient}
   *
   * @param[in] imGt : @TI{Ground Truth} image
   * @param[in] imIn : image to verify
   * @param[in] threshold : difference between pixels accepted as equality.
   *
   * @returns the @TB{Accuracy} index between two images
   */
  template <typename T>
  double indexAccuracy(const Image<T> &imGt, const Image<T> &imIn,
                       const T threshold = 0)
  {
    ASSERT_ALLOCATED(&imGt, &imIn);
    ASSERT_SAME_SIZE(&imGt, &imIn);

    if (isBinary(imGt) && isBinary(imIn)) {
      ConfusionMatrix<T> cTable(imGt, imIn);
      return cTable.Accuracy();
    } else {
      size_t nbPixels = imGt.getPixelCount();
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
   * - Wikipedia : @UrlWikipedia{Confusion_matrix, Confusion matrix}
   * - Wikipedia : @UrlWikipedia{Precision_and_recall, Precision and Recall}
   * @note
   * - only binary images (two classes)
   *
   * @param[in] imGt : @TI{Ground Truth} image
   * @param[in] imIn : image to verify
   * @returns returns @TB{Precision} index between two images
   */
  template <typename T>
  double indexPrecision(const Image<T> &imGt, const Image<T> &imIn)
  {
    ConfusionMatrix<T> cTable(imGt, imIn);
    return cTable.Precision();
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
   * - Wikipedia : @UrlWikipedia{Confusion_matrix, Confusion matrix}
   * - Wikipedia : @UrlWikipedia{Precision_and_recall, Precision and Recall}
   * @note
   * - only binary images (two classes)
   *
   * @param[in] imGt : @TI{Ground Truth} image
   * @param[in] imIn : image to verify
   * @returns returns @TB{Recall} index between two images
   */
  template <typename T>
  double indexRecall(const Image<T> &imGt, const Image<T> &imIn)
  {
    ConfusionMatrix<T> cTable(imGt, imIn);
    return cTable.Recall();
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
   * - Wikipedia : @UrlWikipedia{Confusion_matrix, Confusion matrix}
   * - Wikipedia : @UrlWikipedia{F-score, F-Score}
   * - indexPrecision() and indexRecall()
   * @note
   * - only binary images (two classes)
   *
   * @param[in] imGt : @TI{Ground Truth} image
   * @param[in] imIn : image to verify
   * @param[in] beta : @f$ \beta \f$ coefficient (default value : @TT{1.})
   * @returns returns @TB{F-Score} index between two images
   */
  template <typename T>
  double indexFScore(const Image<T> &imGt, const Image<T> &imIn,
                     const double beta = 1.)
  {
    ConfusionMatrix<T> cTable(imGt, imIn);
    return cTable.FScore(beta);
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
   * - Wikipedia : @UrlWikipedia{Confusion_matrix, Confusion matrix}
   * - Wikipedia : @UrlWikipedia{Sensitivity_and_specificity,
   *                             Sensitivity and Specificity}
   * @note
   * - only binary images (two classes)
   *
   * @param[in] imGt : @TI{Ground Truth} image
   * @param[in] imIn : image to verify
   * @returns returns @TB{Sensitivity} index between two images
   */
  template <typename T>
  double indexSensitivity(const Image<T> &imGt, const Image<T> &imIn)
  {
    ConfusionMatrix<T> cTable(imGt, imIn);
    return cTable.Sensitivity();
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
   * - Wikipedia : @UrlWikipedia{Confusion_matrix, Confusion matrix}
   * - Wikipedia : @UrlWikipedia{Sensitivity_and_specificity,
   *                             Sensitivity and Specificity}
   * @note
   * - only binary images (two classes)
   *
   * @param[in] imGt : @TI{Ground Truth} image
   * @param[in] imIn : image to verify
   * @returns returns @TB{Specificity} index between two images
   */
  template <typename T>
  double indexSpecificity(const Image<T> &imGt, const Image<T> &imIn)
  {
    ConfusionMatrix<T> cTable(imGt, imIn);
    return cTable.Specificity();
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
   * - Wikipedia : @UrlWikipedia{Confusion_matrix, Confusion matrix}
   * - Wikipedia : @UrlWikipedia{Sensitivity_and_specificity,
   *                             Sensitivity and specificity}
   * @note
   * - only binary images (two classes)
   *
   * @param[in] imGt : @TI{Ground Truth} image
   * @param[in] imIn : image to verify
   * @returns returns @TB{FallOut} index between two images
   */
  template <typename T>
  double indexFallOut(const Image<T> &imGt, const Image<T> &imIn)
  {
    ConfusionMatrix<T> cTable(imGt, imIn);
    return cTable.FallOut();
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
   * - Wikipedia : @UrlWikipedia{Confusion_matrix, Confusion matrix}
   * - Wikipedia : @UrlWikipedia{Sensitivity_and_specificity,
   *                             Sensitivity and Specificity}
   * @note
   * - only binary images (two classes)
   *
   * @param[in] imGt : @TI{Ground Truth} image
   * @param[in] imIn : image to verify
   * @returns returns @TB{MissRate} index between two images
   */
  template <typename T>
  double indexMissRate(const Image<T> &imGt, const Image<T> &imIn)
  {
    ConfusionMatrix<T> cTable(imGt, imIn);
    return cTable.MissRate();
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
   * @note
   * - only binary images (two classes)
   *
   * @param[in] imGt : @TI{Ground Truth} image
   * @param[in] imIn : image to verify
   * @returns returns @TB{Overlap} index between two images
   */
  template <typename T>
  double indexOverlap(const Image<T> &imGt, const Image<T> &imIn)
  {
    ConfusionMatrix<T> cTable(imGt, imIn);
    return cTable.Overlap();
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
   * @returns returns @TB{Hamming distance} between two images
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

  /**
   * distanceHausdorff() -
   *
   *
   * @see
   * - Wikipedia : @UrlWikipedia{Hausdorff_distance, Hausdorff distance}
   * @note
   * - only binary images (two classes)
   *
   * @param[in] imGt : @TI{Ground Truth} image
   * @param[in] imIn : image to verify
   * @returns returns @TB{Hausdorff distance} between two images
   */
  template <typename T>
  double distanceHausdorff(const Image<T> &imGt, const Image<T> &imIn)
  {
    ASSERT_ALLOCATED(&imGt, &imIn);
    ASSERT_SAME_SIZE(&imGt, &imIn);

    if (!isBinary(imGt) || !isBinary(imIn)) {
      ERR_MSG("This function is defined only for binary images");
      return 0.;
    }

    Image<T> imt(imGt);

    gradient(imGt, imt, SquSE());
    inf(imGt, imt, imt);
    std::vector<size_t> pixGt = nonZeroOffsets(imt);

    gradient(imIn, imt, SquSE());
    inf(imIn, imt, imt);
    std::vector<size_t> pixIn = nonZeroOffsets(imt);

    off_t szGt = pixGt.size();
    std::vector<double> distGt(szGt, std::numeric_limits<double>::max());
    off_t szIn = pixIn.size();
    std::vector<double> distIn(szIn, std::numeric_limits<double>::max());

    size_t Size[3];
    imGt.getSize(Size);
    ImageBox box(Size);

#ifdef OPEN_MP
#pragma omp parallel for 
#endif
    for (off_t i = 0; i < szGt; i++) {
      for (off_t j = 0; j < szIn; j++) {
        double d = box.getDistance(pixGt[i], pixIn[j]);

        if (d < distGt[i])
          distGt[i] = d;
        if (d < distIn[j])
          distIn[j] = d;
      }
    }

    std::vector<double>::iterator iGt, iIn;
    iGt = std::max_element(distGt.begin(), distGt.end());
    iIn = std::max_element(distIn.begin(), distIn.end());

    return std::max(*iGt, *iIn);
  }



  /** @} */

#undef DXM
} // namespace smil

#endif // _D_IMAGE_COMPARE_HPP
