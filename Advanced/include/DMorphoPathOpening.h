#ifndef _DMORPHO_PATH_OPENING_H_
#define _DMORPHO_PATH_OPENING_H_

#include "Core/include/DCore.h"

namespace smil
{
  /**
   * @ingroup   Advanced
   * @defgroup  AdvMorphoPathOpening   Morpho Path Opening/Closing
   * @{ */

  /** @brief Grayscale Path Opening Operation made by "staking" the results of
   * the binary opening applied to each threshold
   * @param[in]  imIn : the initial image
   * @param[in]  Lenght : Stop criteria values i.e. size of lenght
   * @param[out] imOut : Path Opening of imIn
   */
  template <class T1, class T2>
  RES_T ImPathOpeningBruteForce(const Image<T1> &imIn, const UINT32 Lenght,
                                Image<T2> &imOut);

  /** @brief Grayscale Path Closing Operation made by "staking" the results of
   * the binary closing applied to each threshold
   * @param[in]  imIn : the initial image
   * @param[in]  Lenght : Stop criteria values i.e. size of lenght
   * @param[out] imOut : Path Closing of imIn
   */
  template <typename T1, typename T2>
  RES_T ImPathClosingBruteForce(const Image<T1> &imIn, const UINT32 Lenght,
                                Image<T2> &imOut)
  {
    ASSERT_ALLOCATED(&imIn)
    ASSERT_SAME_SIZE(&imIn, &imOut)

    Image<T1> imNeg(imIn);
    RES_T res = inv(imIn, imNeg);
    if (res != RES_OK)
      return res;
    res = ImPathOpeningBruteForce(imNeg, Lenght, imOut);
    if (res != RES_OK)
      return res;
    return inv(imOut, imOut);
  }

  /** @brief  Fast Grayscale Path Opening Operation: made by updating the
   * distance function from on threshold to an other
   * @param[in]  imIn : the initial image
   * @param[in]  Lenght : Stop criteria values i.e. size of lenght
   * @param[out] imOut : Path Opening of imIn
   */
  template <class T>
  RES_T ImPathOpening(const Image<UINT8> &imIn, const UINT32 Lenght,
                      Image<T> &imOut);

  /** @brief  Fast Grayscale Path Closing Operation made by updating the
   * distance function from on threshold to an other
   * @param[in]  imIn : the initial image
   * @param[in]  Lenght : Stop criteria values i.e. size of lenght
   * @param[out] imOut : Path Opening of imIn
   */
  template <typename T>
  RES_T ImPathClosing(const Image<UINT8> &imIn, const UINT32 Lenght,
                      Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn)
    ASSERT_SAME_SIZE(&imIn, &imOut)
    Image<UINT8> imNeg(imIn);
    RES_T res = inv(imIn, imNeg);
    if (res != RES_OK)
      return res;
    res = ImPathOpening(imNeg, Lenght, imOut);
    if (res != RES_OK)
      return res;
    return inv(imOut, imOut);
  }

  /** @brief Ultimate Path Opening Operation with the geodesic diameter as
   * criterium
   * @param[in]  imIn : the initial image
   * @param[out] imOut : Ultimate Path Opening of imIn (contraste)
   * @param[out] imIndicatrice : Ultimate Path Opening of imIn (Size of the opening)
   * @param[in] stop : size of the max opening [-1 : max(W,H)]
   * @param[out] lambdaAttribute : accumulation [0 : no accumulation]
   */
  template <class T1, class T2>
  RES_T ImUltimatePathOpening(const Image<UINT8> &imIn, Image<T1> &imOut,
                              Image<T2> &imIndicatrice, int stop,
                              int lambdaAttribute);

  /** @brief Ultimate Path Closing Operation with a graph V3
   * @param[in]  imIn : the initial image
   * @param[out] imTrans : Ultimate Path Opening of imIn (contraste)
   * @param[out] imInd : Ultimate Path Opening of imIn (Size of the opening)
   * @param[out] stop : size of the max opening [-1 : max(W,H)]
   * @param[out] lambdaAttribute : accumulation [0 : no accumulation]
   */
  template <typename T1, typename T2>
  RES_T ImUltimatePathClosing(const Image<UINT8> &imIn, Image<T1> &imTrans,
                              Image<T2> &imInd, int stop, int lambdaAttribute)
  {
    ASSERT_ALLOCATED(&imIn)
    ASSERT_SAME_SIZE(&imIn, &imTrans)
    ASSERT_SAME_SIZE(&imIn, &imInd)
    Image<UINT8> imNeg(imIn);
    RES_T res = inv(imIn, imNeg);
    if (res != RES_OK)
      return res;
    res = ImUltimatePathOpening(imNeg, imTrans, imInd, stop, lambdaAttribute);
    if (res != RES_OK)
      return res;
    return inv(imTrans, imTrans);
  }

  /** @brief Binary Path Opening Operation
   * @param[in]  imIn : the initial image
   * @param[in]  Lenght : Stop criteria values i.e. size of lenght
   * @param[in]  Slice : Threashold of the input image : the path closing would be
   * applyed on this slice
   * @param[out] imOut : Binary Path Opening of imIn
   */
  template <class T1, class T2>
  RES_T ImBinaryPathOpening(const Image<T1> &imIn, const UINT32 Lenght,
                            const UINT32 Slice, Image<T2> &imOut);

  /** @brief Binary Path Closing Operation
   * @param[in]  imIn : the initial image
   * @param[in]  Lenght : Stop criteria values i.e. size of lenght
   * @param[in]  Slice : Threashold of the input image : the path closing would be
   * applyed on this slice
   * @param[out] imOut : Binary Path Closing of imIn
   */
  template <typename T1, typename T2>
  RES_T ImBinaryPathClosing(const Image<T1> &imIn, const UINT32 Lenght,
                            const UINT32 Slice, Image<T2> &imOut)
  {
    ASSERT_ALLOCATED(&imIn)
    ASSERT_SAME_SIZE(&imIn, &imOut)
    Image<T1> imNeg(imIn);
    RES_T res = inv(imIn, imNeg);
    if (res != RES_OK)
      return res;
    res = ImBinaryPathOpening(imNeg, Lenght, Slice, imOut);
    if (res != RES_OK)
      return res;
    return inv(imOut, imOut);
  }

  /** @brief ImGeodesicPathOpening : Compute the geodesic diameter for each
   * connected component CC for each threshold
   * @param[in]  imIn : the initial image
   * @param[in]  Lenght : size of the path (above: the CC is destroy)
   * @param[out] imOut : geodesic valuation of the path
   * @param[in]  Method : size of the path (above: the CC is destroy)
   * @param[in]  ScaleX : Scale X
   * @param[in]  ScaleY : Scale Y
   * @param[in]  ScaleZ : Scale Z
   */
  template <typename T>
  RES_T ImGeodesicPathOpening(const Image<UINT8> &imIn, double Lenght,
                              int Method, Image<T> &imOut, float ScaleX = 1,
                              float ScaleY = 1, float ScaleZ = 1);

  /** @brief ImGeodesicPathClosing : Compute the geodesic diameter for each
   * connected component CC for each threshold
   * @param[in]  imIn : the initial image
   * @param[out] imOut : geodesic valuation of the path
   * @param[in] Lenght : size of the path (above: the CC is destroy)
   * @param[in]  Method :  
   * @param[in]  ScaleX : Scale X
   * @param[in]  ScaleY : Scale Y
   * @param[in]  ScaleZ : Scale Z
   */
  template <typename T>
  RES_T ImGeodesicPathClosing(const Image<UINT8> &imIn, double Lenght,
                              int Method, Image<T> &imOut, float ScaleX = 1,
                              float ScaleY = 1, float ScaleZ = 1)
  {
    ASSERT_ALLOCATED(&imIn)
    ASSERT_SAME_SIZE(&imIn, &imOut)
    Image<UINT8> imNeg(imIn);
    RES_T res = inv(imIn, imNeg);
    if (res != RES_OK)
      return res;
    res = ImGeodesicPathOpening(imNeg, Lenght, Method, imOut, ScaleX, ScaleY,
                                ScaleZ);
    if (res != RES_OK)
      return res;
    return inv(imOut, imOut);
  }

  /** @brief ImGeodesicPathUltimateOpening : Compute the geodesic diameter for
   * each connected component CC for each threshold and then the ultimate O
   * @param[in]  imIn : the initial image
   * @param[out] imTrans : geodesic valuation of the path: image Transform
   * @param[out] imInd : geodesic valuation of the path : image Indicatrice
   * @param[in]  ScaleX : Scale X
   * @param[in]  ScaleY : Scale Y
   * @param[in]  ScaleZ : Scale Z
   * @param[in] stop : Parameter of the UO stop lenght value
   * @param[in] lambdaAttribute : Accumulation. [0, no accumulation]
   * @param[in] takeMin : As the geodesic Path is not an increasing attribute,
   * we can chose the min or the max.
   */
  template <typename T>
  RES_T ImUltimateGeodesicPathOpening(const Image<UINT8> &imIn,
                                      Image<UINT8> &imTrans, Image<T> &imInd,
                                      float ScaleX = 1, float ScaleY = 1,
                                      float ScaleZ = 1, int stop = -1,
                                      int lambdaAttribute = 0, int takeMin = 1);

  /** @brief ImGeodesicPathUltimateClosing: Compute the geodesic diameter for
   *     each connected component CC for each threshold and then the ultimate O
   * @param[in]  imIn : the initial image
   * @param[out] imTrans : geodesic valuation of the path: image Transform
   * @param[out] imIndicatrice : geodesic valuation of the path : image Indicatrice
   * @param[in]  ScaleX : Scale X
   * @param[in]  ScaleY : Scale Y
   * @param[in]  ScaleZ : Scale Z
   * @param[in] stop : Parameter of the UO stop lenght value
   * @param[in] lambdaAttribute : Accumulation. [0, no accumulation]
   * @param[in] takeMin : As the geodesic Path is not an increasing attribute,
   *     we can chose the min or the max.
   */
  template <typename T>
  RES_T ImUltimateGeodesicPathClosing(const Image<UINT8> &imIn,
                                      Image<UINT8> &imTrans,
                                      Image<T> &imIndicatrice, float ScaleX = 1,
                                      float ScaleY = 1, float ScaleZ = 1,
                                      int stop = -1, int lambdaAttribute = 0,
                                      int takeMin = 1)
  {
    ASSERT_ALLOCATED(&imIn)
    ASSERT_SAME_SIZE(&imIn, &imTrans)
    ASSERT_SAME_SIZE(&imIn, &imIndicatrice)
    Image<UINT8> imNeg(imIn);
    RES_T res = inv(imIn, imNeg);
    if (res != RES_OK)
      return res;
    res = ImUltimateGeodesicPathOpening(imNeg, imTrans, imIndicatrice, ScaleX,
                                        ScaleY, ScaleZ, stop, lambdaAttribute,
                                        takeMin);
    if (res != RES_OK)
      return res;
    return inv(imTrans, imTrans);
  }

  /** @brief ImGeodesicElongation : we compute the elongation for each CC
   * @param[in]  imIn : the initial image (binary : 0, 255)
   * @param[out] imOut : the result (gray scale )
   * @param[in]  sliceBySlice : if we apply this algorithm for the entire
   * picture or slice by slice.
   * @param[in]  dz_over_dx :
   */
  template <typename T1, typename T2>
  RES_T ImGeodesicElongation(const Image<T1> &imIn, Image<T2> &imOut,
                             int sliceBySlice  = 0,
                             double dz_over_dx = 1); // BMI SK

  /** @brief ImGeodesicExtremities : we compute the elongation for each CC
   * @param[in]  imIn : the initial image (binary : 0, 255)
   * @param[out] imOut : the result (gray scale )
   * @param[in]  sliceBySlice : if we apply this algorithm for the entire
   * picture or slice by slice.
   * @param[in]  dz_over_dx :
   */
  template <typename T1, typename T2>
  RES_T ImGeodesicExtremities(const Image<T1> &imIn, Image<T2> &imOut,
                              int sliceBySlice  = 0,
                              double dz_over_dx = 1); // BMI SK

  /** @brief ImLabelFlatZonesWithElongation : we compute the elongation for each
   * flat zone in a gray scale image. Only 8-connectivity is implemented by the
   * moment.
   * @param[in]  imIn : the initial image (gray scale)
   * @param[out] imOut : the result (gray scale)
   */
  template <typename T1, typename T2>
  RES_T ImLabelFlatZonesWithElongation(const Image<T1> &imIn, Image<T2> &imOut);

  /** @brief ImLabelFlatZonesWithExtremities : we compute the elongation for
   * each flat zone in a gray scale image. Only 8-connectivity is implemented by
   * the moment.
   * @param[in]  imIn  : the initial image (gray scale)
   * @param[out] imOut : the result (gray scale)
   */
  template <typename T1, typename T2>
  RES_T ImLabelFlatZonesWithExtremities(const Image<T1> &imIn,
                                        Image<T2> &imOut);

  /** @brief ImLabelFlatZonesWithGeodesicDiameter : we compute the geodesic
   * diameter for each flat zone in a gray scale image. Only 8-connectivity is
   * implemented by the moment.
   * @param[in]  imIn : the initial image (gray scale)
   * @param[out] imOut : the result (gray scale)
   */
  template <typename T1, typename T2>
  RES_T ImLabelFlatZonesWithGeodesicDiameter(const Image<T1> &imIn,
                                             Image<T2> &imOut);

  /** @brief ImGeodesicDiameter : we compute the geodesic diameter for each CC
   * @param[in]  imIn  : the initial image (binary : 0, 255)
   * @param[out] imOut : the result (gray scale )
   * @param[in]  sliceBySlice : if we apply this algorithm for the entire picture or slice by slice.
   * @param[in]  dz_over_dx :
   */
  template <typename T1, typename T2>
  RES_T ImGeodesicDiameter(const Image<T1> &imIn, Image<T2> &imOut,
                           int sliceBySlice = 0, double dz_over_dx = 1);

  /*  * @brief ImGeodesicTortuosity : we compute the tortuosity for each CC
   * @param[in]  imIn  : the initial image (binary : 0, 255)
   * @param[out] imOut : the result (gray scale )
   * @param[in]  SliceBySlice : if we apply this algorithm for the entire
   * picture or slice by slice.
   * @param[in]  dz_over_dx :
   */
  /* template<typename T>
 */ // BMI TODO integrate function
                              // ImGeodesicAttribute_v2
  /*   RES_T ImGeodesicTortuosity(const Image<UINT8> &imIn, Image<T> &imOut,int
   * sliceBySlice,double dz_over_dx);
 */

  /** @brief Ultimate Path Opening Operation with a graph V2
   * @param[in]  imIn : the initial image
   * @param[out] imTrans : Ultimate Path Opening of imIn (contraste)
   * @param[out] imInd : Ultimate Path Opening of imIn (Size of the opening)
   * @param[out] stop : size of the max opening [-1 : max(W,H)]
   * @param[out] lambdaAttribute : accumulation [0 : no accumulation]
   */
  template <typename T1, typename T2>
  RES_T ImUltimatePathOpening_GraphV2(const Image<UINT8> &imIn,
                                      Image<T1> &imTrans, Image<T2> &imInd,
                                      int stop = -1, int lambdaAttribute = 0);

  /** @brief Ultimate Path Closing Operation with a graph V2
   * @param[in]  imIn : the initial image
   * @param[out] imTrans : Ultimate Path Opening of imIn (contraste)
   * @param[out] imInd : Ultimate Path Opening of imIn (Size of the opening)
   * @param[out] stop : size of the max opening [-1 : max(W,H)]
   * @param[out] lambdaAttribute : accumulation [0 : no accumulation]
   */
  template <typename T1, typename T2>
  RES_T ImUltimatePathClosing_GraphV2(const Image<UINT8> &imIn,
                                      Image<T1> &imTrans, Image<T2> &imInd,
                                      int stop = -1, int lambdaAttribute = 0)
  {
    ASSERT_ALLOCATED(&imIn)
    ASSERT_SAME_SIZE(&imIn, &imTrans)
    ASSERT_SAME_SIZE(&imIn, &imInd)
    Image<UINT8> imNeg(imIn);
    RES_T res = inv(imIn, imNeg);
    if (res != RES_OK)
      return res;
    res = ImUltimatePathOpening_GraphV2(imNeg, imTrans, imInd, stop,
                                        lambdaAttribute);
    if (res != RES_OK)
      return res;
    return inv(imTrans, imTrans);
  }

  /** @brief Path Opening Operation with a graph V2
   * @param[in]  imIn : the initial image
   * @param[out] imOut : pathOpening of imIn
   * @param[in] Lenght : criteria value i.e. size of lenght
   */
  template <typename T>
  RES_T ImPathOpening_GraphV2(const Image<UINT8> &imIn, double Lenght,
                              Image<T> &imOut);

  /** @brief Path Closing Operation with a graph V2
   * @param[in]  imIn : the initial image
   * @param[out] imOut : pathOpening of imIn
   * @param[in] Lenght : criteria value i.e. size of lenght
   */
  template <typename T>
  RES_T ImPathClosing_GraphV2(const Image<UINT8> &imIn, double Lenght,
                              Image<T> &imOut)
  {
    ASSERT_ALLOCATED(&imIn)
    ASSERT_SAME_SIZE(&imIn, &imOut)
    Image<UINT8> imNeg(imIn);
    RES_T res = inv(imIn, imNeg);
    if (res != RES_OK)
      return res;
    res = ImPathOpening_GraphV2(imNeg, Lenght, imOut);
    if (res != RES_OK)
      return res;
    return inv(imOut, imOut);
  }

  /** @brief ImThresholdWithUniqueCCForBackGround : for each slice of a 3D
   * image, we select a threshold to have one CC of the background
   * @param[in]  imIn : the initial image
   * @param[out] imOut : the result (binary : 0, 255)
   * @param[in]  SliceBySlice : if we apply this algorithm for the entire
   * picture or slice by slice.
   */
  template <class T1, class T2>
  RES_T ImThresholdWithUniqueCCForBackGround(const Image<T1> &imIn,
                                             Image<T2> &imOut,
                                             int SliceBySlice = 0);

  /** @brief ImThresholdWithMuAndSigma : for each slice of a 3D image, we
   * compute Mu and Sigma and we apply a threshold of (MU + paramSigma * Sigma)
   * @param[in]  imIn  : the initial image
   * @param[out] imOut : the result (binary : 0, 255)
   * @param[in]  paramSigma  : threshold : Mu + paramSigma * Sigma
   */
  template <class T1, class T2>
  RES_T ImThresholdWithMuAndSigma(const Image<T1> &imIn, Image<T2> &imOut,
                                  float paramSigma = 1);

  /** @brief PseudoPatternSpectrum : With the contrast and the indicatrice
   * images from the UPO, we compute the pattern spectrum
   * @param[in]  imIn : 
   * @param[in]  imIn2 : 
   * @param[out] patternSpect : 
   * image
   */
  template <class T1, class T2>
  RES_T PseudoPatternSpectrum(const Image<T1> &imIn, const Image<T2> &imIn2,
                              int *patternSpect);

  /** @brief ImSupSmallRegion : Delete from an image of the conected component
   *where their area is less than their gray level times a percentage.
   * @param[in]  imIndicatrice : the initial image
   * @param[out] imIndicatriceCorr : the initial image filtered
   * @param[in] percentage :
   */
  template <class T1, class T2>
  RES_T ImSupSmallRegion(const Image<T1> &imIndicatrice,
                         Image<T2> &imIndicatriceCorr, float percentage);

  /** @brief ImElongationFromSkeleton : from a binary image and its skeleton, we
   * compute the elongation of each CC
   * @param[in]  imBin  : the initial image (binary : 0, 255)
   * @param[in]  imSk   : the initial image (the skeleton associated)
   * @param[out] imOut  : the result : each CC have a value which correspond to the
   * elongation of the CC
   */
  template <class T1, class T2>
  RES_T ImElongationFromSkeleton(const Image<UINT8> &imBin,
                                 const Image<T1> &imSk, Image<T2> &imOut);

  /** @brief ImFromSkeletonSupTriplePoint : Suppress the triples points from a
   * skeleton
   * @param[in]  imIn  : the initial image
   * @param[out] imOut : the result
   */
  template <class T1, class T2>
  RES_T ImFromSkeletonSupTriplePoint(const Image<T1> &imIn, Image<T2> &imOut);

  /** @brief FromSkeletonComputeGranulometry : Compute the histogram of lenght
   * of the skeleton
   * @param[in]  imIn : the initial image
   * @param[out] Granulo : 
   * @param[in]  nbElt : 
   * @param[in]  ScaleX : Scale X
   * @param[in]  ScaleY : Scale Y
   * @param[in]  ScaleZ : Scale Z
   * 
   */
  template <class T>
  RES_T FromSkeletonComputeGranulometry(const Image<T> &imIn, UINT32 *Granulo,
                                        int nbElt, float ScaleX, float ScaleY,
                                        float ScaleZ);

  /** @brief ImFalseColorHSL : from a gray scale picture, we apply false color
   * with a shade off. (HSL color space)
   * @param[in]   imIn  : the initial image
   * @param[out]  imOut : the result
   * @param[in]   Scale : we multiply each pixel by a scale factor.
   */
  template <class T>
  RES_T ImFalseColorHSL(const Image<T> &imIn, Image<RGB> &imOut, float Scale);

  template <class T>
  RES_T CountNbCCperThreshold(const Image<T> &imIn, int *NbCC, int Invert = 0);

  template <class T>
  RES_T CountNbPixelOfNDG(const Image<T> &imIn, int NDG, int *NbPixel);

  template <class T1, class T2>
  RES_T measComputeVolume(const Image<T1> &imIn, const Image<T2> &imLevel,
                          float *Value);

  template <class T1, class T2>
  RES_T measComputeIndFromPatternSpectrum(const Image<T1> &imTrans,
                                          const Image<T2> &imInd,
                                          UINT16 BorneMin, UINT16 BorneMax,
                                          UINT8 Normalized, float *Value);

  template <class T1, class T2>
  RES_T ImFromSK_AreaForEachCC(const Image<T1> &imIn, int ScaleX, int ScaleY,
                               int ScaleZ, Image<T2> &imOut);

  template <class T>
  RES_T MeanValueOf(const Image<T> &imIn, bool slideBySlide, double *Value);

  template <class T1, class T2>
  RES_T GetConfusionMatrix(const Image<T1> &imMask, const Image<T2> &imIn2,
                           int seuil, int *FP, int *FN, int *TP, int *TN,
                           Image<RGB> &imOut, int WantImOut = 1);

  template <class T1, class T2>
  RES_T ImLayerDist(Image<T1> &imIn, int labelIn, int labelOut, float dx,
                    float dy, float dz, Image<T2> &imOut);

  /** @} */
} // namespace smil

// MorphoPathOpening Module header
#include "MorphoPathOpening/morphoPathOpening_T.hpp"
#include "MorphoPathOpening/morphoGeodesicPathOpening_T.hpp"
#include "MorphoPathOpening/morphoPathOpeningGraphV2_T.hpp"
#include "MorphoPathOpening/PO_Utilities_T.hpp"

#endif // _DMORPHO_PATH_OPENING_H_
