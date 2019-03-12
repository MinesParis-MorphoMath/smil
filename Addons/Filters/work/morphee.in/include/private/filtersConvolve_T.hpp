#ifndef __MORPHEE_FILTERS_CONVOLVE_T_HPP__
#define __MORPHEE_FILTERS_CONVOLVE_T_HPP__

#include <morphee/image/include/morpheeImage.hpp>
#include <morphee/selement/include/private/selementImage_T.hpp>

namespace morphee
{
  namespace filters
  {
    //! @addtogroup filters_group
    //! @{

    /*! @brief Generic convolution
     *
     * Perform a convolution of an image by a kernel (which can be
     * an image by itself), and make it possible to chose a
     * normalisation coefficient and apply an offset to the value
     * of each convoluted pixels.
     *
     * @param[in] imIn Image to be convolved by the kernel
     *
     * @param[in] kernelIn Image representing the kernel to be convolved with
     * imIn
     *
     * @param[in] valueOffset offset applied to the value of each convoluted
     * pixel
     *
     * @param[in] divider each convoluted pixel will have its value divided by
     * this number (before being applied the valueOffset)
     *
     * @param[out] imOut result of the convolution
     */
    template <typename image_type1, typename TSE, typename image_type2>
    RES_C t_ImConvolve(const image_type1 &imIn,
                       const selement::ConvolutionKernel<TSE> &kernelIn,
                       typename image_type2::value_type valueOffset,
                       typename image_type2::value_type divider,
                       image_type2 &imOut)
    {
      typedef typename image_type1::value_type value_type1;
      typedef typename image_type2::value_type value_type2;
      typedef typename DataTraits<value_type1>::float_accumulator_type
          accumulator_type;
      typedef selement::Neighborhood<selement::ConvolutionKernel<TSE>,
                                     image_type1>
          neighborhood_convolution_type;

      MORPHEE_ENTER_FUNCTION("t_ImConvolve<" + NameTraits<value_type1>::name() +
                             ", ... ," + NameTraits<value_type2>::name() + ">");

      if ((!imIn.isAllocated()) || (!imOut.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Images not allocated");
        return RES_NOT_ALLOCATED;
      }
      if (!t_CheckWindowSizes(imIn, imOut)) {
        MORPHEE_REGISTER_ERROR("Bad window sizes");
        return morphee::RES_ERROR_BAD_WINDOW_SIZE;
      }

      typename image_type1::const_iterator itIn, endIn;

      neighborhood_convolution_type neighb(imIn, kernelIn);
      typename neighborhood_convolution_type::iterator itKernel, endKernel;

      accumulator_type sum = 0;

      if (!t_CheckOffsetCompatible(imIn, imOut)) {
        typename image_type2::iterator itOut;
        for (itIn = imIn.begin(), endIn = imIn.end(), itOut = imOut.begin();
             itIn != endIn; ++itIn, ++itOut) {
          // for all the pixels

          // place the convolution kernel around the current pixels
          neighb.setCenter(itIn);

          sum = 0;
          for (itKernel = neighb.begin(), endKernel = neighb.end();
               itKernel != endKernel; ++itKernel) {
            sum += static_cast<accumulator_type>(*itKernel) *
                   (itKernel.getSEValue());
          }

          sum /= divider;
          sum += valueOffset;
          *itOut = static_cast<value_type2>(sum);
        } // for all pixels
      } else {
        // ici on se traine un iterateur en moins.
        for (itIn = imIn.begin(), endIn = imIn.end(); itIn != endIn; ++itIn) {
          // for all the pixels

          // place the convolution kernel around the current pixels
          neighb.setCenter(itIn);

          sum = 0;
          for (itKernel = neighb.begin(), endKernel = neighb.end();
               itKernel != endKernel; ++itKernel) {
            sum += static_cast<accumulator_type>(*itKernel) *
                   (itKernel.getSEValue());
          }

          sum /= divider;
          sum += valueOffset;
          imOut.pixelFromOffset(itIn.Offset()) = static_cast<value_type2>(sum);
        } // for all pixels
      }

      return RES_OK;
    }

    /* @brief Simple version of the convolution
     *
     * Convolve the image with an image kernel, but do not perform
     * any division on the convoluted values.
     *
     * @param[in] imIn input image
     *
     * @param[in] kernelIn image representation of the convolution kernel
     *
     * @param[out] imOut result of the convolution of imIn by kernelIn
     *
     * @warning You have to manage normalisation problems y
     * yourself, or use the generic version of t_ImConvolve.
     *
     * @see t_ImConvolve
     */
    template <typename T1, typename TSE, typename T2>
    RES_C t_ImConvolve(const Image<T1> &imIn,
                       const selement::ConvolutionKernel<TSE> &kernelIn,
                       Image<T2> &imOut)
    {
      return t_ImConvolve<Image<T1>, TSE, Image<T2>>(imIn, kernelIn, (T2) 0,
                                                     (T2) 1, imOut);
    }

    //! @} // defgroup filters

  } // namespace filters

} // namespace morphee

#endif // __MORPHEE_IMAGE_FILTERS_HPP__
