#ifndef __MORPHEE_FILTERS_NOISE_T_HPP__
#define __MORPHEE_FILTERS_NOISE_T_HPP__

#include <morphee/filters/include/morpheeFilters.hpp>

namespace morphee
{
  namespace filters
  {
    /*!
     * @addtogroup noise_group
     *
     * @{
     */

    /*!
     * @brief Add a salt n pepper noise to the image.
     *
     * Noisy pixel appear with a frequency @c freq among the
     * pixels of the image.
     *
     * Each noisy pixel has equal chance to be valued with the
     * maximum value authorized by the image's data type of by the
     * minimum value authorized (50% each).
     *
     */
    template <class __Image>
    RES_C t_ImAddNoiseSaltAndPepper(const __Image &imIn,
                                    const F_DOUBLE frequency, __Image &imOut);

    /*!
     * @brief Add a gaussian noise to the image.
     *
     * Each pixel of the image is attributed a new value
     * corresponding to the sum between its original value and the
     * realisation of a random gaussian variable of mean 0 and
     * standard deviation @c sigma
     *
     */
    template <class __Image>
    RES_C t_ImAddNoiseGaussian(const __Image &imIn, const F_DOUBLE sigma,
                               __Image &imOut);

    /*!
     * @brief Adds poissonian noise to an image
     *
     * Each pixel of the output image will be the realisation of a
     * Poisson random variable whose mean is set to the value of
     * the corresponding pixel in the input image.
     *
     * @warning Defining a Poisson variable for negative values or
     * floating point values is absurd, really ! Here, when
     * negative pixels are encoutered whe chosed to work on the
     * absolute value of the pixel (hence inversing the axis for
     * the negative pixels)
     *
     */
    template <class __Image>
    RES_C t_ImAddNoisePoissonian(const __Image &imIn, __Image &imOut);

    /*!
     * @brief Add uniform noise to an image.
     *
     * Each pixel of the image is attributed a value corresponding
     * to the addition between its original value and the
     * realisation of a random uniform variable.
     *
     */
    template <class __Image, class Tparam>
    RES_C t_ImAddNoiseUniform(const __Image &imIn, Tparam minVal, Tparam maxVal,
                              __Image &imOut);

    //@}  noise_group

  } // namespace filters
} // namespace morphee

#include "filtersNoise_T_impl.hpp"

#endif // __MORPHEE_FILTERS_NOISE_T_HPP__
