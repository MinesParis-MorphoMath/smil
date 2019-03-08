#ifndef __MORPHEE_FILTERS_NOISE_T_IMP_HPP__
#define __MORPHEE_FILTERS_NOISE_T_IMP_HPP__

#include <boost/random.hpp>
#include <boost/numeric/conversion/bounds.hpp>
#include <morphee/filters/include/morpheeFilters.hpp>

#include <boost/random/poisson_distribution.hpp>
#include <boost/static_assert.hpp>

namespace morphee
{
  namespace filters
  {
    /*!
     * @addtogroup noise_group
     *
     * @{
     */

    //! Randomness generator
    typedef boost::minstd_rand base_generator_type;

    template <class __Image>
    RES_C t_ImAddNoiseSaltAndPepper(const __Image &imIn, const F_DOUBLE freq,
                                    __Image &imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_ImAddNoiseSaltAndPepper");

      if (!imIn.isAllocated() || !imOut.isAllocated()) {
        MORPHEE_REGISTER_ERROR("Images are not allocated");
        return RES_NOT_ALLOCATED;
      }
      if (!t_CheckWindowSizes(imIn, imOut)) {
        MORPHEE_REGISTER_ERROR("Incompatible active window sizes");
        return RES_ERROR_BAD_WINDOW_SIZE;
      }

      // initialize the uniform number generator
      base_generator_type generator(getSeed());
      boost::uniform_real<> uni_dist(0, 1);
      boost::variate_generator<base_generator_type &, boost::uniform_real<>>
          uni(generator, uni_dist);

      typename __Image::const_iterator it = imIn.begin(), iend = imIn.end();
      typename __Image::iterator itOut = imOut.begin();

      for (; it != iend; ++it, ++itOut) {
        if (uni() < freq) {
          // FIXME TODO have a pixel_3 implementation ??
          if (uni() < 0.5) // black
            *itOut = std::numeric_limits<typename __Image::value_type>::min();
          else // white
            *itOut = std::numeric_limits<typename __Image::value_type>::max();
        } else { // keep original
          *itOut = *it;
        }
      }

      return RES_OK;
    }

    //! Generate realisations of a gaussian random variable.
    template <typename T> struct s_gaussianSource {
      typedef typename DataTraits<T>::float_accumulator_type float_t;

      base_generator_type m_generator;
      boost::normal_distribution<> m_norm_dist;
      boost::variate_generator<base_generator_type &,
                               boost::normal_distribution<>>
          m_norm;
      float_t m_minVal, m_maxVal;

      s_gaussianSource(F_DOUBLE sigma)
          : m_generator(getSeed()), m_norm_dist(0., sigma),
            m_norm(m_generator, m_norm_dist),
            m_minVal(static_cast<float_t>(boost::numeric::bounds<T>::lowest())),
            m_maxVal(static_cast<float_t>(std::numeric_limits<T>::max()))
      {
      }

      T operator()(const T &val)
      {
        float_t tmpVal   = static_cast<float_t>(val);
        float_t noiseVal = static_cast<float_t>(m_norm());

        if (noiseVal > 0) // beware overflow
        {
          tmpVal += noiseVal; // FIXME: this overflow test
                              // could be better: if(noiseVal > max - tmpVal)
                              // but this overflow is mainly for UINT8, so using
                              // floats it's OK to do it like this. I guess...
          if (tmpVal > m_maxVal)
            return std::numeric_limits<T>::max();
          else
            return static_cast<T>(tmpVal);

        } else {
          // beware underflow
          tmpVal += noiseVal;
          if (tmpVal < m_minVal)
            return boost::numeric::bounds<T>::lowest();
          else
            return static_cast<T>(tmpVal);
        }
      }

    }; // s_gaussianSource

    //! Specialisation of the gaussian variable for pixel_3 data types.
    template <typename T> struct s_gaussianSource<pixel_3<T>> {
      typedef s_gaussianSource<T> scalar_source;

      scalar_source m_source;

      s_gaussianSource(F_DOUBLE sigma) : m_source(sigma)
      {
      }

      pixel_3<T> operator()(const pixel_3<T> &val)
      {
        const T val1 = m_source(val.channel1);
        const T val2 = m_source(val.channel2);
        const T val3 = m_source(val.channel3);
        return pixel_3<T>(val1, val2, val3);
        // This is the shorter expression, but since the arguments to a function
        // call are evaluated in an unspecified order, I cannot guarantee that
        // the calls will be performed in the left to right order. This means I
        // cannot check that the results are OK in automated tests, since the
        // results will be implementation-dependent (and they actually are !).
        // As I don't believe the above method is very much slower (probably
        // isn't, actually) I prefer to keep the explicit ordering and have
        // something I can actually test.
        // return pixel_3<T>( m_source(val.channel1), m_source(val.channel2),
        // m_source(val.channel3) );
      }
    };

    template <class __Image>
    RES_C t_ImAddNoiseGaussian(const __Image &imIn, const F_DOUBLE sigma,
                               __Image &imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_ImAddNoiseGaussian");

      if (!imIn.isAllocated() || !imOut.isAllocated()) {
        MORPHEE_REGISTER_ERROR("Images are not allocated");
        return RES_NOT_ALLOCATED;
      }
      if (!t_CheckWindowSizes(imIn, imOut)) {
        MORPHEE_REGISTER_ERROR("Incompatible active window sizes");
        return RES_ERROR_BAD_WINDOW_SIZE;
      }

      typename __Image::const_iterator it = imIn.begin(), iend = imIn.end();
      typename __Image::iterator itOut = imOut.begin();

      typedef typename __Image::value_type T;

      typedef typename DataTraits<T>::float_accumulator_type float_t;

      s_gaussianSource<T> source(sigma);

      for (; it != iend; ++it, ++itOut)
        *itOut = source(*it);

      return RES_OK;
    }

    /*!
     * @brief Generate one poisson variable per pixel (ie per call
     * to te operator())
     *
     */
    template <typename T, bool issigned> struct s_poissonianSourceImpl {
      typedef typename DataTraits<T>::float_accumulator_type float_t;
      base_generator_type m_generator;

      typedef boost::poisson_distribution<T, float_t> Poisson_distrib_t;
      typedef boost::variate_generator<base_generator_type &, Poisson_distrib_t>
          Poisson_var_t;

      s_poissonianSourceImpl() : m_generator(getSeed())
      {
      }

      inline T operator()(const T &val)
      {
        // we already know that val is an integer (cf static cast in
        // poisson_distribution)
        T sign;
        if (val < 0)
          sign = -1;
        else
          sign = 1;
        Poisson_distrib_t poisson(static_cast<float_t>(val));
        Poisson_var_t var(m_generator, poisson);

        return sign * var();
      }

    }; // s_poissonianSourceImpl

    //! A small optimisation when we don't need to care about the
    //! sign (is it really useful ? I haven't tested it,
    //! really...)
    template <typename T> struct s_poissonianSourceImpl<T, false> {
      typedef typename DataTraits<T>::float_accumulator_type float_t;
      base_generator_type m_generator;

      typedef boost::poisson_distribution<T, float_t> Poisson_distrib_t;
      typedef boost::variate_generator<base_generator_type &, Poisson_distrib_t>
          Poisson_var_t;

      s_poissonianSourceImpl() : m_generator(getSeed())
      {
      }

      inline T operator()(const T &val)
      {
        Poisson_distrib_t poisson(static_cast<float_t>(val));
        Poisson_var_t var(m_generator, poisson);

        return var();
      }

    }; // s_poissonianSourceImpl

    //! Generate a Poisson random variable
    template <typename T> struct s_poissonianSource {
      typedef s_poissonianSourceImpl<T, std::numeric_limits<T>::is_signed>
          s_sourceImpl_t;
      s_sourceImpl_t source_impl;

      s_poissonianSource()
      {
      }

      inline T operator()(const T &val)
      {
        return source_impl(val);
      }
    }; // s_poissonianSource

    template <class __Image>
    RES_C t_ImAddNoisePoissonian(const __Image &imIn, __Image &imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_ImAddNoisePoissonian");

      if (!imIn.isAllocated() || !imOut.isAllocated()) {
        MORPHEE_REGISTER_ERROR("Images are not allocated");
        return RES_NOT_ALLOCATED;
      }
      if (!t_CheckWindowSizes(imIn, imOut)) {
        MORPHEE_REGISTER_ERROR("Incompatible active window sizes");
        return RES_ERROR_BAD_WINDOW_SIZE;
      }

      typename __Image::const_iterator it = imIn.begin(), iend = imIn.end();
      typename __Image::iterator itOut = imOut.begin();

      typedef typename __Image::value_type T;

      typedef typename DataTraits<T>::float_accumulator_type float_t;

      s_poissonianSource<T> source;

      for (; it != iend; ++it, ++itOut)
        *itOut = source(*it);

      return RES_OK;
    } // t_ImAddNoisePoissonian

    template <class __Image, class Tparam>
    RES_C t_ImAddNoiseUniform(const __Image &imIn, Tparam minVal, Tparam maxVal,
                              __Image &imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_ImAddNoiseUniform");

      if (!imIn.isAllocated() || !imOut.isAllocated()) {
        MORPHEE_REGISTER_ERROR("Images are not allocated");
        return RES_NOT_ALLOCATED;
      }
      if (!t_CheckWindowSizes(imIn, imOut)) {
        MORPHEE_REGISTER_ERROR("Incompatible active window sizes");
        return RES_ERROR_BAD_WINDOW_SIZE;
      }
      typedef typename __Image::value_type Real_t;

      // initialize the uniform number generator
      base_generator_type generator(getSeed());
      boost::uniform_real<Real_t> uni_dist(static_cast<Real_t>(minVal),
                                           static_cast<Real_t>(maxVal));
      boost::variate_generator<base_generator_type &, boost::uniform_real<>>
          uni(generator, uni_dist);

      typename __Image::const_iterator it = imIn.begin(), iend = imIn.end();
      typename __Image::iterator itOut = imOut.begin();

      for (; it != iend; ++it, ++itOut) {
        *itOut = *it + uni();
      }

      return RES_OK;
    } // t_ImAddNoiseUniform

    // @} addtogroup noise_group

  } // namespace filters
} // namespace morphee

#endif //__MORPHEE_FILTERS_NOISE_T_IMP_HPP__
