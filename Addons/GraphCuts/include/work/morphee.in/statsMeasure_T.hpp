

#ifndef __MORPHEE_MEASURE_T_HPP__
#define __MORPHEE_MEASURE_T_HPP__

#include <complex>
#include <vector>
#include <limits>
#include <set>

#include <boost/numeric/conversion/bounds.hpp>

#include <morphee/image/include/private/image_T.hpp>
#include <morphee/image/include/private/imagePixelwise_T.hpp>
#include <morphee/common/include/commonException.hpp>
#include <morphee/image/include/private/imageFunctorInterface_T.hpp>

//
// FIXME TODO: pour ceux qui s'ennuient: il y a plein de nouveaux "throw" dans
// les operateurs. Ce serait bien de les catcher, de faire un
// MORPHEE_REGISTER_ERROR et un return RES_ERROR...
//
//

#define __DEBUG_VERBOSE_STATS_MEASURE_T_HPP(                                   \
    mouf) // std::cout << mouf << std::endl;

namespace morphee
{
  namespace stats
  {
    /*!
     * @brief Template function for performing pixelwise measures on two images
     * (think distances)
     * @ingroup group_measSimple
     *
     */
    template <class Image1, class Image2, class Oper>
    inline RES_C t_ImBinaryMeasureOperation(const Image1 &imin1,
                                            const Image2 &imin2, Oper &op)
    {
      // This was stolen from t_ImBinaryOperation in imagePixelwise_T.hpp

      if (!imin1.isAllocated() || !imin2.isAllocated())
        return RES_NOT_ALLOCATED;

      if (t_CheckOffsetCompatible(imin1, imin2)) {
        typename Image1::const_iterator it1   = imin1.begin();
        typename Image1::const_iterator iend1 = imin1.end();
        offset_t off;

        for (; it1 != iend1; ++it1) {
          off = it1.getOffset();
          op(*it1, imin2.pixelFromOffset(off));
        }
        return RES_OK;
      }

      if (!t_CheckWindowSizes(imin1, imin2))
        return RES_ERROR_BAD_WINDOW_SIZE;

      typename Image1::const_iterator it1 = imin1.begin(), iend1 = imin1.end();
      typename Image2::const_iterator it2 = imin2.begin(); // iend2=imin2.end();

      for (; it1 != iend1; ++it1, ++it2) {
        op(*it1, *it2);
      }
      return RES_OK;
    } // t_ImBinaryMeasureOperation

    // 		/*!
    // 		 * @brief Template function for investigating bivariate
    // 		 * properties of an image (think covariance, bivariate law
    // 		 * variograms, ...)
    // 		 *
    // 		 * @ingroup group_measSimple
    // 		 *
    // 		 * @param[in] imin the image on with the bivariate statistics
    // 		 * are investigated.
    // 		 *
    // 		 * @param[in] shift_vector the vector characterizing the scale
    // 		 * and direction of the bivariate behaviour under
    // 		 * investigation.
    // 		 *
    // 		 * @param[out] op the opertator that will actually be iterated
    // 		 * along the image.
    // 		 *
    // 		 */
    // 		template<class Image, class Oper>
    // 			inline RES_C t_ImBivariateMeasureOperation(const Image&imin, const
    // typename Image::corrdinate_system shift_vector, Oper & op)
    // 		{
    // 			// This was stolen from t_ImBivariateOperation in
    // imagePixelwise_T.hpp 			if(!imin.isAllocated()) 				return RES_NOT_ALLOCATED;

    // // 			typename Image1::const_iterator
    // it1=imin1.begin(),iend1=imin1.end();
    // // 			typename Image2::const_iterator
    // it2=imin2.begin();//iend2=imin2.end();

    // // 			for(;it1!=iend1;++it1,++it2)
    // // 			{
    // // 				op(*it1,*it2);
    // // 			}

    // 			coordinate_system p(0);
    // 			offset_t ofs = 0;
    // 			value_type_out accu = 0;
    // 			UINT32 taille = 0;
    // 			value_type valeur = 0 ;

    // 			// Set the new active window's dimension
    // 			coordinate_system newWSize = imData.Size() - hvector;
    // 			/*newWXSize = imData.getXSize() - hvector.x;
    // 			  newWYSize = imData.getYSize() - hvector.y;
    // 			  newWZSize = imData.getZSize() - hvector.z;*/

    // 			// Test if the vector fits in the image :)
    // 			for(unsigned int i = 0; i < coordinate_system::dimension; i++)
    // 			{
    // 				if (newWSize[i] <= 0)
    // 				{
    // 					throw(MException("Vector is too large for the image."));
    // 				}
    // 			}

    // 			//Changing the active Window
    // 			window_info_type oldwi = imData.ActiveWindow();
    // 			window_info_type newwi(coordinate_system(0), newWSize);

    // 			if(imData.setActiveWindow(newwi) != RES_OK)
    // 			{
    // 				throw(MException("Unable to change the active window"));
    // 			}

    // 			typename image_type::iterator itBeg = imData.begin(), itEnd =
    // imData.end();

    // 			taille = t_WSizePixel(imData);//imData.getWxSize() *
    // imData.getWySize() * imData.getWzSize();
    // 			__DEBUG_VERBOSE_STATS_MEASURE_T_HPP("Taille = "<<taille<<"  Accu =
    // "<<int(accu));
    // 			__DEBUG_VERBOSE_STATS_MEASURE_T_HPP("vecteur = (" << hvector[0] <<",
    // "<< hvector[1] << ", " << hvector[2] <<")");

    // 			// Iterate on the window to get the "first" point used to process the
    // variogram value 			for(;itBeg != itEnd; ++itBeg)
    // 			{
    // 				valeur = *itBeg;
    // 				// Compute the second point
    // 				p = itBeg.Position();
    // 				p+= hvector;
    // 				/*x = itBeg.getX() + hvector.x;
    // 				  y = itBeg.getY() + hvector.y;
    // 				  z = itBeg.getZ() + hvector.z;*/

    // 				ofs = t_GetOffsetFromCoords(imData, p);
    // 				// Get the square of the differences

    // 				value_type val = imData.pixelFromOffset(ofs);
    // 				val -= valeur;
    // 				val *= val;

    // 				//accu +=
    // (imData.pixelFromOffset(ofs)-valeur)*(imData.pixelFromOffset(ofs)-valeur);
    // 				accu += val;
    // 			}

    // 			if(RES_OK != imData.setActiveWindow(oldwi))
    // 			{
    // 				throw(MException("Unable to reset the active window"));
    // 			}

    // 			return RES_OK;

    // 		}// t_ImBivariateMeasureOperation

    /*!
     * @brief Collect densities values and create the
     * corresponding distribution.
     *
     * If used in a @c for_each like construct upon an iterable
     * collection of values M this will build the distribution
     * function @c F corresponding to the formula:
     *
     * @code
     * F(i) = 1 - M(i)/M_norm
     * @endcode
     *
     * @warning @c M_norm must be different from 0
     *
     * @note When M is an decreasing function @c M_norm is usually
     * equal to @c M(0) but should be given separately at the
     * objet initialisation. When M is decreasing there is not
     * much point in using this operator.
     *
     *
     * @par Use cases
     *
     * This operator can be usefull for granulometry and all the
     * measure that look like it.
     */
    template <class T> class s_opDistributionFromDecreasingMeasure
    {
    public:
      typedef typename DataTraits<T>::float_accumulator_type acc_type;
      acc_type m_norm;

    private:
      std::vector<acc_type> m_distrib;

      s_opDistributionFromDecreasingMeasure()
      {
      }

    public:
      s_opDistributionFromDecreasingMeasure(const T _norm) : m_norm(_norm)
      {
        if (_norm == 0) {
          MORPHEE_REGISTER_ERROR(
              "Normalising coefficient should be different from 0.");
          throw MException(RES_ERROR_BAD_ARG);
        }

        // this is a ugly hack to work around a strange
        // (and hardly reproducible) bug thatbasically transformed
        // the m_norm value into a 'nan' midway between the call
        // to the constructor and the call to operator() in the
        // optimised FastMorphm addon...
        m_norm = static_cast<acc_type>(_norm);
      }

      void operator()(const T val)
      {
        __DEBUG_VERBOSE_STATS_MEASURE_T_HPP("Value " + AnyToString(val));
        __DEBUG_VERBOSE_STATS_MEASURE_T_HPP("Norm " + AnyToString(m_norm));
        const acc_type val_f(val);
        __DEBUG_VERBOSE_STATS_MEASURE_T_HPP("Casted Value " +
                                            AnyToString(val_f));
        const acc_type temp0(val_f / m_norm);
        const acc_type temp(static_cast<acc_type>(1) - temp0);
        m_distrib.push_back(temp);
      }

      std::vector<acc_type> getDistribution()
      {
        return m_distrib;
      }
    }; // s_opDistributionFromDecreasingMeasure

    /*!
     * @defgroup group_calculusUtilities Calculus utilities
     * @ingroup group_measSimple
     *
     * Define several tables that may be calculated once and
     * used several times.
     *
     * @{
     */

    template <unsigned long int T> class table_precalcul_trigo
    {
    private:
      static double table_cos[T];
      static double table_sin[T];
      static bool b_created;

    public:
      table_precalcul_trigo()
      {
        if (!b_created) {
          for (int i = 0; i < T; i++) {
            table_cos[i] = cos((double) i * (MPI * 2. / ((double) T)));
            table_sin[i] = sin((double) i * (MPI * 2. / ((double) T)));
          }
          b_created = true;
        }
      }
      inline double &get_cos(const long int &i)
      {
        return table_cos[i];
      }
      inline double &get_sin(const long int &i)
      {
        return table_sin[i];
      }
    };
    template <unsigned long int T>
    bool table_precalcul_trigo<T>::b_created = false;
    template <unsigned long int T>
    double table_precalcul_trigo<T>::table_cos[T] = {0.};
    template <unsigned long int T>
    double table_precalcul_trigo<T>::table_sin[T] = {0.};

    template <long int T> class table_precalcul_exponentiel
    {
    private:
      static double table_expo[T];
      static bool b_created;

    public:
      table_precalcul_exponentiel()
      {
        if (!b_created) {
          for (int i = 0; i < T; i++) {
            table_expo[i] = exp((double) i);
          }
          b_created = true;
        }
      }
      inline double &get_expo(const long int &i)
      {
        return table_expo[i];
      }
    };
    template <long int T>
    bool table_precalcul_exponentiel<T>::b_created = false;
    template <long int T>
    double table_precalcul_exponentiel<T>::table_expo[T] = {0.};

    // @} // defgroup group_calculusUtilities

    /*!
     * @addtogroup group_measStats
     * @{
     */

    /*! @brief Mean (average) operator
     *
     * The average is (obviously) given by the sum of the elements' value
     * divided by the number of elements.
     *
     */
    template <class T> struct s_opMeanLinear {
      typedef typename DataTraits<T>::float_accumulator_type acc_type;
      acc_type m_mean;
      UINT32 m_nb_pixels;

      s_opMeanLinear() : m_mean(0), m_nb_pixels(0)
      {
      }

      void operator()(const T &pixel)
      {
        m_mean += static_cast<acc_type>(pixel);
        m_nb_pixels++;
      }

      acc_type average()
      {
        if (m_nb_pixels == 0)
          throw(MException("empty dataset in s_opMeanLinear"));
        return (m_nb_pixels != 0 ? static_cast<acc_type>(m_mean) /
                                       static_cast<acc_type>(m_nb_pixels)
                                 : 0);
      }
    };

    /*! @brief Mean (average) operator on set
     *
     * The average is (obviously) given by the sum of the elements' value
     * divided by the number of elements.
     *
     *
     */
    template <class T> struct s_opMeanLinearOnSet {
      typedef typename DataTraits<T>::float_accumulator_type acc_type;
      acc_type m_mean;
      UINT32 m_nb_pixels;

      s_opMeanLinearOnSet() : m_mean(0), m_nb_pixels(0)
      {
      }

      void operator()(const T &pixel)
      {
        if (pixel != DataTraits<T>::default_value::background()) {
          m_mean += static_cast<acc_type>(pixel);
          m_nb_pixels++;
        }
      }

      acc_type average()
      {
        // In this case, returning 0 when m_nb_pixels == 0 is less dangerous ...
        // if( m_nb_pixels == 0 )
        //	throw(MException("empty dataset in s_opMeanLinear"));
        return (m_nb_pixels != 0 ? static_cast<acc_type>(m_mean) /
                                       static_cast<acc_type>(m_nb_pixels)
                                 : 0);
      }
    };

    /*! @brief Mean (average) operator. Specialized for pixel_3
     *
     * The average is taken separately for each channel (number of points
     * remaining obviously the same)
     *
     */
    template <class T> struct s_opMeanLinear<pixel_3<T>> {
      typedef typename DataTraits<pixel_3<T>>::float_accumulator_type acc_type;
      acc_type m_mean;
      UINT32 m_nb_pixels;

      s_opMeanLinear() : m_mean(0, 0, 0), m_nb_pixels(0)
      {
      }

      void operator()(const pixel_3<T> &pixel)
      {
        m_mean += static_cast<acc_type>(pixel);
        m_nb_pixels++;
      }

      acc_type average()
      {
        if (m_nb_pixels == 0)
          throw(MException("empty dataset in s_opMeanLinear"));

        return m_mean / acc_type(m_nb_pixels, m_nb_pixels, m_nb_pixels);
      }
    };

    /*! @brief Mean (average) operator. Specialized for pixel_3
     *
     * The average is taken separately for each channel (number of points
     * remaining obviously the same)
     *
     */
    template <class T> struct s_opMeanLinearOnSet<pixel_3<T>> {
      typedef typename DataTraits<pixel_3<T>>::float_accumulator_type acc_type;
      acc_type m_mean;
      UINT32 m_nb_pixels;

      s_opMeanLinearOnSet() : m_mean(0, 0, 0), m_nb_pixels(0)
      {
      }

      void operator()(const pixel_3<T> &pixel)
      {
        if (pixel != (DataTraits<pixel_3<T>>::default_value::background())) {
          m_mean += static_cast<acc_type>(pixel);
          m_nb_pixels++;
        }
      }

      acc_type average()
      {
        if (m_nb_pixels == 0)
          throw(MException("empty dataset in s_opMeanLinear"));

        return m_mean / acc_type(m_nb_pixels, m_nb_pixels, m_nb_pixels);
      }
    };

    /*! @brief Mean (average) operator over a range
     *
     * This operator actually use the operator s_opMeanLinear over the range
     * specified by the provided iterators
     *
     * @see s_opMeanLinear
     */
    template <class It, class Tout = typename DataTraits<
                            typename It::value_type>::float_accumulator_type>
    struct s_measMeanLinear : std::binary_function<It, const It &, Tout> {
      Tout operator()(It beg, const It &end)
      {
        s_opMeanLinear<typename It::value_type> op;
        op = std::for_each(beg, end, op);
        return static_cast<Tout>(op.average());
      }
    };

    /*! @brief Mean (average) function over an image from iterators
     *
     * This operator actually use the operator s_opMeanLinear over the range
     * specified by the provided iterators
     *
     * @see s_opMeanLinear
     */
    template <class It, class Tout>
    RES_C t_measMeanLinear(It ibeg, const It &iend, Tout &meanValue)
    {
      s_measMeanLinear<It, Tout> op;
      meanValue = op(ibeg, iend);
      return RES_OK;
    }

    /*! @brief Mean (average) operator over an image
     *
     * This operator actually use the operator s_opMeanLinear over the image
     *
     * @see s_opMeanLinear, t_measMeanLinear
     */
    template <class _image, class Tout>
    RES_C t_measMeanLinear(const _image &im, Tout &meanValue)
    {
      return t_measMeanLinear(im.begin(), im.end(), meanValue);
    }

    /*! @brief Mean (average) operator over a range	On Set
     *
     * This operator actually uses the operator s_opMeanLinearOnSet over the
     * range specified by the provided iterators
     *
     * @see s_opMeanLinearOnSet
     */
    template <class It, class Tout = typename DataTraits<
                            typename It::value_type>::accumulator_type>
    struct s_measMeanLinearOnSet : std::binary_function<It, const It &, Tout> {
      Tout operator()(It beg, const It &end)
      {
        s_opMeanLinearOnSet<typename It::value_type> op;
        op = std::for_each(beg, end, op);
        return static_cast<Tout>(op.average());
      }
    };

    /*! @brief Mean On Set (average) function over an image from iterators
     *
     * This operator actually use the operator s_opMeanLinearOnSet over the
     * range specified by the provided iterators
     *
     * @see s_opMeanLinearOnSet
     */
    template <class It, class Tout>
    RES_C t_measMeanLinearOnSet(It ibeg, const It &iend, Tout &meanValue)
    {
      s_measMeanLinearOnSet<It, Tout> op;
      meanValue = op(ibeg, iend);
      return RES_OK;
    }

    /*! @brief Mean On Set (average) operator over an image
     *
     * This operator actually use the operator s_opMeanLinearOnSet over the
     * image
     *
     * @see s_opMeanLinearOnSet, t_measMeanLinearOnSet
     */
    template <class _image, class Tout>
    RES_C t_measMeanLinearOnSet(const _image &im, Tout &meanValue)
    {
      return t_measMeanLinearOnSet(im.begin(), im.end(), meanValue);
    }

    /*! @brief Mean absolute deviation operator
     *
     * The average is (obviously) given by the sum of the elements' value
     * divided by the number of elements.
     *
     * The absolute deviation is the average of the absolute difference between
     * the samples and their average.
     *
     */
    template <class T> struct s_opMeanAbsoluteDeviation {
      typedef typename DataTraits<T>::float_accumulator_type acc_type;
      acc_type m_mean, m_dev;
      UINT32 m_nb_pixels;

      s_opMeanAbsoluteDeviation(acc_type _mean)
          : m_mean(_mean), m_dev(0), m_nb_pixels(0)
      {
      }

      void operator()(const T &pixel)
      {
        m_dev += std::abs(static_cast<acc_type>(pixel) - m_mean);
        m_nb_pixels++;
      }

      acc_type average()
      {
        if (m_nb_pixels == 0)
          throw(MException("empty dataset in s_opMeanAbsoluteDeviation"));
        return (m_nb_pixels != 0 ? static_cast<acc_type>(m_dev) /
                                       static_cast<acc_type>(m_nb_pixels)
                                 : 0);
      }
    };

    /*! @brief Mean absolute deviation over a range
     *
     * This operator actually use the mean absolute deviation over the range
     * specified by the provided iterators It first computes the mean of the
     * provided range.
     *
     * @see s_opMeanLinear
     */
    template <class It, class Tout = typename DataTraits<
                            typename It::value_type>::accumulator_type>
    struct s_measMeanAbsoluteDeviation
        : std::binary_function<It, const It &, Tout> {
      Tout operator()(const It &beg, const It &end)
      {
        s_opMeanLinear<typename It::value_type> opMean;
        opMean = std::for_each(beg, end, opMean);

        s_opMeanAbsoluteDeviation<typename It::value_type> opMeanDev(
            opMean.average());
        opMeanDev = std::for_each(beg, end, opMeanDev);
        return static_cast<Tout>(opMeanDev.average());
      }
    };

    /*! @brief Mean absolute deviation function over a range of iterators
     *
     * This function actually use the operator s_measVarianceLinear over the
     * range specified by the provided iterators
     *
     * @see s_opMeanAbsoluteDeviation, s_measMeanAbsoluteDeviation
     */
    template <class It, class Tout>
    RES_C t_measMeanAbsoluteDeviation(It ibeg, const It &iend,
                                      Tout &meanDeviationValue)
    {
      s_measMeanAbsoluteDeviation<It, Tout> op;
      meanDeviationValue = op(ibeg, iend);
      return RES_OK;
    }

    /*! @brief Mean absolute deviation function over an image
     *
     * This operator actually use the operator s_measMeanAbsoluteDeviation over
     * the image
     *
     * @see s_opMeanAbsoluteDeviation, s_measMeanAbsoluteDeviation,
     * t_measMeanAbsoluteDeviation
     */
    template <class _image, class Tout>
    RES_C t_measMeanAbsoluteDeviation(const _image &im,
                                      Tout &meanDeviationValue)
    {
      return t_measMeanAbsoluteDeviation(im.begin(), im.end(),
                                         meanDeviationValue);
    }

    /*! @brief Variance operator
     *
     * This is the "biased" version of the variance computing, which means that
     * the mean is known a priori. If this is not the case, use
     * s_opVarianceLinearUnbiased instead.
     *
     * When used with a mean computed on the samples, then this
     * operator will compute the variance of the samples and not
     * of the underlying random variable.
     *
     */
    template <class T> struct s_opVarianceLinear {
      typedef typename DataTraits<T>::float_accumulator_type acc_type;
      acc_type m_variance, m_mean;
      UINT32 m_nb_pixels;

      s_opVarianceLinear(acc_type _mean)
          : m_mean(_mean), m_nb_pixels(0), m_variance(0)
      {
      }

      void operator()(const T &pixel)
      {
        acc_type val = static_cast<acc_type>(pixel) - m_mean;
        val *= val;
        m_variance += val;
        m_nb_pixels++;
      }

      acc_type result()
      {
        if (m_nb_pixels == 0)
          throw(MException("empty dataset in s_opVarianceLinear"));
        return (m_nb_pixels != 0 ? static_cast<acc_type>(m_variance) /
                                       static_cast<acc_type>(m_nb_pixels)
                                 : 0);
      }
    };

    /*! @brief "Unbiaised" variance operator
     *
     * This is the "unbiased" version of the variance computing, which means
     * that the mean is also computed from the provided samples.
     *
     */
    template <class T> struct s_opVarianceLinearUnbiased {
      typedef typename DataTraits<T>::float_accumulator_type acc_type;
      acc_type m_variance, m_mean;
      UINT32 m_nb_pixels;

      s_opVarianceLinearUnbiased(acc_type _mean)
          : m_variance(0), m_mean(_mean), m_nb_pixels(0)
      {
      }

      void operator()(const T &pixel)
      {
        acc_type val = static_cast<acc_type>(pixel) - m_mean;
        val *= val;
        m_variance += val;
        m_nb_pixels++;
      }

      acc_type result()
      {
        if (m_nb_pixels == 0)
          throw(MException("empty dataset in s_opVarianceLinearUnbiased"));
        return (m_nb_pixels > 1 ? static_cast<acc_type>(m_variance) /
                                      static_cast<acc_type>(m_nb_pixels - 1)
                                : 0);
      }
    };

    /*! @brief "Unbiaised" variance operator On set
     *
     * This is the "unbiased" version of the variance computed on a set , which
     * means that the mean on set is also computed from the provided samples.
     *
     */
    template <class T> struct s_opVarianceLinearUnbiasedOnSet {
      typedef typename DataTraits<T>::float_accumulator_type acc_type;
      acc_type m_variance, m_mean;
      UINT32 m_nb_pixels;

      s_opVarianceLinearUnbiasedOnSet(acc_type _mean)
          : m_variance(0), m_mean(_mean), m_nb_pixels(0)
      {
      }

      void operator()(const T &pixel)
      {
        if (pixel != (DataTraits<T>::default_value::background())) {
          acc_type val = static_cast<acc_type>(pixel) - m_mean;
          val *= val;
          m_variance += val;
          m_nb_pixels++;
        }
      }

      acc_type result()
      {
        // When the set is empty, then give back 0
        // if(m_nb_pixels == 0)
        //	throw(MException("empty dataset in
        //s_opVarianceLinearUnbiasedOnSet"));
        return (m_nb_pixels > 1 ? static_cast<acc_type>(m_variance) /
                                      static_cast<acc_type>(m_nb_pixels - 1)
                                : 0);
      }
    };

    /*! @brief Variance operator over a range
     *
     * This operator actually use the operator s_opVarianceLinearUnbiased over
     * the range specified by the provided iterators It first computes the mean
     * of the provided range.
     *
     * @see s_opVarianceLinearUnbiased, s_opMeanLinear
     */
    template <class It, class Tout = typename DataTraits<
                            typename It::value_type>::accumulator_type>
    struct s_measVarianceLinear : std::binary_function<It, const It &, Tout> {
      Tout operator()(const It &beg, const It &end)
      {
        s_opMeanLinear<typename It::value_type> opMean;
        opMean = std::for_each(beg, end, opMean);

        s_opVarianceLinearUnbiased<typename It::value_type> opVariance(
            opMean.average());
        opVariance = std::for_each(beg, end, opVariance);
        return static_cast<Tout>(opVariance.result());
      }
    };

    /*! @brief Variance operator on Set over a range
     *
     * This operator actually use the operator s_opVarianceLinearUnbiasedOnSet
     * over the range specified by the provided iterators It first computes the
     * mean of the provided range.
     *
     * @see s_opVarianceLinearUnbiasedOnSet, s_opMeanLinearOnSet
     */
    template <class It, class Tout = typename DataTraits<
                            typename It::value_type>::accumulator_type>
    struct s_measVarianceLinearOnSet
        : std::binary_function<It, const It &, Tout> {
      Tout operator()(const It &beg, const It &end)
      {
        s_opMeanLinearOnSet<typename It::value_type> opMean;
        opMean = std::for_each(beg, end, opMean);

        s_opVarianceLinearUnbiasedOnSet<typename It::value_type> opVariance(
            opMean.average());
        opVariance = std::for_each(beg, end, opVariance);
        return static_cast<Tout>(opVariance.result());
      }
    };

    /*! @brief Variance function over an image from iterators
     *
     * This function actually use the operator s_measVarianceLinear over the
     * range specified by the provided iterators
     *
     * @see s_measVarianceLinear, s_opVarianceLinear
     */
    template <class It, class Tout>
    RES_C t_measVarianceLinear(It ibeg, const It &iend, Tout &varianceValue)
    {
      s_measVarianceLinear<It, Tout> op;
      varianceValue = op(ibeg, iend);
      return RES_OK;
    }

    /*! @brief Variance function over an image
     *
     * This operator actually use the operator s_opVarianceLinear over the image
     *
     * @see s_opVarianceLinear, t_measVarianceLinear
     */
    template <class _image, class Tout>
    RES_C t_measVarianceLinear(const _image &im, Tout &varianceValue)
    {
      return t_measVarianceLinear(im.begin(), im.end(), varianceValue);
    }

    /*! @brief Variance function over an image On Set from iterators
     *
     * This function actually use the operator s_measVarianceLinearOnSet over
     * the range specified by the provided iterators
     *
     * @see s_measVarianceLinearOnSet, s_opVarianceLinearOnSet
     */
    template <class It, class Tout>
    RES_C t_measVarianceLinearOnSet(It ibeg, const It &iend,
                                    Tout &varianceValue)
    {
      s_measVarianceLinearOnSet<It, Tout> op;
      varianceValue = op(ibeg, iend);
      return RES_OK;
    }

    /*! @brief Variance function over an image
     *
     * This operator actually use the operator s_opVarianceLinear over the image
     *
     * @see s_opVarianceLinearOnSet, t_measVarianceLinearOnSet
     */
    template <class _image, class Tout>
    RES_C t_measVarianceLinearOnSet(const _image &im, Tout &varianceValue)
    {
      return t_measVarianceLinearOnSet(im.begin(), im.end(), varianceValue);
    }

    /*! @brief Circular mean operator (mean angle)
     *
     * The average is given by the complex sum of the elements' value, normed to
     * 1 (the modulus of each element is set to 1), divided by the number of
     * elements. Input type is the angle value (which is scalar).
     *
     */
    template <class T> struct s_opMeanCircular {
      typedef std::complex<F_DOUBLE> result_type;
      result_type m_acc;
      UINT32 m_nb_pixels;

      s_opMeanCircular() : m_acc(0, 0), m_nb_pixels(0)
      {
      }

      //! Called at each point. Mean is computed over the points fed to this
      //! method.
      void operator()(const T &pixel)
      {
        m_acc += result_type(::cos(static_cast<double>(pixel)),
                             ::sin(static_cast<double>(pixel)));
        m_nb_pixels++;
      }

      //! resets the operator
      void reset()
      {
        m_nb_pixels(0);
        m_acc(0, 0);
      }

      //! Returns the result of the accumulation
      result_type average()
      {
        if (m_nb_pixels == 0)
          throw(MException("empty dataset in s_opMeanCircular"));
        typename result_type::value_type divider =
            static_cast<typename result_type::value_type>(m_nb_pixels);

        // @TODO : error checking
        return result_type(m_acc.real() / divider, m_acc.imag() / divider);
      }
    };

    /*! @brief Circular mean operator (with export) over a range
     *
     * This operator actually use the operator s_opMeanCircular over the range
     * specified by the provided iterators
     *
     * @see s_opMeanCircular
     */
    template <class It, class result = std::complex<F_DOUBLE>>
    struct s_measMeanCircular
        : public std::binary_function<It, const It &, result> {
      typedef It iterator_type;
      result operator()(It beg, const It &end) const
      {
        typedef s_opMeanCircular<typename iterator_type::value_type> _Op;

        _Op op;
        op = std::for_each(beg, end, op);
        return op.average();
      }
    };

    /*! @brief Circular mean operator (with export) over an image
     *
     * This operator actually use the operator s_opMeanCircular over the whole
     * image, through a call to s_measMeanCircular inside the beginning and the
     * end of the image
     *
     * @see s_opMeanCircular, s_measMeanCircular
     */
    template <class _image>
    RES_C t_measMeanCircular(const _image &_im,
                             std::complex<F_DOUBLE> &meanValue)
    {
      s_measMeanCircular<typename _image::const_iterator> op;
      meanValue = op(_im.begin(), _im.end());
      return RES_OK;
    }

    /*! @brief Circular mean operator with modulus norm
     *
     * The average is given by the modulus of the complex sum divided by the
     * number of elements. Input values should be of type pixel3
     *
     * @note This object also offers a method to get a
     * specifically defined median on the dataset it operates on.
     *
     */
    template <class T> struct s_opMeanCircularNormed {
      typedef std::complex<F_DOUBLE> result_type;
      typename DataTraits<
          typename result_type::value_type>::float_accumulator_type m_w_acc;
      result_type m_acc;
      // UINT32				m_nb_pixels;
      s_opMeanCircularNormed() : m_w_acc(0), m_acc(0, 0) /*,m_nb_pixels(0)*/
      {
      }

      //! Called at each point. Mean is computed over the points fed to this
      //! method.
      void operator()(const T &pixel)
      {
        m_acc += result_type(
            static_cast<typename result_type::value_type>(pixel.channel3) *
                cos(static_cast<double>(pixel.channel1)),
            static_cast<typename result_type::value_type>(pixel.channel3) *
                sin(static_cast<double>(pixel.channel1)));
        m_w_acc += pixel.channel3;
        // m_nb_pixels++;
      }

      //! resets the operator
      void reset()
      {
        m_w_acc = 0;
        m_acc   = result_type(0, 0);
      }

      //! Returns the result of the accumulation
      result_type average() const
      {
        if (/*m_nb_pixels*/ m_w_acc == 0)
          throw(MException("empty dataset in s_opMeanCircularNormed"));
        typename result_type::value_type divider =
            static_cast<typename result_type::value_type>(
                /*m_nb_pixels*/ m_w_acc);

        // @TODO: error checking
        return result_type(m_acc.real() / divider, m_acc.imag() / divider);
      }

      //! mediane minimisant la distance entre les couples (angle,rayon) et la
      //! valeur moyenne calculee par average La distance est donnee par D**2 =
      //! r1**2 + r2**2 - 2*r1*r2*cos(a1-a2)
      template <class Iter>
      typename Iter::value_type median(Iter iterI, const Iter &iterE)
      {
        // on prend le point qui minimise la norme de la difference entre un
        // point et la moyenne i.e. le point le plus proche en terme de distance
        // complexe, basee sur le module

        typedef typename Iter::value_type pix_value_type;
        pix_value_type pix, pixOut;
        typedef typename DataTraits<typename pix_value_type::value_type>::
            accumulator_type distance_type;
        distance_type distMin  = std::numeric_limits<distance_type>::max();
        distance_type distTemp = distMin;

        assert(iterI != iterE);
        for (; iterI != iterE; ++iterI) {
          pix            = *iterI;
          result_type ct = result_type(
              static_cast<typename result_type::value_type>(pix.channel1),
              static_cast<typename result_type::value_type>(pix.channel3));
          ct -= m_acc;

          distTemp = static_cast<distance_type>(std::norm(ct));
          if (distMin > distTemp) {
            distMin = distTemp;
            pixOut  = pix;
          }
        }

        return pixOut;
      }
    }; // s_opMeanCircular

    /*! @brief Circular normed mean operator (with export) over a range
     *
     * This operator actually use the operator s_opMeanCircularNormed over the
     * range specified by the input iterators
     *
     * @see s_opMeanCircularNormed
     */
    template <class It, class result = std::complex<F_DOUBLE>>
    struct s_measMeanCircularNormed
        : public std::binary_function<It, const It &, result> {
      typedef It iterator_type;

      s_measMeanCircularNormed() throw()
      {
      }
      result operator()(It beg, const It &end) const
      {
        typedef s_opMeanCircularNormed<typename iterator_type::value_type> _Op;
        _Op op;
        op = std::for_each(beg, end, op);
        return op.average();
      }
    };

    /*! @brief Circular normed mean operator (with export) over an image
     *
     * This operator actually use the operator s_opMeanCircularNormed over the
     * whole image, through a call to s_measMeanCircularNormed inside the
     * beginning and the end of the image
     *
     * @see s_opMeanCircularNormed, s_measMeanCircularNormed
     */
    template <class _image>
    RES_C t_measMeanCircularNormed(const _image &_im,
                                   std::complex<F_DOUBLE> &meanValue)
    {
      s_measMeanCircularNormed<typename _image::const_iterator> op;
      meanValue = op(_im.begin(), _im.end());
      return RES_OK;
    }

    //! obsolete and deprecated. Prefer operators over iterators (implemented
    //! almost at each line of this file)
    template <class T, class Tmask>
    RES_C t_maskedMeasMeanCircularNormed(const Image<T> &_image,
                                         const Image<Tmask> &_imageM,
                                         std::complex<F_DOUBLE> &meanValueOut)
    {
      typedef typename Image<T>::const_iterator const_iterator;
      typedef ImageMaskConstIterator<const_iterator,
                                     typename Image<Tmask>::const_iterator>
          iter;

      iter mit     = iter(_image.begin(), _imageM.begin(), _imageM.end()),
           mit_end = iter(_image.end(), _imageM.end(), _imageM.end());

      s_measMeanCircularNormed<iter> op;

      meanValueOut = op(mit, mit_end); // op(_image.begin_masked(_imageM),
                                       // _image.end_masked(_imageM));
      return RES_OK;
    }

    /*! @brief Centered moment operator (biased)
     *
     * Computes the ith moment over a certain amount of elements.
     * This operator is unbiased if you consider the elements being pixels of
     * the same sample. For computing the moments over different independant
     * samples, this operator introduces an important bias, but no trivial
     * solution exists, since the moment is highly dependant of the samples
     * probability law.
     *
     *
     * @note Moment of order 2
     *
     * Please refer to s_opVarianceLinearUnbiased for an unbiased
     * and maybe quicker way of computing this moment.
     *
     * @see s_opMeanLinear
     */
    template <class T, int i_moment> struct s_opCenteredMomentLinear {
      typedef typename DataTraits<T>::float_accumulator_type facc_type;
      facc_type m_var;
      facc_type m_meanValueLinear;
      UINT32 m_nb_pixels;

      s_opCenteredMomentLinear(const facc_type &d_mVL)
          : m_var(0.), m_meanValueLinear(d_mVL), m_nb_pixels(0)
      {
      }

      //! Should be called on each element
      inline void operator()(const T &pixel)
      {
        facc_type _temp = static_cast<facc_type>(pixel);
        _temp -= m_meanValueLinear;
        m_var += static_cast<facc_type>(
            pow(static_cast<facc_type>(_temp), i_moment));
        m_nb_pixels++;
      }

      //! Resets the operator
      void reset()
      {
        m_nb_pixels = 0;
        m_var       = 0.;
      }

      //! Provides the result of the operator
      inline facc_type average()
      {
        if (m_nb_pixels == 0)
          throw(MException("empty dataset in s_opCenteredMomentLinear"));
        facc_type divider =
            (m_nb_pixels != 0 ? static_cast<facc_type>(m_nb_pixels) : 1.);
        return m_var / divider;
      }
    };

    /*! @brief Centered moment operator over a range
     *
     * This operator actually use the operator s_opCenteredMomentLinear over the
     * range specified by the provided iterators
     *
     * @note Moment of order 2
     *
     * Please refer to s_measVarianceLinearUnbiased for an unbiased
     * and maybe quicker way of computing this moment.
     *
     * @see s_opCenteredMomentLinear
     */
    template <class It, int i_moment,
              class Tout =
                  typename DataTraits<typename It::value_type>::acumulator_type>
    struct s_measCenteredMomentLinear
        : std::binary_function<It, const It &, Tout> {
      typedef
          typename DataTraits<typename It::value_type>::float_accumulator_type
              facc_type;
      facc_type m_center;

      s_measCenteredMomentLinear(const facc_type &center) : m_center(center)
      {
      }
      Tout operator()(It beg, const It &end)
      {
        s_opCenteredMomentLinear<typename It::value_type, i_moment> op(
            m_center);
        op = std::for_each(beg, end, op);
        return static_cast<Tout>(op.average());
      }
    };

    /*! @brief Centered moment function over a range (from iterators)
     *
     * This operator actually use the operator s_opCenteredMomentLinear over the
     * range specified by the provided iterators
     *
     * @see s_opMeanLinear
     */
    template <class It, int i_moment, class Tin, class Tout>
    RES_C t_measCenteredMomentLinear(It ibeg, const It &iend, const Tin &center,
                                     Tout &centeredMoment)
    {
      s_measCenteredMomentLinear<It, i_moment, Tout> op(center);
      centeredMoment = op(ibeg, iend);
      return RES_OK;
    }

    /*! @brief Centered moment function over an image
     *
     * This operator actually use the operator s_opCenteredMomentLinear over the
     * image
     *
     * @see s_opMeanLinear
     */
    template <class _image, int i_moment, class Tin, class Tout>
    RES_C t_measCenteredMomentLinear(const _image &im, const Tin &center,
                                     Tout &centeredMoment)
    {
      return t_measCenteredMomentLinear<typename _image::const_iterator,
                                        i_moment, Tin, Tout>(
          im.begin(), im.end(), center, centeredMoment);
    }

    // @} // addtogroup group_measStats

    /*!
     * @addtogroup group_measSimple
     * @{
     */

    /*! @brief Volume (integration) operator
     *
     * Operator for summing the values (integration)
     *
     */
    template <class T> struct s_opVolume {
      typedef typename DataTraits<T>::accumulator_type acc_type;
      acc_type m_volume;

      s_opVolume() : m_volume(0)
      {
      }
      inline void operator()(const T &pixel)
      {
        m_volume += static_cast<acc_type>(pixel);
      }

      //! Resets the operator
      void reset()
      {
        m_volume = T(0);
      }
    };

    //! Speclializing of s_opVolume for pixel_3 structures. The returned volume
    //! is the marginal one.
    template <class T> struct s_opVolume<pixel_3<T>> {
      typedef pixel_3<T> value_type;
      typedef typename DataTraits<value_type>::accumulator_type acc_type;
      acc_type m_volume;

      s_opVolume() : m_volume(0, 0, 0)
      {
      }
      inline void operator()(const value_type &pixel)
      {
        m_volume += static_cast<acc_type>(pixel);
      }

      //! Resets the operator
      void reset()
      {
        m_volume = value_type(0, 0, 0);
      }
    };

    /*! @brief Volume (integration) operator over a range
     *
     * Operator for summing the values (integration)
     *
     */
    template <class It, class Tout>
    struct s_measVolume : public std::binary_function<It, const It &, Tout> {
      inline Tout operator()(It beg, const It &end)
      {
        s_opVolume<typename It::value_type> op;
        op = std::for_each(beg, end, op);
        return static_cast<Tout>(op.m_volume);
      }
    };

    /*! @brief Volume (integration) operator from iterators
     *
     * Operator for summing the values (integration) of the image
     *
     */
    template <class It, class Tout>
    inline RES_C t_measVolume(It ibeg, const It &iend, Tout &d_volume)
    {
      s_measVolume<It, Tout> op;
      d_volume = op(ibeg, iend);
      return RES_OK;
    }

    /*! @brief Volume (integration) operator over an image
     *
     * Operator for summing the values (integration) of the image
     *
     */
    template <class __image, class Tout>
    inline RES_C t_measVolume(const __image &image, Tout &d_volume)
    {
      return t_measVolume(image.begin(), image.end(), d_volume);
    }

    /*! @brief Minimum and maximum operator
     *
     * Minimum and maximum operator
     *
     */
    template <class T, class predLess = std::less<T>,
              class predGreat = std::greater<T>>
    struct s_opMinMax {
      T m_min, m_max;
      predLess pl;
      predGreat pg;
      s_opMinMax()
          : m_min(std::numeric_limits<T>::max()),
            m_max(boost::numeric::bounds<T>::lowest())
      {
      }

      inline T min() const
      {
        return m_min;
      }
      inline T max() const
      {
        return m_max;
      }

      inline void operator()(const T &pixel)
      {
        if (pg(pixel, m_max))
          m_max = pixel;
        if (pl(pixel, m_min))
          m_min = pixel;
      }
    };

    /*! @brief Minimum and maximum operator over a range
     *
     * Minimum and maximum operator over a range
     *
     */
    template <class It, class Tout = typename It::value_type>
    struct s_measMinMax
        : public std::binary_function<It, const It &, std::pair<Tout, Tout>> {
      typedef std::pair<Tout, Tout> result_type;
      inline result_type operator()(It beg, const It &end)
      {
        s_opMinMax<typename It::value_type> op;
        op = std::for_each(beg, end, op);
        return result_type(static_cast<Tout>(op.min()),
                           static_cast<Tout>(op.max()));
      }
    };

    /*! @brief Minimum and maximum operator over a range. Function variant of
     * s_measMinMax
     *
     * Minimum and maximum operator over a range
     *
     */
    template <class It>
    inline RES_C t_measMinMax(It beg, const It &end,
                              typename It::value_type &_min,
                              typename It::value_type &_max)
    {
      MORPHEE_ENTER_FUNCTION("t_measMinMax(iterators)");

      if (beg == end) {
        MORPHEE_REGISTER_ERROR("Empty dataset");
        return RES_ERROR_BAD_ARG;
      }
      typedef s_measMinMax<It> op_type;
      op_type op;
      typename op_type::result_type res = op(beg, end);
      _min                              = res.first;
      _max                              = res.second;
      return RES_OK;
    }

    /*! @brief Minimum and maximum operator over an image
     *
     * Minimum and maximum operator over an image
     *
     */
    template <class _image, typename T2>
    inline RES_C t_measMinMax(const _image &im, T2 &_min, T2 &_max)
    {
      MORPHEE_ENTER_FUNCTION("t_measMinMax(image)");

      if (!im.isAllocated()) {
        MORPHEE_REGISTER_ERROR("Image not allocated");
        return RES_NOT_ALLOCATED;
      }
      typename _image::value_type _im_max, _im_min;
      RES_C res = t_measMinMax(im.begin(), im.end(), _im_min, _im_max);
      if (res != RES_OK)
        return res;
      _min = static_cast<T2>(_im_min);
      _max = static_cast<T2>(_im_max);
      return RES_OK;
    }

    /*! @brief Minimum and maximum operator over an image, inside a mask
     *
     * Minimum and maximum operator over an image, inside the non-zero values of
     * a mask
     *
     * TODO: maybe use more powerful predicates ?
     *
     */
    template <class ImageValues, class ImageMask, typename T2>
    inline RES_C t_measMinMaxWithMask(const ImageValues &im,
                                      const ImageMask &mask, T2 &_min, T2 &_max)
    {
      MORPHEE_ENTER_FUNCTION("t_measMinMaxWithMask(image)");

      if ((!im.isAllocated()) || (!mask.isAllocated())) {
        MORPHEE_REGISTER_ERROR("Image not allocated");
        return RES_NOT_ALLOCATED;
      }
      // TODO: check sizes (need to think for a second about that one...)

      typename ImageValues::value_type _im_max, _im_min;

      typedef typename ImageValues::const_iterator const_iterator;
      typedef typename ImageMask::const_iterator mask_const_iterator;
      typedef ImageMaskConstIterator<const_iterator, mask_const_iterator>
          masked_iterator;

      masked_iterator mit(im.begin(), mask.begin(), mask.end()),
          mit_end(im.end(), mask.end(), mask.end());

      RES_C res = t_measMinMax(mit, mit_end, _im_min, _im_max);
      if (res != RES_OK)
        return res;
      _min = static_cast<T2>(_im_min);
      _max = static_cast<T2>(_im_max);
      return RES_OK;
    }

    //! Compute the maximum and minimum values of an image and
    //! returns also their positions.
    /* Store the positions of the pixels whose value correspond to
     * the min or max in two separate ListOffset.
     *
     * The first pixel of each vector is the first pixel
     * encountered while iterating through the image (and that
     * satisfied the conditions required to be in this vector
     * of course)
     *
     * @param _image Image where to look for the min and max.
     *
     * @param min minimum value
     *
     * @param max maximum value
     *
     * @param vectMinPos vector containing the positions of the
     * pixels whose value is equal to the minimum
     *
     * @param vectMaxPos vector containing the positions of the
     * pixels whose value is equal to the maximum
     */
    template <typename Tin, typename T2>
    inline RES_C t_measMinMax(const Image<Tin> &_image1, T2 &minVal, T2 &maxVal,
                              ListOffset &vectMinPos, ListOffset &vectMaxPos)
    {
      RES_C res = stats::t_measMinMax(_image1, minVal, maxVal);
      if (RES_OK != res) {
        MORPHEE_REGISTER_ERROR("Unable to find minimum and maximum values");
        return res;
      }
      typename Image<Tin>::const_iterator itIn    = _image1.begin(),
                                          itInEnd = _image1.end();
      for (; itIn != itInEnd; ++itIn) {
        if (*itIn == maxVal) {
          vectMaxPos.push_back(itIn.getOffset());
        } else if (*itIn == minVal) {
          vectMinPos.push_back(itIn.getOffset());
        }
      }

      // I'm pretty sure someone will try to catch me on
      // this, so...
      if (minVal == maxVal) {
        vectMinPos.assign(vectMaxPos.begin(), vectMaxPos.end());
      }

      return RES_OK;

    } // t_measMinMax

    // @} // addtogroup group_measSimple

    /*!
     * @addtogroup group_measHisto
     *
     * @{
     */

    /*! @brief Label counting operator
     *
     * Counts the number of different input values
     *
     */
    template <class T> struct s_opCountLabel {
      typedef std::set<T> container_type;
      container_type m_label;

      inline void operator()(const T &pixel)
      {
        m_label.insert(pixel);
      }

      inline UINT32 countLabels()
      {
        return static_cast<UINT32>(m_label.size());
      }

      //! Resets the operator
      void reset()
      {
        m_label.clear();
      }
    };

    /*! @brief Label counting operator over a range
     *
     * Counts the number of different input values
     * @see s_opCountLabel
     */
    template <class It>
    struct s_measLabelCount
        : public std::binary_function<It, const It &, UINT32> {
      inline UINT32 operator()(It beg, It end)
      {
        s_opCountLabel<typename It::value_type> op;
        op = std::for_each(beg, end, op);
        return op.countLabels();
      }
    };

    /*! @brief Label counting operator over a range
     *
     * Counts the number of different input values
     * @see s_opCountLabel
     */
    template <class It> inline UINT32 t_measLabelCount(It beg, const It &end)
    {
      s_measLabelCount<It> op;
      return op(beg, end);
    }

    /*! @brief Label counting operator over an image
     *
     * Counts the number of different input values
     * @see s_opCountLabel
     */
    template <class _image>
    inline RES_C t_measLabelCount(const _image &im, UINT32 &li_countLabels)
    {
      li_countLabels = t_measLabelCount(im.begin(), im.end());
      return RES_OK;
    }

    /*! @brief Label counting operator, Nobackground
     *
     * Counts the number of different input values, background default value
     * excluded
     *
     */
    // Thomas : y a peut ̻tre plus efficace, mais sa a le merite d'etre simple
    template <class T> struct s_opCountLabelNoBackground {
      typedef std::set<T> container_type;
      container_type m_label;

      inline void operator()(const T &pixel)
      {
        if (pixel != (DataTraits<T>::default_value::background())) {
          m_label.insert(pixel);
        }
      }

      inline UINT32 countLabels()
      {
        return static_cast<UINT32>(m_label.size());
      }

      //! Resets the operator
      void reset()
      {
        m_label.clear();
      }
    };

    /*! @brief Label counting operator over a range, Nobackground
     *
     * Counts the number of different input values, background default value
     * excluded
     * @see s_opCountLabelNoBackGround
     */
    template <class It>
    struct s_measLabelCountNoBackground
        : public std::binary_function<It, const It &, UINT32> {
      inline UINT32 operator()(It beg, It end)
      {
        s_opCountLabelNoBackground<typename It::value_type> op;
        op = std::for_each(beg, end, op);
        return op.countLabels();
      }
    };

    /*! @brief Label counting operator over a range, Nobackground
     *
     * Counts the number of different input values, background default value
     * excluded
     * @see s_opCountLabelNoBackGround
     */
    template <class It>
    inline UINT32 t_measLabelCountNoBackground(It beg, const It &end)
    {
      s_measLabelCountNoBackground<It> op;
      return op(beg, end);
    }

    /*! @brief Label counting operator over an image, Nobackground
     *
     * Counts the number of different input values, background default value
     * excluded
     * @see s_opCountLabelNoBackGround
     */
    template <class _image>
    inline RES_C t_measLabelCountNoBackground(const _image &im,
                                              UINT32 &li_countLabels)
    {
      li_countLabels = t_measLabelCountNoBackground(im.begin(), im.end());
      return RES_OK;
    }

    /*! @brief Histogram operator
     *
     * Histogram operator
     *
     */
    template <class T> struct s_opHistogram {
    public:
      typedef T value_type;
      typedef std::map<T, UINT32> map_type;
      typedef std::pair<T, UINT32> t_pair;
      typedef typename map_type::iterator iterator;
      typedef std::pair<iterator, bool> t_insertpair;

      map_type m_map;
      t_insertpair m_pair;
      inline void operator()(const T &pixel)
      {
        iterator it = m_map.find(pixel);
        if (it == m_map.end()) {
          m_pair = m_map.insert(t_pair(pixel, 1));
          assert(m_pair.second == true);
        } else {
          (it->second)++;
        }
      }
    };

    /*! @brief Histogram operator over a range
     *
     * Histogram operator over a range
     *
     */
    template <class It,
              class result =
                  typename s_opHistogram<typename It::value_type>::map_type>
    struct s_measHistogram
        : public std::binary_function<It, const It &, result> {
      inline result operator()(It beg, It end)
      {
        s_opHistogram<typename It::value_type> op;
        op = std::for_each(beg, end, op);
        return op.m_map;
      }
    };

    /*! @brief Histogram operator. Iterator version.
     *
     * Histogram operator over a range
     *
     */
    template <class It>
    inline RES_C
    t_measHistogram(It ibeg, const It &iend,
                    std::map<typename It::value_type, UINT32> &map_histogram)
    {
      s_measHistogram<It> op;
      map_histogram = op(ibeg, iend);
      return RES_OK;
    }

    /*! @brief Histogram operator over an image
     *
     * Histogram operator over an image
     *
     */
    template <class _image>
    inline RES_C t_measHistogram(
        const _image &im,
        std::map<typename _image::value_type, UINT32> &map_histogram)
    {
      return t_measHistogram(im.begin(), im.end(), map_histogram);
    }

    /*! @brief Histogram operator. Specialization for pixel_3 images
     *
     * Histogram operator for colour images (pixel_3).
     * Each channel is taken separately from the other.
     *
     */
    template <class T> struct s_opHistogramIndependant {
    public:
      typedef typename T::value_type value_type;
      typedef std::vector<std::map<value_type, UINT32>> map_type;
      typedef std::pair<value_type, UINT32> t_pair;
      typedef typename map_type::value_type::iterator iterator;
      typedef std::pair<iterator, bool> t_insertpair;

      map_type m_map;
      t_insertpair m_pair;
      s_opHistogramIndependant() : m_map(3)
      {
      }
      inline void operator()(const T &pixel)
      {
        iterator it = m_map[0].find(pixel.channel1);
        if (it == m_map[0].end()) {
          m_pair = m_map[0].insert(t_pair(pixel.channel1, 1));
          assert(m_pair.second == true);
        } else {
          (it->second)++;
        }

        it = m_map[1].find(pixel.channel2);
        if (it == m_map[1].end()) {
          m_pair = m_map[1].insert(t_pair(pixel.channel2, 1));
          assert(m_pair.second == true);
        } else {
          (it->second)++;
        }

        it = m_map[2].find(pixel.channel3);
        if (it == m_map[2].end()) {
          m_pair = m_map[2].insert(t_pair(pixel.channel3, 1));
          assert(m_pair.second == true);
        } else {
          (it->second)++;
        }
      }
    };

    /*! @brief Histogram (independant channels) operator over a range
     *
     * Histogram operator over a range. The histogram of each channel is taken
     * independantly from the others. Hence, 3 histograms are computed for a
     * pixel_3 image
     *
     */
    template <class It, class result = typename s_opHistogramIndependant<
                            typename It::value_type>::map_type>
    struct s_measHistogramIndependant
        : public std::binary_function<It, const It &, result> {
      inline result operator()(It beg, const It &end)
      {
        s_opHistogramIndependant<typename It::value_type> op;
        op = std::for_each(beg, end, op);
        return op.m_map;
      }
    };

    /*! @brief Histogram (independant channels) operator. Iterator version
     *
     * Histogram operator over an image
     *
     */
    template <class It>
    inline RES_C t_measHistogramIndependant(
        It ibeg, const It &iend,
        typename s_opHistogramIndependant<typename It::value_type>::map_type
            &map_histogram)
    {
      s_measHistogramIndependant<It> op;
      map_histogram = op(ibeg, iend);
      return RES_OK;
    }

    /*! @brief Histogram (independant channels) operator over an image
     *
     * Histogram operator over an image
     *
     */
    template <class _image>
    inline RES_C t_measHistogramIndependant(
        const _image &im,
        typename s_opHistogramIndependant<typename _image::value_type>::map_type
            &map_histogram)
    {
      return t_measHistogramIndependant(im.begin(), im.end(), map_histogram);
    }

    // @} // addtogroup  group_measHisto

    /*!
     * @addtogroup group_measGeom
     *
     * @{
     */

    /*! @brief Barycentre operator
     *
     * Simple barycentre operator with integer values (a more accurate version
     * may exists if you specify the returning type)
     *
     */
    template <class It, class Tout = Point3D>
    struct s_Barycentre : public std::binary_function<It, const It &, Tout> {
      typedef Tout result_type;
      inline result_type operator()(It beg, const It &end)
      {
        if (beg == end)
          throw(MException("empty dataset in s_Barycentre"));

        UINT32 x, y, z, n;
        x = y = z = n = 0;
        for (; beg != end; ++beg) {
          n++;
          x += beg.getX();
          y += beg.getY();
          z += beg.getZ();
        }

        result_type res;
        res.x = x / n;
        res.y = y / n;
        res.z = z / n;

        return res;
      }
    };

    /*! @brief Barycentre over a range. Function variant of s_Barycentre
     *
     * Barycentre operator over a range
     *
     */
    template <class It>
    inline RES_C t_Barycentre(It beg, const It &end, Point3D &point)
    {
      typedef s_Barycentre<It> op_type;
      op_type op;
      try {
        typename op_type::result_type res = op(beg, end);
        point                             = res;
      } catch (MException e) {
        MORPHEE_REGISTER_ERROR(e.what());
        return RES_ERROR;
      }
      return RES_OK;
    }

    /*! @brief Estimate of the normal vector of the plan
            approaching the group of points given by the input vector
     *
     * Computes coef_x and coef_x in:
     *
     * coef_x * x + coef_y * y + c = z
     *
     * So that we have:
     *
     *   NormalVector = (coef_x,coef_y,-1)
     *
     */
    template <class It,
              class Tout = typename DataTraits<typename std::iterator_traits<
                  It>::value_type>::float_accumulator_type>
    struct s_measNormalVector
        : public std::binary_function<It, const It &, Tout> {
    private:
      typedef typename std::iterator_traits<It>::value_type Tin;

    public:
      typedef Tout
          result_type; /*!< The type the coordinates will be returned with */

      inline result_type operator()(It beg, const It &end)
      {
        F_DOUBLE sumX2 = 0, sumY2 = 0, sumXZ = 0, sumYZ = 0, xx = 0, yy = 0,
                 zz = 0, corrXY2 = 0, corrZ2 = 0, sumZ2 = 0;

        // First compute the average on the given set of vectors
        s_measMeanLinear<It, Tout> moyOp;
        Tout average = moyOp(beg, end);

        // Then put the mean of the dataset at zero and
        // estimate its normal vector.
        for (; beg != end; ++beg) {
          xx = (*beg).channel1 - average.channel1;
          yy = (*beg).channel2 - average.channel2;
          zz = (*beg).channel3 - average.channel3;
          sumX2 += (xx * xx);
          sumY2 += (yy * yy);
          sumXZ += (xx * zz);
          sumYZ += (yy * zz);
          corrXY2 += (xx * yy);
          sumZ2 += (zz * zz);
        }

        Tout normalVector;
        Tout determinant;

        // We'd better run a little check here
        if (sumX2 < boost::numeric::bounds<F_DOUBLE>::smallest() ||
            sumY2 < boost::numeric::bounds<F_DOUBLE>::smallest()) {
          if (sumX2 < boost::numeric::bounds<F_DOUBLE>::smallest() &&
              sumY2 < boost::numeric::bounds<F_DOUBLE>::smallest()) {
            throw(MException("Problem computing the normal. Are the points on "
                             "a same vertical line ?"));
          } else if (sumX2 < boost::numeric::bounds<F_DOUBLE>::smallest()) {
            // All points on a vertical plane of equation x=constant
            normalVector = Tout(-1, 0, 0);
            determinant  = Tout(1, 1, 1);
          } else if (sumY2 < boost::numeric::bounds<F_DOUBLE>::smallest()) {
            // All points on a vertical plane of equation y=constant
            normalVector = Tout(0, -1, 0);
            determinant  = Tout(1, 1, 1);
          }
        } else {
          // Finish the computation of the correlation coefficient:
          corrXY2 = (corrXY2 * corrXY2) / (sumX2 * sumY2);

          if ((-1 * boost::numeric::bounds<F_DOUBLE>::smallest()) <
                  (corrXY2 - 1.) &&
              (corrXY2 - 1.) < boost::numeric::bounds<F_DOUBLE>::smallest()) {
            // Then the plane cannot be described by an equation: a.x+b.y+c=z
            // We have to check that all the points are not on the same line in
            // 3D:
            corrZ2 = (sumXZ * sumXZ) / (sumZ2 * sumX2);
            if ((-1 * boost::numeric::bounds<F_DOUBLE>::smallest()) <
                    (corrZ2 - 1.) &&
                (corrZ2 - 1.) < boost::numeric::bounds<F_DOUBLE>::smallest()) {
              throw(MException("Problem computing the normal. Are the points "
                               "on a same line (in 3D) ?"));
            } else {
              // We have a vertical plane here:
              normalVector = Tout(sumXZ, sumYZ, 0);
              determinant  = Tout(sumX2, sumY2, 1);
            }
          } else {
            normalVector = Tout(sumXZ, sumYZ, 1);
            determinant  = Tout(sumX2, sumY2, -1);
          }
        }

        return (normalVector / determinant);
      }
    };

    /*! @brief Estimate of the normal vector of the plan
            approaching the group of dots given by the input vector
     *
     * Computes coef_x and coef_x in:
     *
     * coef_x * x + coef_y * y + z = c
     *
     * So that we have:
     *
     *   NormalVector = (coef_x,coef_y,1)
     *
     *\param v vector of vectors (those are pixel3s representing
     *the points of the surface where we want to compute a linear
     *regression)
     *
     *\param coef_x X coordinate of the normal vector
     *
     *\param coef_y Y coordinate of the normal vector
     */
    template <class Tin, class TOut>
    RES_C t_computeLinearRegression(std::vector<pixel_3<Tin>> &v, TOut &coef_x,
                                    TOut &coef_y)
    {
      MORPHEE_ENTER_FUNCTION("computeLinearRegression");
      s_measNormalVector<Tin> op;
      pixel_3<Tin> normalVector(0, 0, 0);

      try {
        normalVector = op(v.begin(), v.end());
      } catch (MException e) {
        MORPHEE_REGISTER_ERROR(e.what());
        return RES_ERROR;
      }

      coef_x = static_cast<TOut>(normalVector.channel1);
      coef_y = static_cast<TOut>(normalVector.channel2);

      return RES_OK;
    }

    /*! @brief Give the slope of the line of highest slope on a
     *   plane estimated over a set of pixel_3.
     *
     * Return the ma
     * @see s_measNormalVector
     *
     */
    template <class It, typename Tout = F_DOUBLE>
    //			class Tout = typename DataTraits< typename
    //std::iterator_traits<It>::value_type::value_type
    //>::float_accumulator_type>
    struct s_measSlope : public std::binary_function<It, const It &, Tout> {
    private:
      typedef typename std::iterator_traits<It>::value_type Tin;

    public:
      // Floating points make more sense for a slope, it's up to
      // the user to see if he wants to approximate it with an
      // integer or not.
      typedef Tout result_type; /*!< The type the slope will be returned with */

      //! use computeLinearRegression to compute the slope
      inline result_type operator()(It beg, const It &end)
      {
        s_measNormalVector<It> op;
        Tin normalVector(0, 0, 0);
        normalVector = op(beg, end);
        // If the plane is vertical, return an infinite slope
        if (normalVector.channel3 == 0) {
          return std::numeric_limits<Tout>::infinity();
        }

        // Else: do this little hack to get the slope.
        normalVector.channel3 = 0;
        return t_Norm_L2(normalVector);
      }
    };

    //@} // addtogroup group_measGeom

    /*!
     * @addtogroup group_measStats
     *
     * @{
     */

    /*! @brief Mean Square Error operator
     *
     * Returns the normalized MSE (mean square error) between two data sets:
     *
     * (1/N) * ( sum{ ||set1(i), set1(i)||^2} )
     *
     */
    template <typename T> struct s_MSE_helper {
      typedef typename DataTraits<T>::float_accumulator_type return_type;

      s_MSE_helper() : m_accum(0), m_count(0)
      {
      }

      RES_C operator()(const T &v1, const T &v2)
      {
        return_type tmp =
            v1 > v2 ? v1 - v2 : v2 - v1; // abs val of the difference
        tmp *= tmp;
        m_accum += tmp;
        ++m_count;
        return RES_OK;
      }

      const return_type getValue() const
      {
        return m_accum / m_count;
      }

      return_type m_accum;
      UINT32 m_count;
    };

    /*! @brief Mean Square Error operator
     *
     * (specialisation for pixel_3)
     */
    template <typename T> struct s_MSE_helper<pixel_3<T>> {
      typedef typename DataTraits<T>::float_accumulator_type return_type;

      s_MSE_helper() : m_accum(0), m_count(0)
      {
      }

      RES_C operator()(const pixel_3<T> &v1, const pixel_3<T> &v2)
      {
        return_type tmp = t_Distance_L2_squared(v1, v2);
        m_accum += tmp;
        ++m_count;
        return RES_OK;
      }

      const return_type getValue() const
      {
        return m_accum / (3 * m_count);
      }

      return_type m_accum;
      UINT32 m_count;
    };

    /*! @brief Mean Square Error operator
     *
     * Returns the normalized MSE (mean square error) between two data sets:
     *
     * (1/N) * ( sum{ ||set1(i), set1(i)||^2} )
     *
     */
    template <class IMAGE, class tReturn>
    RES_C t_measMeanSquaredError(const IMAGE &imIn1, const IMAGE &imIn2,
                                 tReturn &out)
    {
      s_MSE_helper<typename IMAGE::value_type> op;
      RES_C res = t_ImBinaryMeasureOperation(imIn1, imIn2, op);
      if (res != RES_OK)
        return res;
      out = static_cast<tReturn>(op.getValue());
      return RES_OK;
    }

    /*!
     * @brief Compute the PSN
     *
     * @code
     * PSNR = 10 log10 ( Max(DataType)^2 / MSE(image1,image2) )
     * @endcode
     */
    template <typename T> struct s_PSNR_Helper {
      static T maxVal()
      {
        return std::numeric_limits<T>::max();
      }
    };

    /*!
     * @brief Compute the PSN
     *
     * (specialisation for pixel_3)
     */
    template <typename T> struct s_PSNR_Helper<pixel_3<T>> {
      static T maxVal()
      {
        return s_PSNR_Helper<T>::maxVal();
      }
    };

    /*!
     * @brief Compute the PSN
     *
     * @code
     * PSNR = 10 log10 ( Max(DataType)^2 / MSE(image1,image2) )
     * @endcode
     */
    template <class IMAGE>
    RES_C t_measPSNR(const IMAGE &imIn1, const IMAGE &imIn2, F_DOUBLE &dPSNR)
    {
      MORPHEE_ENTER_FUNCTION("t_measPSNR");

      F_DOUBLE dMSE;

      RES_C res = t_measMeanSquaredError(imIn1, imIn2, dMSE);
      if (res != RES_OK) {
        return res;
      }

      F_DOUBLE dMax = static_cast<F_DOUBLE>(
          s_PSNR_Helper<typename IMAGE::value_type>::maxVal());

      if (dMSE <= boost::numeric::bounds<F_DOUBLE>::smallest())
        dPSNR = std::numeric_limits<F_DOUBLE>::max(); // +infinity for equal
                                                      // images ?
      else {
        // RAFFI: depassement d'operation arithmetique les enfants, regardez les
        // millions de warning de GCC :)
        dMax *= dMax;

        if (dMax <= boost::numeric::bounds<F_DOUBLE>::smallest())
          dPSNR =
              boost::numeric::bounds<F_DOUBLE>::lowest(); // -infinity for
                                                          // log(0) when dMax==0
        else
          dPSNR =
              10. * std::log10(dMax / dMSE); // OK, not in a pathological case
      }

      return RES_OK;
    }

    /*! @brief Compute the correlation operator between two samples
     *
     * Correlation formula:
     *
     * @code
     * r = [E(X-E(X)).E(Y-E(Y))] / [std_dev(X).std_dev(Y)]
     * @endcode
     *
     */
    template <typename T> struct s_opCorrelationCoefficient {
      typedef typename DataTraits<T>::float_accumulator_type return_type;

      s_opCorrelationCoefficient()
          : m_sum_X2(0), m_sum_Y2(0), m_sum_X(0), m_sum_Y(0), m_sum_XY(0),
            m_count(0)
      {
      }

      inline RES_C operator()(const T &v1, const T &v2)
      {
        // number of sample points
        m_count++;
        // now accumulate the various sums
        const return_type X(static_cast<return_type>(v1));
        const return_type Y(static_cast<return_type>(v2));
        m_sum_X2 += X * X;
        m_sum_Y2 += Y * Y;
        m_sum_XY += X * Y;
        m_sum_X += X;
        m_sum_Y += Y;
        return RES_OK;
      }

      /*!
       * @brief Retuns the correlation coefficient.
       *
       * @warning The coefficient is of course a value between
       * -1 and 1 BUT returns -2 if the variance of one of the
       * samples is null since the correlation coef. is not
       * defined in this case !
       *
       */
      const return_type getValue() const
      {
        const return_type var_X((m_count * m_sum_X2 - m_sum_X * m_sum_X));
        const return_type var_Y((m_count * m_sum_Y2 - m_sum_Y * m_sum_Y));
        if (var_X != 0 && var_Y != 0) {
          return ((m_count * m_sum_XY - m_sum_X * m_sum_Y) /
                  std::sqrt(var_X * var_Y));
        } else {
          return -2;
        }
      }

      return_type m_sum_X2;
      return_type m_sum_Y2;
      return_type m_sum_X;
      return_type m_sum_Y;
      return_type m_sum_XY;
      size_t m_count;

    }; // s_opCorrelationCoefficient

    /*! @brief Compute the correlation operator between two images
     *
     * Correlation formula:
     *
     * @code
     * r = [E(X-E(X)).E(Y-E(Y))] / [std_dev(X).std_dev(Y)]
     * @endcode
     *
     */
    template <class IMAGE>
    RES_C t_measCorrelationCoefficient(const IMAGE &imIn1, const IMAGE &imIn2,
                                       F_DOUBLE &corr)
    {
      MORPHEE_ENTER_FUNCTION("t_measCorrelationCoefficient");

      s_opCorrelationCoefficient<typename IMAGE::value_type> op;

      RES_C res = t_ImBinaryMeasureOperation(imIn1, imIn2, op);
      if (res != RES_OK) {
        MORPHEE_REGISTER_ERROR(
            "Impossible to measure the correlation coefficient.");
        return res;
      }

      corr = static_cast<F_DOUBLE>(op.getValue());
      if (corr == -2) {
        MORPHEE_REGISTER_ERROR("Correlation is not defined when one at least "
                               "of the sample's variance is null.");
        return RES_ERROR_BAD_ARG;
      }

      return RES_OK;
    }

    /*! @brief Computes the variogram of an image along a given
     * vector.
     *
     * It must be initialised with the image you want to compute
     * the variogram on. And then call the operator giving it the
     * vector along which you want the variogram value.
     *
     */
    template <class t_image> struct s_opVariogram {
    public:
      typedef t_image image_type;
      typedef typename image_type::value_type value_type;
      typedef typename image_type::coordinate_system coordinate_system;
      typedef typename image_type::window_info_type window_info_type;

    private:
      image_type &imData;

    public:
      typedef typename DataTraits<value_type>::float_accumulator_type
          value_type_out;

      /*! @brief Constructor
       *
       *	@param imIn The image whose variogram is to be computed.
       */
      s_opVariogram(image_type &imIn) : imData(imIn){};

      /*! @brief  Compute one value of the image's variogram
       *
       * @param hvector the vector with which a variogram value will be
       * processed.
       *
       * @return The value of the variogram for the given vector.
       */
      inline value_type_out operator()(coordinate_system &hvector)
      {
        coordinate_system p(0);
        offset_t ofs        = 0;
        value_type_out accu = 0;
        UINT32 taille       = 0;
        value_type valeur   = 0;

        // Set the new active window's dimension
        coordinate_system newWSize = imData.Size() - hvector;
        /*newWXSize = imData.getXSize() - hvector.x;
        newWYSize = imData.getYSize() - hvector.y;
        newWZSize = imData.getZSize() - hvector.z;*/

        // Test if the vector fits in the image :)
        for (unsigned int i = 0; i < coordinate_system::dimension; i++) {
          if (newWSize[i] <= 0) {
            throw(MException("Vector is too large for the image."));
          }
        }

        // Changing the active Window
        window_info_type oldwi = imData.ActiveWindow();
        window_info_type newwi(coordinate_system(0), newWSize);

        if (imData.setActiveWindow(newwi) != RES_OK) {
          throw(MException("Unable to change the active window"));
        }

        typename image_type::iterator itBeg = imData.begin(),
                                      itEnd = imData.end();

        taille =
            t_WSizePixel(imData); // imData.getWxSize() * imData.getWySize() *
                                  // imData.getWzSize();
        __DEBUG_VERBOSE_STATS_MEASURE_T_HPP(
            "Taille = " << taille << "  Accu = " << int(accu));
        __DEBUG_VERBOSE_STATS_MEASURE_T_HPP("vecteur = (" << hvector[0] << ", "
                                                          << hvector[1] << ", "
                                                          << hvector[2] << ")");

        // Iterate on the window to get the "first" point used to process the
        // variogram value
        for (; itBeg != itEnd; ++itBeg) {
          valeur = *itBeg;
          // Compute the second point
          p = itBeg.Position();
          p += hvector;
          /*x = itBeg.getX() + hvector.x;
          y = itBeg.getY() + hvector.y;
          z = itBeg.getZ() + hvector.z;*/

          ofs = t_GetOffsetFromCoords(imData, p);
          // Get the square of the differences

          value_type val = imData.pixelFromOffset(ofs);
          val -= valeur;
          val *= val;

          // accu +=
          // (imData.pixelFromOffset(ofs)-valeur)*(imData.pixelFromOffset(ofs)-valeur);
          accu += val;
        }

        if (RES_OK != imData.setActiveWindow(oldwi)) {
          throw(MException("Unable to reset the active window"));
        }

        // Return the *mean* square difference
        return (accu / (2 * taille));
      }
    }; // s_opVariogram

    /*!
     * @brief Compute the variogram in a given direction from 0 to
     * a given maximum length.
     *
     *
     * @param imIn image on which to compute the variogram
     *
     * @param vect_elem the elementary vector that will be used to
     * define the direction and steps at which the variogram will
     * be calculated.
     *
     * @param max_length the maximum number of steps at which to calculate the
     * variogram's values
     *
     * @param vario the resulting variogram (advised type is
     * F_DOUBLE)
     *  @warning the returned vector is a size max_length+1 !
     */
    template <class image_t, typename Tout>
    RES_C t_measVariogram(image_t imIn,
                          const typename image_t::coordinate_system vect_elem,
                          const offset_t max_length, std::vector<Tout> &vario)
    {
      typename image_t::coordinate_system current_vector(0);
      s_opVariogram<image_t> op(imIn);
      Tout temp;

      for (offset_t i = 0; i < (max_length + 1); ++i) {
        try {
          temp = op(current_vector);
        } catch (...) {
          MORPHEE_REGISTER_ERROR("Spectified length is too wide");
          return RES_ERROR_BAD_ARG;
        }
        vario.push_back(temp);
        current_vector += vect_elem;
      }
      return RES_OK;
    } // t_measVariogram

    /*! @brief Computes the covariance of an image along a given
     * vector.
     *
     * It must be initialised with the image you want to compute
     * the covariance on. And then call the operator giving it the
     * vector along which you want the covariance value.
     *
     * The covariance is calculated without centering the input
     * signal, this is up to the users to make chose wether to
     * center or not the in put image.
     */
    template <class t_image> struct s_opAutoCovariance {
    public:
      typedef t_image image_type;
      typedef typename image_type::value_type value_type;
      typedef typename image_type::coordinate_system coordinate_system;
      typedef typename image_type::window_info_type window_info_type;

    private:
      image_type &imData;

    public:
      typedef typename DataTraits<value_type>::float_accumulator_type
          value_type_out;

      /*! @brief Constructor
       *
       *	@param imIn The image whose covariance is to be computed.
       */
      s_opAutoCovariance(image_type &imIn) : imData(imIn){};

      /*! @brief  Compute one value of the image's covariance
       *
       * @param hvector the vector with which a covariance value will be
       * processed.
       *
       * @return The value of the covariance for the given vector.
       */
      inline value_type_out operator()(coordinate_system &hvector)
      {
        coordinate_system p(0);
        offset_t ofs        = 0;
        value_type_out accu = 0;
        UINT32 taille       = 0;
        value_type_out val  = 0;

        // Set the new active window's dimension
        coordinate_system newWSize = imData.Size() - hvector;
        /*newWXSize = imData.getXSize() - hvector.x;
        newWYSize = imData.getYSize() - hvector.y;
        newWZSize = imData.getZSize() - hvector.z;*/

        // Test if the vector fits in the image :)
        for (unsigned int i = 0; i < coordinate_system::dimension; i++) {
          if (newWSize[i] <= 0) {
            throw(MException("Vector is too large for the image."));
          }
        }

        // Changing the active Window
        window_info_type oldwi = imData.ActiveWindow();
        window_info_type newwi(coordinate_system(0), newWSize);

        if (imData.setActiveWindow(newwi) != RES_OK) {
          throw(MException("Unable to change the active window"));
        }

        typename image_type::iterator itBeg = imData.begin(),
                                      itEnd = imData.end();

        taille =
            t_WSizePixel(imData); // imData.getWxSize() * imData.getWySize() *
                                  // imData.getWzSize();
        __DEBUG_VERBOSE_STATS_MEASURE_T_HPP(
            "Taille = " << taille << "  Accu = " << int(accu));
        __DEBUG_VERBOSE_STATS_MEASURE_T_HPP("vecteur = (" << hvector[0] << ", "
                                                          << hvector[1] << ", "
                                                          << hvector[2] << ")");

        // Iterate on the window to get the "first" point used to process the
        // covariance value
        for (; itBeg != itEnd; ++itBeg) {
          val = static_cast<value_type_out>(*itBeg);
          // Compute the second point
          p = itBeg.Position();
          p += hvector;
          /*x = itBeg.getX() + hvector.x;
          y = itBeg.getY() + hvector.y;
          z = itBeg.getZ() + hvector.z;*/

          ofs = t_GetOffsetFromCoords(imData, p);
          // Get the square of the differences

          const value_type_out val2(
              static_cast<value_type_out>(imData.pixelFromOffset(ofs)));
          val *= val2;

          // accu +=
          // (imData.pixelFromOffset(ofs)-valeur)*(imData.pixelFromOffset(ofs)-valeur);
          accu += val;
        }

        if (RES_OK != imData.setActiveWindow(oldwi)) {
          throw(MException("Unable to reset the active window"));
        }

        // Return the *mean* product
        return (accu / (taille));
      }
    }; // s_opAutoCovariance

    /*!
     * @brief Compute the auto covariance in a given direction from 0 to
     * a given maximum length.
     *
     *
     * @param imIn image on which to compute the covariance
     *
     * @param vect_elem the elementary vector that will be used to
     * define the direction and steps at which the covariance will
     * be calculated.
     *
     * @param max_length the maximum number of steps at which to calculate the
     * covariance's values
     *
     * @param cov the resulting covariance (advised type is
     * F_DOUBLE)
     *  @warning the returned vector is a size max_length+1 !
     */
    template <class image_t, typename Tout>
    RES_C
    t_measAutoCovariance(image_t imIn,
                         const typename image_t::coordinate_system vect_elem,
                         const offset_t max_length, std::vector<Tout> &cov)
    {
      typename image_t::coordinate_system current_vector(0);
      s_opAutoCovariance<image_t> op(imIn);
      Tout temp;

      for (offset_t i = 0; i < (max_length + 1); ++i) {
        try {
          temp = op(current_vector);
        } catch (...) {
          MORPHEE_REGISTER_ERROR("Spectified length is too wide");
          return RES_ERROR_BAD_ARG;
        }
        cov.push_back(temp);
        current_vector += vect_elem;
      }
      return RES_OK;
    } // t_measAutoCovariance

    // 		/*! @brief Mean Square Error operator
    // 		 *
    // 		 * Returns the normalized MSE (mean square error) between two data
    // sets:
    // 		 *
    // 		 * (1/N) * ( sum{ ||set1(i), set1(i)||^2} )
    // 		 *
    // 		 */
    // 		template< class IMAGE >
    // 			RES_C t_measBiVariateLaw( const IMAGE& imIn,
    // 									  const typename IMAGE::coordinate_system& translation,
    // 									  out )
    // 		{
    // 			s_MSE_helper< typename IMAGE::value_type > op;
    // 			RES_C res = t_ImBinaryMeasureOperation( imIn1, imIn2 , op);
    // 			if(res != RES_OK)
    // 				return res;
    // 			out=static_cast<tReturn>(op.getValue());
    // 			return RES_OK;
    // 		}

    //@} //addtogroup group_measStats

  } // namespace stats
} // namespace morphee

#endif
