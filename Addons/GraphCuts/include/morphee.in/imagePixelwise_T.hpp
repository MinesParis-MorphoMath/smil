#ifndef __MORPHEE_IMAGE_PIXELWISE_T_HPP__
#define __MORPHEE_IMAGE_PIXELWISE_T_HPP__

#include <morphee/common/include/commonTypes.hpp>
#include <morphee/image/include/private/image_T.hpp>
#include <morphee/image/include/private/imageUtils_T.hpp>

namespace morphee
{
  /*!
   * @defgroup bitwise_template Template implementations for pixel-wise generic
   * operations
   * @ingroup bitwise
   *
   * @{
   */

  /*!
   * @defgroup bitwise_template_utilities
   *
   * Utilities structures and functions used in the definition
   * and/or optimisation of the pixel-wise operations.
   *
   * @{
   */

  typedef enum {
    e_accessNone,
    e_accessGeneric,
    e_accessContinuous,
    e_accessOffsetCompatible,
    e_accessOffsetDifferenceCompatible
    /*e_accessTableTypeContinuous,
    e_accessTableTypeContinuousOffsetCompatible,
    e_accessTableTypeOffsetDifferenceCompatible*/
  } e_accessType;

  //! Generic application of an unary operator from one image (or any structure
  //! verifying the image class implementation) into another
  template <class __image1, class __image2, class Oper,
            e_accessType e_ac = e_accessGeneric>
  struct s_applyUtilityUnaryOperator {
    const __image1 &im1;
    __image2 &im2;

    typedef typename __image1::const_iterator _iter1;
    typedef typename __image2::iterator _iter2;

    s_applyUtilityUnaryOperator(const __image1 &_im1, __image2 &_im2)
        : im1(_im1), im2(_im2)
    {
    }

    RES_C operator()(_iter1 itin, const _iter1 &itEin, _iter2 itout,
                     const _iter2 &itEout, Oper &op) const
    {
      // std::cout << "e_accessGeneric" << std::endl;
      // if(itin.sizePixel() != itout.sizePixel())
      //{
      for (; (itin != itEin) && (itout != itEout); ++itin, ++itout) {
        *itout = op(*itin);
      }
      /*
    }
    else
    {
      for(;
        (itin != itEin);
        ++itin, ++itout)
      {
        *itout = op(*itin);
      }
    }
    */
      return RES_OK;
    }
  };

#if 0
	//! Specialization for compatible offsets: we need only one iterator
	template <
		typename T1,
		typename T2,
		//class __image1,
		//class __image2,
		class Oper>
		struct s_applyUtilityUnaryOperator<Image<T1> , Image<T2> , Oper, e_accessOffsetCompatible>
	{
		typedef Image<T1> __image1;
		typedef Image<T2> __image2;

		const __image1& im1;
		__image2&		im2;

		typedef typename __image1::const_iterator	_iter1;
		typedef typename __image2::iterator			_iter2;

		s_applyUtilityUnaryOperator(const __image1& _im1, __image2& _im2) :
			im1(_im1), im2(_im2)
		{}

		RES_C operator()(_iter1 itin, const _iter1& itEin, _iter2 itout, const _iter2& itEout, Oper& op) const
		{
			//std::cout << "e_accessOffsetCompatible" << std::endl;
			if(itin.sizePixel() != itout.sizePixel())
			{
				s_applyUtilityUnaryOperator<__image1, __image2, Oper, e_accessGeneric> opgen(im1, im2);
				return opgen(itin, itEin, itout, itEout, op);
			}
			else
			{
				for(;
					(itin != itEin);
					++itin)
				{
					const offset_t off = itin.getOffset();
					im2.pixelFromOffset(off) = op(*itin);
				}
			}
			return RES_OK;
		}
	};
#endif

  //! Specialization for compatible difference of offset: we only need one
  //! iterator
  template <typename T1, typename T2, int dim, class Oper>
  struct s_applyUtilityUnaryOperator<Image<T1, dim>, Image<T2, dim>, Oper,
                                     e_accessOffsetDifferenceCompatible> {
    typedef Image<T1> __image1;
    typedef Image<T2> __image2;

    const __image1 &im1;
    __image2 &im2;

    typedef typename __image1::const_iterator _iter1;
    typedef typename __image2::iterator _iter2;

    s_applyUtilityUnaryOperator(const __image1 &_im1, __image2 &_im2)
        : im1(_im1), im2(_im2)
    {
    }

    RES_C operator()(_iter1 itin, const _iter1 &itEin, _iter2 itout,
                     const _iter2 &itEout, Oper &op) const
    {
      // std::cout << "e_accessOffsetDifferenceCompatible" << std::endl;
      if (itin.sizePixel() != itout.sizePixel() || itin.getXIncrement() != 1 ||
          itout.getXIncrement() != 1) {
        s_applyUtilityUnaryOperator<__image1, __image2, Oper, e_accessGeneric>
            opgen(im1, im2);
        return opgen(itin, itEin, itout, itEout, op);
      } else {
        offset_t off_diff = itout.getOffset() - itin.getOffset();
        for (; (itin != itEin); ++itin) {
          const offset_t off                  = itin.getOffset();
          im2.pixelFromOffset(off + off_diff) = op(*itin);
        }
      }
      return RES_OK;
    }
  };

  //! Specialization for compatible offsets and for known image structure
  template <class T, class U, class Oper>
  struct s_applyUtilityUnaryOperator<Image<T, 3>, Image<U, 3>, Oper,
                                     e_accessOffsetCompatible> {
    typedef Image<T, 3> __image1;
    typedef Image<U, 3> __image2;

    const __image1 &im1;
    __image2 &im2;

    typedef typename __image1::const_iterator _iter1;
    typedef typename __image2::iterator _iter2;

    s_applyUtilityUnaryOperator(const __image1 &_im1, __image2 &_im2)
        : im1(_im1), im2(_im2)
    {
    }

    RES_C operator()(_iter1 itin, const _iter1 &itEin, _iter2 itout,
                     const _iter2 &itEout, Oper &op) const
    {
      // std::cout << "e_accessOffsetCompatible < T, U >" << std::endl;

      if (itin.sizePixel() != itout.sizePixel() || itin.getXIncrement() != 1 ||
          itout.getXIncrement() != 1) {
        s_applyUtilityUnaryOperator<__image1, __image2, Oper, e_accessGeneric>
            opgen(im1, im2);
        return opgen(itin, itEin, itout, itEout, op);
      } else {
        typedef typename __image1::value_type value_type1;
        typedef typename __image2::value_type value_type2;

        const value_type1 *p1 = im1.rawPointer() + itin.getOffset();
        value_type2 *p2       = im2.rawPointer() + itout.getOffset();

        const offset_t inc_in  = itin.getYIncrement() - 1;
        const offset_t inc_out = itout.getYIncrement() - 1;

        for (offset_t line_z = itin.getZSize(); line_z > 0; line_z--) {
          for (offset_t line_y = itin.getYSize(); line_y > 0; line_y--) {
            // ca évite getXSize soustraction
            const value_type1 *p3 = p1 + itin.getXSize();
            for (; p1 != p3; p1++, p2++) {
              *p2 = op(*p1);
            }
            p1 += inc_in;  // t_YIncrementPixel(im1);
            p2 += inc_out; // t_YIncrementPixel(im2);
          }
          p1 += itin.getZIncrement();  // t_ZIncrementPixel(im1);
          p2 += itout.getZIncrement(); // t_ZIncrementPixel(im2);
          p1 -= itin.getYIncrement();  // t_YIncrementPixel(im1);
          p2 -= itout.getYIncrement(); // t_YIncrementPixel(im2);
        }
      }
      return RES_OK;
    }
  };

  //! Specialization for compatible offsets and for known image structure
  template <class T, class U, class Oper>
  struct s_applyUtilityUnaryOperator<Image<T, 2>, Image<U, 2>, Oper,
                                     e_accessOffsetCompatible> {
    typedef Image<T, 2> __image1;
    typedef Image<U, 2> __image2;

    const __image1 &im1;
    __image2 &im2;

    typedef typename __image1::const_iterator _iter1;
    typedef typename __image2::iterator _iter2;

    s_applyUtilityUnaryOperator(const __image1 &_im1, __image2 &_im2)
        : im1(_im1), im2(_im2)
    {
    }

    RES_C operator()(_iter1 itin, const _iter1 &itEin, _iter2 itout,
                     const _iter2 &itEout, Oper &op) const
    {
      // std::cout << "e_accessOffsetCompatible < T, U >" << std::endl;

      if (itin.sizePixel() != itout.sizePixel() || itin.getXIncrement() != 1 ||
          itout.getXIncrement() != 1) {
        s_applyUtilityUnaryOperator<__image1, __image2, Oper, e_accessGeneric>
            opgen(im1, im2);
        return opgen(itin, itEin, itout, itEout, op);
      } else {
        typedef typename __image1::value_type value_type1;
        typedef typename __image2::value_type value_type2;

        const value_type1 *p1 = im1.rawPointer() + itin.getOffset();
        value_type2 *p2       = im2.rawPointer() + itout.getOffset();

        const offset_t inc_in  = itin.getYIncrement() - 1;
        const offset_t inc_out = itout.getYIncrement() - 1;

        for (offset_t line_y = itin.getYSize(); line_y > 0; line_y--) {
          const value_type1 *p3 = p1 + itin.getXSize();
          for (; p1 != p3; p1++, p2++) {
            *p2 = op(*p1);
          }
          p1 += inc_in;  // t_YIncrementPixel(im1);
          p2 += inc_out; // t_YIncrementPixel(im2);
        }
      }
      return RES_OK;
    }
  };

  //! Specialization for compatible offsets and for known image structure
  template <class T, class Oper>
  struct s_applyUtilityUnaryOperator<Image<T, 3>, Image<T, 3>, Oper,
                                     e_accessOffsetCompatible> {
    typedef Image<T, 3> __image;

    const __image &im1;
    __image &im2;

    typedef typename __image::const_iterator _iter1;
    typedef typename __image::iterator _iter2;

    s_applyUtilityUnaryOperator(const __image &_im1, __image &_im2)
        : im1(_im1), im2(_im2)
    {
    }

    RES_C operator()(_iter1 itin, const _iter1 &itEin, _iter2 itout,
                     const _iter2 &itEout, Oper &op) const
    {
      // std::cout << "e_accessOffsetCompatible < T >" << std::endl;
      if (itin.sizePixel() != itout.sizePixel() || itin.getXIncrement() != 1 ||
          itout.getXIncrement() != 1) {
        s_applyUtilityUnaryOperator<__image, __image, Oper, e_accessGeneric>
            opgen(im1, im2);
        return opgen(itin, itEin, itout, itEout, op);
      } else {
        typedef typename __image::value_type value_type;
        if (!t_areImageTheSame(im1, im2)) {
          // Raffi: evidemment ce genre de branchement ne marche pas.
          // Un gd merci au C++
          // On recopie le code de la template concernée
          // s_applyUtilityUnaryOperator<__image, __image, Oper,
          // e_accessOffsetCompatible> opgen(im1, im2); return opgen(itin, itEin,
          // itout, itEout, op);
          typedef typename __image::value_type value_type1;
          typedef typename __image::value_type value_type2;

          const value_type1 *p1 = im1.rawPointer() + itin.getOffset();
          value_type2 *p2       = im2.rawPointer() + itout.getOffset();

          const offset_t inc_in  = itin.getYIncrement() - 1;
          const offset_t inc_out = itout.getYIncrement() - 1;

          for (offset_t line_z = itin.getZSize(); line_z > 0; line_z--) {
            for (offset_t line_y = itin.getYSize(); line_y > 0; line_y--) {
              const value_type1 *p3 = p1 + itin.getXSize();
              for (; p1 != p3; p1++, p2++) {
                *p2 = op(*p1);
              }
              p1 += inc_in;  // t_YIncrementPixel(im1);
              p2 += inc_out; // t_YIncrementPixel(im2);
            }
            p1 += itin.getZIncrement();  // t_ZIncrementPixel(im1);
            p2 += itout.getZIncrement(); // t_ZIncrementPixel(im2);
            p1 -= itin.getYIncrement();  // t_YIncrementPixel(im1);
            p2 -= itout.getYIncrement(); // t_YIncrementPixel(im2);
          }
        } else {
          assert(itin.sizePixel() == itout.sizePixel());
          assert(itin.getXIncrement() == 1);
          assert(itout.getXIncrement() == 1);
          typedef typename __image::value_type value_type;

          value_type *p1 = im2.rawPointer() + itout.getOffset();

          const offset_t inc_in = itout.getYIncrement() - 1;

          for (offset_t line_z = itout.getZSize(); line_z > 0; line_z--) {
            for (offset_t line_y = itout.getYSize(); line_y > 0; line_y--) {
              const value_type *p3 = p1 + itout.getXSize();
              for (; p1 != p3; p1++ /*, ++itout*/) // Raffi : MAIS NON BORDEL,
                                                   // c'est justement ce qu'on
                                                   // ne veut pas faire !!!
              {
                *p1 = op(*p1);
              }
              p1 += inc_in;
            }
            p1 += itout.getZIncrement(); // p1 += t_ZIncrementPixel(im2);
            p1 -= itout.getYIncrement();
          }
        }
      }
      return RES_OK;
    }
  };

  //! Specialization for continuous iteration type (whole images) and for known
  //! image structure
  template <class T, class U, class Oper, int dim>
  struct s_applyUtilityUnaryOperator<Image<T, dim>, Image<U, dim>, Oper,
                                     e_accessContinuous> {
    typedef Image<T, dim> __image1;
    typedef Image<U, dim> __image2;

    const __image1 &im1;
    __image2 &im2;

    typedef typename __image1::const_iterator _iter1;
    typedef typename __image2::iterator _iter2;

    s_applyUtilityUnaryOperator(const __image1 &_im1, __image2 &_im2)
        : im1(_im1), im2(_im2)
    {
      if (!t_isWindowMaximal(im1) || !t_isWindowMaximal(im2)) {
        throw morphee::MException(
            "Trying to call explicitly s_applyUtilityUnaryOperator<Image<T, dim>, Image<U, dim>, Oper, e_accessContinuous> while the properties\
 of the images are not appropriate");
      }
    }

    RES_C operator()(_iter1 itin, const _iter1 &itEin, _iter2 itout,
                     const _iter2 &itEout, Oper &op) const
    {
      // std::cout << "e_accessContinuous < T, U >" << std::endl;
      if (itin.sizePixel() != itout.sizePixel() || itin.getXIncrement() != 1 ||
          itout.getXIncrement() != 1) {
        s_applyUtilityUnaryOperator<__image1, __image2, Oper, e_accessGeneric>
            opgen(im1, im2);
        return opgen(itin, itEin, itout, itEout, op);
      } else {
        assert(itin.getXIncrement() == 1);
        typedef typename __image1::value_type value_type1;
        typedef typename __image2::value_type value_type2;

        const value_type1 *p1 = im1.rawPointer();
        value_type2 *p2       = im2.rawPointer();

        // Raffi: ca évite size_pixel soustractions inutiles
        const value_type1 *p3 = p1 + t_SizePixel(im1);

        for (; p1 != p3; p1++, p2++) {
          *p2 = op(*p1);
        }
      }
      return RES_OK;
    }
  };

  //! Specialization for continuous iteration type (whole images) and for known
  //! image structure
  template <class T, class Oper, int dim>
  struct s_applyUtilityUnaryOperator<Image<T, dim>, Image<T, dim>, Oper,
                                     e_accessContinuous> {
    typedef Image<T, dim> __image;

    const __image &im1;
    __image &im2;

    typedef typename __image::const_iterator _iter1;
    typedef typename __image::iterator _iter2;

    s_applyUtilityUnaryOperator(const __image &_im1, __image &_im2)
        : im1(_im1), im2(_im2)
    {
    }

    RES_C operator()(_iter1 itin, const _iter1 &itEin, _iter2 itout,
                     const _iter2 &itEout, Oper &op) const
    {
      // std::cout << "e_accessContinuous < T >" << std::endl;

      if (itin.sizePixel() != itout.sizePixel() || itin.getXIncrement() != 1 ||
          itout.getXIncrement() != 1) {
        s_applyUtilityUnaryOperator<__image, __image, Oper, e_accessGeneric>
            opgen(im1, im2);
        return opgen(itin, itEin, itout, itEout, op);
      } else {
        typedef typename __image::value_type value_type;

        if (!morphee::t_areImageTheSame(im1, im2)) {
          // Raffi: je ne pense pas que ca marche
          // s_applyUtilityUnaryOperator<Image<T>, Image<T>, Oper,
          // e_accessContinuous> _aop(im1, im2); return _aop(itin, itEin, itout,
          // itEout, op);

          const value_type *p1 = im1.rawPointer();
          value_type *p2       = im2.rawPointer();

          // Raffi: ca évite size_pixel soustractions inutiles
          const value_type *p3 = p1 + t_SizePixel(im1);

          for (; p1 != p3; p1++, p2++) {
            *p2 = op(*p1);
          }
        } else {
          value_type *p1       = im2.rawPointer();
          const value_type *p3 = p1 + t_SizePixel(im1);

          for (; p1 != p3; p1++) {
            *p1 = op(*p1);
          }
        }
      }
      return RES_OK;
    }
  };

  //! Generic application of an unary operator from one image (or any structure
  //! verifying the image class implementation) into another
  template <class __image1, class __image2, class __image3, class Oper,
            e_accessType e_ac = e_accessGeneric>
  struct s_applyUtilityBinaryOperator {
    const __image1 &im1;
    const __image2 &im2;
    __image3 &im3;

    typedef typename __image1::const_iterator _iter1;
    typedef typename __image2::const_iterator _iter2;
    typedef typename __image3::iterator _iter3;

    s_applyUtilityBinaryOperator(const __image1 &_im1, const __image2 &_im2,
                                 __image3 &_im3)
        : im1(_im1), im2(_im2), im3(_im3)
    {
    }

    RES_C operator()(_iter1 itin1, const _iter1 &itEin1, _iter2 itin2,
                     const _iter2 &itEin2, _iter3 itout, const _iter3 &itEOut,
                     Oper &op) const
    {
      // if(	itin1.sizePixel() != itin2.sizePixel() ||
      //	itin1.sizePixel() != itout.sizePixel()) // do we need for the third ?
      //{
      for (; (itin1 != itEin1) && (itin2 != itEin2) && (itout != itEOut);
           ++itin1, ++itin2, ++itout) {
        *itout = op(*itin1, *itin2);
      }
      /*}
      else
      {
        for(;
          (itin1 != itEin1);
          ++itin1, ++itin2, ++itout)
        {
          *itout = op(*itin1, *itin2);
        }
      }*/
      return RES_OK;
    }
  };

  //! Generic application of an unary operator from one image (or any structure
  //! verifying the image class implementation) into another
  template <class __image1, class __image2, class __image3, class Oper>
  struct s_applyUtilityBinaryOperator<__image1, __image2, __image3, Oper,
                                      e_accessOffsetCompatible> {
    const __image1 &im1;
    const __image2 &im2;
    __image3 &im3;

    typedef typename __image1::const_iterator _iter1;
    typedef typename __image2::const_iterator _iter2;
    typedef typename __image3::iterator _iter3;

    s_applyUtilityBinaryOperator(const __image1 &_im1, const __image2 &_im2,
                                 __image3 &_im3)
        : im1(_im1), im2(_im2), im3(_im3)
    {
    }

    RES_C operator()(_iter1 itin1, const _iter1 &itEin1, _iter2 itin2,
                     const _iter2 &itEin2, _iter3 itout, const _iter3 &itEOut,
                     Oper &op) const
    {
      if (itin1.sizePixel() != itin2.sizePixel() ||
          itin1.sizePixel() != itout.sizePixel() ||

          itin1.getXIncrement() != 1 || itin2.getXIncrement() != 1 ||
          itout.getXIncrement() != 1) {
        s_applyUtilityBinaryOperator<__image1, __image2, __image3, Oper,
                                     e_accessGeneric>
            _aop(im1, im2, im3);
        return _aop(itin1, itEin1, itin2, itEin2, itout, itEOut, op);
      } else {
        for (; (itin1 != itEin1); ++itin1) {
          const offset_t off       = itin1.getOffset();
          im3.pixelFromOffset(off) = op(*itin1, im2.pixelFromOffset(off));
        }
      }
      return RES_OK;
    }
  };

  template <class __image1, class __image2, class __image3, class Oper>
  struct s_applyUtilityBinaryOperator<__image1, __image2, __image3, Oper,
                                      e_accessOffsetDifferenceCompatible> {
    const __image1 &im1;
    const __image2 &im2;
    __image3 &im3;

    typedef typename __image1::const_iterator _iter1;
    typedef typename __image2::const_iterator _iter2;
    typedef typename __image3::iterator _iter3;

    s_applyUtilityBinaryOperator(const __image1 &_im1, const __image2 &_im2,
                                 __image3 &_im3)
        : im1(_im1), im2(_im2), im3(_im3)
    {
    }

    RES_C operator()(_iter1 itin1, const _iter1 &itEin1, _iter2 itin2,
                     const _iter2 &itEin2, _iter3 itout, const _iter3 &itEOut,
                     Oper &op) const
    {
      if (itin1.sizePixel() != itin2.sizePixel() ||
          itin1.sizePixel() != itout.sizePixel() ||

          itin1.getXIncrement() != 1 || itin2.getXIncrement() != 1 ||
          itout.getXIncrement() != 1) {
        s_applyUtilityBinaryOperator<__image1, __image2, __image3, Oper,
                                     e_accessGeneric>
            _aop(im1, im2, im3);
        return _aop(itin1, itEin1, itin2, itEin2, itout, itEOut, op);
      } else {
        offset_t off_diffo = itout.getOffset() - itin1.getOffset();
        offset_t off_diff2 = itin2.getOffset() - itin1.getOffset();

        for (; (itin1 != itEin1); ++itin1) {
          const offset_t off = itin1.getOffset();
          im3.pixelFromOffset(off + off_diffo) =
              op(*itin1, im3.pixelFromOffset(off + off_diff2));
        }
      }
      return RES_OK;
    }
  };

  template <class T, class U, class V, class Oper>
  struct s_applyUtilityBinaryOperator<Image<T, 3>, Image<U, 3>, Image<V, 3>,
                                      Oper, e_accessOffsetCompatible> {
    typedef Image<T, 3> __image1;
    typedef Image<U, 3> __image2;
    typedef Image<V, 3> __image3;

    const __image1 &im1;
    const __image2 &im2;
    __image3 &im3;

    typedef typename __image1::const_iterator _iter1;
    typedef typename __image2::const_iterator _iter2;
    typedef typename __image3::iterator _iter3;

    s_applyUtilityBinaryOperator(const __image1 &_im1, const __image2 &_im2,
                                 __image3 &_im3)
        : im1(_im1), im2(_im2), im3(_im3)
    {
    }

    RES_C operator()(_iter1 itin1, const _iter1 &itEin1, _iter2 itin2,
                     const _iter2 &itEin2, _iter3 itout, const _iter3 &itEOut,
                     Oper &op) const
    {
      if (itin1.sizePixel() != itin2.sizePixel() ||
          itin1.sizePixel() != itout.sizePixel() ||

          itin1.getXIncrement() != 1 || itin2.getXIncrement() != 1 ||
          itout.getXIncrement() != 1) {
        s_applyUtilityBinaryOperator<__image1, __image2, __image3, Oper,
                                     e_accessGeneric>
            _aop(im1, im2, im3);
        return _aop(itin1, itEin1, itin2, itEin2, itout, itEOut, op);
      } else {
        typedef typename __image1::value_type value_type1;
        typedef typename __image2::value_type value_type2;
        typedef typename __image3::value_type value_type3;

        const value_type1 *p1 = im1.rawPointer() + itin1.getOffset();
        const value_type2 *p2 = im2.rawPointer() + itin2.getOffset();
        value_type3 *p3       = im3.rawPointer() + itout.getOffset();

        for (offset_t line_z = itin1.getZSize(); line_z > 0; line_z--) {
          for (offset_t line_y = itin1.getYSize(); line_y > 0; line_y--) {
            const value_type1 *p4 = p1 + itin1.getXSize();
            for (; p1 != p4; p1++, p2++, p3++) {
              *p3 = op(*p1, *p2);
            }
            p1 += itin1.getYIncrement(); // t_YIncrementPixel(im1);
            p2 += itin2.getYIncrement(); // t_YIncrementPixel(im2);
            p3 += itout.getYIncrement(); // t_YIncrementPixel(im3);
            p1--;
            p2--;
            p3--;
          }
          p1 += itin1.getZIncrement(); // t_ZIncrementPixel(im1);
          p2 += itin2.getZIncrement(); // t_ZIncrementPixel(im2);
          p3 += itout.getZIncrement(); // t_ZIncrementPixel(im3);

          p1 -= itin1.getYIncrement();
          p2 -= itin2.getYIncrement();
          p3 -= itout.getYIncrement();
        }
      }
      return RES_OK;
    }
  };

  template <class T, class Oper>
  struct s_applyUtilityBinaryOperator<Image<T, 3>, Image<T, 3>, Image<T, 3>,
                                      Oper, e_accessOffsetCompatible> {
    typedef Image<T, 3> __image;

    const __image &im1;
    const __image &im2;
    __image &im3;

    typedef typename __image::const_iterator _iterI;
    typedef typename __image::iterator _iterO;

    s_applyUtilityBinaryOperator(const __image &_im1, const __image &_im2,
                                 __image &_im3)
        : im1(_im1), im2(_im2), im3(_im3)
    {
    }

  private:
    RES_C sameNumberOfPixels(_iterI itin1, const _iterI &itEin1, _iterI itin2,
                             const _iterI &itEin2, _iterO itout,
                             const _iterO &itEOut, Oper &op) const
    {
      // Raffi: cas normal, toutes les images sont différentes
      typedef typename __image::value_type value_type;

      const value_type *p1 = im1.rawPointer() + itin1.getOffset();
      const value_type *p2 = im2.rawPointer() + itin2.getOffset();
      value_type *p3       = im3.rawPointer() + itout.getOffset();

      for (offset_t line_z = itin1.getZSize(); line_z > 0; line_z--) {
        for (offset_t line_y = itin1.getYSize(); line_y > 0; line_y--) {
          const value_type *p4 = p1 + itin1.getXSize();
          for (; p1 != p4; p1++, p2++, p3++) {
            *p3 = op(*p1, *p2);
          }

          p1 += itin1.getYIncrement(); // t_YIncrementPixel(im1);
          p2 += itin2.getYIncrement(); // t_YIncrementPixel(im2);
          p3 += itout.getYIncrement(); // t_YIncrementPixel(im3);
          p1--;
          p2--;
          p3--;
        }
        p1 += itin1.getZIncrement(); // t_ZIncrementPixel(im1);
        p2 += itin2.getZIncrement(); // t_ZIncrementPixel(im2);
        p3 += itout.getZIncrement(); // t_ZIncrementPixel(im3);

        p1 -= itin1.getYIncrement();
        p2 -= itin2.getYIncrement();
        p3 -= itout.getYIncrement();
      }

      return RES_OK;
    }

    RES_C Same12Compatibles(_iterI itin1, const _iterI &itEin1, _iterI itin2,
                            const _iterI &itEin2, _iterO itout,
                            const _iterO &itEOut, Oper &op) const
    {
      typedef typename __image::value_type value_type;

      const value_type *p1 = im1.rawPointer() + itin1.getOffset();
      value_type *p3       = im3.rawPointer() + itout.getOffset();

      for (offset_t line_z = itin1.getZSize(); line_z > 0; line_z--) {
        for (offset_t line_y = itin1.getYSize(); line_y > 0; line_y--) {
          const value_type *p4 = p1 + itin1.getXSize();
          for (; p1 != p4; p1++, p3++) {
            const value_type val1 = *p1;
            *p3                   = op(val1, val1);
          }
          p1 += itin1.getYIncrement(); // t_YIncrementPixel(im1);
          p3 += itout.getYIncrement(); // t_YIncrementPixel(im3);
          p1--;
          p3--;
        }
        p1 += itin1.getZIncrement(); // t_ZIncrementPixel(im1);
        p3 += itout.getZIncrement(); // t_ZIncrementPixel(im3);

        p1 -= itin1.getYIncrement(); // t_YIncrementPixel(im1);
        p3 -= itout.getYIncrement(); // t_YIncrementPixel(im3);
      }

      return RES_OK;
    }

    RES_C Same13Compatibles(_iterI itin1, const _iterI &itEin1, _iterI itin2,
                            const _iterI &itEin2, _iterO itout,
                            const _iterO &itEOut, Oper &op) const
    {
      typedef typename __image::value_type value_type;

      const value_type *p2 = im2.rawPointer() + itin2.getOffset();
      value_type *p3       = im3.rawPointer() + itout.getOffset();

      for (offset_t line_z = itin2.getZSize(); line_z > 0; line_z--) {
        for (offset_t line_y = itin2.getYSize(); line_y > 0; line_y--) {
          const value_type *p4 = p2 + itin2.getXSize();
          for (; p2 != p4; p2++, p3++) {
            const value_type val1 = *p3;
            *p3                   = op(val1, *p2);
          }
          p2 += itin2.getYIncrement(); // t_YIncrementPixel(im2);
          p3 += itout.getYIncrement(); // t_YIncrementPixel(im3);
          p2--;
          p3--;
        }
        p2 += itin2.getZIncrement(); // t_ZIncrementPixel(im2);
        p3 += itout.getZIncrement(); // t_ZIncrementPixel(im3);

        p2 -= itin1.getYIncrement(); // t_YIncrementPixel(im1);
        p3 -= itout.getYIncrement(); // t_YIncrementPixel(im3);
      }

      return RES_OK;
    }

    RES_C Same23Compatibles(_iterI itin1, const _iterI &itEin1, _iterI itin2,
                            const _iterI &itEin2, _iterO itout,
                            const _iterO &itEOut, Oper &op) const
    {
      typedef typename __image::value_type value_type;

      const value_type *p1 = im1.rawPointer() + itin1.getOffset();
      value_type *p3       = im3.rawPointer() + itout.getOffset();

      for (offset_t line_z = itin1.getZSize(); line_z > 0; line_z--) {
        for (offset_t line_y = itin1.getYSize(); line_y > 0; line_y--) {
          const value_type *p4 = p1 + itin1.getXSize();
          for (; p1 != p4; p1++, p3++) {
            const value_type val1 = *p3;
            *p3                   = op(*p1, val1);
          }
          p1 += itin1.getYIncrement(); // t_YIncrementPixel(im1);
          p3 += itout.getYIncrement(); // t_YIncrementPixel(im3);
          p1--;
          p3--;
        }
        p1 += itin1.getZIncrement(); // t_ZIncrementPixel(im1);
        p3 += itout.getZIncrement(); // t_ZIncrementPixel(im3);

        p1 -= itin1.getYIncrement(); // t_YIncrementPixel(im1);
        p3 -= itout.getYIncrement(); // t_YIncrementPixel(im3);
      }

      return RES_OK;
    }

    RES_C allSameCompatibles(_iterI itin1, const _iterI &itEin1, _iterI itin2,
                             const _iterI &itEin2, _iterO itout,
                             const _iterO &itEOut, Oper &op) const
    {
      typedef typename __image::value_type value_type3;
      value_type3 *p3 = im3.rawPointer() + itout.getOffset();

      for (offset_t line_z = itout.getZSize(); line_z > 0; line_z--) {
        for (offset_t line_y = itout.getYSize(); line_y > 0; line_y--) {
          const value_type3 *p4 = p3 + itout.getXSize();
          for (; p3 != p4; p3++) {
            const value_type3 val3 = *p3;
            *p3                    = op(val3, val3);
          }
          p3 += itout.getYIncrement(); // t_YIncrementPixel(im3);
          p3--;
        }
        p3 += itout.getZIncrement(); // t_ZIncrementPixel(im3);
        p3 -= itout.getYIncrement(); // t_YIncrementPixel(im3);
      }

      return RES_OK;
    }

  public:
    RES_C operator()(_iterI itin1, const _iterI &itEin1, _iterI itin2,
                     const _iterI &itEin2, _iterO itout, const _iterO &itEOut,
                     Oper &op) const
    {
      if (itin1.sizePixel() != itin2.sizePixel() ||
          itin1.sizePixel() != itout.sizePixel() ||

          itin1.getXIncrement() != 1 || itin2.getXIncrement() != 1 ||
          itout.getXIncrement() != 1) {
        s_applyUtilityBinaryOperator<__image, __image, __image, Oper,
                                     e_accessGeneric>
            _aop(im1, im2, im3);
        return _aop(itin1, itEin1, itin2, itEin2, itout, itEOut, op);
      } else {
        // Raffi: TODO: mettre un peu d'ordre là dedans
        if (morphee::t_areImageTheSame(im1, im2) &&
            morphee::t_areImageTheSame(im1, im3)) {
          if (itin1.getOffset() == itin2.getOffset() &&
              itin1.getOffset() == itout.getOffset())
            return allSameCompatibles(itin1, itEin1, itin2, itEin2, itout,
                                      itEOut, op);
          else
            return sameNumberOfPixels(itin1, itEin1, itin2, itEin2, itout,
                                      itEOut, op);
        } else if (morphee::t_areImageTheSame(im1, im2)) {
          // Raffi: cas spécial 1, im1 = im2
          if (itin1.getOffset() == itin2.getOffset())
            return Same12Compatibles(itin1, itEin1, itin2, itEin2, itout,
                                     itEOut, op);
          else
            return sameNumberOfPixels(itin1, itEin1, itin2, itEin2, itout,
                                      itEOut, op);
        } else if (morphee::t_areImageTheSame(im1, im3)) {
          // Raffi: cas spécial 2, im1 = im3
          if (itin1.getOffset() == itout.getOffset())
            return Same13Compatibles(itin1, itEin1, itin2, itEin2, itout,
                                     itEOut, op);
          else
            return sameNumberOfPixels(itin1, itEin1, itin2, itEin2, itout,
                                      itEOut, op);
        } else if (morphee::t_areImageTheSame(im2, im3)) {
          // Raffi: cas spécial 3, im2 = im3
          if (itin2.getOffset() == itout.getOffset())
            return Same23Compatibles(itin1, itEin1, itin2, itEin2, itout,
                                     itEOut, op);
          else
            return sameNumberOfPixels(itin1, itEin1, itin2, itEin2, itout,
                                      itEOut, op);
        } else
          return sameNumberOfPixels(itin1, itEin1, itin2, itEin2, itout, itEOut,
                                    op);
      }
      MORPHEE_NEVER_REACH_HERE();
    }
  };

  template <class T, class Oper, int dim>
  struct s_applyUtilityBinaryOperator<Image<T, dim>, Image<T, dim>,
                                      Image<T, dim>, Oper, e_accessContinuous> {
    typedef Image<T, dim> __image;

    const __image &im1;
    const __image &im2;
    __image &im3;

    typedef typename __image::const_iterator _iterI;
    typedef typename __image::iterator _iterO;

    s_applyUtilityBinaryOperator(const __image &_im1, const __image &_im2,
                                 __image &_im3)
        : im1(_im1), im2(_im2), im3(_im3)
    {
      if (!t_isWindowMaximal(im1) || !t_isWindowMaximal(im2) ||
          !t_isWindowMaximal(im3)) {
        throw morphee::MException(
            "Trying to call explicitly s_applyUtilityBinaryOperator<Image<T, dim>, Image<T, dim>, Image<T, dim>, Oper, e_accessContinuous> while the properties\
 of the images are not appropriate");
      }
    }

  private:
    RES_C sameNumberOfPixels(_iterI itin1, const _iterI &itEin1, _iterI itin2,
                             const _iterI &itEin2, _iterO itout,
                             const _iterO &itEOut, Oper &op) const
    {
      // Raffi: cas normal, toutes les images sont différentes
      typedef typename __image::value_type value_type;

      // Raffi : unsigned long int à passer en offset_t (au cas où les images
      // seraient plus grande que 2^32)
      const unsigned long int sz = t_SizePixel(im1);
      const value_type *p1       = im1.rawPointer();
      const value_type *p2       = im2.rawPointer();
      value_type *p3             = im3.rawPointer();

      /*
      offset_t			l128 = static_cast<offset_t>(t_XYZSizePixel(im1) / 128);
      offset_t			r128 = t_XYZSizePixel(im1) % 128;
      for(;l128 > 0; l128--)
      {
        // Raffi: stuning optimization for stuning compilators
        for(offset_t l = 128;
          l > 0;
          p1++, p2++, p3++, l--)
        {
          *p3 = op(*p1, *p2);
        }
      }
      for(; r128 > 0; p1++, p2++, p3++, r128--)
      {
        *p3 = op(*p1, *p2);
      }
      */

      // Romain: quand je dis que c'est jamais aussi simple que ça en a l'air.
      // Ce code ci-dessus est 2,5 fois plus *lent* que le code ci-dessous:
      // Raffi: dsl, je comptais fusionner ce code (gardé en dessous), mais
      // oubli spontanné on peut certainement le répliquer dans les autres
      // fonctions (je laisse ca à titre d'exo). Par contre, Icc développe
      // correctement le code précédent, Visual je ne pense pas, apparemment GCC
      // non plus Je sens que je vais bientot me mettre à l'assembleur.

      for (unsigned long int i = 0; i < sz; ++i)
        p3[i] = op(p1[i], p2[i]);

      return RES_OK;
    }

    RES_C Same12Compatibles(_iterI itin1, const _iterI &itEin1, _iterI itin2,
                            const _iterI &itEin2, _iterO itout,
                            const _iterO &itEOut, Oper &op) const
    {
      typedef typename __image::value_type value_type;

      const value_type *p1 = im1.rawPointer();
      value_type *p3       = im3.rawPointer();

      offset_t l128 = static_cast<offset_t>(t_SizePixel(im1) / 128);
      offset_t r128 = t_SizePixel(im1) % 128;
      for (; l128 > 0; l128--) {
#if 0
				for(offset_t l = 128;
					l > 0;
					p1++, p3++, l--)
				{
					const value_type val1 = *p1;
					*p3 = op(val1, val1);
				}
#endif

        for (unsigned int l = 0; l < 128; l++) {
          const value_type val1 = p1[l];
          p3[l]                 = op(val1, val1);
        }
        p1 += 128;
        p3 += 128;
      }

      for (; r128 > 0; p1++, p3++, r128--) {
        const value_type val1 = *p1;
        *p3                   = op(val1, val1);
      }

      return RES_OK;
    }

    RES_C Same13Compatibles(_iterI itin1, const _iterI &itEin1, _iterI itin2,
                            const _iterI &itEin2, _iterO itout,
                            const _iterO &itEOut, Oper &op) const
    {
      typedef typename __image::value_type value_type;

      const value_type *p2 = im2.rawPointer();
      value_type *p3       = im3.rawPointer();

      offset_t l128 = static_cast<offset_t>(t_SizePixel(im2) / 128);
      offset_t r128 = t_SizePixel(im2) % 128;
      for (; l128 > 0; l128--) {
        // Raffi: stuning optimization for stuning compilators
#if 0
				for(offset_t l = 128;
					l > 0;
					p2++, p3++, l--)
				{
					const value_type val1 = *p3;
					*p3 = op(val1, *p2);
				}
#endif
        for (unsigned int l = 0; l < 128; l++) {
          // const value_type val1 = p3[l];
          p3[l] = op(p3[l], p2[l]);
        }
        p2 += 128;
        p3 += 128;
      }

      for (; r128 > 0; p2++, p3++, r128--) {
        const value_type val1 = *p3;
        *p3                   = op(val1, *p2);
      }

      return RES_OK;
    }

    RES_C Same23Compatibles(_iterI itin1, const _iterI &itEin1, _iterI itin2,
                            const _iterI &itEin2, _iterO itout,
                            const _iterO &itEOut, Oper &op) const
    {
      typedef typename __image::value_type value_type;

      const value_type *p1 = im1.rawPointer();
      value_type *p3       = im3.rawPointer();

      offset_t l128 = static_cast<offset_t>(t_SizePixel(im1) / 128);
      offset_t r128 = t_SizePixel(im1) % 128;

      for (; l128 > 0; l128--) {
        // Raffi: stuning optimization for stuning compilators
#if 0
				for(offset_t l = 128;
					l > 0;
					p1++, p3++, l--)
				{
					const value_type val1 = *p3;
					*p3 = op(*p1, val1);
				}
#endif
        for (unsigned int l = 0; l < 128; l++) {
          // const value_type val1 = p3[l];
          p3[l] = op(p1[l], p3[l]);
        }
        p1 += 128;
        p3 += 128;
      }

      for (; r128 > 0; p1++, p3++, r128--) {
        const value_type val1 = *p3;
        *p3                   = op(*p1, val1);
      }

      return RES_OK;
    }

    RES_C allSameCompatibles(_iterI itin1, const _iterI &itEin1, _iterI itin2,
                             const _iterI &itEin2, _iterO itout,
                             const _iterO &itEOut, Oper &op) const
    {
      typedef typename __image::value_type value_type3;
      value_type3 *p3 = im3.rawPointer();

      offset_t l128 = static_cast<offset_t>(t_SizePixel(im3) / 128);
      offset_t r128 = t_SizePixel(im3) % 128;

      for (; l128 > 0; l128--) {
        // Raffi: stuning optimization for stuning compilators
#if 0
				for(offset_t l = 128;
					l > 0;
					p3++, l--)
				{
					const value_type3 val1 = *p3;
					*p3 = op(val1, val1);
				}
#endif
        for (unsigned int l = 0; l < 128; l++) {
          const value_type3 val1 = p3[l];
          p3[l]                  = op(val1, val1);
        }
        p3 += 128;
      }

      for (; r128 > 0; p3++, r128--) {
        const value_type3 val1 = *p3;
        *p3                    = op(val1, val1);
      }

      return RES_OK;
    }

  public:
    RES_C operator()(_iterI itin1, const _iterI &itEin1, _iterI itin2,
                     const _iterI &itEin2, _iterO itout, const _iterO &itEOut,
                     Oper &op) const
    {
      if (itin1.sizePixel() != itin2.sizePixel() ||
          itin1.sizePixel() != itout.sizePixel() ||

          itin1.getXIncrement() != 1 || itin2.getXIncrement() != 1 ||
          itout.getXIncrement() != 1) {
        s_applyUtilityBinaryOperator<__image, __image, __image, Oper,
                                     e_accessGeneric>
            _aop(im1, im2, im3);
        return _aop(itin1, itEin1, itin2, itEin2, itout, itEOut, op);
      } else {
        // Raffi: TODO: mettre un peu d'ordre là dedans
        if (morphee::t_areImageTheSame(im1, im2) &&
            morphee::t_areImageTheSame(im1, im3)) {
          if (itin1.getOffset() == itin2.getOffset() &&
              itin1.getOffset() == itout.getOffset())
            return allSameCompatibles(itin1, itEin1, itin2, itEin2, itout,
                                      itEOut, op);
          else
            return sameNumberOfPixels(itin1, itEin1, itin2, itEin2, itout,
                                      itEOut, op);
        } else if (morphee::t_areImageTheSame(im1, im2)) {
          // Raffi: cas spécial 1, im1 = im2
          if (itin1.getOffset() == itin2.getOffset())
            return Same12Compatibles(itin1, itEin1, itin2, itEin2, itout,
                                     itEOut, op);
          else
            return sameNumberOfPixels(itin1, itEin1, itin2, itEin2, itout,
                                      itEOut, op);
        } else if (morphee::t_areImageTheSame(im1, im3)) {
          // Raffi: cas spécial 2, im1 = im3
          if (itin1.getOffset() == itout.getOffset())
            return Same13Compatibles(itin1, itEin1, itin2, itEin2, itout,
                                     itEOut, op);
          else
            return sameNumberOfPixels(itin1, itEin1, itin2, itEin2, itout,
                                      itEOut, op);
        } else if (morphee::t_areImageTheSame(im2, im3)) {
          // Raffi: cas spécial 3, im2 = im3
          if (itin2.getOffset() == itout.getOffset())
            return Same23Compatibles(itin1, itEin1, itin2, itEin2, itout,
                                     itEOut, op);
          else
            return sameNumberOfPixels(itin1, itEin1, itin2, itEin2, itout,
                                      itEOut, op);
        } else
          return sameNumberOfPixels(itin1, itEin1, itin2, itEin2, itout, itEOut,
                                    op);
      }
      MORPHEE_NEVER_REACH_HERE();
    }
  }; // struct s_applyUtilityBinaryOperator<access_continuous>

  // @} defgroup bitwise_template_utilities

  //! Calls the operator 'op' for each pixel of imin,
  template <class Image1, class Oper>
  inline RES_C t_ImZeroaryOperation(Image1 &imin, Oper &op)
  {
    if (!imin.isAllocated())
      return RES_NOT_ALLOCATED;

    typename Image1::iterator it1 = imin.begin(), iend1 = imin.end();
    for (; it1 != iend1; ++it1)
      op(*it1);

    return RES_OK;
  }

  //! Calls the operator 'op' for each pixel of imin, output is sent to imout
  template <class t_image1, class t_image2, class Oper>
  struct s_ImUnaryOperation {
    RES_C operator()(const t_image1 &imin, Oper op, t_image2 &imout) const
    {
      MORPHEE_ENTER_FUNCTION("s_ImUnaryOperation<im1,im2,Oper>()");

      if (!imin.isAllocated() || !imout.isAllocated()) {
        MORPHEE_REGISTER_ERROR("Images not allocated");
        return RES_NOT_ALLOCATED;
      }
      if (!t_CheckWindowSizesPixel(imin, imout)) {
        MORPHEE_REGISTER_ERROR("Bad window sizes (number of elements)");
        return RES_ERROR_BAD_WINDOW_SIZE;
      }
      if (t_CheckOffsetCompatible(imin, imout)) {
        if (morphee::t_isWindowMaximal(imin)) {
          assert(morphee::t_isWindowMaximal(imout));
          s_applyUtilityUnaryOperator<t_image1, t_image2, Oper,
                                      e_accessContinuous>
              _aop(imin, imout);
          return _aop(imin.begin(), imin.end(), imout.begin(), imout.end(), op);
        } else {
          s_applyUtilityUnaryOperator<t_image1, t_image2, Oper,
                                      e_accessOffsetCompatible>
              _aop(imin, imout);
          return _aop(imin.begin(), imin.end(), imout.begin(), imout.end(), op);
        }
      } else {
        s_applyUtilityUnaryOperator<t_image1, t_image2,
                                    Oper /*, e_accessGeneric*/>
            _aop(imin, imout);
        return _aop(imin.begin(), imin.end(), imout.begin(), imout.end(), op);
      }
    }
  };

  template <int dim, class Oper>
  struct s_ImUnaryOperation<Image<CVariant, dim>, Image<CVariant, dim>, Oper> {
    RES_C operator()(const Image<CVariant, dim> &imin, Oper op,
                     Image<CVariant, dim> &imout) const
    {
      MORPHEE_ENTER_FUNCTION(
          "s_ImUnaryOperation<imCVariant,imCVariant,Oper>()");

      if (!imin.isAllocated() || !imout.isAllocated()) {
        MORPHEE_REGISTER_ERROR("Images not allocated");
        return RES_NOT_ALLOCATED;
      }
      if (!t_CheckWindowSizesPixel(imin, imout)) {
        MORPHEE_REGISTER_ERROR("Bad window sizes (number of elements)");
        return RES_ERROR_BAD_WINDOW_SIZE;
      }
      // Romain: on peut peut-être garder la version "offset-compatible", mais
      // dans le doute...
      s_applyUtilityUnaryOperator<Image<CVariant, dim>, Image<CVariant, dim>,
                                  Oper /*, e_accessGeneric*/>
          _aop(imin, imout);
      return _aop(imin.begin(), imin.end(), imout.begin(), imout.end(), op);
    }
  };

  //! Calls the operator 'op' for each pixel of imin, output is sent to imout
  template <class t_image1, class t_image2, class Oper>
  inline RES_C t_ImUnaryOperation(const t_image1 &imin, Oper oper,
                                  t_image2 &imout)
  {
    MORPHEE_ENTER_FUNCTION("t_ImUnaryOperation(template)");

    s_ImUnaryOperation<t_image1, t_image2, Oper> im_op;
    return im_op(imin, oper, imout);
  }

  //! Calls the operator 'op' for each pixel of imin, output is sent to imout
  template <class Image1, class Image2, class Oper, class Oper2>
  struct s_ImUnaryBiOperation {
    RES_C operator()(const Image1 &imin, Oper op, Oper2 op2, Image2 &imout)
    {
      MORPHEE_ENTER_FUNCTION("s_ImUnaryBiOperation<im1,im2,Oper,Oper2>()");

      if (!imin.isAllocated() || !imout.isAllocated()) {
        MORPHEE_REGISTER_ERROR("Images not allocated");
        return RES_NOT_ALLOCATED;
      }
      if (t_CheckOffsetCompatible(imin, imout)) {
        typename Image1::const_iterator it1 = imin.begin(), iend1 = imin.end();

        for (; it1 != iend1; ++it1) {
          const offset_t off              = it1.getOffset();
          op2(imout.pixelFromOffset(off)) = op(*it1);
        }
        return RES_OK;
      }

      if (!t_CheckWindowSizes(imin, imout)) {
        MORPHEE_REGISTER_ERROR("Bad window sizes");
        return RES_ERROR_BAD_WINDOW_SIZE;
      }

      typename Image1::const_iterator it1 = imin.begin(), iend1 = imin.end();
      typename Image2::iterator it2 = imout.begin();

      for (; it1 != iend1; ++it1, ++it2)
        op2(*it2) = op(*it1);

      return RES_OK;
    }
  };

  //! Generic version by default
  template <class __image1, class __image2, class __image3, class Oper>
  struct s_ImBinaryOperation {
    RES_C operator()(const __image1 &imin1, const __image2 &imin2, Oper op,
                     __image3 &imout)
    {
      s_applyUtilityBinaryOperator<__image1, __image2, __image3, Oper,
                                   e_accessGeneric>
          _aop(imin1, imin2, imout);
      return _aop(imin1.begin(), imin1.end(), imin2.begin(), imin2.end(),
                  imout.begin(), imout.end(), op);
    }
  };

  //! Generic version also for CVariants
  template <int dim, class Oper>
  struct s_ImBinaryOperation<Image<CVariant, dim>, Image<CVariant, dim>,
                             Image<CVariant, dim>, Oper> {
    RES_C operator()(const Image<CVariant, dim> &imin1,
                     const Image<CVariant, dim> &imin2, Oper op,
                     Image<CVariant, dim> &imout)
    {
      s_applyUtilityBinaryOperator<Image<CVariant, dim>, Image<CVariant, dim>,
                                   Image<CVariant, dim>, Oper, e_accessGeneric>
          _aop(imin1, imin2, imout);
      return _aop(imin1.begin(), imin1.end(), imin2.begin(), imin2.end(),
                  imout.begin(), imout.end(), op);
    }
  };

  // template<class __image1, class __image2, class __image3, class Oper>
  template <class T1, class T2, class T3, class Oper>
  struct s_ImBinaryOperation<Image<T1>, Image<T2>, Image<T3>, Oper> {
    typedef Image<T1> __image1;
    typedef Image<T2> __image2;
    typedef Image<T3> __image3;
    RES_C operator()(const __image1 &imin1, const __image2 &imin2, Oper op,
                     __image3 &imout)
    {
      MORPHEE_ENTER_FUNCTION("s_ImBinaryOperation<im1,im2,im3,Oper>()");

      if (!imin1.isAllocated() || !imin2.isAllocated() ||
          !imout.isAllocated()) {
        MORPHEE_REGISTER_ERROR("Images not allocated");
        return RES_NOT_ALLOCATED;
      }
      if (!t_CheckWindowSizes(imin1, imout) ||
          !t_CheckWindowSizes(imin2, imout)) {
        MORPHEE_REGISTER_ERROR("Bad window sizes");
        return RES_ERROR_BAD_WINDOW_SIZE;
      }
      bool b1 = t_CheckOffsetCompatible(imin1, imin2);
      bool b2 = t_CheckOffsetCompatible(imin1, imout);
#ifndef NDEBUG
      bool b3 = t_CheckOffsetCompatible(imin2, imout);
#endif
      if (b1 && b2) {
        assert(b3);

        if (morphee::t_isWindowMaximal(imin1)) {
          assert(morphee::t_isWindowMaximal(imin2));
          assert(morphee::t_isWindowMaximal(imout));
          s_applyUtilityBinaryOperator<__image1, __image2, __image3, Oper,
                                       e_accessContinuous>
              _aop(imin1, imin2, imout);
          return _aop(imin1.begin(), imin1.end(), imin2.begin(), imin2.end(),
                      imout.begin(), imout.end(), op);
        } else {
          s_applyUtilityBinaryOperator<__image1, __image2, __image3, Oper,
                                       e_accessOffsetCompatible>
              _aop(imin1, imin2, imout);
          return _aop(imin1.begin(), imin1.end(), imin2.begin(), imin2.end(),
                      imout.begin(), imout.end(), op);
        }
      } else {
        // Generic access
        s_applyUtilityBinaryOperator<__image1, __image2, __image3, Oper> _aop(
            imin1, imin2, imout);

        return _aop(imin1.begin(), imin1.end(), imin2.begin(), imin2.end(),
                    imout.begin(), imout.end(), op);
      }
      MORPHEE_NEVER_REACH_HERE();
    }
  };

  // Romain: j'aime bien cette syntaxe
  template <class __image1, class __image2, class __image3, class Oper>
  RES_C t_ImBinaryOperation(const __image1 &imin1, const __image2 &imin2,
                            Oper op, __image3 &imout)
  {
    s_ImBinaryOperation<__image1, __image2, __image3, Oper> func;
    return func(imin1, imin2, op, imout);
  }

  // @} defgroup bitwise_template

} // namespace morphee

#endif // __MORPHEE_IMAGE_PIXELWISE_T_HPP__
