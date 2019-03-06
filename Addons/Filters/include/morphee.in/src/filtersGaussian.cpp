

#include <morphee/filters/include/morpheeFilters.hpp>
#include <morphee/filters/include/private/filtersGaussian_T.hpp>

#include <morphee/image/include/private/imageDispatch_T.hpp>
#include <morphee/image/include/private/image_T.hpp>

namespace morphee
{
  namespace filters
  {
    template <typename T1, typename T2>
    RES_C t_ImGaussianFilter(const ImageInterface *imIn, INT32 filterRadius,
                             ImageInterface *imOut)
    {
      const Image<T1> *_imIn = dynamic_cast<const Image<T1> *>(imIn);
      Image<T2> *_imOut      = dynamic_cast<Image<T2> *>(imOut);

      if (!imIn || !imOut)
        return RES_ERROR_DYNAMIC_CAST;

      return t_ImGaussianFilter(*_imIn, filterRadius, *_imOut);
    }

    RES_C ImGaussianFilter(const ImageInterface *imIn, INT32 filterRadius,
                           ImageInterface *imOut)
    {
      typedef generalDispatch<2, const ImageInterface *, INT32,
                              ImageInterface *>
          _localDispatch;

      static const _localDispatch::dispatchData localTab[] = {
          {{dtScalar, dtScalar},
           {{{sdtUINT8, sdtUINT8}, t_ImGaussianFilter<UINT8, UINT8>},
            {{sdtINT8, sdtINT8}, t_ImGaussianFilter<INT8, INT8>},
            {{sdtUINT16, sdtUINT16}, t_ImGaussianFilter<UINT16, UINT16>},
            {{sdtINT16, sdtINT16}, t_ImGaussianFilter<INT16, INT16>},
            {{sdtUINT32, sdtUINT32}, t_ImGaussianFilter<UINT32, UINT32>},
            {{sdtINT32, sdtINT32}, t_ImGaussianFilter<INT32, INT32>},
            {{sdtFloat, sdtFloat}, t_ImGaussianFilter<F_SIMPLE, F_SIMPLE>},
            {{sdtDouble, sdtDouble}, t_ImGaussianFilter<F_DOUBLE, F_DOUBLE>},

            {{sdtNone, sdtNone}, 0}}},
          {{dtPixel3, dtPixel3},
           {{{sdtUINT8, sdtUINT8},
             t_ImGaussianFilter<pixel_3<UINT8>, pixel_3<UINT8>>},
            {{sdtINT8, sdtINT8},
             t_ImGaussianFilter<pixel_3<INT8>, pixel_3<INT8>>},
            {{sdtUINT16, sdtUINT16},
             t_ImGaussianFilter<pixel_3<UINT16>, pixel_3<UINT16>>},
            {{sdtINT16, sdtINT16},
             t_ImGaussianFilter<pixel_3<INT16>, pixel_3<INT16>>},
            {{sdtUINT32, sdtUINT32},
             t_ImGaussianFilter<pixel_3<UINT32>, pixel_3<UINT32>>},
            {{sdtINT32, sdtINT32},
             t_ImGaussianFilter<pixel_3<INT32>, pixel_3<INT32>>},
            {{sdtFloat, sdtFloat},
             t_ImGaussianFilter<pixel_3<F_SIMPLE>, pixel_3<F_SIMPLE>>},
            {{sdtDouble, sdtDouble},
             t_ImGaussianFilter<pixel_3<F_DOUBLE>, pixel_3<F_DOUBLE>>},

            {{sdtNone, sdtNone}, 0}}},
          {{dtNone, dtNone}, {{{sdtNone, sdtNone}, 0}}}};
      return _localDispatch::dispatch(imIn, filterRadius, imOut, localTab);
    }
  } // namespace filters
} // namespace morphee
