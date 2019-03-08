
#include <morphee/filters/include/morpheeFilters.hpp>
#include <morphee/filters/include/private/filtersHarris_T.hpp>

#include <morphee/image/include/private/imageDispatch_T.hpp>

namespace morphee
{
  namespace filters
  {
    template <class T1, class T2>
    RES_C t_ImHarrisOperator(const ImageInterface *imIn, UINT32 gaussSize,
                             ImageInterface *imOut)
    {
      const Image<T1> *_imTempIn = dynamic_cast<const Image<T1> *>(imIn);
      Image<T2> *_imTempOut      = dynamic_cast<Image<T2> *>(imOut);
      if (!_imTempIn || !_imTempOut) {
        MORPHEE_REGISTER_ERROR("Error in downcast");
        return morphee::RES_ERROR_DYNAMIC_CAST;
      }
      return t_ImHarrisOperator(*_imTempIn, gaussSize, *_imTempOut);
    }

    RES_C ImHarrisOperator(const ImageInterface *imin, UINT32 gaussSize,
                           ImageInterface *imOut)
    {
      typedef generalDispatch<2, const ImageInterface *, UINT32,
                              ImageInterface *>
          _localDispatch;
      static const _localDispatch::dispatchData tabLocal[] = {
          {{dtScalar, dtScalar},
           {{{sdtUINT8, sdtDouble}, t_ImHarrisOperator<UINT8, F_DOUBLE>},
            {{sdtFloat, sdtFloat}, t_ImHarrisOperator<F_SIMPLE, F_SIMPLE>},
            {{sdtDouble, sdtDouble}, t_ImHarrisOperator<F_DOUBLE, F_DOUBLE>},
            {{sdtNone, sdtNone}, 0}}},
          {{dtPixel3, dtScalar},
           {{{sdtUINT8, sdtDouble},
             t_ImHarrisOperator<pixel_3<UINT8>, F_DOUBLE>},
            {{sdtFloat, sdtFloat},
             t_ImHarrisOperator<pixel_3<F_SIMPLE>, F_SIMPLE>},
            {{sdtDouble, sdtDouble},
             t_ImHarrisOperator<pixel_3<F_DOUBLE>, F_DOUBLE>},
            {{sdtNone, sdtNone}, 0}}},
          {{dtNone, dtNone}, {{{sdtNone, sdtNone}, 0}}}};

      return _localDispatch::dispatch(imin, gaussSize, imOut, tabLocal);
    }
  } // namespace filters
} // namespace morphee
