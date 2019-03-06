

#include <morphee/filters/include/morpheeFilters.hpp>
#include <morphee/filters/include/private/filtersDifferential_T.hpp>

#include <morphee/image/include/private/imageDispatch_T.hpp>
#include <morphee/image/include/private/image_T.hpp>

namespace morphee
{
  namespace filters
  {
    template <typename T1, typename T2>
    RES_C t_ImLaplacianFilter(const ImageInterface *imIn, ImageInterface *imOut)
    {
      const Image<T1> *_imIn = dynamic_cast<const Image<T1> *>(imIn);
      Image<T2> *_imOut      = dynamic_cast<Image<T2> *>(imOut);

      if (!imIn || !imOut) {
        MORPHEE_REGISTER_ERROR("Error in downcast");
        return RES_ERROR_DYNAMIC_CAST;
      }
      return t_ImLaplacianFilter(*_imIn, *_imOut);
    }

    RES_C ImLaplacianFilter(const ImageInterface *imIn, ImageInterface *imOut)
    {
      typedef generalDispatch<2, const ImageInterface *, ImageInterface *>
          _localDispatch;

      static const _localDispatch::dispatchData localTab[] = {
          {{dtScalar, dtScalar},
           {{{sdtUINT8, sdtUINT8}, t_ImLaplacianFilter<UINT8, UINT8>},
            {{sdtINT8, sdtINT8}, t_ImLaplacianFilter<INT8, INT8>},
            {{sdtUINT16, sdtUINT16}, t_ImLaplacianFilter<UINT16, UINT16>},
            {{sdtINT16, sdtINT16}, t_ImLaplacianFilter<INT16, INT16>},
            {{sdtUINT32, sdtUINT32}, t_ImLaplacianFilter<UINT32, UINT32>},
            {{sdtINT32, sdtINT32}, t_ImLaplacianFilter<INT32, INT32>},
            {{sdtFloat, sdtFloat}, t_ImLaplacianFilter<F_SIMPLE, F_SIMPLE>},
            {{sdtDouble, sdtDouble}, t_ImLaplacianFilter<F_DOUBLE, F_DOUBLE>},

            {{sdtNone, sdtNone}, 0}}},
          {{dtNone, dtNone}, {{{sdtNone, sdtNone}, 0}}}};
      return _localDispatch::dispatch(imIn, imOut, localTab);
    }

    template <class T1, class T2>
    RES_C t_ImDifferentialGradientX(const ImageInterface *imIn,
                                    ImageInterface *imOut)
    {
      const Image<T1> *_imTempIn = dynamic_cast<const Image<T1> *>(imIn);
      Image<T2> *_imTempOut      = dynamic_cast<Image<T2> *>(imOut);
      if (!_imTempIn || !_imTempOut) {
        MORPHEE_REGISTER_ERROR("Error in downcast");
        return morphee::RES_ERROR_DYNAMIC_CAST;
      }
      return t_ImDifferentialGradientX(*_imTempIn, *_imTempOut);
    }

    RES_C ImDifferentialGradientX(const ImageInterface *imin,
                                  ImageInterface *imOut)
    {
      typedef generalDispatch<2, const ImageInterface *, ImageInterface *>
          _localDispatch;
      static const _localDispatch::dispatchData tabLocal[] = {
          {{dtScalar, dtScalar},
           {{{sdtUINT8, sdtUINT8}, t_ImDifferentialGradientX<UINT8, UINT8>},
            {{sdtUINT8, sdtFloat}, t_ImDifferentialGradientX<UINT8, F_SIMPLE>},
            {{sdtFloat, sdtFloat},
             t_ImDifferentialGradientX<F_SIMPLE, F_SIMPLE>},
            {{sdtNone, sdtNone}, 0}}},
          {{dtNone, dtNone}, {{{sdtNone, sdtNone}, 0}}}};

      return _localDispatch::dispatch(imin, imOut, tabLocal);
    }

    template <class T1, class T2>
    RES_C t_ImDifferentialGradientY(const ImageInterface *imIn,
                                    ImageInterface *imOut)
    {
      const Image<T1> *_imTempIn = dynamic_cast<const Image<T1> *>(imIn);
      Image<T2> *_imTempOut      = dynamic_cast<Image<T2> *>(imOut);
      if (!_imTempIn || !_imTempOut) {
        MORPHEE_REGISTER_ERROR("Error in downcast");
        return morphee::RES_ERROR_DYNAMIC_CAST;
      }
      return t_ImDifferentialGradientY(*_imTempIn, *_imTempOut);
    }

    RES_C ImDifferentialGradientY(const ImageInterface *imin,
                                  ImageInterface *imOut)
    {
      typedef generalDispatch<2, const ImageInterface *, ImageInterface *>
          _localDispatch;
      static const _localDispatch::dispatchData tabLocal[] = {
          {{dtScalar, dtScalar},
           {{{sdtUINT8, sdtUINT8}, t_ImDifferentialGradientY<UINT8, UINT8>},
            {{sdtUINT8, sdtFloat}, t_ImDifferentialGradientY<UINT8, F_SIMPLE>},
            {{sdtFloat, sdtFloat},
             t_ImDifferentialGradientY<F_SIMPLE, F_SIMPLE>},
            {{sdtNone, sdtNone}, 0}}},
          {{dtNone, dtNone}, {{{sdtNone, sdtNone}, 0}}}};

      return _localDispatch::dispatch(imin, imOut, tabLocal);
    }

  } // namespace filters
} // namespace morphee
