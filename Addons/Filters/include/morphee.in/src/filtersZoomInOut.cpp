

#include <morphee/filters/include/morpheeFilters.hpp>

#include <morphee/image/include/private/imageDispatch_T.hpp>
#include <morphee/filters/include/private/filtersZoomInOut_T.hpp>

namespace morphee
{
  namespace filters
  {
    template <typename T>
    inline RES_C t_ImSimpleDecimator(const ImageInterface *imin,
                                     const std::vector<F_SIMPLE> &dec_factors,
                                     ImageInterface *&imout)
    {
      MORPHEE_ENTER_FUNCTION("t_ImSimpleDecimator(interface)");

      const Image<T> *_iminLocal = dynamic_cast<const Image<T> *>(imin);
      if (_iminLocal == 0) {
        MORPHEE_REGISTER_ERROR("Error in dynamic cast")
        return morphee::RES_ERROR_DYNAMIC_CAST;
      }

      s_returnFirstPoint<typename Image<T>::const_iterator> op;
      Image<T> *_imoutLocal = 0;
      if (imout != 0) {
        _imoutLocal = dynamic_cast<Image<T> *>(imout);
        if (_imoutLocal == 0) {
          MORPHEE_REGISTER_ERROR("Error in dynamic_cast");
          return RES_ERROR_DYNAMIC_CAST;
        }
      }

      RES_C res = t_ImDecimation /*<T, s_returnFirstPoint<typename
                                    Image<T>::const_iterator> >*/
          (*_iminLocal, dec_factors, op, _imoutLocal);
      imout = dynamic_cast<ImageInterface *>(_imoutLocal);
      return res;
    }

    ImageInterface *ImSimpleDecimator(const ImageInterface *imIn,
                                      const std::vector<F_SIMPLE> &dec_factors)
    {
      typedef generalDispatch<1, const ImageInterface *,
                              const std::vector<F_SIMPLE> &, ImageInterface *&>
          _localDispatch;

      static const _localDispatch::dispatchData localTab[] = {
          {{dtScalar},
           {{{sdtUINT8}, t_ImSimpleDecimator<UINT8>},
            {{sdtUINT16}, t_ImSimpleDecimator<UINT16>},
            {{sdtUINT32}, t_ImSimpleDecimator<UINT32>},
            {{sdtFloat}, t_ImSimpleDecimator<F_SIMPLE>},
            {{sdtDouble}, t_ImSimpleDecimator<F_DOUBLE>},
            {{sdtNone}, 0}}},
          {{dtPixel3},
           {{{sdtUINT8}, t_ImSimpleDecimator<pixel_3<UINT8>>},
            {{sdtUINT16}, t_ImSimpleDecimator<pixel_3<UINT16>>},
            {{sdtUINT32}, t_ImSimpleDecimator<pixel_3<UINT32>>},
            {{sdtNone}, 0}}},
          {{dtNone}, {{{sdtNone}, 0}}}};

      ImageInterface *imTemp = 0;
      RES_C res = _localDispatch::dispatch(imIn, dec_factors, imTemp, localTab);
      if (res != RES_OK)
        throw MException(res);
      return imTemp;
    }

    template <typename T>
    inline RES_C t_ImHLSDecimator(const ImageInterface *imin,
                                  const std::vector<F_SIMPLE> &dec_factor,
                                  ImageInterface *&imout)
    {
      MORPHEE_ENTER_FUNCTION("t_ImHLSDecimator(interface)");

      const Image<T> *_iminLocal = dynamic_cast<const Image<T> *>(imin);
      if (_iminLocal == 0) {
        MORPHEE_REGISTER_ERROR("Error in dynamic_cast");
        return RES_ERROR_DYNAMIC_CAST;
      }

      if (_iminLocal->ColorInfo().cs != csHLS) {
        MORPHEE_REGISTER_ERROR("Wrong input image color space (should be HLS)");
        return RES_BAD_COLOR_SPACE;
      }

      s_hlsDecimation<typename Image<T>::const_iterator> op;
      Image<T> *_imoutLocal = 0;
      if (imout != 0) {
        _imoutLocal = dynamic_cast<Image<T> *>(imout);
        if (_imoutLocal == 0) {
          MORPHEE_REGISTER_ERROR(
              "Error: the provided output image is in a bad format");
          return RES_ERROR_DYNAMIC_CAST;
        }
      }

      RES_C res = t_ImDecimation /*<T, s_hlsDecimation<typename
                                    Image<T>::const_iterator> >*/
          (*_iminLocal, dec_factor, op, _imoutLocal);

      imout = dynamic_cast<ImageInterface *>(_imoutLocal);

      return res;
    }

    ImageInterface *ImHLSDecimator(const ImageInterface *imIn,
                                   const std::vector<F_SIMPLE> &dec_factor)
    {
      typedef generalDispatch<1, const ImageInterface *,
                              const std::vector<F_SIMPLE> &, ImageInterface *&>
          _localDispatch;

      static const _localDispatch::dispatchData localTab[] = {
          {{dtPixel3},
           {{{sdtFloat}, t_ImHLSDecimator<pixel_3<F_SIMPLE>>},
            {{sdtDouble}, t_ImHLSDecimator<pixel_3<F_DOUBLE>>},
            {{sdtNone}, 0}}},
          {{dtNone}, {{{sdtNone}, 0}}}};

      ImageInterface *imTemp = 0;
      RES_C res = _localDispatch::dispatch(imIn, dec_factor, imTemp, localTab);
      if (res != RES_OK)
        throw MException(res);
      return imTemp;
    }

    template <typename T>
    inline RES_C t_ImZoom(const ImageInterface *imin,
                          const std::vector<F_SIMPLE> &zoom_factor,
                          ImageInterface *&imout)
    {
      MORPHEE_ENTER_FUNCTION("t_ImZoom");

      const Image<T> *_iminLocal = dynamic_cast<const Image<T> *>(imin);
      Image<T> *_imoutLocal      = 0;
      if (imout != 0) {
        _imoutLocal = dynamic_cast<Image<T> *>(imout);
        if (_imoutLocal == 0) {
          MORPHEE_REGISTER_ERROR("Error in dynamic_cast");
          return RES_ERROR_DYNAMIC_CAST;
        }
      }

      RES_C res = t_ImZoom(*_iminLocal, zoom_factor, _imoutLocal);
      imout     = dynamic_cast<ImageInterface *>(_imoutLocal);
      return res;
    }

    ImageInterface *ImZoom(const ImageInterface *imIn,
                           const std::vector<F_SIMPLE> &zoom_factor)
    {
      MORPHEE_ENTER_FUNCTION("ImZoom");
      if (!imIn)
        throw MException(RES_ERROR_NULL_PARAM);

      typedef generalDispatch<1, const ImageInterface *,
                              const std::vector<F_SIMPLE> &, ImageInterface *&>
          _localDispatch;
      static const _localDispatch::dispatchData localTab[] = {
          {{dtScalar},
           {{{sdtUINT8}, t_ImZoom<UINT8>},
            {{sdtUINT16}, t_ImZoom<UINT16>},
            {{sdtUINT32}, t_ImZoom<UINT32>},
            {{sdtINT32}, t_ImZoom<INT32>},
            {{sdtFloat}, t_ImZoom<F_SIMPLE>},
            {{sdtDouble}, t_ImZoom<F_DOUBLE>},
            {{sdtNone}, 0}}},
          {{dtPixel3},
           {{{sdtUINT8}, t_ImZoom<pixel_3<UINT8>>},
            {{sdtUINT16}, t_ImZoom<pixel_3<UINT16>>},
            {{sdtUINT32}, t_ImZoom<pixel_3<UINT32>>},
            {{sdtNone}, 0}}},
          {{dtNone}, {{{sdtNone}, 0}}}};

      ImageInterface *imTemp = 0;
      RES_C res = _localDispatch::dispatch(imIn, zoom_factor, imTemp, localTab);
      if (res != RES_OK)
        throw MException(res);
      return imTemp;
    }

  } // namespace filters
} // namespace morphee
