#include <morphee/filters/include/morpheeFilters.hpp>
#include <morphee/filters/include/private/filterMeanShift_T.hpp>

#include <morphee/image/include/private/imageDispatch_T.hpp>
#include <morphee/image/include/private/image_T.hpp>

namespace morphee
{
  namespace filters
  {
    // Author : Vincent MORARD
    // 20 october 2011

    //****************************************************************
    //*												Template (interface)
    //****************************************************************

    template <typename T1, typename T2>
    RES_C t_ImMeanShift(const ImageInterface *imIn, const UINT8 radius,
                        const int tonalDistance, ImageInterface *imOut)
    {
      const Image<T1> *_localImIn = dynamic_cast<const Image<T1> *>(imIn);
      Image<T2> *_localImOut      = dynamic_cast<Image<T2> *>(imOut);
      if (!_localImOut || !_localImIn) {
        MORPHEE_REGISTER_ERROR("Error in downcast");
        return RES_ERROR_DYNAMIC_CAST;
      }
      return t_ImMeanShift(*_localImIn, radius, tonalDistance, *_localImOut);
    }

    template <typename T1, typename T2>
    RES_C t_ImMeanShiftRGB(const ImageInterface *imIn, const UINT8 radius,
                           const int tonalDistance, ImageInterface *imOut)
    {
      const Image<T1, 3> *_localImIn = dynamic_cast<const Image<T1, 3> *>(imIn);
      Image<T2, 3> *_localImOut      = dynamic_cast<Image<T2, 3> *>(imOut);
      if (!_localImOut || !_localImIn) {
        MORPHEE_REGISTER_ERROR("Error in downcast");
        return RES_ERROR_DYNAMIC_CAST;
      }
      return t_ImMeanShiftRGB(*_localImIn, radius, tonalDistance, *_localImOut);
    }

    //*************************************************************************
    //*												Dispatch (interface)
    //*************************************************************************

    RES_C ImMeanShift(const ImageInterface *imIn, const UINT8 radius,
                      const int tonalDistance, ImageInterface *imOut)
    {
      // proper type for the dispatch array
      typedef generalDispatch<2, const ImageInterface *, const UINT8, const int,
                              ImageInterface *>
          _localDispatch;

      // the dispatch array itself
      static const _localDispatch::dispatchData _tab_local[] = {
          {{dtScalar, dtScalar},
           {{{sdtUINT8, sdtUINT8},
             morphee::filters::t_ImMeanShift<UINT8, UINT8>},
            {{sdtINT8, sdtINT8}, morphee::filters::t_ImMeanShift<INT8, INT8>},
            {{sdtUINT16, sdtUINT16},
             morphee::filters::t_ImMeanShift<UINT16, UINT16>},
            {{sdtINT16, sdtINT16},
             morphee::filters::t_ImMeanShift<INT16, INT16>},
            {{sdtUINT32, sdtUINT32},
             morphee::filters::t_ImMeanShift<UINT32, UINT32>},
            {{sdtINT32, sdtINT32},
             morphee::filters::t_ImMeanShift<INT32, INT32>},
            {{sdtFloat, sdtFloat},
             morphee::filters::t_ImMeanShift<F_SIMPLE, F_SIMPLE>},
            {{sdtDouble, sdtDouble},
             morphee::filters::t_ImMeanShift<F_DOUBLE, F_DOUBLE>},
            {{sdtNone, sdtNone}, 0}}},
          {{dtPixel3, dtPixel3},
           {{{sdtUINT8, sdtUINT8},
             morphee::filters::t_ImMeanShiftRGB<pixel_3<UINT8>,
                                                pixel_3<UINT8>>},
            {{sdtINT8, sdtINT8},
             morphee::filters::t_ImMeanShiftRGB<pixel_3<INT8>, pixel_3<INT8>>},
            {{sdtUINT16, sdtUINT16},
             morphee::filters::t_ImMeanShiftRGB<pixel_3<UINT16>,
                                                pixel_3<UINT16>>},
            {{sdtINT16, sdtINT16},
             morphee::filters::t_ImMeanShiftRGB<pixel_3<INT16>,
                                                pixel_3<INT16>>},
            {{sdtUINT32, sdtUINT32},
             morphee::filters::t_ImMeanShiftRGB<pixel_3<UINT32>,
                                                pixel_3<UINT32>>},
            {{sdtINT32, sdtINT32},
             morphee::filters::t_ImMeanShiftRGB<pixel_3<INT32>,
                                                pixel_3<INT32>>},
            {{sdtFloat, sdtFloat},
             morphee::filters::t_ImMeanShiftRGB<pixel_3<F_SIMPLE>,
                                                pixel_3<F_SIMPLE>>},
            {{sdtDouble, sdtDouble},
             morphee::filters::t_ImMeanShiftRGB<pixel_3<F_DOUBLE>,
                                                pixel_3<F_DOUBLE>>},
            {{sdtNone, sdtNone}, 0}}},
          {{dtNone, dtNone}, {{{sdtNone, sdtNone}, 0}}}};

      // actually call the mapped function and return its result
      return _localDispatch::dispatch(imIn, radius, tonalDistance, imOut,
                                      _tab_local);
    }

  } // namespace filters

} // namespace morphee