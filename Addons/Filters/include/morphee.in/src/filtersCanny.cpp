#include <morphee/filters/include/morpheeFilters.hpp>
#include <morphee/filters/include/private/filtersCanny_T.hpp>

#include <morphee/image/include/private/imageDispatch_T.hpp>
#include <morphee/image/include/private/image_T.hpp>

namespace morphee
{
  namespace filters
  {
    // Author : Vincent MORARD
    // Date : 26 july 2012

    //****************************************************************
    //*	Template (interface)
    //****************************************************************

    template <typename T1, typename T2>
    RES_C t_ImCannyEdgeDetection(const ImageInterface *imIn, const double Sigma,
                                 ImageInterface *imOut)
    {
      const Image<T1> *_localImIn = dynamic_cast<const Image<T1> *>(imIn);
      Image<T2> *_localImOut      = dynamic_cast<Image<T2> *>(imOut);
      if (!_localImOut || !_localImIn) {
        MORPHEE_REGISTER_ERROR("Error in downcast");
        return RES_ERROR_DYNAMIC_CAST;
      }
      return t_ImCannyEdgeDetection(*_localImIn, Sigma, *_localImOut);
    }

    //*************************************************************************
    //*												Dispatch (interface)
    //*************************************************************************
    RES_C ImCannyEdgeDetection(const ImageInterface *imIn, const double Sigma,
                               ImageInterface *imOut)
    {
      // proper type for the dispatch array
      typedef generalDispatch<2, const ImageInterface *, const double,
                              ImageInterface *>
          _localDispatch;
      // the dispatch array itself
      static const _localDispatch::dispatchData tab_arith_local[] = {
          {{dtScalar, dtScalar},
           {{{sdtUINT8, sdtUINT8},
             morphee::filters::t_ImCannyEdgeDetection<UINT8, UINT8>},
            {{sdtUINT8, sdtUINT16},
             morphee::filters::t_ImCannyEdgeDetection<UINT8, UINT16>},
            {{sdtUINT8, sdtUINT32},
             morphee::filters::t_ImCannyEdgeDetection<UINT8, UINT32>},
            {{sdtUINT16, sdtUINT16},
             morphee::filters::t_ImCannyEdgeDetection<UINT16, UINT16>},
            {{sdtUINT16, sdtUINT32},
             morphee::filters::t_ImCannyEdgeDetection<UINT16, UINT32>},
            {{sdtUINT32, sdtUINT32},
             morphee::filters::t_ImCannyEdgeDetection<UINT32, UINT32>},
            {{sdtFloat, sdtFloat},
             morphee::filters::t_ImCannyEdgeDetection<F_SIMPLE, F_SIMPLE>},
            {{sdtDouble, sdtDouble},
             morphee::filters::t_ImCannyEdgeDetection<F_DOUBLE, F_DOUBLE>},
            {{sdtNone, sdtNone}, 0}}},
          {{dtNone, dtNone}, {{{sdtNone, sdtNone}, 0}}}};
      // actually call the mapped function and return its result
      return _localDispatch::dispatch(imIn, Sigma, imOut, tab_arith_local);
    }

  } // namespace filters

} // namespace morphee
