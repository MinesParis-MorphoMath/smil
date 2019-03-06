
#include <morphee/filters/include/morpheeFilters.hpp>
#include <morphee/filters/include/private/filtersGabor_T.hpp>

#include <morphee/image/include/private/imageDispatch_T.hpp>
#include <morphee/image/include/private/image_T.hpp>

namespace morphee
{
  namespace filters
  {
    // Author : Vincent MORARD
    // 31 july 2012

    //****************************************************************
    //*												Template (interface)
    //****************************************************************

    template <typename T1, typename T2>
    RES_C t_ImNormalized(const ImageInterface *imIn, double Value,
                         ImageInterface *imOut)
    {
      const Image<T1> *_localImIn = dynamic_cast<const Image<T1> *>(imIn);
      Image<T2> *_localImOut      = dynamic_cast<Image<T2> *>(imOut);
      if (!_localImOut || !_localImIn) {
        MORPHEE_REGISTER_ERROR("Error in downcast");
        return RES_ERROR_DYNAMIC_CAST;
      }
      return t_ImNormalized(*_localImIn, Value, *_localImOut);
    }

    template <typename T1, typename T2>
    RES_C t_ImDisplayKernel(const ImageInterface *imIn, ImageInterface *imOut)
    {
      const Image<T1> *_localImIn = dynamic_cast<const Image<T1> *>(imIn);
      Image<T2> *_localImOut      = dynamic_cast<Image<T2> *>(imOut);
      if (!_localImOut || !_localImIn) {
        MORPHEE_REGISTER_ERROR("Error in downcast");
        return RES_ERROR_DYNAMIC_CAST;
      }
      return t_ImDisplayKernel(*_localImIn, *_localImOut);
    }

    template <typename T1>
    RES_C t_createGaborKernel(ImageInterface *imOut, double sigma, double theta,
                              double lambda, double psi, double gamma)
    {
      Image<T1> *_localImOut = dynamic_cast<Image<T1> *>(imOut);
      if (!_localImOut) {
        MORPHEE_REGISTER_ERROR("Error in downcast");
        return RES_ERROR_DYNAMIC_CAST;
      }
      return t_createGaborKernel(*_localImOut, sigma, theta, lambda, psi,
                                 gamma);
    }

    template <typename T1, typename T2>
    RES_C t_ImGaborFilterConvolution(const ImageInterface *imIn, double sigma,
                                     double theta, double lambda, double psi,
                                     double gamma, ImageInterface *imOut)
    {
      const Image<T1> *_localImIn = dynamic_cast<const Image<T1> *>(imIn);
      Image<T2> *_localImOut      = dynamic_cast<Image<T2> *>(imOut);
      if (!_localImOut || !_localImIn) {
        MORPHEE_REGISTER_ERROR("Error in downcast");
        return RES_ERROR_DYNAMIC_CAST;
      }
      return t_ImGaborFilterConvolution(*_localImIn, sigma, theta, lambda, psi,
                                        gamma, *_localImOut);
    }

    template <typename T1, typename T2>
    RES_C t_ImGaborFilterConvolutionNorm(const ImageInterface *imIn,
                                         double sigma, double theta,
                                         double lambda, double psi,
                                         double gamma, double Min, double Max,
                                         ImageInterface *imOut)
    {
      ImageInterface *ImGabor;
      ImGabor = ImCreateSame(imIn, "F_DOUBLE");
      ImGabor->allocateImage();

      const Image<T1> *_localImIn    = dynamic_cast<const Image<T1> *>(imIn);
      Image<T2> *_localImOut         = dynamic_cast<Image<T2> *>(imOut);
      Image<F_DOUBLE> *_localImGabor = dynamic_cast<Image<F_DOUBLE> *>(ImGabor);

      if (!_localImOut || !_localImIn || !_localImGabor) {
        MORPHEE_REGISTER_ERROR("Error in downcast");
        return RES_ERROR_DYNAMIC_CAST;
      }

      RES_C res = t_ImGaborFilterConvolutionNorm(*_localImIn, sigma, theta,
                                                 lambda, psi, gamma, Min, Max,
                                                 *_localImOut, *_localImGabor);
      ImGabor->deallocateImage();
      delete ImGabor;
      return res;
    }

    template <typename T1, typename T2>
    RES_C t_ImGaborFilterConvolutionNormAuto(const ImageInterface *imIn,
                                             double sigma, double theta,
                                             double lambda, double psi,
                                             double gamma, double *Min,
                                             double *Max, ImageInterface *imOut)
    {
      ImageInterface *ImGabor;
      ImGabor = ImCreateSame(imIn, "F_DOUBLE");
      ImGabor->allocateImage();

      const Image<T1> *_localImIn    = dynamic_cast<const Image<T1> *>(imIn);
      Image<T2> *_localImOut         = dynamic_cast<Image<T2> *>(imOut);
      Image<F_DOUBLE> *_localImGabor = dynamic_cast<Image<F_DOUBLE> *>(ImGabor);
      if (!_localImOut || !_localImIn || !_localImGabor) {
        MORPHEE_REGISTER_ERROR("Error in downcast");
        return RES_ERROR_DYNAMIC_CAST;
      }

      RES_C res = t_ImGaborFilterConvolutionNormAuto(
          *_localImIn, sigma, theta, lambda, psi, gamma, Min, Max, *_localImOut,
          *_localImGabor);
      ImGabor->deallocateImage();
      delete ImGabor;
      return res;
    }

    //*************************************************************************
    //*												Dispatch (interface)
    //*************************************************************************

    RES_C ImGaborFilterConvolution(const ImageInterface *imIn, double sigma,
                                   double theta, double lambda, double psi,
                                   double gamma, ImageInterface *imOut)
    {
      // proper type for the dispatch array
      typedef generalDispatch<2, const ImageInterface *, double, double, double,
                              double, double, ImageInterface *>
          _localDispatch;

      // the dispatch array itself
      static const _localDispatch::dispatchData _tab_local[] = {
          {{dtScalar, dtScalar},
           {{{sdtUINT8, sdtDouble},
             morphee::filters::t_ImGaborFilterConvolution<UINT8, F_DOUBLE>},
            {{sdtUINT16, sdtDouble},
             morphee::filters::t_ImGaborFilterConvolution<UINT16, F_DOUBLE>},
            {{sdtUINT32, sdtDouble},
             morphee::filters::t_ImGaborFilterConvolution<UINT32, F_DOUBLE>},
            {{sdtUINT8, sdtFloat},
             morphee::filters::t_ImGaborFilterConvolution<UINT8, F_SIMPLE>},
            {{sdtUINT16, sdtFloat},
             morphee::filters::t_ImGaborFilterConvolution<UINT16, F_SIMPLE>},
            {{sdtUINT32, sdtFloat},
             morphee::filters::t_ImGaborFilterConvolution<UINT32, F_SIMPLE>},
            {{sdtNone, sdtNone}, 0}}},
          {{dtNone, dtNone}, {{{sdtNone, sdtNone}, 0}}}};

      // actually call the mapped function and return its result
      return _localDispatch::dispatch(imIn, sigma, theta, lambda, psi, gamma,
                                      imOut, _tab_local);
    }

    RES_C ImGaborFilterConvolution_Normalized(const ImageInterface *imIn,
                                              double sigma, double theta,
                                              double lambda, double psi,
                                              double gamma, double Min,
                                              double Max, ImageInterface *imOut)
    {
      // proper type for the dispatch array
      typedef generalDispatch<2, const ImageInterface *, double, double, double,
                              double, double, double, double, ImageInterface *>
          _localDispatch;

      // the dispatch array itself
      static const _localDispatch::dispatchData _tab_local[] = {
          {{dtScalar, dtScalar},
           {{{sdtUINT8, sdtUINT8},
             morphee::filters::t_ImGaborFilterConvolutionNorm<UINT8, UINT8>},
            {{sdtUINT8, sdtUINT16},
             morphee::filters::t_ImGaborFilterConvolutionNorm<UINT8, UINT16>},
            {{sdtUINT8, sdtUINT32},
             morphee::filters::t_ImGaborFilterConvolutionNorm<UINT8, UINT32>},
            {{sdtUINT16, sdtUINT16},
             morphee::filters::t_ImGaborFilterConvolutionNorm<UINT16, UINT16>},
            {{sdtUINT32, sdtUINT32},
             morphee::filters::t_ImGaborFilterConvolutionNorm<UINT32, UINT32>},

            {{sdtNone, sdtNone}, 0}}},
          {{dtNone, dtNone}, {{{sdtNone, sdtNone}, 0}}}};

      // actually call the mapped function and return its result
      return _localDispatch::dispatch(imIn, sigma, theta, lambda, psi, gamma,
                                      Min, Max, imOut, _tab_local);
    }

    RES_C ImGaborFilterConvolution_Auto_Normalized(const ImageInterface *imIn,
                                                   double sigma, double theta,
                                                   double lambda, double psi,
                                                   double gamma, double *Min,
                                                   double *Max,
                                                   ImageInterface *imOut)
    {
      // proper type for the dispatch array
      typedef generalDispatch<2, const ImageInterface *, double, double, double,
                              double, double, double *, double *,
                              ImageInterface *>
          _localDispatch;

      // the dispatch array itself
      static const _localDispatch::dispatchData _tab_local[] = {
          {{dtScalar, dtScalar},
           {{{sdtUINT8, sdtUINT8},
             morphee::filters::t_ImGaborFilterConvolutionNormAuto<UINT8,
                                                                  UINT8>},
            {{sdtUINT8, sdtUINT16},
             morphee::filters::t_ImGaborFilterConvolutionNormAuto<UINT8,
                                                                  UINT16>},
            {{sdtUINT8, sdtUINT32},
             morphee::filters::t_ImGaborFilterConvolutionNormAuto<UINT8,
                                                                  UINT32>},
            {{sdtUINT16, sdtUINT16},
             morphee::filters::t_ImGaborFilterConvolutionNormAuto<UINT16,
                                                                  UINT16>},
            {{sdtUINT32, sdtUINT32},
             morphee::filters::t_ImGaborFilterConvolutionNormAuto<UINT32,
                                                                  UINT32>},

            {{sdtNone, sdtNone}, 0}}},
          {{dtNone, dtNone}, {{{sdtNone, sdtNone}, 0}}}};

      // actually call the mapped function and return its result
      return _localDispatch::dispatch(imIn, sigma, theta, lambda, psi, gamma,
                                      Min, Max, imOut, _tab_local);
    }

    // For the FFT!
    RES_C createGaborKernel(ImageInterface *imOut, double sigma, double theta,
                            double lambda, double psi, double gamma)
    {
      // proper type for the dispatch array
      typedef generalDispatch<1, ImageInterface *, double, double, double,
                              double, double>
          _localDispatch;

      // the dispatch array itself
      static const _localDispatch::dispatchData _tab_local[] = {
          {{dtScalar},
           {{{sdtDouble}, morphee::filters::t_createGaborKernel<F_DOUBLE>},
            {{sdtFloat}, morphee::filters::t_createGaborKernel<F_SIMPLE>},
            {{sdtNone}, 0}}},
          {{dtNone}, {{{sdtNone}, 0}}}};

      // actually call the mapped function and return its result
      return _localDispatch::dispatch(imOut, sigma, theta, lambda, psi, gamma,
                                      _tab_local);
    }

    RES_C ImDisplayKernel(const ImageInterface *imIn, ImageInterface *imOut)
    {
      // proper type for the dispatch array
      typedef generalDispatch<2, const ImageInterface *, ImageInterface *>
          _localDispatch;

      // the dispatch array itself
      static const _localDispatch::dispatchData _tab_local[] = {
          {{dtScalar, dtScalar},
           {{{sdtDouble, sdtUINT8},
             morphee::filters::t_ImDisplayKernel<F_DOUBLE, UINT8>},
            {{sdtFloat, sdtUINT8},
             morphee::filters::t_ImDisplayKernel<F_SIMPLE, UINT8>},
            {{sdtNone, sdtNone}, 0}}},
          {{dtNone, dtNone}, {{{sdtNone, sdtNone}, 0}}}};

      // actually call the mapped function and return its result
      return _localDispatch::dispatch(imIn, imOut, _tab_local);
    }

    RES_C ImNormalized(const ImageInterface *imIn, double Value,
                       ImageInterface *imOut)
    {
      // proper type for the dispatch array
      typedef generalDispatch<2, const ImageInterface *, double,
                              ImageInterface *>
          _localDispatch;

      // the dispatch array itself
      static const _localDispatch::dispatchData _tab_local[] = {
          {{dtScalar, dtScalar},
           {{{sdtDouble, sdtUINT8},
             morphee::filters::t_ImNormalized<F_DOUBLE, UINT8>},
            {{sdtFloat, sdtUINT8},
             morphee::filters::t_ImNormalized<F_SIMPLE, UINT8>},
            {{sdtNone, sdtNone}, 0}}},
          {{dtNone, dtNone}, {{{sdtNone, sdtNone}, 0}}}};

      // actually call the mapped function and return its result
      return _localDispatch::dispatch(imIn, Value, imOut, _tab_local);
    }

  } // namespace filters

} // namespace morphee