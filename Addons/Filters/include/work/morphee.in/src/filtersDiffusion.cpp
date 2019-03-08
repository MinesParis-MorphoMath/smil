

#include <morphee/filters/include/morpheeFilters.hpp>
#include <morphee/filters/include/private/filtersDiffusion_T.hpp>

#include <morphee/image/include/private/imageDispatch_T.hpp>

namespace morphee
{
  namespace filters
  {
    template <class T1, class T2>
    RES_C t_Gradient(const ImageInterface *imIn,
                     std::vector<ImageInterface *> &v_imOut)
    {
      // on peut supposer que l'utilisateur fournit des images déjà allouées
      // mais nous n'avons pas envie de l'aider (la fonction de gradient les
      // supprimera de toute manière)
      if (v_imOut.size() != 0) {
        MORPHEE_REGISTER_ERROR("The ouput list should be void at the begining")
        return RES_ERROR_BAD_ARG;
      }

      const Image<T1> *_imTempIn = dynamic_cast<const Image<T1> *>(imIn);

      std::vector<Image<T2> *> v_output;
      RES_C res = t_Gradient(*_imTempIn, v_output);
      if (res != RES_OK)
        return res;

      for (unsigned int i = 0; i < v_output.size(); i++) {
        v_imOut.push_back(dynamic_cast<ImageInterface *>(v_output[i]));
      }

      return RES_OK;
    }

    RES_C Gradient(const ImageInterface *imin,
                   std::vector<ImageInterface *> &v_imOut)
    {
      typedef generalDispatch<1, const ImageInterface *,
                              std::vector<ImageInterface *> &>
          _localDispatch;
      static const _localDispatch::dispatchData tabLocal[] = {
          {{dtScalar},
           {{{sdtUINT8}, t_Gradient<UINT8, F_DOUBLE>},
            {{sdtFloat}, t_Gradient<F_SIMPLE, F_DOUBLE>},
            {{sdtUINT8}, t_Gradient<UINT8, F_DOUBLE>},
            {{sdtDouble}, t_Gradient<F_DOUBLE, F_DOUBLE>},
            {{sdtNone}, 0}}},
          {{dtNone}, {{{sdtNone}, 0}}}};

      return _localDispatch::dispatch(imin, v_imOut, tabLocal);
    }

    template <class T1, class T2>
    RES_C t_HeatDiffusion(const ImageInterface *imIn,
                          const morphee::UINT32 nosteps,
                          const F_SIMPLE step_value, ImageInterface *imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_HeatDiffusion<" + NameTraits<T1>::name() + "," +
                             NameTraits<T2>::name() + ">(interface)");

      const Image<T1> *_imTempIn = dynamic_cast<const Image<T1> *>(imIn);
      Image<T2> *_imTempOut      = dynamic_cast<Image<T2> *>(imOut);
      if ((!_imTempIn)) {
        MORPHEE_REGISTER_ERROR("Error in first downcast:" + ImageInfo(imIn));
        return morphee::RES_ERROR_DYNAMIC_CAST;
      }
      if (!_imTempOut) {
        MORPHEE_REGISTER_ERROR("Error in second downcast: " + ImageInfo(imOut));
        return morphee::RES_ERROR_DYNAMIC_CAST;
      }
      return t_HeatDiffusion(*_imTempIn, nosteps, step_value, *_imTempOut);
    }

    RES_C HeatDiffusion(const ImageInterface *imin,
                        const morphee::UINT32 nosteps,
                        const F_SIMPLE step_value, ImageInterface *imOut)
    {
      typedef generalDispatch<2, const ImageInterface *, const morphee::UINT32,
                              const F_SIMPLE, ImageInterface *>
          _localDispatch;

      static const _localDispatch::dispatchData tabLocal[] = {
          {{dtScalar, dtScalar},
           {{{sdtUINT8, sdtFloat}, t_HeatDiffusion<UINT8, F_SIMPLE>},
            {{sdtFloat, sdtFloat}, t_HeatDiffusion<F_SIMPLE, F_SIMPLE>},
            {{sdtDouble, sdtDouble}, t_HeatDiffusion<F_DOUBLE, F_DOUBLE>},
            {{sdtNone, sdtNone}, 0}}},
          {{dtNone, dtNone}, {{{sdtNone, sdtNone}, 0}}}};

      return _localDispatch::dispatch(imin, nosteps, step_value, imOut,
                                      tabLocal);
    }

    template <class T1, class T2>
    RES_C t_PeronaMalikDiffusion(const ImageInterface *imIn,
                                 const morphee::UINT32 nosteps,
                                 const F_SIMPLE step_value,
                                 const F_SIMPLE lambda, ImageInterface *imOut)
    {
      const Image<T1> *_imTempIn = dynamic_cast<const Image<T1> *>(imIn);
      Image<T2> *_imTempOut      = dynamic_cast<Image<T2> *>(imOut);
      if (!_imTempIn || !_imTempOut) {
        MORPHEE_REGISTER_ERROR("Error in downcast");
        return RES_ERROR_DYNAMIC_CAST;
      }
      return t_PeronaMalikDiffusion(*_imTempIn, nosteps, step_value, lambda,
                                    *_imTempOut);
    }

    RES_C PeronaMalikDiffusion(const ImageInterface *imin,
                               const morphee::UINT32 nosteps,
                               const F_SIMPLE step_value, const F_SIMPLE lambda,
                               ImageInterface *imOut)
    {
      typedef generalDispatch<2, const ImageInterface *, const morphee::UINT32,
                              const F_SIMPLE, const F_SIMPLE, ImageInterface *>
          _localDispatch;

      static const _localDispatch::dispatchData tabLocal[] = {
          {{dtScalar, dtScalar},
           {{{sdtUINT8, sdtFloat}, t_PeronaMalikDiffusion<UINT8, F_SIMPLE>},
            {{sdtFloat, sdtFloat}, t_PeronaMalikDiffusion<F_SIMPLE, F_SIMPLE>},
            {{sdtDouble, sdtDouble},
             t_PeronaMalikDiffusion<F_DOUBLE, F_DOUBLE>},
            {{sdtNone, sdtNone}, 0}}},
          {{dtNone, dtNone}, {{{sdtNone, sdtNone}, 0}}}};

      return _localDispatch::dispatch(imin, nosteps, step_value, lambda, imOut,
                                      tabLocal);
    }

    template <class T1, class T2>
    RES_C t_WeickertDiffusion(const ImageInterface *imIn,
                              const morphee::UINT32 nosteps,
                              const F_SIMPLE step_value, const F_SIMPLE lambda,
                              const F_SIMPLE m, const F_SIMPLE cm,
                              ImageInterface *imOut)
    {
      const Image<T1> *_imTempIn = dynamic_cast<const Image<T1> *>(imIn);
      Image<T2> *_imTempOut      = dynamic_cast<Image<T2> *>(imOut);
      if (!_imTempIn || !_imTempOut) {
        MORPHEE_REGISTER_ERROR("Error in downcast");
        return RES_ERROR_DYNAMIC_CAST;
      }
      return t_WeickertDiffusion(*_imTempIn, nosteps, step_value, lambda, m, cm,
                                 *_imTempOut);
    }

    RES_C WeickertDiffusion(const ImageInterface *imin,
                            const morphee::UINT32 nosteps,
                            const F_SIMPLE step_value, const F_SIMPLE lambda,
                            const F_SIMPLE m, const F_SIMPLE cm,
                            ImageInterface *imOut)
    {
      typedef generalDispatch<2, const ImageInterface *, const morphee::UINT32,
                              const F_SIMPLE, const F_SIMPLE, const F_SIMPLE,
                              const F_SIMPLE, ImageInterface *>
          _localDispatch;
      static const _localDispatch::dispatchData tab_local[] = {
          {{dtScalar, dtScalar},
           {{{sdtUINT8, sdtFloat}, t_WeickertDiffusion<UINT8, F_SIMPLE>},
            {{sdtFloat, sdtFloat}, t_WeickertDiffusion<F_SIMPLE, F_SIMPLE>},
            {{sdtDouble, sdtDouble}, t_WeickertDiffusion<F_DOUBLE, F_DOUBLE>},
            {{sdtNone, sdtNone}, 0}}},
          {{dtNone, dtNone}, {{{sdtNone, sdtNone}, 0}}}};

      return _localDispatch::dispatch(imin, nosteps, step_value, lambda, m, cm,
                                      imOut, tab_local);
    }

  } // namespace filters
} // namespace morphee
