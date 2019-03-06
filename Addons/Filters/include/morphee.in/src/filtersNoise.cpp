

#include <morphee/filters/include/morpheeFilters.hpp>
#include <morphee/filters/include/private/filtersNoise_T.hpp>

#include <morphee/image/include/private/imageDispatch_T.hpp>
#include <morphee/image/include/private/image_T.hpp>

namespace morphee
{
  namespace filters
  {
    static unsigned long int __global_random_seed = 1u;

    void setSeed(unsigned long int s)
    {
      // if(s>=0)
      __global_random_seed = s;
    }
    unsigned long int getSeed()
    {
      return __global_random_seed;
    }
    template <typename T>
    RES_C t_ImAddNoiseSaltAndPepper(const ImageInterface *imIn,
                                    const F_DOUBLE freq, ImageInterface *imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_ImAddNoiseSaltAndPepper(interface)");
      const Image<T> *_imIn = dynamic_cast<const Image<T> *>(imIn);
      Image<T> *_imOut      = dynamic_cast<Image<T> *>(imOut);

      if (!imIn || !imOut) {
        MORPHEE_REGISTER_ERROR("Unable to downcast images");
        return RES_ERROR;
      }

      return t_ImAddNoiseSaltAndPepper(*_imIn, freq, *_imOut);
    }

    RES_C ImAddNoiseSaltAndPepper(const ImageInterface *imIn,
                                  const F_DOUBLE freq, ImageInterface *imOut)
    {
      typedef generalDispatch<2, const ImageInterface *, const F_DOUBLE,
                              ImageInterface *>
          __localDispatch;

      static const __localDispatch::dispatchData localTab[] = {
          {{dtScalar, dtScalar},
           {{{sdtUINT8, sdtUINT8}, t_ImAddNoiseSaltAndPepper<UINT8>},
            {{sdtINT8, sdtINT8}, t_ImAddNoiseSaltAndPepper<INT8>},
            {{sdtUINT16, sdtUINT16}, t_ImAddNoiseSaltAndPepper<UINT16>},
            {{sdtINT16, sdtINT16}, t_ImAddNoiseSaltAndPepper<INT16>},
            {{sdtUINT32, sdtUINT32}, t_ImAddNoiseSaltAndPepper<UINT32>},
            {{sdtINT32, sdtINT32}, t_ImAddNoiseSaltAndPepper<INT32>},

            {{sdtFloat, sdtFloat}, t_ImAddNoiseSaltAndPepper<F_SIMPLE>},
            {{sdtDouble, sdtDouble}, t_ImAddNoiseSaltAndPepper<F_DOUBLE>},
            {{sdtNone, sdtNone}, 0}}},
          {{dtNone, dtNone}, {{{sdtNone, sdtNone}, 0}}}};

      return __localDispatch::dispatch(imIn, freq, imOut, localTab);
    }

    template <typename T>
    RES_C t_ImAddNoiseGaussian(const ImageInterface *imIn, const F_DOUBLE sigma,
                               ImageInterface *imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_ImAddNoiseGaussian(interface)");
      const Image<T> *_imIn = dynamic_cast<const Image<T> *>(imIn);
      Image<T> *_imOut      = dynamic_cast<Image<T> *>(imOut);

      if (!imIn || !imOut) {
        MORPHEE_REGISTER_ERROR("Unable to downcast images");
        return RES_ERROR;
      }

      return t_ImAddNoiseGaussian(*_imIn, sigma, *_imOut);
    }

    RES_C ImAddNoiseGaussian(const ImageInterface *imIn, const F_DOUBLE sigma,
                             ImageInterface *imOut)
    {
      typedef generalDispatch<2, const ImageInterface *, const F_DOUBLE,
                              ImageInterface *>
          __localDispatch;

      static const __localDispatch::dispatchData localTab[] = {
          {{dtScalar, dtScalar},
           {{{sdtUINT8, sdtUINT8}, t_ImAddNoiseGaussian<UINT8>},
            {{sdtINT8, sdtINT8}, t_ImAddNoiseGaussian<INT8>},
            {{sdtUINT16, sdtUINT16}, t_ImAddNoiseGaussian<UINT16>},
            {{sdtINT16, sdtINT16}, t_ImAddNoiseGaussian<INT16>},
            {{sdtUINT32, sdtUINT32}, t_ImAddNoiseGaussian<UINT32>},
            {{sdtINT32, sdtINT32}, t_ImAddNoiseGaussian<INT32>},

            {{sdtFloat, sdtFloat}, t_ImAddNoiseGaussian<F_SIMPLE>},
            {{sdtDouble, sdtDouble}, t_ImAddNoiseGaussian<F_DOUBLE>},
            {{sdtNone, sdtNone}, 0}}},
          {{dtPixel3, dtPixel3},
           {{{sdtUINT8, sdtUINT8}, t_ImAddNoiseGaussian<pixel_3<UINT8>>},
            {{sdtINT8, sdtINT8}, t_ImAddNoiseGaussian<pixel_3<INT8>>},
            {{sdtUINT16, sdtUINT16}, t_ImAddNoiseGaussian<pixel_3<UINT16>>},
            {{sdtINT16, sdtINT16}, t_ImAddNoiseGaussian<pixel_3<INT16>>},
            {{sdtUINT32, sdtUINT32}, t_ImAddNoiseGaussian<pixel_3<UINT32>>},
            {{sdtINT32, sdtINT32}, t_ImAddNoiseGaussian<pixel_3<INT32>>},

            {{sdtFloat, sdtFloat}, t_ImAddNoiseGaussian<pixel_3<F_SIMPLE>>},
            {{sdtDouble, sdtDouble}, t_ImAddNoiseGaussian<pixel_3<F_DOUBLE>>},
            {{sdtNone, sdtNone}, 0}}},
          {{dtNone, dtNone}, {{{sdtNone, sdtNone}, 0}}}};
      return __localDispatch::dispatch(imIn, sigma, imOut, localTab);
    }

    template <typename T>
    RES_C t_ImAddNoisePoissonian(const ImageInterface *imIn,
                                 ImageInterface *imOut)
    {
      MORPHEE_ENTER_FUNCTION("t_ImAddNoisePoissonian(interface)");
      const Image<T> *_imIn = dynamic_cast<const Image<T> *>(imIn);
      Image<T> *_imOut      = dynamic_cast<Image<T> *>(imOut);

      if (!imIn || !imOut) {
        MORPHEE_REGISTER_ERROR("Unable to downcast images");
        return RES_ERROR;
      }

      return t_ImAddNoisePoissonian(*_imIn, *_imOut);
    }

    RES_C ImAddNoisePoissonian(const ImageInterface *imIn,
                               ImageInterface *imOut)
    {
      typedef generalDispatch<2, const ImageInterface *, ImageInterface *>
          __localDispatch;

      static const __localDispatch::dispatchData localTab[] = {
          {{dtScalar, dtScalar},
           {{{sdtUINT8, sdtUINT8}, t_ImAddNoisePoissonian<UINT8>},
            {{sdtINT8, sdtINT8}, t_ImAddNoisePoissonian<INT8>},
            {{sdtUINT16, sdtUINT16}, t_ImAddNoisePoissonian<UINT16>},
            {{sdtINT16, sdtINT16}, t_ImAddNoisePoissonian<INT16>},
            {{sdtUINT32, sdtUINT32}, t_ImAddNoisePoissonian<UINT32>},
            {{sdtINT32, sdtINT32}, t_ImAddNoisePoissonian<INT32>},
            {{sdtNone, sdtNone}, 0}}},
          {{dtNone, dtNone}, {{{sdtNone, sdtNone}, 0}}}};
      return __localDispatch::dispatch(imIn, imOut, localTab);
    }

  } // namespace filters
} // namespace morphee
