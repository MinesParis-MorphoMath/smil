#ifndef D_MOSAIC_GEOCUTS_T_HPP
#define D_MOSAIC_GEOCUTS_T_HPP

namespace smil
{
  namespace graphalgo
  {
    template <class ImageIn, class ImageGrad, class ImageMarker, class SE,
              class ImageOut>
    RES_C t_GeoCuts_MinSurfaces_with_steps(const ImageIn &imIn,
                                           const ImageGrad &imGrad,
                                           const ImageMarker &imMarker,
                                           const SE &nl, F_SIMPLE step_x,
                                           F_SIMPLE step_y, F_SIMPLE step_z,
                                           ImageOut &imOut);

    template <class ImageIn, class ImageGrad, class ImageMarker, class SE,
              class ImageOut>
    RES_C t_GeoCuts_MinSurfaces_with_steps_vGradient(
        const ImageIn &imIn, const ImageGrad &imGrad,
        const ImageMarker &imMarker, const SE &nl, F_SIMPLE step_x,
        F_SIMPLE step_y, F_SIMPLE step_z, ImageOut &imOut);

    template <class ImageIn, class ImageGrad, class ImageMarker, class SE,
              class ImageOut>
    RES_C t_GeoCuts_MinSurfaces(const ImageIn &imIn, const ImageGrad &imGrad,
                                const ImageMarker &imMarker, const SE &nl,
                                ImageOut &imOut);

    template <class ImageIn, class ImageGrad, class ImageMarker, class SE,
              class ImageOut>
    RES_C t_GeoCuts_MinSurfaces_With_Line(const ImageIn &imIn,
                                          const ImageGrad &imGrad,
                                          const ImageMarker &imMarker,
                                          const SE &nl, ImageOut &imOut);

    template <class ImageIn, class ImageGrad, class ImageCurvature,
              class ImageMarker, typename _Beta, class SE, class ImageOut>
    RES_C t_GeoCuts_Regularized_MinSurfaces(const ImageIn &imIn,
                                            const ImageGrad &imGrad,
                                            const ImageCurvature &imCurvature,
                                            const ImageMarker &imMarker,
                                            const _Beta Beta, const SE &nl,
                                            ImageOut &imOut);

    template <class ImageIn, class ImageGrad, class ImageMarker, class SE,
              class ImageOut>
    RES_C t_GeoCuts_MultiWay_MinSurfaces(const ImageIn &imIn,
                                         const ImageGrad &imGrad,
                                         const ImageMarker &imMarker,
                                         const SE &nl, ImageOut &imOut);

    template <class ImageIn, class ImageGrad, class ImageMosaic,
              class ImageMarker, class SE, class ImageOut>
    RES_C t_GeoCuts_Optimize_Mosaic(const ImageIn &imIn,
                                    const ImageGrad &imGrad,
                                    const ImageMosaic &imMosaic,
                                    const ImageMarker &imMarker, const SE &nl,
                                    ImageOut &imOut);

    template <class ImageIn, class ImageMosaic, class ImageMarker, class SE,
              class ImageOut>
    RES_C t_GeoCuts_Segment_Graph(const ImageIn &imIn,
                                  const ImageMosaic &imMosaic,
                                  const ImageMarker &imMarker, const SE &nl,
                                  ImageOut &imOut);

    template <class ImageIn, class ImageMosaic, class ImageMarker,
              typename _Beta, typename _Sigma, class SE, class ImageOut>
    RES_C t_MAP_MRF_Ising(const ImageIn &imIn, const ImageMosaic &imMosaic,
                          const ImageMarker &imMarker, const _Beta Beta,
                          const _Sigma Sigma, const SE &nl, ImageOut &imOut);

    template <class ImageIn, class ImageMosaic, class ImageMarker,
              typename _Beta, typename _Sigma, class SE, class ImageOut>
    RES_C t_MAP_MRF_edge_preserving(const ImageIn &imIn,
                                    const ImageMosaic &imMosaic,
                                    const ImageMarker &imMarker,
                                    const _Beta Beta, const _Sigma Sigma,
                                    const SE &nl, ImageOut &imOut);

    template <class ImageIn, class ImageMosaic, class ImageMarker,
              typename _Beta, typename _Sigma, class SE, class ImageOut>
    RES_C t_MAP_MRF_Potts(const ImageIn &imIn, const ImageMosaic &imMosaic,
                          const ImageMarker &imMarker, const _Beta Beta,
                          const _Sigma Sigma, const SE &nl, ImageOut &imOut);

  } // namespace graphalgo
} // namespace smil

hello coucou

// #include "private/Mosaic_GeoCutsAlgo_impl_T.hpp"
// #include "private/Mosaic_GraphCuts_impl_T.hpp"
#endif // D_MOSAIC_GEOCUTS_T_HPP
