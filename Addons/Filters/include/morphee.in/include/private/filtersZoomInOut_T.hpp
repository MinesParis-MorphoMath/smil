#ifndef __MORPHEE_FILTERS_SIZING_T_HPP__
#define __MORPHEE_FILTERS_SIZING_T_HPP__

#include <morphee/image/include/private/imageUtils_T.hpp>
#include <morphee/image/include/private/imageManipulation_T.hpp>
#include <morphee/stats/include/private/statsMeasure_T.hpp>

namespace morphee
{
  namespace filters
  {
    //! Example of operator used by t_ImBasicDecimation
    //! This one is basically returning the first pixel of each window provided
    //! by t_ImBasicDecimation
    template <class Iter> class s_returnFirstPoint
    {
    public:
      s_returnFirstPoint()
      {
      }
      void reset()
      {
      }
      inline void
      accumulator(Iter /*begin*/,
                  const Iter & /*end*/) // void pour que ce soit rapide. Peut
                                        // mettre aut chose si tu veux
      {
        // ici, remplissage des variables accumulateur internes à cette classe.
      }
      inline typename Iter::value_type result(Iter begin, const Iter & /*end*/)
      {
        // ici, choix du pixel parmi les valeurs de la fenetre, correspondant au
        // mieux à ce qui a été calculé permet par exemple une fonction distance
        // euclidienne etc...
        return *begin;
      }
    };

    //! Example of operator used by t_ImBasicDecimation
    //! This operator returns the closest vector considering the distance on HLS
    //! color space
    template <class Iter> class s_hlsDecimation
    {
      typedef stats::s_opMeanCircularNormed<typename Iter::value_type> _Op;
      _Op op;

    public:
      s_hlsDecimation() : op()
      {
      }
      void reset()
      {
        // op.m_acc = typename _Op::result_type(0,0);
        // op.m_nb_pixels = 0;
        op.reset();
      }
      inline void accumulator(Iter begin, const Iter &end)
      {
        op = std::for_each(begin, end, op);
      }
      inline typename Iter::value_type result(Iter begin, const Iter &end)
      {
        // Choix du vecteur median
        return op.median(begin, end);
      }
    };

    /*!
     * @brief Perform a decimation of the input image
     *
     * This function performs the operation defined by 'op::accumulator' on the
     * pixels of the window. Once done on the window, it calls 'op::result' to
     * get the result. This may be for instance a distance function. Eventually,
     * 'op::reset' is called before computing the next window
     *
     */
    template <typename __imageInOut, class Op>
    RES_C t_ImDecimation(const __imageInOut &imin,
                         const std::vector<F_SIMPLE> &coords_factors, Op op,
                         __imageInOut *&imout)
    {
      typedef typename __imageInOut::coordinate_system coordinate_system;

      MORPHEE_ENTER_FUNCTION("t_ImDecimation");

      if (coords_factors.size() != imin.getCoordinateDimension()) {
        MORPHEE_REGISTER_ERROR(
            "Factors' size and image's dimensions are different");
        return RES_ERROR_BAD_ARG;
      }

      // les facteurs doivent etre supérieurs à 1 (ce sont des facteurs de
      // division)
      for (unsigned int k = 0; k < coords_factors.size(); k++) {
        if (coords_factors[k] < 1.) {
          MORPHEE_REGISTER_ERROR("Decimation factors must be >= 1");
          return RES_ERROR_BAD_ARG;
        }
      }

      if (!imin.isAllocated()) {
        MORPHEE_REGISTER_ERROR("Input image not allocated");
        return RES_NOT_ALLOCATED;
      }
      if (imout != 0) {
        if (imout->isAllocated()) {
          MORPHEE_REGISTER_ERROR(
              "Output image exists and is already allocated");
          return RES_ALLOCATED;
        }
      }

      s_coordinate_proxy<__imageInOut> op_size;

      __imageInOut imTemp = imin.getSame();
      t_ImCopy(imin, imTemp);

      coordinate_system w_size = op_size.WSize(imin);
      coordinate_system w_size_new;

      for (unsigned int k = 0; k < w_size.getDimension(); k++) {
        w_size_new[k] = static_cast<coord_t>(
            ::ceilf(static_cast<F_SIMPLE>(w_size[k]) / coords_factors[k]));
      }

      typename __imageInOut::window_info_type wi(0, 0);
      for (unsigned int k = 0; k < w_size.getDimension(); k++) {
        wi.Size()[k] = static_cast<coord_t>(::ceilf(coords_factors[k]));
      }

      // cette partie à reprendre suivant les préférences de romain en matière
      // d'allocation
      if (imout == 0) {
        imout = new (std::nothrow) __imageInOut(w_size_new);
        if (!imout) {
          MORPHEE_REGISTER_ERROR("Unable to create image structure");
          return RES_ERROR_MEMORY;
        }
        if (imout->allocateImage() != RES_OK) {
          delete imout;
          MORPHEE_REGISTER_ERROR("Unable to allocate image");
          return RES_ERROR_MEMORY;
        }
        imout->ColorInfo() = imin.ColorInfo();
      } else {
        imout->setSize(w_size_new);
        if (imout->allocateImage() != RES_OK) {
          MORPHEE_REGISTER_ERROR("Unable to allocate image");
          return RES_ERROR_MEMORY;
        }
      }

      const coordinate_system &w_start = op_size.WStart(imin);

      typename __imageInOut::iterator it     = imout->begin(),
                                      it_end = imout->end();

      for (; it != it_end; ++it) {
        coordinate_system coord = it.Position();

        for (unsigned int k = 0; k < coord.getDimension(); k++) {
          coord[k] = static_cast<coord_t>(coord[k] * coords_factors[k]);
        }

        wi.Start() = w_start + coord;
        // wi.xStart = imin.getWxStart() +
        // static_cast<coord_t>(static_cast<F_SIMPLE>(it.getX()) * x_factor);
        // wi.yStart = imin.getWyStart() +
        // static_cast<coord_t>(static_cast<F_SIMPLE>(it.getY()) * y_factor);
        // wi.zStart = imin.getWzStart() +
        // static_cast<coord_t>(static_cast<F_SIMPLE>(it.getZ()) * z_factor);
        imTemp.setActiveWindow(wi);
        // peut-etre traitement particulier pour les bords extremes de imout,
        // pour eviter les erreurs d'arrondi
        op.accumulator(imTemp.begin(), imTemp.end());
        *it = op.result(imTemp.begin(), imTemp.end());
        op.reset();
      }
      return RES_OK;
    }

    /*!
     * @brief Perform a (basic) zoom of the input image
     *
     */
    template <typename __imageInOut>
    RES_C t_ImZoom(const __imageInOut &imin,
                   const std::vector<F_SIMPLE> &zoom_factors,
                   __imageInOut *&imout)
    {
      typedef typename __imageInOut::coordinate_system coordinate_system;

      MORPHEE_ENTER_FUNCTION("t_ImZoom");

      if (!imin.isAllocated()) {
        MORPHEE_REGISTER_ERROR("Input image not allocated");
        return RES_NOT_ALLOCATED;
      }

      if (zoom_factors.size() != imin.getCoordinateDimension()) {
        MORPHEE_REGISTER_ERROR(
            "Factors' size and image's dimensions are different");
        return RES_ERROR_BAD_ARG;
      }

      // les facteurs doivent etre supérieurs à 1 (ce sont des facteurs de
      // division)
      for (unsigned int k = 0; k < zoom_factors.size(); k++) {
        if (FLOAT_EQ(zoom_factors[k], 0.)) {
          MORPHEE_REGISTER_ERROR("Zoom factors cannot be null");
          return RES_ERROR_BAD_ARG;
        }
      }

      if (imout != 0) {
        if (imout->isAllocated()) {
          MORPHEE_REGISTER_ERROR("Output image already allocated");
          return RES_ALLOCATED;
        }
      }

      s_coordinate_proxy<__imageInOut> op_size;

      const coordinate_system w_size = op_size.WSize(imin);
      coordinate_system w_size_new;

      for (unsigned int k = 0; k < w_size.getDimension(); k++) {
        w_size_new[k] = static_cast<coord_t>(
            ::ceilf(static_cast<F_SIMPLE>(w_size[k]) * zoom_factors[k]));
      }

      // cette partie à reprendre suivant les préférences de romain en matière
      // d'allocation
      if (imout == 0) {
        imout = new (std::nothrow) __imageInOut(w_size_new);
        if (!imout) {
          MORPHEE_REGISTER_ERROR("Unable to create image structure");
          return RES_ERROR_MEMORY;
        }
        if (imout->allocateImage() != RES_OK) {
          MORPHEE_REGISTER_ERROR("Unable to allocate image");
          delete imout;
          return RES_ERROR_MEMORY;
        }
        imout->ColorInfo() = imin.ColorInfo();
      } else {
        imout->setSize(w_size_new);
        if (imout->allocateImage() != RES_OK) {
          MORPHEE_REGISTER_ERROR("Unable to allocate image");
          return RES_ERROR_MEMORY;
        }
      }

      const coordinate_system &w_start = op_size.WStart(imin);
      typename __imageInOut::window_info_type wi;

      typename __imageInOut::const_iterator it     = imin.begin(),
                                            it_end = imin.end();

      for (; it != it_end; ++it) {
        // Raffi: cette manière d'écrire le parcours de l'image nous *assure*
        // que tous les points sont correctement parcourus. Si on fait des cast
        // trop tot, on perd des points.
        coordinate_system coord1 = it.Position();
        coord1 -= w_start;
        coordinate_system coord2 = coord1;
        coord1 += 1;

        for (unsigned int k = 0; k < coord1.getDimension(); k++) {
          const F_SIMPLE fac = zoom_factors[k];
          const F_SIMPLE f   = static_cast<F_SIMPLE>(coord2[k]) * fac;
          coord1[k] =
              static_cast<coord_t>(static_cast<F_SIMPLE>(coord1[k]) * fac - f);
          coord2[k] = static_cast<coord_t>(f);
        }
        wi.Start() = coord2;
        wi.Size()  = coord1;

        // Raffi: il faut certainement l'écrire de manière explicite
        // wi.Start() = (it.Position() - w_start) * zoom_factors;
        // wi.xStart = static_cast<coord_t>(static_cast<F_SIMPLE>(it.getX() -
        // imin.getWxStart()) * x_factor); wi.yStart =
        // static_cast<coord_t>(static_cast<F_SIMPLE>(it.getY() -
        // imin.getWyStart()) * y_factor); wi.zStart =
        // static_cast<coord_t>(static_cast<F_SIMPLE>(it.getZ() -
        // imin.getWzStart()) * z_factor);

        /*
        wi.xSize = static_cast<coord_t>(static_cast<F_SIMPLE>(it.getX() -
        imin.getWxStart() + 1) * x_factor)
          - static_cast<coord_t>(static_cast<F_SIMPLE>(it.getX() -
        imin.getWxStart()) * x_factor) ;

        wi.ySize = static_cast<coord_t>(static_cast<F_SIMPLE>(it.getY() -
        imin.getWyStart() + 1) * y_factor)
          - static_cast<coord_t>(static_cast<F_SIMPLE>(it.getY() -
        imin.getWyStart()) * y_factor) ;

        wi.zSize = static_cast<coord_t>(static_cast<F_SIMPLE>(it.getZ() -
        imin.getWzStart() + 1) * z_factor)
          - static_cast<coord_t>(static_cast<F_SIMPLE>(it.getZ() -
        imin.getWzStart()) * z_factor) ;

        */

        imout->setActiveWindow(wi);

        // Est-ce qu'il y a des méthodes de zoom avec interpolation ?
        // Comment factoriser les deux fonctions de zoom/unzoom ?

        // peut-etre traitement particulier pour les bords extremes de imout,
        // pour eviter les erreurs d'arrondi
        // op.accumulator(imTemp.begin(), imTemp.end());
        //*it = op.result(imTemp.begin(), imTemp.end());
        t_ImSetConstant(*imout, *it);
        // op.reset();
      }

      return imout->resetActiveWindow();
      ;
    }

  } // namespace filters
} // namespace morphee

#endif /* __MORPHEE_FILTERS_SIZING_T_HPP__ */
