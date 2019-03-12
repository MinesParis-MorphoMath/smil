#ifndef __MORPHEE_HEADER_FILTERS_HARRIS_T_HPP__
#define __MORPHEE_HEADER_FILTERS_HARRIS_T_HPP__

#include <morphee/image/include/private/imageArithmetic_T.hpp>
#include <morphee/image/include/private/imageColorSpaceTransform_T.hpp>

#include <morphee/filters/include/private/filtersDiffusion_T.hpp>
#include <morphee/filters/include/private/filtersDifferential_T.hpp>
#include <morphee/filters/include/private/filtersGaussian_T.hpp>

namespace morphee
{
  namespace filters
  {
    template <class image> struct s_image_vector_deletor {
      std::vector<image *> vect;
      s_image_vector_deletor(std::vector<image *> &_vect) : vect(_vect){};
      ~s_image_vector_deletor()
      {
        for (unsigned int i = 0; i < vect.size(); i++) {
          delete vect[i];
        }
      }
    };

    //
    // Détecteur de points d'intéret:
    //
    // M = 	[ G ( gradX ^ 2 )		G ( gradXgradY ) ]
    //		[ G ( gradXgradY)		G ( gradY ^ 2 )  ] où G est une convolution par
    // une gaussienne
    //
    // L'operateur de Harris c'est det(M) - k Trace^2(M)  et k = 0.04 (va savoir
    // pourquoi...)
    //
    // Raffi: apparemment (wikipedia) la matrice M est (\nabla I) * t(\nabla I),
    // t étant la transposition. M est donc symétrique, et la définition du
    // détecteur de coin reste ensuite la meme. Si ca intéresse qq'un de mettre
    // tout ca ...
    template <class IMAGE, class IMAGEOUT, typename tIn>
    struct s_HarrisOperator_Helper {
      RES_C operator()(const IMAGE &imIn, UINT32 gaussSz, IMAGEOUT &imOut)
      {
        MORPHEE_ENTER_FUNCTION("t_ImHarrisOperator");

        // FIXME TODO: dimension 2 uniquement (pour l'instant !)

        typedef typename s_from_type_to_type<IMAGE, F_DOUBLE>::image_type
            gradient_image_type;

        if (morphee::t_ImDimension(imIn) > 2 ||
            morphee::t_ImDimension(imOut) > 2) {
          MORPHEE_REGISTER_ERROR(
              "This function works only for bidimensionnal images");
          return morphee::RES_ERROR_BAD_ARG;
        }

        if (imIn.getZSize() > 1 || imOut.getZSize() > 1) {
          MORPHEE_REGISTER_ERROR(
              "This function works only on images in xy plane");
          return morphee::RES_ERROR_BAD_ARG;
        }

        RES_C res;

        std::vector<gradient_image_type *> v_grad;

        res = t_Gradient(imIn, v_grad);
        if (res != RES_OK)
          return res;

        // frees the gradient images at exit
        const s_image_vector_deletor<gradient_image_type>
            gradient_vector_deletor(v_grad);

        /*void delete_v_grad()
        {
          for(unsigned int i = 0; i < v_grad.size(); i++)
          {
            delete v_grad[i];
          }
        }*/

        if (v_grad.size() != 2) {
          /*for(unsigned int i = 0; i < v_grad.size(); i++)
          {
            delete v_grad[i];
          }*/
          return RES_ERROR;
        }

        gradient_image_type &imGradX = *v_grad[0];
        gradient_image_type &imGradY = *v_grad[1];

        // Image<F_DOUBLE> imGradX = imIn.template t_getSameImage<
        // Image<F_DOUBLE> >(); Image<F_DOUBLE> imGradY = imGradX.getSame();
        gradient_image_type imGradXY = imGradX.getSame();

        // t_colorSplitTo3( imGradientXYZ, imGradX, imGradY, imGradXY); //
        // imGradXY only here to fill the position, its value will be
        // overwritten below

        /*
        res = t_ImDifferentialGradientX( imIn, imGradX );

        if( res != RES_OK )
          return res;

        res = t_ImDifferentialGradientY( imIn, imGradY );
        if( res != RES_OK )
          return res;
          */

        // gradX*gradY
        res = t_arithMultImage(imGradX, imGradY, imGradXY);
        if (res != RES_OK) {
          /*for(unsigned int i = 0; i < v_grad.size(); i++)
          {
            delete v_grad[i];
          }*/
          return res;
        }

        // gradX ^ 2
        res = t_arithMultImage(imGradX, imGradX, imGradX);
        if (res != RES_OK) {
          /*for(unsigned int i = 0; i < v_grad.size(); i++)
          {
            delete v_grad[i];
          }*/
          return res;
        }

        // gradY ^ 2
        res = t_arithMultImage(imGradY, imGradY, imGradY);
        if (res != RES_OK) {
          /*for(unsigned int i = 0; i < v_grad.size(); i++)
          {
            delete v_grad[i];
          }*/
          return res;
        }

        Image<F_DOUBLE> imGradX_G  = imGradX.getSame();
        Image<F_DOUBLE> imGradY_G  = imGradX.getSame();
        Image<F_DOUBLE> imGradXY_G = imGradX.getSame();

        res = t_ImGaussianFilter(imGradX, gaussSz, imGradX_G);
        if (res != RES_OK) {
          /*for(unsigned int i = 0; i < v_grad.size(); i++)
          {
            delete v_grad[i];
          }*/
          return res;
        }
        res = t_ImGaussianFilter(imGradY, gaussSz, imGradY_G);
        if (res != RES_OK) {
          /*for(unsigned int i = 0; i < v_grad.size(); i++)
          {
            delete v_grad[i];
          }*/
          return res;
        }
        res = t_ImGaussianFilter(imGradXY, gaussSz, imGradXY_G);
        if (res != RES_OK) {
          /*for(unsigned int i = 0; i < v_grad.size(); i++)
          {
            delete v_grad[i];
          }*/
          return res;
        }

        typename IMAGEOUT::iterator it = imOut.begin(), iend = imOut.end();

        const F_DOUBLE k = 0.04;

        for (; it != iend; ++it) {
          const offset_t offs = it.getOffset();
          const F_DOUBLE px   = imGradX_G.pixelFromOffset(offs);
          const F_DOUBLE py   = imGradY_G.pixelFromOffset(offs);
          const F_DOUBLE pxy  = imGradXY_G.pixelFromOffset(offs);

          F_DOUBLE trace = px + py;
          trace *= trace;
          trace *= k;

          *it = static_cast<typename IMAGEOUT::value_type>(px * py - pxy * pxy -
                                                           trace);
        }

        /*for(unsigned int i = 0; i < v_grad.size(); i++)
        {
          delete v_grad[i];
        }*/

        return RES_OK;
      }
    }; // class s_HarrisOperator_Helper

    // Specialisation pixel_3
    template <class IMAGE, class IMAGEOUT, typename tIn>
    struct s_HarrisOperator_Helper<IMAGE, IMAGEOUT, pixel_3<tIn>> {
      RES_C operator()(const IMAGE &imIn, UINT32 gaussSz, IMAGEOUT &imOut)
      {
        MORPHEE_ENTER_FUNCTION("t_ImHarrisOperator");

        // Raffi: 2 étapes nécéssaire. Ne peut être directement déduit par
        // Visual en faisant IMAGE::value_type::value_type
        typedef typename IMAGE::value_type image_value_type;
        typedef typename image_value_type::value_type image_scalar_value_type;

        // typedef typename IMAGE::value_type::value_type scalar_t;
        typedef typename s_from_type_to_type<IMAGE, F_DOUBLE>::image_type
            gradient_image_type;
        typedef typename s_from_type_to_type<
            IMAGE, image_scalar_value_type>::image_type channel_image_type;

        RES_C res;

        // Image<pixel_3<F_DOUBLE> > imGradientR_XYZ = imIn.template
        // t_getSameImage< Image<pixel_3<F_DOUBLE> > >();
        // Image<pixel_3<F_DOUBLE> > imGradientG_XYZ = imIn.template
        // t_getSameImage< Image<pixel_3<F_DOUBLE> > >();
        // Image<pixel_3<F_DOUBLE> > imGradientB_XYZ = imIn.template
        // t_getSameImage< Image<pixel_3<F_DOUBLE> > >();

        // Thomas R:Parce que le  compilo visual 7.1 fait la tête même si c'est
        // du C++ valide

        // typedef tIn scalar_t;
        channel_image_type imR =
            imIn.template t_getSameImage<channel_image_type>();
        channel_image_type imG =
            imIn.template t_getSameImage<channel_image_type>();
        channel_image_type imB =
            imIn.template t_getSameImage<channel_image_type>();

        res = t_colorSplitTo3(imIn, imR, imG, imB);

        std::vector<gradient_image_type *> v_grad_r, v_grad_g, v_grad_b;

        res = t_Gradient(imR, v_grad_r);
        const s_image_vector_deletor<gradient_image_type>
            gradient_vector_deletor1(
                v_grad_r); // frees the gradient images at exit
        res = t_Gradient(imG, v_grad_g);
        const s_image_vector_deletor<gradient_image_type>
            gradient_vector_deletor2(v_grad_g);
        res = t_Gradient(imB, v_grad_b);
        const s_image_vector_deletor<gradient_image_type>
            gradient_vector_deletor3(v_grad_b);

        if (v_grad_r.size() != 2 || v_grad_g.size() != 2 ||
            v_grad_b.size() != 2) {
          return RES_ERROR;
        }

        if (res != RES_OK)
          return res;

        // Image<F_DOUBLE> imGradR_X = imIn.template t_getSameImage<
        // Image<F_DOUBLE> >(); Image<F_DOUBLE> imGradR_Y = imGradR_X.getSame();
        gradient_image_type imGradR_XY = v_grad_r[0]->getSame();

        // Image<F_DOUBLE> imGradG_X = imGradR_X.getSame();
        // Image<F_DOUBLE> imGradG_Y = imGradR_X.getSame();
        gradient_image_type imGradG_XY = v_grad_g[0]->getSame();

        // Image<F_DOUBLE> imGradB_X = imGradR_X.getSame();
        // Image<F_DOUBLE> imGradB_Y = imGradR_X.getSame();
        gradient_image_type imGradB_XY = v_grad_b[0]->getSame();

        // t_colorSplitTo3( imGradientR_XYZ, imGradR_X, imGradR_Y, imGradR_XY);
        // // imGradXY only here to fill the position, its value will be
        // overwritten below t_colorSplitTo3( imGradientG_XYZ, imGradG_X,
        // imGradG_Y, imGradG_XY); // imGradXY only here to fill the position,
        // its value will be overwritten below t_colorSplitTo3( imGradientB_XYZ,
        // imGradB_X, imGradB_Y, imGradB_XY); // imGradXY only here to fill the
        // position, its value will be overwritten below

        // gradX*gradY
        res = t_arithMultImage(*v_grad_r[0], *v_grad_r[1], imGradR_XY);
        res = t_arithMultImage(*v_grad_g[0], *v_grad_g[1], imGradG_XY);
        res = t_arithMultImage(*v_grad_b[0], *v_grad_b[1], imGradB_XY);
        if (res != RES_OK)
          return res;

        // gradX ^ 2
        res = t_arithMultImage(*v_grad_r[0], *v_grad_r[0], *v_grad_r[0]);
        res = t_arithMultImage(*v_grad_g[0], *v_grad_g[0], *v_grad_g[0]);
        res = t_arithMultImage(*v_grad_b[0], *v_grad_b[0], *v_grad_b[0]);
        if (res != RES_OK)
          return res;

        // gradY ^ 2
        res = t_arithMultImage(*v_grad_r[1], *v_grad_r[1], *v_grad_r[1]);
        res = t_arithMultImage(*v_grad_g[1], *v_grad_g[1], *v_grad_g[1]);
        res = t_arithMultImage(*v_grad_b[1], *v_grad_b[1], *v_grad_b[1]);
        if (res != RES_OK)
          return res;

        gradient_image_type imGradX_G  = v_grad_r[0]->getSame();
        gradient_image_type imGradY_G  = v_grad_r[0]->getSame();
        gradient_image_type imGradXY_G = v_grad_r[0]->getSame();

        t_arithAddImage(*v_grad_r[0], *v_grad_g[0], *v_grad_r[0]);
        t_arithAddImage(*v_grad_r[0], *v_grad_b[0], *v_grad_r[0]);

        res = t_ImGaussianFilter(*v_grad_r[0], gaussSz, imGradX_G);
        if (res != RES_OK)
          return res;

        t_arithAddImage(*v_grad_r[1], *v_grad_g[1], *v_grad_r[1]);
        t_arithAddImage(*v_grad_r[1], *v_grad_b[1], *v_grad_r[1]);
        res = t_ImGaussianFilter(*v_grad_r[1], gaussSz, imGradY_G);
        if (res != RES_OK)
          return res;

        t_arithAddImage(imGradR_XY, imGradG_XY, imGradR_XY);
        t_arithAddImage(imGradR_XY, imGradB_XY, imGradR_XY);
        res = t_ImGaussianFilter(imGradR_XY, gaussSz, imGradXY_G);
        if (res != RES_OK)
          return res;

        typename IMAGEOUT::iterator it = imOut.begin(), iend = imOut.end();

        const F_DOUBLE k = 0.04;

        for (; it != iend; ++it) {
          const offset_t offs = it.getOffset();
          const F_DOUBLE px   = imGradX_G.pixelFromOffset(offs);
          const F_DOUBLE py   = imGradY_G.pixelFromOffset(offs);
          const F_DOUBLE pxy  = imGradXY_G.pixelFromOffset(offs);

          F_DOUBLE trace = px + py;
          trace *= trace;
          trace *= k;

          *it = static_cast<typename IMAGEOUT::iterator::value_type>(
              px * py - pxy * pxy - trace);
        }

        return RES_OK;
      }
    }; // class s_HarrisOperator_Helper

    template <class IMAGE, class IMAGEOUT> struct s_HarrisOperator {
      RES_C operator()(const IMAGE &imIn, UINT32 gaussSz, IMAGEOUT &imOut)
      {
        return s_HarrisOperator_Helper<IMAGE, IMAGEOUT,
                                       typename IMAGE::value_type>()(
            imIn, gaussSz, imOut);
      }
    };

    template <class IMAGE, class IMAGEOUT>
    RES_C t_ImHarrisOperator(const IMAGE &imIn, UINT32 gaussSz, IMAGEOUT &imOut)
    {
      s_HarrisOperator<IMAGE, IMAGEOUT> op;
      return op(imIn, gaussSz, imOut);
    }
  } // namespace filters
} // namespace morphee

#endif
