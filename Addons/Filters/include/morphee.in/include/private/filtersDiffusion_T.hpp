
#ifndef __MORPHEE_FILTER_DIFFUSION_HPP__
#define __MORPHEE_FILTER_DIFFUSION_HPP__

//#define USE_PNG

#include <cmath>
#include <set>
#include <morphee/image/include/private/image_T.hpp>
#include <morphee/image/include/private/imageUtils_T.hpp>
#include <morphee/image/include/private/imageManipulation_T.hpp>
#include <morphee/image/include/private/imageArithmetic_T.hpp>

/*
#include <morphee/image/include/private/imageColorSpaceTransform_T.hpp>
#include "d:\Working CVS
Directory\MorpheeProject\morphee\imageIOExt\include\morpheeImageIOExt.hpp"
#include <strstream>
*/

namespace morphee
{
  namespace filters
  {
    //! @addtogroup diffusion_group
    //! @{

    /* @defgroup diffusion_tools
     * @ingroup diffusion_group
     * Tools provided in order to construct diffusion filters
     * @{
     */

    /*!
     * @brief Class providing important tools in order to create concrete
     * diffusive filters.
     *
     *
     */
    template <class t_image> class diffusion_tools
    {
    public:
      //! The input image type
      typedef t_image image_type;
      typedef typename image_type::value_type image_value_type;
      typedef typename image_type::coordinate_system image_coordinate_system;
      typedef typename image_type::pixel_position image_pixel_position;
      typedef typename image_type::window_info_type image_window_info_type;

      typedef typename DataTraits<image_value_type>::float_accumulator_type
          image_accumulator_type;

      // Since we do NOT know at compilation time what dimension of the input
      // image will be used we rather use a vector of image

      //! The type used for each gradient image
      typedef image_accumulator_type gradient_point_type;
      //! The type of each gradient image
      typedef Image<gradient_point_type, image_coordinate_system::dimension>
          gradient_scalar_image_type;
      //! The (fake) gradient image, which is in fact a collection of singular
      //! gradient image
      typedef std::vector<gradient_scalar_image_type *> gradient_image_type;

      //! Image type used for storing intermediate scalar computing (divergence,
      //! etc)
      typedef Image<image_accumulator_type, image_coordinate_system::dimension>
          functionnal_scalar_image_type;

      //! Image type used for storing intermediate vectorial computing
      typedef std::vector<functionnal_scalar_image_type>
          functionnal_vectorial_image_type;

      enum e_gradient_keep {
        e_keep_all_gradient, //!< Flag relevant for vectorial functionnals: the
                             //!< whole gradient should be computed before the
                             //!< div
        e_compute_each_step  //!< Flag to be used if the process is separable
                             //!< (gradient - modification - divergence along
                             //!< every direction)
      };

      //! Vector containing the displacement along every dimension, for
      //! computing the centered difference scheme (gradient image, etc.)
      std::vector<coord_t> displacement_offset;

    private:
      //! Computes the displacement along every relevant dimension
      RES_C compute_displacements(const image_type &im)
      {
        MORPHEE_ENTER_FUNCTION("diffusion_tools::compute_displacements");
        assert(im.isAllocated());

        displacement_offset.assign(im.getCoordinateDimension(), 0);
        const image_coordinate_system &w_size = im.WSize();
        const image_coordinate_system coord2(0);

        bool b_all_too_small = true;
        for (unsigned int k = 0; k < im.getCoordinateDimension(); k++) {
          if (w_size[k] > 2) {
            image_coordinate_system coord1(0);
            coord1[k] = 1;

            displacement_offset[k] = t_GetOffsetFromCoords(im, coord1) -
                                     t_GetOffsetFromCoords(im, coord2);
            b_all_too_small = false;
          }
        }

        if (b_all_too_small) {
          MORPHEE_REGISTER_ERROR("None of the dimension are of size > 2");
          return morphee::RES_ERROR_BAD_WINDOW_SIZE;
        }

        return RES_OK;
      }

    public:
      diffusion_tools() : displacement_offset(0)
      {
      }
      ~diffusion_tools()
      {
        // clean();
      }

      //! Prepare
      RES_C initFromImage(const image_type &im)
      {
        if (!im.isAllocated())
          return morphee::RES_NOT_ALLOCATED;

        // clean();

        RES_C res = compute_displacements(im);
        if (res != RES_OK)
          return res;

        return RES_OK;
      }

      void clean_gradients(gradient_image_type &whole_gradient)
      {
        if (whole_gradient.size() != 0) {
          for (unsigned int i = 0; i < whole_gradient.size(); i++) {
            delete whole_gradient[i];
          }
        }
      }

      //! Computes the centered difference scheme ie ( f(i+1) - f(i-1) ) / 2
      //! along the specified dimension
      //! This may also be applied on the input image, or the gradient image
      //! (for second derivatives, etc.)
      template <class t_image_type>
      RES_C centered_difference_scheme_along_dimension(
          const t_image_type &input_image, UINT16 dimension,
          gradient_scalar_image_type &gradient)
      {
        MORPHEE_ENTER_FUNCTION(
            "diffusion_tools::centered_difference_scheme_along_dimension");
        if (input_image.getCoordinateDimension() !=
            displacement_offset.size()) {
          MORPHEE_REGISTER_ERROR(
              "Input image's dimension does not correspond to the one used "
              "during initializing / or initFromImage was not called yet");
          return morphee::RES_ERROR_BAD_ARG;
        }
        if (dimension > displacement_offset.size()) {
          MORPHEE_REGISTER_ERROR(
              "Bad input dimension (> to the one of the image)");
          return morphee::RES_ERROR_BAD_ARG;
        }
        const coord_t displacement = displacement_offset[dimension];
        if (displacement == 0) {
          MORPHEE_REGISTER_ERROR("The specified dimension is not relevant");
          return morphee::RES_ERROR_BAD_ARG;
        }
        if (!t_CheckOffsetCompatible(input_image, gradient)) {
          MORPHEE_REGISTER_ERROR(
              "Offset should be compatible for current implementation "
              "(changing this is rather easy, but not done)");
          return morphee::RES_ERROR_BAD_ARG;
        }

        RES_C res = t_ImSetConstant(gradient, gradient_point_type(0));

        image_window_info_type win_grad     = gradient.ActiveWindow();
        image_window_info_type win_grad_new = win_grad;
        win_grad_new.Start()[dimension] += 1;
        win_grad_new.Size()[dimension] -= 2;

        res = gradient.setActiveWindow(win_grad_new);
        if (res != RES_OK) {
          MORPHEE_REGISTER_ERROR("Error while reducing the gradient's window");
          return res;
        }

        typedef typename gradient_scalar_image_type::iterator grad_it;
        grad_it itO          = gradient.begin();
        const grad_it itOend = gradient.end();

        for (; itO != itOend; ++itO) {
          const offset_t offset = itO.Offset();
          gradient.pixelFromOffset(offset) =
              input_image.pixelFromOffset(offset + displacement) -
              input_image.pixelFromOffset(offset - displacement);
        }

        res = t_arithDivImageConst(gradient, 2, gradient);
        if (res != RES_OK) {
          MORPHEE_REGISTER_ERROR("Error during division by 2");
          return res;
        }

        res = gradient.setActiveWindow(win_grad);
        if (res != RES_OK) {
          MORPHEE_REGISTER_ERROR(
              "Error while re-initializing the gradient's window");
          return res;
        }

        return RES_OK;
      }

      //! Computes the gradient along every relevant dimension and stores it
      //! into a vector of image The vector of image contains only relevant
      //! dimensions. They can be retrieved by checking the displacement_offset
      //! member (ordered by increasing index of the dimensions)
      RES_C compute_whole_gradient(const image_type &im,
                                   gradient_image_type &whole_gradient)
      {
        MORPHEE_ENTER_FUNCTION("diffusion_tools::compute_whole_gradient");
        RES_C res = initFromImage(im);
        if (res != RES_OK)
          return res;

        // delete previous images
        clean_gradients(whole_gradient);

        for (unsigned int i = 0; i < im.getCoordinateDimension(); i++) {
          if (displacement_offset[i] != 0) {
            // Raffi: alors là, t_getSame, deep copy ?
            gradient_scalar_image_type *gradient =
                new gradient_scalar_image_type(im);
            whole_gradient.push_back(gradient);

            res = centered_difference_scheme_along_dimension(im, i, *gradient);
            if (res != RES_OK)
              return res;
          }
        }
        return RES_OK;
      }

      // Interressant si le nombre de points dans l'image est vachement plus
      // grand que le nombre de valeurs possibles A voir... pe. on peut calculer
      // seulement sur un nombre n' de points tel que n' << N (nombre de points
      // dans l'image) et tel que occurence(n') important (10 premières valeurs
      // de l'histogramme par ex, on a alors log_2(10) < 4)

      /*! Applies an operator on a scalar image, by use of pre-computed table
       *
       * This function is interesting when the computing the value of each
       * point, through the operator s_scalar_operator is expensive. However,
       * the algorithm doesn't do anything automatically and the tradeoff should
       * be juged by the user. This function is called "scalar" since it is not
       * designed to cope with operators of arity diifferent of 1, but the input
       * image can have multiple channels. For the latter case, the multiple
       * channels should benefit of an ordering through the operator std::less
       * (the ordering can be a dummy one, just enough to discern two elements.)
       *
       * @remark: this function can be used in place
       */
      template <class t_scalar_image_functionnal, class s_scalar_operator>
      RES_C apply_scalar_functionnal_by_precomputing(
          const t_scalar_image_functionnal &im_scalar,
          const s_scalar_operator &op,
          functionnal_scalar_image_type &functionnal_im_out) const
      {
        typedef
            typename t_scalar_image_functionnal::value_type input_value_type;
        typedef
            typename t_scalar_image_functionnal::const_iterator input_iterator;
        typedef image_accumulator_type output_value_type;
        typedef
            typename functionnal_scalar_image_type::iterator output_iterator;
        typedef std::map<input_value_type, output_value_type>
            precomputing_table_type;

        MORPHEE_ENTER_FUNCTION(
            "diffusion_tools::apply_scalar_functionnal_by_precomputing");
        RES_C res = t_alignWindows(im_scalar, functionnal_im_out);
        if (res != RES_OK) {
          MORPHEE_REGISTER_ERROR("Error in align window");
          return res;
        }

        precomputing_table_type table;

        if (t_CheckOffsetCompatible(im_scalar, functionnal_im_out)) {
          input_iterator i_it = im_scalar.begin(), i_itE = im_scalar.end();

          for (; i_it != i_itE; ++i_it) {
            const offset_t off         = i_it.Offset();
            const input_value_type val = im_scalar.pixelFromOffset(off);

            // build the table while parsing the points
            if (table.count(val) == 0) {
              const output_value_type val_transform   = op(val);
              table[val]                              = val_transform;
              functionnal_im_out.pixelFromOffset(off) = val_transform;
            } else {
              functionnal_im_out.pixelFromOffset(off) = table[val];
            }
          }
        } else {
          input_iterator i_it = im_scalar.begin(), i_itE = im_scalar.end();
          output_iterator o_it = functionnal_im_out.begin();

          for (; i_it != i_itE; ++i_it, ++o_it) {
            const input_value_type val = *i_it;
            if (table.count(val) == 0) {
              const output_value_type val_transform = op(val);
              table[val]                            = val_transform;
              *o_it                                 = val_transform;
            } else {
              *o_it = table[val];
            }
          }
        }
        return RES_OK;
      }

      /*! Transform the whole gradient to a scalar image
       * The scalar image is computed by the following formula (pixel-wise):
       *  output = \sum_i op(gradient_i)
       * where i goes along the relevant dimensions, gradient_i is the gradient
       * along dimension i. op is provided by additive_separable_operator.
       */
      template <class additive_separable_operator>
      RES_C transform_gradient_to_squared_modulus(
          const gradient_image_type &whole_gradient,
          const additive_separable_operator &op,
          functionnal_scalar_image_type &sq_image) const
      {
        typedef typename functionnal_scalar_image_type::iterator out_it_type;
        typedef typename gradient_image_type::iterator in_it_type;
        typedef typename DataTraits<
            typename gradient_scalar_image_type::value_type>::
            float_accumulator_type accumulator_type;
        MORPHEE_ENTER_FUNCTION(
            "diffusion_tools::transform_gradient_to_squared_modulus");

        if (whole_gradient.size() == 0) {
          return RES_ERROR_BAD_ARG;
        }

        gradient_scalar_image_type const *im_scalar_gradient =
            whole_gradient[0];
        const image_coordinate_system &w_size = im_scalar_gradient->WSize();

        for (unsigned int i = 1; i < whole_gradient.size(); i++) {
          im_scalar_gradient = whole_gradient[i];
          if (w_size != im_scalar_gradient->WSize()) {
            MORPHEE_REGISTER_ERROR(
                "Some gradient scalar images' window size differ");
            return RES_ERROR;
          }
        }

        if (!t_CheckOffsetCompatible(*im_scalar_gradient, sq_image)) {
          MORPHEE_REGISTER_ERROR("Images should be offset compatible");
          return morphee::RES_ERROR_BAD_WINDOW_SIZE;
        }

        out_it_type itO          = sq_image.begin();
        const out_it_type itOend = sq_image.end();

        for (; itO != itOend; ++itO) {
          const offset_t off   = itO.Offset();
          accumulator_type acc = 0;
          for (in_it_type it  = whole_gradient.begin(),
                          itE = whole_gradient.end();
               it != itE; ++it) {
            const accumulator_type acc2 = it->pixelFromOffset(off);
            acc += op(acc2);
          }

          sq_image.pixelFromOffset(off) = acc;
        }

        return RES_OK;
      }
    };

    //! @} diffusion_tool

    /*!
     * @brief Class implementing the heat diffusion
     * In order to be used jointly with t_Diffusion
     * The heat diffusion is defined by: \n
     *  dI / dt = -k * div ( grad (I) )
     * The blurring process is the reverse of the heat equation:
     *  dI / dt = +k * div ( grad (I) )
     * K positive is equivalent to a bluring process, k negative is equivalent
     * to a deblurring process which is, in this particular case, highly
     * unstable (ill posed problem)
     *
     */
    template <class t_image> class diffusion_heat_operator
    {
    private:
      const F_DOUBLE _k_step;

    public:
      typedef t_image input_image_type;

      //! The type of the diffusion tool to be used
      typedef diffusion_tools<input_image_type> diffusion_tool_type;

      //! the resulting gradient image in its scalar form
      typedef typename diffusion_tool_type::gradient_scalar_image_type
          gradient_scalar_image_type;
      typedef typename diffusion_tool_type::functionnal_scalar_image_type
          functionnal_scalar_image_type;

      //! Tells the client that a copy should be done before and after the
      //! descent steps
      enum { copy_input = true, copy_output = true };

      //! The input image type for the processing step
      typedef functionnal_scalar_image_type processing_step_input_image_type;
      //! The output image type for the processing step
      typedef functionnal_scalar_image_type processing_step_output_image_type;

      diffusion_tool_type diffusion_op;

      diffusion_heat_operator(const F_SIMPLE &k_step) : _k_step(k_step)
      {
      }

      RES_C process_init(const input_image_type &im)
      {
        MORPHEE_ENTER_FUNCTION("diffusion_heat_operator::process_init");
        return diffusion_op.initFromImage(im);
      }

      RES_C process_step(const processing_step_input_image_type &im_in,
                         processing_step_output_image_type &im_functionnal_out)
      {
        MORPHEE_ENTER_FUNCTION("diffusion_heat_operator::process_step");
        RES_C res;

        // temporary needed images
        functionnal_scalar_image_type im_temp1 = im_in.template t_getSame<
            typename functionnal_scalar_image_type::value_type>();
        functionnal_scalar_image_type im_temp2 = im_in.template t_getSame<
            typename functionnal_scalar_image_type::value_type>();

        res = t_ImSetConstant(
            im_functionnal_out,
            typename processing_step_output_image_type::value_type(0));
        if (res != RES_OK) {
          MORPHEE_REGISTER_ERROR("Error while initialising the output image");
          return res;
        }

        // for each relevant dimension
        for (unsigned int dimension = 0;
             dimension < diffusion_op.displacement_offset.size(); dimension++) {
          // do not compute void dimensions
          if (diffusion_op.displacement_offset[dimension] == 0)
            continue;

          // compute a scalar gradient image
          res = diffusion_op.centered_difference_scheme_along_dimension(
              im_in, dimension, im_temp1);
          if (res != RES_OK) {
            MORPHEE_REGISTER_ERROR(
                "Error while computing the first derivative");
            return res;
          }

          // compute the gradient of the gradient (div)
          res = diffusion_op.centered_difference_scheme_along_dimension(
              im_temp1, dimension, im_temp2);
          if (res != RES_OK) {
            MORPHEE_REGISTER_ERROR(
                "Error while computing the second derivative");
            return res;
          }

          // cumulate the div into the output
          res =
              t_arithAddImage(im_temp2, im_functionnal_out, im_functionnal_out);
          if (res != RES_OK) {
            MORPHEE_REGISTER_ERROR("Error while cumulating the divergence");
            return res;
          }
        }

        res = t_arithMultImageConst(im_functionnal_out, _k_step,
                                    im_functionnal_out);
        if (res != RES_OK) {
          MORPHEE_REGISTER_ERROR("Error in t_arithMultImageConst call");
          return res;
        }
        return RES_OK;
      }
    };

    /*!
     * @brief Class implementing the Perona-Malik non linear diffusion
     *
     */
    template <class t_image> class diffusion_PeronaMalik_operator
    {
    public:
      typedef t_image input_image_type;

      //! The type of the diffusion tool to be used
      typedef diffusion_tools<input_image_type> diffusion_tool_type;

      //! the resulting gradient image in its scalar form
      typedef typename diffusion_tool_type::gradient_scalar_image_type
          gradient_scalar_image_type;
      typedef typename diffusion_tool_type::functionnal_scalar_image_type
          functionnal_scalar_image_type;
      typedef typename functionnal_scalar_image_type::value_type
          functionnal_scalar_value_type;

      //! Tells the client that a copy should be done before and after the
      //! descent steps
      enum { copy_input = true, copy_output = true };

      //! The input image type for the processing step
      typedef functionnal_scalar_image_type processing_step_input_image_type;
      //! The output image type for the processing step
      typedef functionnal_scalar_image_type processing_step_output_image_type;

    private:
      const F_DOUBLE _k_step;

      template <class Tin, class Tout> struct s_peronaMalikFunction {
        const F_DOUBLE lambda_sq;
        s_peronaMalikFunction(const F_DOUBLE lambda)
            : lambda_sq(lambda * lambda)
        {
        }
        Tout operator()(Tin squared_modulus_gradient) const
        {
          return static_cast<Tout>(1. /
                                   (1. + squared_modulus_gradient / lambda_sq));
        }
      };

      s_peronaMalikFunction<functionnal_scalar_value_type,
                            functionnal_scalar_value_type>
          g_op;
      diffusion_tool_type diffusion_op;

    public:
      diffusion_PeronaMalik_operator(const F_DOUBLE lambda,
                                     const F_DOUBLE k_step)
          : _k_step(k_step), g_op(lambda)
      {
      }

      RES_C process_init(const input_image_type &im)
      {
        MORPHEE_ENTER_FUNCTION("diffusion_PeronaMalik_operator::process_init");
        return diffusion_op.initFromImage(im);
      }

      RES_C process_step(const processing_step_input_image_type &im_in,
                         processing_step_output_image_type &im_functionnal_out)
      {
        MORPHEE_ENTER_FUNCTION("diffusion_PeronaMalik_operator::process_step");
        RES_C res;

        // temporary needed images
        functionnal_scalar_image_type im_temp1 =
            im_in.template t_getSame<functionnal_scalar_value_type>();
        functionnal_scalar_image_type im_temp2 =
            im_in.template t_getSame<functionnal_scalar_value_type>();

        res = t_ImSetConstant(
            im_functionnal_out,
            typename processing_step_output_image_type::value_type(0));
        if (res != RES_OK) {
          MORPHEE_REGISTER_ERROR("Error while initialising the output image");
          return res;
        }
        // for each relevant dimension
        for (unsigned int dimension = 0;
             dimension < diffusion_op.displacement_offset.size(); dimension++) {
          // do not compute void dimensions
          if (diffusion_op.displacement_offset[dimension] == 0)
            continue;

          // compute a scalar gradient image - we need to keep it till the
          // divergence
          res = diffusion_op.centered_difference_scheme_along_dimension(
              im_in, dimension, im_temp1);
          if (res != RES_OK) {
            MORPHEE_REGISTER_ERROR(
                "Error while computing the first derivative");
            return res;
          }

          // compute the step modulus of the gradient along the dimension
          res = t_ImUnaryOperation(
              im_temp1,
              operationSquare<functionnal_scalar_value_type,
                              functionnal_scalar_value_type>(),
              im_temp2);
          if (res != RES_OK) {
            MORPHEE_REGISTER_ERROR("Error while computing the modulus");
            return res;
          }

          res = diffusion_op.apply_scalar_functionnal_by_precomputing(
              im_temp2, g_op, im_temp2);
          if (res != RES_OK) {
            MORPHEE_REGISTER_ERROR("Error while computing the perona");
            return res;
          }

          res = t_arithMultImage(im_temp1, im_temp2, im_temp2);
          if (res != RES_OK) {
            MORPHEE_REGISTER_ERROR("Error while computing g(\nabla) * \nabla");
            return res;
          }

          // compute the gradient of the gradient (div)
          res = diffusion_op.centered_difference_scheme_along_dimension(
              im_temp2, dimension, im_temp1);
          if (res != RES_OK) {
            MORPHEE_REGISTER_ERROR(
                "Error while computing the second derivative");
            return res;
          }

          // cumulate the div into the output
          res =
              t_arithAddImage(im_temp1, im_functionnal_out, im_functionnal_out);
          if (res != RES_OK) {
            MORPHEE_REGISTER_ERROR("Error while cumulating the divergence");
            return res;
          }
        }

        res = t_arithMultImageConst(im_functionnal_out, _k_step,
                                    im_functionnal_out);
        if (res != RES_OK) {
          MORPHEE_REGISTER_ERROR("Error in t_arithMultImageConst call");
          return res;
        }

        return RES_OK;
      }
    };

    /*!
     * @brief Class implementing the Weickert non linear diffusion
     *
     */
    template <class t_image> class diffusion_Weickert_operator
    {
    public:
      typedef t_image input_image_type;

      //! The type of the diffusion tool to be used
      typedef diffusion_tools<input_image_type> diffusion_tool_type;

      //! the resulting gradient image in its scalar form
      typedef typename diffusion_tool_type::gradient_scalar_image_type
          gradient_scalar_image_type;
      typedef typename diffusion_tool_type::functionnal_scalar_image_type
          functionnal_scalar_image_type;
      typedef typename functionnal_scalar_image_type::value_type
          functionnal_scalar_value_type;

      //! Tells the client that a copy should be done before and after the
      //! descent steps
      enum { copy_input = true, copy_output = true };

      //! The input image type for the processing step
      typedef functionnal_scalar_image_type processing_step_input_image_type;
      //! The output image type for the processing step
      typedef functionnal_scalar_image_type processing_step_output_image_type;

    private:
      const F_DOUBLE _k_step;

      template <class Tin, class Tout> struct s_weickertFunction {
        const F_DOUBLE _lambda, _m, _cm;
        s_weickertFunction(F_DOUBLE lambda, F_DOUBLE m, F_DOUBLE cm)
            : _lambda(pow(lambda, m)), _m(m / 2.), _cm(cm)
        {
        } // precomputing some stuff there

        // Il faut en entrée |\nabla u|^2, sinon il faut changer l'équation
        Tout operator()(Tin squared_modulus_gradient) const
        {
          if (squared_modulus_gradient <= 0)
            return Tout(1.);
          return static_cast<Tout>(
              1. - exp(-_cm /
                       (pow(static_cast<double>(squared_modulus_gradient), _m) /
                        _lambda)));
        }
      };

      s_weickertFunction<functionnal_scalar_value_type,
                         functionnal_scalar_value_type>
          g_op;
      diffusion_tool_type diffusion_op;

    public:
      diffusion_Weickert_operator(const F_DOUBLE lambda, const F_DOUBLE m,
                                  const F_DOUBLE cm, const F_DOUBLE k_step)
          : _k_step(k_step), g_op(lambda, m, cm)
      {
      }

      RES_C process_init(const input_image_type &im)
      {
        MORPHEE_ENTER_FUNCTION("diffusion_PeronaMalik_operator::process_init");
        return diffusion_op.initFromImage(im);
      }

      RES_C process_step(const processing_step_input_image_type &im_in,
                         processing_step_output_image_type &im_functionnal_out)
      {
        MORPHEE_ENTER_FUNCTION("diffusion_PeronaMalik_operator::process_step");
        RES_C res;

        // temporary needed images
        functionnal_scalar_image_type im_temp1 =
            im_in.template t_getSame<functionnal_scalar_value_type>();
        functionnal_scalar_image_type im_temp2 =
            im_in.template t_getSame<functionnal_scalar_value_type>();

        res = t_ImSetConstant(
            im_functionnal_out,
            typename processing_step_output_image_type::value_type(0));
        if (res != RES_OK) {
          MORPHEE_REGISTER_ERROR("Error while initialising the output image");
          return res;
        }

        // for each relevant dimension
        for (unsigned int dimension = 0;
             dimension < diffusion_op.displacement_offset.size(); dimension++) {
          // do not compute void dimensions
          if (diffusion_op.displacement_offset[dimension] == 0)
            continue;

          // compute a scalar gradient image - we need to keep it till the
          // divergence
          res = diffusion_op.centered_difference_scheme_along_dimension(
              im_in, dimension, im_temp1);
          if (res != RES_OK) {
            MORPHEE_REGISTER_ERROR(
                "Error while computing the first derivative");
            return res;
          }

          // compute the step modulus of the gradient along the dimension
          res = t_ImUnaryOperation(
              im_temp1,
              operationSquare<functionnal_scalar_value_type,
                              functionnal_scalar_value_type>(),
              im_temp2);
          if (res != RES_OK) {
            MORPHEE_REGISTER_ERROR("Error while computing the modulus");
            return res;
          }

          res = diffusion_op.apply_scalar_functionnal_by_precomputing(
              im_temp2, g_op, im_temp2);
          if (res != RES_OK) {
            MORPHEE_REGISTER_ERROR("Error while computing the perona");
            return res;
          }

          res = t_arithMultImage(im_temp1, im_temp2, im_temp2);
          if (res != RES_OK) {
            MORPHEE_REGISTER_ERROR("Error while computing g(\nabla) * \nabla");
            return res;
          }

          // compute the gradient of the gradient (div)
          res = diffusion_op.centered_difference_scheme_along_dimension(
              im_temp2, dimension, im_temp1);
          if (res != RES_OK) {
            MORPHEE_REGISTER_ERROR(
                "Error while computing the second derivative");
            return res;
          }

          // cumulate the div into the output
          res =
              t_arithAddImage(im_temp1, im_functionnal_out, im_functionnal_out);
          if (res != RES_OK) {
            MORPHEE_REGISTER_ERROR("Error while cumulating the divergence");
            return res;
          }
        }

        res = t_arithMultImageConst(im_functionnal_out, _k_step,
                                    im_functionnal_out);
        if (res != RES_OK) {
          MORPHEE_REGISTER_ERROR("Error in t_arithMultImageConst call");
          return res;
        }
        return RES_OK;
      }
    };

    /*!
     * @brief A generic function to process diffusion over an image
     *
     * Have a look to t_HeatDiffusion to understand how to call this function.\n
     * The diffusions always can be written as:\n
     *  dI/dt = -div ( f(I, grad(I)) )
     *
     */
    template <class t_imageIn, class t_filter_type, class t_imageOut>
    RES_C t_DiffusionFunction(const t_imageIn &imIn, const UINT32 nosteps,
                              t_filter_type &filterOp, t_imageOut &imOut)

    {
      typedef typename t_filter_type::processing_step_input_image_type
          processing_step_input_image_type;
      typedef typename t_filter_type::processing_step_output_image_type
          processing_step_output_image_type;
      // typedef typename t_filter_type::functionnal_scalar_image_type
      // processing_step_input_image_type; typedef typename
      // t_filter_type::functionnal_scalar_image_type
      // processing_step_output_image_type;

      MORPHEE_ENTER_FUNCTION("t_DiffusionFunction");

      if (!imIn.isAllocated() || !imOut.isAllocated()) {
        MORPHEE_REGISTER_ERROR("Not allocated");
        return RES_NOT_ALLOCATED;
      }

      if (nosteps == 0)
        return RES_ERROR_BAD_ARG;

      RES_C res = t_alignWindows(imIn, imOut);
      if (res != RES_OK) {
        MORPHEE_REGISTER_ERROR("Unable to align windows");
        return res;
      }

      processing_step_input_image_type im_temp1 = imIn.template t_getSame<
          typename processing_step_input_image_type::value_type>();
      if (t_filter_type::copy_input) {
        res = t_ImCopy(imIn, im_temp1);
        if (res != RES_OK) {
          MORPHEE_REGISTER_ERROR("Unable to copy images");
          return res;
        }
      } else {
        throw morphee::MException(
            "Raffi: je n'ai pas encore décidé de ce que j'allais faire dans ce "
            "cas (hahahah). Si tu consultes les warnings tu verras que tu "
            "n'atteins pas le reste du code.");
      }

      processing_step_output_image_type im_temp2 = imIn.template t_getSame<
          typename processing_step_output_image_type::value_type>();

      res = filterOp.process_init(imIn);
      if (res != RES_OK) {
        MORPHEE_REGISTER_ERROR("An error occured during the initialization.");
        return res;
      }

      for (UINT32 i = 0; i < nosteps; i++) {
        res = filterOp.process_step(im_temp1, im_temp2);
        if (res != RES_OK) {
          MORPHEE_REGISTER_ERROR("Error in filterOp.process_step");
          return res;
        }

        im_temp1.swap(im_temp2);
      }

      if (t_filter_type::copy_output) {
        res = t_ImCopy(im_temp1, imOut);
        if (res != RES_OK) {
          MORPHEE_REGISTER_ERROR("Unable to copy images");
          return res;
        }
      } else {
        throw morphee::MException(
            "Raffi: je n'ai pas encore décidé de ce que j'allais faire dans ce "
            "cas (hahahah). Si tu consultes les warnings tu verras que tu "
            "n'atteins pas le reste du code.");
      }
      return RES_OK;
    }

    /*!
     * @brief A gradient function using the diffusion_tools class methods
     *
     * This gradient is not the same as the morphological one.
     * The output gradient's active window is set to the dimensions of the input
     * image's active window minus one along each directions, since the gradient
     * does not exist on the borders (the borders values are left unchanged.)
     * The size of the output image should be at least equal to the size of the
     * input image's active window.
     */
    template <class t_imageIn, class t_imageOut>
    RES_C t_Gradient(const t_imageIn &imIn, std::vector<t_imageOut *> &v_imOut)
    {
      typedef diffusion_tools<t_imageIn> diffusion_op_type;
      typedef
          typename diffusion_op_type::gradient_image_type gradient_image_type;

      MORPHEE_ENTER_FUNCTION("t_Gradient");

      diffusion_op_type diffusion_op;

      if (diffusion_op.initFromImage(imIn) != RES_OK) {
        MORPHEE_REGISTER_ERROR(
            "Error while initializing. Check if the image's window size is "
            "wide enough or if there is any active dimension");
      }

      gradient_image_type grad;

      // cleaning the output
      for (unsigned int i = 0; i < grad.size(); i++) {
        if (v_imOut[i] != 0) {
          delete v_imOut[i];
        }
      }
      v_imOut.clear();

      if (diffusion_op.compute_whole_gradient(imIn, grad) != RES_OK) {
        MORPHEE_REGISTER_ERROR("Error while computing the whole gradient");
      }

      for (unsigned int i = 0; i < grad.size(); i++) {
        t_imageOut *im_new = new t_imageOut(*grad[i]);
        if (t_ImCopy(*grad[i], *im_new) != RES_OK) {
          diffusion_op.clean_gradients(grad);
          MORPHEE_REGISTER_ERROR("Error while copying");
          return RES_ERROR;
        }
        v_imOut.push_back(im_new);
      }

      diffusion_op.clean_gradients(grad);
      return RES_OK;
    }

    /*!
     * @brief A concrete implementation of the heat diffusion filter
     *
     */
    template <class __imageIn, class __imageOut>
    RES_C t_HeatDiffusion(const __imageIn &imIn, const UINT32 nosteps,
                          const F_SIMPLE step_value, __imageOut &imOut)
    {
      diffusion_heat_operator<__imageIn> op(step_value);
      return t_DiffusionFunction(imIn, nosteps, op, imOut);
    }
    /*!
     * @brief A concrete implementation of the Perona-Malik diffusion filter
     *
     */
    template <class __imageIn, class __imageOut>
    RES_C t_PeronaMalikDiffusion(const __imageIn &imIn, const UINT32 nosteps,
                                 const F_SIMPLE step_value,
                                 const F_SIMPLE lambda, __imageOut &imOut)
    {
      diffusion_PeronaMalik_operator<__imageIn> op(lambda, step_value);
      return t_DiffusionFunction(imIn, nosteps, op, imOut);
    }

    /*!
     * @brief A concrete implementation of the Wieckert diffusion filter
     *
     */
    template <class __imageIn, class __imageOut>
    RES_C t_WeickertDiffusion(const __imageIn &imIn, const UINT32 nosteps,
                              const F_SIMPLE step_value, const F_SIMPLE lambda,
                              const F_SIMPLE m, const F_SIMPLE cm,
                              __imageOut &imOut)
    {
      diffusion_Weickert_operator<__imageIn> op(lambda, m, cm, step_value);
      return t_DiffusionFunction(imIn, nosteps, op, imOut);
    }
    //! @} diffusion_group

  } // namespace filters
} // namespace morphee

#endif /* __DIFFU_HPP__ */
