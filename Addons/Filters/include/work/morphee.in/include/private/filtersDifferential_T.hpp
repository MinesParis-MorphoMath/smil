#ifndef __MORPHEE_FILTERS_DIFFERENTIAL_T_HPP__
#define __MORPHEE_FILTERS_DIFFERENTIAL_T_HPP__

#include <cmath>
#include <morphee/image/include/morpheeImage.hpp>
#include <morphee/selement/include/private/selementImage_T.hpp>
#include <morphee/selement/include/morpheeSelement.hpp>
#include <morphee/filters/include/private/filtersConvolve_T.hpp>

namespace morphee
{
  namespace filters
  {
    //! @addtogroup filters_group
    //! @{

    /*! Filter an image by a gaussian
     * The kernel used is a cube with edges of length 3 in the relevant
     * dimension.
     */
    template <typename image_type1, typename image_type2>
    RES_C t_ImLaplacianFilter(const image_type1 &imIn, image_type2 &imOut)
    {
      typedef typename image_type1::value_type value_type1;
      typedef typename image_type2::value_type value_type2;
      typedef typename image_type1::coordinate_system coordinate_system1;
      typedef typename image_type2::coordinate_system coordinate_system2;
      typedef typename DataTraits<value_type1>::float_accumulator_type T1acc;

      MORPHEE_ENTER_FUNCTION("t_ImLaplacianFilter<" +
                             NameTraits<value_type1>::name() + "," +
                             NameTraits<value_type2>::name() + ">");

      if (t_ImDimension(imIn) != t_ImDimension(imOut)) {
        MORPHEE_REGISTER_ERROR("Input images have different dimension");
        return morphee::RES_ERROR_BAD_ARG;
      }

      coordinate_system1 kernel_size(1);
      coordinate_system1 kernel_center(0);
      const coordinate_system1 &w_size = imIn.WSize();

      bool b_are_all_too_small = true;
      for (unsigned int k = 0; k < imIn.getCoordinateDimension(); k++) {
        if (w_size[k] > 2) {
          b_are_all_too_small = false;
          kernel_size[k]      = 3;
          kernel_center[k]    = 1;
        }
      }

      if (b_are_all_too_small) {
        MORPHEE_REGISTER_ERROR(
            "Specified dimension is above the input imge's one");
        return morphee::RES_ERROR_BAD_WINDOW_SIZE;
      }

      RES_C res = t_alignWindows(imIn, imOut);
      if (res != RES_OK) {
        MORPHEE_REGISTER_ERROR("Error during the alignment of images");
        return res;
      }

      Image<T1acc> kernel_image(kernel_size);
      kernel_image.allocateImage();

      // Raffi: je pense qu'on peut faire le remplissage de l'image, puis
      // instancier le noyau de convolution
      res = t_ImSetConstant(kernel_image, T1acc(1));
      if (res != RES_OK) {
        MORPHEE_REGISTER_ERROR(
            "Error during the initialization of the kernel's image");
        return res;
      }
      const coord_t count               = t_SizePixel(kernel_image);
      kernel_image.pixelFromOffset(t_GetOffsetFromCoords(
          kernel_image, kernel_center)) = -static_cast<T1acc>(count - 1);
      selement::ConvolutionKernel<T1acc> kernel(kernel_image, kernel_center,
                                                selement::SEBorderTypeMirrored);
      return t_ImConvolve(imIn, kernel, value_type2(0), value_type2(1), imOut);

#if 0
			MORPHEE_ENTER_FUNCTION("t_ImLaplacianFilter<"+NameTraits<T1>::name()+","+NameTraits<T2>::name()+">");
			typedef typename DataTraits<T1>::float_accumulator_type T1acc;
			INT32 kernelSizeX = 3;
			INT32 kernelSizeY = 1;
			INT32 kernelSizeZ = 1;

			INT32 centerX,centerY=0,centerZ=0;
			
			Image<T1acc> kernelIm;

			centerX			= kernelSizeX/2;

			if(t_ImDimension(imIn)==2)
			{
				kernelSizeY	= kernelSizeX;
				centerY		= kernelSizeY/2;
			}
			else if(t_ImDimension(imIn)==3)
			{
				kernelSizeY	= kernelSizeX;
				centerY		= kernelSizeY/2;

				kernelSizeZ = kernelSizeZ;
				centerZ		= kernelSizeZ/2;
			}

			kernelIm.setSize(kernelSizeX,kernelSizeY,kernelSizeZ);
			kernelIm.allocateImage();

			// pour faire taire valgrind:
#ifndef NDEBUG
			t_ImSetConstant(kernelIm,0);
#endif
			//selement::ImageKernel<T1acc> kernel(kernelIm, centerX,centerY,centerZ,selement::SEBorderTypeClip);
			selement::ConvolutionKernel<T1acc> kernel(kernelIm, centerX,centerY,centerZ,selement::SEBorderTypeMirrored);

			typename selement::ConvolutionKernel<T1acc>::iterator it,iend;
			int count=0;
			for(it=kernel.begin(), iend=kernel.end(); it != iend ; ++it, ++count)
			{
				*it		= 1;
			}
			kernel.pixelFromOffset(t_GetOffsetFromCoords(kernel,centerX,centerY,centerZ))=-static_cast<T1acc>( count-1 );

			return t_ImConvolve(imIn,kernel,(T2)0,(T2)1,imOut);
#endif
    }

    /*! Gradient by convolution over the specified dimension
     *
     * The input images' dimension of interest should be of size > 2
     */
    template <typename image_type1, typename image_type2>
    RES_C t_ImDifferentialGradient(const image_type1 &imIn, UINT16 dimension,
                                   image_type2 &imOut)
    {
      typedef typename image_type1::value_type value_type1;
      typedef typename image_type2::value_type value_type2;
      typedef typename image_type1::coordinate_system coordinate_system1;
      typedef typename image_type2::coordinate_system coordinate_system2;
      typedef typename DataTraits<value_type1>::float_accumulator_type T1acc;

      MORPHEE_ENTER_FUNCTION("t_ImDifferentialGradient<" +
                             NameTraits<value_type1>::name() + "," +
                             NameTraits<value_type2>::name() + ">");

      if (dimension > coordinate_system1::dimension) {
        MORPHEE_REGISTER_ERROR(
            "Specified dimension is above the input imge's one");
        return morphee::RES_ERROR_BAD_ARG;
      }

      if (imIn.WSize()[dimension] <= 2) {
        MORPHEE_REGISTER_ERROR(
            "Input image's dimension are too small for this operation");
        return morphee::RES_ERROR_BAD_ARG;
      }

      coordinate_system1 kernel_size(1);
      coordinate_system1 kernel_center(0);
      kernel_size[dimension]   = 3;
      kernel_center[dimension] = 1;

      Image<T1acc> kernel_image;

      kernel_image.setSize(kernel_size);
      kernel_image.allocateImage();

      // pour faire taire valgrind:
#ifndef NDEBUG
      t_ImSetConstant(kernel_image, 0);
#endif

      // Raffi: je pense que c'est exactement pareil que ce qui était fait
      kernel_image.pixelFromOffset(0) = -1;
      kernel_image.pixelFromOffset(1) = 0;
      kernel_image.pixelFromOffset(2) = 1;

      // Kids don't do this at home
      assert(*(kernel_image.rawPointer()) == -1);
      assert(*(kernel_image.rawPointer() + 1) == 0);
      assert(*(kernel_image.rawPointer() + 2) == 1);

      selement::ConvolutionKernel<T1acc> kernel(kernel_image, kernel_center,
                                                selement::SEBorderTypeMirrored);

      return t_ImConvolve(imIn, kernel, value_type2(0), value_type2(2), imOut);
    }

    //! Special case of t_ImDifferentialGradient along X
    template <typename image_type1, typename image_type2>
    RES_C t_ImDifferentialGradientX(const image_type1 &imIn, image_type2 &imOut)
    {
#if 0
			MORPHEE_ENTER_FUNCTION("t_ImDifferentialGradientX<"+NameTraits<T1>::name()+","+NameTraits<T2>::name()+">");
			typedef typename DataTraits<T1>::float_accumulator_type T1acc;
			INT32 kernelSizeX = 3;
			INT32 kernelSizeY = 1;
			INT32 kernelSizeZ = 1;

			INT32 centerX,centerY=0,centerZ=0;
			
			Image<T1acc> kernelIm;

			centerX			= kernelSizeX/2;

			kernelIm.setSize(kernelSizeX,kernelSizeY,kernelSizeZ);
			kernelIm.allocateImage();

			// pour faire taire valgrind:
#ifndef NDEBUG
			t_ImSetConstant(kernelIm,0);
#endif
			//selement::ImageKernel<T1acc> kernel(kernelIm, centerX,centerY,centerZ,selement::SEBorderTypeClip);
			selement::ConvolutionKernel<T1acc> kernel(kernelIm, centerX,centerY,centerZ,selement::SEBorderTypeMirrored);

			// Kids don't do this at home
			*(kernel.rawPointer()) = -1;
			*(kernel.rawPointer()+1) = 0;
			*(kernel.rawPointer()+2) = 1;

			return t_ImConvolve(imIn,kernel,(T2)0,(T2)2,imOut);
#endif
      return t_ImDifferentialGradient(imIn, 0, imOut);
    }

    template <typename image_type1, typename image_type2>
    RES_C t_ImDifferentialGradientY(const image_type1 &imIn, image_type2 &imOut)
    {
#if 0
			MORPHEE_ENTER_FUNCTION("t_ImDifferentialGradientY<"+NameTraits<T1>::name()+","+NameTraits<T2>::name()+">");
			typedef typename DataTraits<T1>::float_accumulator_type T1acc;
			INT32 kernelSizeX = 1;
			INT32 kernelSizeY = 3;
			INT32 kernelSizeZ = 1;

			INT32 centerX=0,centerY=0,centerZ=0;
			
			Image<T1acc> kernelIm;

			centerY			= kernelSizeY/2;

			kernelIm.setSize(kernelSizeX,kernelSizeY,kernelSizeZ);
			kernelIm.allocateImage();

			// pour faire taire valgrind:
#ifndef NDEBUG
			t_ImSetConstant(kernelIm,0);
#endif
			//selement::ImageKernel<T1acc> kernel(kernelIm, centerX,centerY,centerZ,selement::SEBorderTypeClip);
			selement::ConvolutionKernel<T1acc> kernel(kernelIm, centerX,centerY,centerZ,selement::SEBorderTypeMirrored);

			// Kids don't do this at home
			*(kernel.rawPointer()) = -1;
			*(kernel.rawPointer()+1) = 0;
			*(kernel.rawPointer()+2) = 1;

			return t_ImConvolve(imIn,kernel,(T2)0,(T2)2,imOut);
#endif
      return t_ImDifferentialGradient(imIn, 1, imOut);
    }

    //! @} filters_group

  } // namespace filters
} // namespace morphee

#endif //__MORPHEE_FILTERS_DIFFERENTIAL_T_HPP__
