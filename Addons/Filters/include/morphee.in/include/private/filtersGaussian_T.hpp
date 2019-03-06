#ifndef __MORPHEE_FILTERS_GAUSSIAN_T_HPP__
#define __MORPHEE_FILTERS_GAUSSIAN_T_HPP__

#include <cmath>
#include <algorithm>
#include <morphee/image/include/morpheeImage.hpp>
#include <morphee/image/include/private/imageColorSpaceTransform_T.hpp>
#include <morphee/selement/include/private/selementImage_T.hpp>
#include <morphee/selement/include/morpheeSelement.hpp>
#include <morphee/filters/include/private/filtersConvolve_T.hpp>

namespace morphee
{
  namespace filters
  {
    //! @addtogroup filters_group
    //! @{

    //! Private: abysmally sub-optimal gaussian computing function
    template <typename T>
    T t_computeGaussian(INT32 deltaX, INT32 deltaY, INT32 deltaZ, T r)
    {
      return exp(-static_cast<float>(deltaX * deltaX + deltaY * deltaY +
                                     deltaZ * deltaZ) /
                 (float) (2. * r * r));
    }
    //! Function that fills an ImageKernel with the right values for a gaussian
    template <typename T>
    RES_C t_fillGaussianKernel(selement::ConvolutionKernel<T> &kernel)
    {
      typename selement::ConvolutionKernel<T>::iterator it, iend;
      INT32 kx = (INT32) kernel.getXCenter();
      INT32 ky = (INT32) kernel.getYCenter();
      INT32 kz = (INT32) kernel.getZCenter();

      T sum = 0, t_temp;

      // arbitrary definition of r
      F_SIMPLE maxSize = static_cast<F_SIMPLE>(std::max(
          kernel.getXSize(), std::max(kernel.getYSize(), kernel.getZSize())));
      T r              = static_cast<T>(maxSize / 6.);

      for (it = kernel.begin(), iend = kernel.end(); it != iend; ++it) {
        t_temp = t_computeGaussian(it.getX() - kx, it.getY() - ky,
                                   it.getZ() - kz, r);
        *it    = t_temp;
        sum += t_temp;
      }
      if (sum == 0)
        return RES_ERROR;

      // normalization
      for (it = kernel.begin(), iend = kernel.end(); it != iend; ++it) {
        *it /= sum;
      }

      return RES_OK;
    }

    //! Filter an image by a gaussian
    template <typename T1, typename T2>
    RES_C t_ImGaussianFilter_Slow(const Image<T1> &imIn, INT32 filterRadius,
                                  Image<T2> &imOut)
    {
      typedef typename DataTraits<T1>::float_accumulator_type T1acc;
      typedef Image<T1> image_type;
      typedef typename image_type::coordinate_system coordinate_system;

      const INT32 kernelSizeDim = filterRadius * 2 + 1;

      coordinate_system kernel_size(1);
      coordinate_system kernel_center(0);
      const coordinate_system &im_wsize = imIn.WSize();

      for (unsigned int k = 0; k < im_wsize.getDimension(); k++) {
        if (im_wsize[k] > 1) {
          kernel_size[k]   = kernelSizeDim;
          kernel_center[k] = kernelSizeDim / 2;
        }
      }

      // INT32 centerX,centerY=0,centerZ=0;

      Image<T1acc> kernelIm;

#if 0
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

				kernelSizeZ = kernelSizeY;
				centerZ		= kernelSizeZ/2;
			}
#endif

      kernelIm.setSize(kernel_size);
      kernelIm.allocateImage();

      // pour faire taire valgrind:
#ifndef NDEBUG
      t_ImSetConstant(kernelIm, 0);
#endif
      // selement::ImageKernel<T1acc> kernel(kernelIm,
      // centerX,centerY,centerZ,selement::SEBorderTypeClip);
      selement::ConvolutionKernel<T1acc> kernel(kernelIm, kernel_center,
                                                selement::SEBorderTypeMirrored);

      RES_C res = t_fillGaussianKernel(kernel);
      if (res != RES_OK)
        return res;

      return t_ImConvolve(imIn, kernel, (T2) 0, (T2) 1, imOut);
    }

    //! Filter an image by a gaussian
    template <typename T1, typename T2>
    RES_C t_ImGaussianFilter_Separable(const Image<T1> &imIn,
                                       INT32 filterRadius, Image<T2> &imOut)
    {
      typedef typename DataTraits<T1>::float_accumulator_type T1acc;

      typedef Image<T1> image_type;
      typedef typename image_type::coordinate_system coordinate_system;

      const INT32 kernelSizeDim = filterRadius * 2 + 1;

      const coordinate_system &im_wsize = imIn.WSize();

      Image<T2> imTmp = imOut.getSame();
      t_ImCopy(imIn, imTmp);

      // pour compter le nombre de permutations
      int counter = 0;
      RES_C res   = RES_OK;

      // on construit un noyau plan suivant chaque direction significative
      for (unsigned int k = 0; k < imIn.getCoordinateDimension(); k++) {
        coordinate_system kernel_size(1);
        coordinate_system kernel_center(0);

        if (!(im_wsize[k] > 1))
          continue;

        kernel_size[k]   = kernelSizeDim;
        kernel_center[k] = kernelSizeDim / 2;

        // creation de l'image 1D à partir duquel on va construire le noyau
        Image<T1acc> kernel_image_1D;
        kernel_image_1D.setSize(kernel_size);
        kernel_image_1D.allocateImage();

#ifndef NDEBUG
        // pour faire taire valgrind:
        t_ImSetConstant(kernel_image_1D, 0);
#endif
        // creation du noyau
        selement::ConvolutionKernel<T1acc> kernel_1D(
            kernel_image_1D, kernel_center, selement::SEBorderTypeMirrored);

        // remplissage du noyau
        res = t_fillGaussianKernel(kernel_1D);
        if (res != RES_OK)
          break;

        // convolution
        res = t_ImConvolve(imTmp, kernel_1D, (T2) 0, (T2) 1, imOut);
        if (res != RES_OK)
          break;

        // swap des images (normalement pas couteux)
        counter++;
        imTmp.swap(imOut);
      }

      if (counter % 2 != 0) {
        imTmp.swap(imOut);
      } else {
        // Raffi: pas sûr de pouvoir se débarraser de cette copie de fin.
        t_ImCopy(imTmp, imOut);
      }

      return res;

      // selement::ImageKernel<T1acc> kernel(kernelIm,
      // centerX,centerY,centerZ,selement::SEBorderTypeClip);

#if 0

			if(t_ImDimension(imIn)>=2)
			{
				kernelSizeY	= kernelSizeX;
				centerY		= kernelSizeY/2;

				Image<T1acc> kernelIm1D_Y;
				kernelIm1D_Y.setSize(1,kernelSizeY,1);
				kernelIm1D_Y.allocateImage();
				selement::ConvolutionKernel<T1acc> kernel1D_Y(kernelIm1D_Y, 0,centerY,0,selement::SEBorderTypeMirrored);

				RES_C res=t_fillGaussianKernel(kernel1D_Y); // inutile ? ImCopy ?
				if(res != RES_OK)
					return res;

				res = t_ImConvolve(imOut,kernel1D_Y,(T2)0,(T2)1,imTmp);
				if(res != RES_OK)
					return res;
			}
			if(t_ImDimension(imIn)< 3)
			{
				return t_ImCopy(imTmp,imOut);
			} 
			else
			{
				kernelSizeZ	= filterRadius*2+1;//kernelSizeY;
				centerZ		= kernelSizeZ/2;

				Image<T1acc> kernelIm1D_Z;
				kernelIm1D_Z.setSize(1,1,kernelSizeZ);
				kernelIm1D_Z.allocateImage();

				selement::ConvolutionKernel<T1acc> kernel1D_Z(kernelIm1D_Z, 0,0,centerZ,selement::SEBorderTypeMirrored);

				RES_C res=t_fillGaussianKernel(kernel1D_Z); // inutile ? ImCopy ?
				if(res != RES_OK)
					return res;

				return t_ImConvolve(imTmp,kernel1D_Z,(T2)0,(T2)1,imOut);
			}
#endif
    }

    template <typename T1, typename T2> struct s_GaussianFilterHelper {
      RES_C operator()(const Image<T1> &imIn, INT32 filterRadius,
                       Image<T2> &imOut)
      {
        return t_ImGaussianFilter_Separable(imIn, filterRadius, imOut);
      }
    };
    template <typename T1, typename T2>
    struct s_GaussianFilterHelper<pixel_3<T1>, pixel_3<T2>> {
      RES_C operator()(const Image<pixel_3<T1>> &imIn, INT32 filterRadius,
                       Image<pixel_3<T2>> &imOut)
      {
        Image<T1> imTmp1 = imIn.template t_getSame<T1>();
        Image<T1> imTmp2 = imTmp1.getSame();
        Image<T1> imTmp3 = imTmp2.getSame();

        Image<T2> imOutTmp1 = imOut.template t_getSame<T1>();
        Image<T2> imOutTmp2 = imOutTmp1.getSame();
        Image<T2> imOutTmp3 = imOutTmp2.getSame();

        RES_C res = t_colorSplitTo3(imIn, imTmp1, imTmp2, imTmp3);
        if (res != RES_OK)
          return res;

        s_GaussianFilterHelper<T1, T2> opGauss;
        res = opGauss(imTmp1, filterRadius, imOutTmp1);
        if (res != RES_OK)
          return res;
        res = opGauss(imTmp2, filterRadius, imOutTmp2);
        if (res != RES_OK)
          return res;
        res = opGauss(imTmp3, filterRadius, imOutTmp3);
        if (res != RES_OK)
          return res;
        return t_colorComposeFrom3(imOutTmp1, imOutTmp2, imOutTmp3, imOut);
      }
    };

    template <typename T1, typename T2>
    RES_C t_ImGaussianFilter(const Image<T1> &imIn, INT32 filterRadius,
                             Image<T2> &imOut)
    {
      return s_GaussianFilterHelper<T1, T2>()(imIn, filterRadius, imOut);
    }

    //! @}

  } // namespace filters
} // namespace morphee

#endif
