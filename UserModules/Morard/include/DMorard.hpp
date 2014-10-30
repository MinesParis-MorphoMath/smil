#ifndef __D_MORARD_HPP__
#define __D_MORARD_HPP__

#include "Core/include/DCore.h"
#include "FastBilateralFilter.hpp"

namespace smil
{

	template<class T>
	RES_T  fastBilateralFilter(const Image<T> & imIn, UINT8 Method, UINT8 nS, UINT32 EctS, UINT32 EctG, Image<T> &imOut)
	{
		ASSERT_ALLOCATED(&imIn, &imOut);
		ASSERT_SAME_SIZE(&imIn, &imOut);
		
		ImageFreezer freeze(imOut);
		
		copy(imIn, imOut);
		size_t s[3];
		imIn.getSize(s);
		
		typename ImDtTypes<T>::lineType bufferOut = imOut.getPixels();

		_fastBilateralFilter(bufferOut, s[0], s[1], s[2], Method, Method, nS, EctS, EctS, EctS, EctG, EctG, EctG);
		
		return RES_OK;
	}



} // smil


#endif