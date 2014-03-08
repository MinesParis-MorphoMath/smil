/*
 * Copyright (c) 2011, Matthieu FAESSEL and ARMINES
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of Matthieu FAESSEL, or ARMINES nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS AND CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#ifndef _MORPH_IMAGE_OPERATIONS_HPP
#define _MORPH_IMAGE_OPERATIONS_HPP

#include "DCore.h"
#include "Morpho/include/DStructuringElement.h"
#include "Morpho/include/DMorphoInstance.h"

#ifdef USE_OPEN_MP
#include <omp.h>
#endif // USE_OPEN_MP

namespace smil
{
  
    template <class T_in, class T_out=T_in>
    class unaryMorphImageFunctionBase : public imageFunctionBase<T_in>
    {
    public:
	unaryMorphImageFunctionBase(T_in _borderValue = ImDtTypes<T_in>::min())
	  : initialValue(_borderValue),
	    borderValue(_borderValue)
	    
	{
	}
	
	unaryMorphImageFunctionBase(T_in _borderValue, T_out _initialValue = ImDtTypes<T_out>::min())
	  : initialValue(_borderValue),
	    borderValue(_borderValue)
	{
	}
	
	~unaryMorphImageFunctionBase()
	{
	}
	
	typedef Image<T_in> imageInType;
	typedef typename imageInType::lineType lineInType;
	typedef typename imageInType::sliceType sliceInType;
	typedef typename imageInType::volType volInType;
	
	typedef Image<T_out> imageOutType;
	typedef typename imageOutType::lineType lineOutType;
	typedef typename imageOutType::sliceType sliceOutType;
	typedef typename imageOutType::volType volOutType;
	
	inline RES_T operator()(const imageInType &imIn, imageOutType &imOut, const StrElt &se) { return this->_exec(imIn, imOut, se); }
	
	virtual RES_T initialize(const imageInType &imIn, imageOutType &imOut, const StrElt &se);
	virtual RES_T finalize(const imageInType &imIn, imageOutType &imOut, const StrElt &se);
	virtual RES_T _exec(const imageInType &imIn, imageOutType &imOut, const StrElt &se);
	
	virtual RES_T processImage(const imageInType &imIn, imageOutType &imOut, const StrElt &se);
	virtual inline void processSlice(sliceInType linesIn, sliceOutType linesOut, size_t &lineNbr, const StrElt &se);
	virtual inline void processLine(lineInType pixIn, lineOutType pixOut, size_t &pixNbr, const StrElt &se);
	virtual inline void processPixel(size_t &pointOffset, vector<int>::iterator dOffset, vector<int>::iterator dOffsetEnd);
	
	static bool isInplaceSafe(const StrElt &se) { return false; }
    protected:
	  size_t imSize[3];
	  volInType slicesIn;
	  volOutType slicesOut;
	  lineInType pixelsIn;
	  lineOutType pixelsOut;
	  
	  size_t curSlice;
	  size_t curLine;
	  size_t curPixel;
	  
	  vector<IntPoint> sePoints;
	  UINT sePointNbr;
	  vector<int> relativeOffsets;
	  
	  int se_xmin;
	  int se_xmax;
	  int se_ymin;
	  int se_ymax;
	  int se_zmin;
	  int se_zmax;
    public:
	T_out initialValue;
	T_in borderValue;
    };




    template <class T, class lineFunction_T>
    class unaryMorphImageFunction : public unaryMorphImageFunctionBase<T>
    {
      public:
	typedef imageFunctionBase<T> parentClass;
	typedef Image<T> imageType;
	typedef typename imageType::lineType lineType;
	typedef typename imageType::sliceType sliceType;
	typedef typename imageType::volType volType;
	
	unaryMorphImageFunction(T border=ImDtTypes<T>::min()) 
	  : unaryMorphImageFunctionBase<T>(border, border) 
	{
	}
	
	
	static bool isInplaceSafe(const StrElt &se);
	
      protected:
	virtual RES_T _exec(const imageType &imIn, imageType &imOut, const StrElt &se);
	
	virtual RES_T _exec_single(const imageType &imIn, imageType &imOut, const StrElt &se);
	virtual RES_T _exec_single_generic(const imageType &imIn, imageType &imOut, const StrElt &se); 				// Inplace unsafe !!
	virtual RES_T _exec_single_hexagonal_SE(const imageType &imIn, imageType &imOut); 					// Inplace safe
	virtual RES_T _exec_single_square_SE(const imageType &imIn, imageType &imOut); 						// Inplace unsafe !!
	virtual RES_T _exec_single_cube_SE(const imageType &imIn, imageType &imOut); 						// Inplace unsafe !!
	virtual RES_T _exec_single_horizontal_2points(const imageType &imIn, int dx, imageType &imOut, bool oddLines=false);	// Inplace safe
	virtual RES_T _exec_single_vertical_2points(const imageType &imIn, int dx, imageType &imOut);				// Inplace unsafe !!
	virtual RES_T _exec_single_horizontal_segment(const imageType &imIn, int xsize, imageType &imOut);			// Inplace safe
	virtual RES_T _exec_single_vertical_segment(const imageType &imIn, imageType &imOut);					// Inplace unsafe !!
	virtual RES_T _exec_single_cross(const imageType &imIn, imageType &imOut);						// Inplace unsafe !!
	virtual RES_T _exec_single_cross_3d(const imageType &imIn, imageType &imOut);
	virtual RES_T _exec_single_depth_segment(const imageType &imIn, int zsize, imageType &imOut); 				// Inplace safe
        virtual RES_T _exec_rhombicuboctahedron(const imageType &imIn, imageType &imOut, unsigned int size);			// Inplace unsafe !!

	

	lineFunction_T lineFunction;
	
	lineType borderBuf, cpBuf;
	size_t lineLen;
	
	inline void _extract_translated_line(const Image<T> *imIn, const int &x, const int &y, const int &z, lineType outBuf);
	inline void _exec_shifted_line(const lineType inBuf1, const lineType inBuf2, const int &dx, const int &lineLen, lineType outBuf, lineType tmpBuf=NULL);
	inline void _exec_shifted_line(const lineType inBuf, const int &dx, const int &lineLen, lineType outBuf, lineType tmpBuf=NULL)
	{
	    return _exec_shifted_line(inBuf, inBuf, dx, lineLen, outBuf, tmpBuf);
	}
	inline void _exec_shifted_line_2ways(const lineType inBuf1, const lineType inBuf2, const int &dx, const int &lineLen, lineType outBuf, lineType tmpBuf=NULL);
	inline void _exec_shifted_line_2ways(const lineType inBuf, const int &dx, const int &lineLen, lineType outBuf, lineType tmpBuf=NULL)
	{
	    return _exec_shifted_line_2ways(inBuf, inBuf, dx, lineLen, outBuf, tmpBuf);
	}
	inline void _exec_line(const lineType inBuf, const Image<T> *imIn, const int &x, const int &y, const int &z, lineType outBuf);
    };


} // namespace smil

# endif // _MORPH_IMAGE_OPERATIONS_HPP
