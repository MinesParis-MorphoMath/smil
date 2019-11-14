/*
 * Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES
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

#include "Core/include/DCore.h"
#include "Morpho/include/DStructuringElement.h"
#include "Morpho/include/DMorphoInstance.h"

#ifdef USE_OPEN_MP
#include <omp.h>
#endif // USE_OPEN_MP

namespace smil
{
    /**
    * @ingroup Morpho 
    * @{
    */
  
    /**
     * Base morpho operator class.
     * 
     * @demo{custom_morpho_operator.py}
     */
    template <class T_in, class T_out=T_in>
    class MorphImageFunctionBase 
#ifndef SWIG
    : public imageFunctionBase<T_in>
#endif // SWIG
    {
    public:
        typedef Image<T_in> imageInType;
        typedef typename ImDtTypes<T_in>::lineType lineInType;
        typedef typename ImDtTypes<T_in>::sliceType sliceInType;
        typedef typename ImDtTypes<T_in>::volType volInType;
        
        typedef Image<T_out> imageOutType;
        typedef typename ImDtTypes<T_out>::lineType lineOutType;
        typedef typename ImDtTypes<T_out>::sliceType sliceOutType;
        typedef typename ImDtTypes<T_out>::volType volOutType;
        
        MorphImageFunctionBase(T_in _borderValue = ImDtTypes<T_in>::min(), T_out _initialValue = ImDtTypes<T_out>::min())
          : initialValue(_initialValue),
            borderValue(_borderValue)
        {
        }
        
        virtual ~MorphImageFunctionBase()
        {
        }
        
        
        RES_T operator()(const imageInType &imIn, imageOutType &imOut, const StrElt &se=DEFAULT_SE) 
        { 
            return this->_exec(imIn, imOut, se); 
        }
        RES_T operator()(const imageInType &imIn, const StrElt &se=DEFAULT_SE) 
        { 
            return this->_exec(imIn, se); 
        }
        
        virtual RES_T initialize(const imageInType &imIn, imageOutType &imOut, const StrElt &se);
        virtual RES_T finalize(const imageInType &imIn, imageOutType &imOut, const StrElt &se);
        virtual RES_T _exec(const imageInType &imIn, imageOutType &imOut, const StrElt &se);
        virtual RES_T _exec(const imageInType &imIn, const StrElt &se);
        
        virtual RES_T processImage(const imageInType &imIn, imageOutType &imOut, const StrElt &se);
        virtual inline void processSlice(sliceInType linesIn, sliceOutType linesOut, size_t &lineNbr, const StrElt &se);
        virtual inline void processLine(lineInType pixIn, lineOutType pixOut, size_t &pixNbr, const StrElt &se);
        virtual inline void processPixel(size_t pointOffset, vector<int> &dOffsets);
        
        static bool isInplaceSafe(const StrElt &/*se*/) { return false; }
        
        const Image<T_in> *imageIn;
        Image<T_out> *imageOut;
        
    protected:
          size_t imSize[3];
          volInType slicesIn;
          volOutType slicesOut;
          lineInType pixelsIn;
          lineOutType pixelsOut;
          
          vector<IntPoint> sePoints;
          UINT sePointNbr;
          vector<int> relativeOffsets;
          
          int se_xmin;
          int se_xmax;
          int se_ymin;
          int se_ymax;
          int se_zmin;
          int se_zmax;

          bool oddSe;
    public:
        T_out initialValue;
        T_in borderValue;
    };




    template <class T_in, class lineFunction_T, class T_out=T_in, bool Enable=IS_SAME(T_in, T_out) >
    class MorphImageFunction : public MorphImageFunctionBase<T_in, T_out>
    {
      public:
        typedef MorphImageFunctionBase<T_in, T_out> parentClass;
        
        typedef Image<T_in> imageInType;
        typedef typename ImDtTypes<T_in>::lineType lineInType;
        typedef typename ImDtTypes<T_in>::sliceType sliceInType;
        typedef typename ImDtTypes<T_in>::volType volInType;
        
        typedef Image<T_out> imageOutType;
        typedef typename ImDtTypes<T_out>::lineType lineOutType;
        typedef typename ImDtTypes<T_out>::sliceType sliceOutType;
        typedef typename ImDtTypes<T_out>::volType volOutType;
        
        
        MorphImageFunction(T_in border=ImDtTypes<T_in>::min(), T_out initialValue = ImDtTypes<T_out>::min()) 
          : MorphImageFunctionBase<T_in, T_out>(border, initialValue),
            lineLen(0)
        {
        }
        
        
        virtual RES_T initialize(const imageInType &imIn, imageOutType &imOut, const StrElt &se);
        virtual RES_T finalize(const imageInType &imIn, imageOutType &imOut, const StrElt &se);
        
        static bool isInplaceSafe(const StrElt &se);
        
      protected:
        virtual RES_T _exec(const imageInType &imIn, imageOutType &imOut, const StrElt &se);
        virtual RES_T _exec_single(const imageInType &/*imIn*/, imageOutType &/*imOut*/, const StrElt &/*se*/) { return RES_OK; }
        

        lineFunction_T lineFunction;
        
        lineInType borderBuf, cpBuf;
        size_t lineLen;
        
        inline void _extract_translated_line(const Image<T_in> *imIn, const int &x, const int &y, const int &z, lineInType outBuf)
        {
            if (z<0 || z>=int(imIn->getDepth()) || y<0 || y>=int(imIn->getHeight()))
              copyLine<T_in>(borderBuf, lineLen, outBuf);
            //     memcpy(outBuf, borderBuf, lineLen*sizeof(T));
            else
                shiftLine<T_in>(imIn->getSlices()[z][y], x, lineLen, outBuf, this->borderValue);
        }
        
        inline void _exec_line(const lineInType inBuf, const Image<T_in> *imIn, const int &x, const int &y, const int &z, lineOutType outBuf)
        {
            _extract_translated_line(imIn, x, y, z, cpBuf);
            lineFunction._exec(inBuf, cpBuf, lineLen, outBuf);
        }
        
        inline void _exec_shifted_line(const lineInType inBuf1, const lineInType inBuf2, const int &dx, const int &lineLen, lineOutType outBuf, lineInType tmpBuf)
        {
            if (tmpBuf==NULL)
              tmpBuf = cpBuf;
            shiftLine<T_in>(inBuf2, dx, lineLen, tmpBuf, this->borderValue);
            lineFunction._exec(inBuf1, tmpBuf, lineLen, outBuf);
        }
        
        template <class T1, class T2>
        inline void _exec_shifted_line(const typename ImDtTypes<T1>::lineType inBuf1, const typename ImDtTypes<T1>::lineType inBuf2, const int &dx, const int &lineLen, typename ImDtTypes<T2>::lineType outBuf)
        {
            return _exec_shifted_line(inBuf1, inBuf2, dx, lineLen, outBuf, cpBuf);
        }
        
        template <class T1, class T2>
        inline void _exec_shifted_line(const typename ImDtTypes<T1>::lineType inBuf, const int &dx, const int &lineLen, typename ImDtTypes<T2>::lineType outBuf, typename ImDtTypes<T1>::lineType tmpBuf)
        {
            return _exec_shifted_line(inBuf, inBuf, dx, lineLen, outBuf, tmpBuf);
        }
        
        template <class T1, class T2>
        inline void _exec_shifted_line(const typename ImDtTypes<T1>::lineType inBuf, const int &dx, const int &lineLen, typename ImDtTypes<T2>::lineType outBuf)
        {
            return _exec_shifted_line(inBuf, inBuf, dx, lineLen, outBuf, cpBuf);
        }
        
        inline void _exec_shifted_line_2ways(lineInType inBuf1, lineInType inBuf2, const int &dx, const int &lineLen, lineOutType outBuf, lineInType tmpBuf)
        {
            if (tmpBuf==NULL)
              tmpBuf = cpBuf;
            shiftLine<T_in>(inBuf2, dx, lineLen, tmpBuf, this->borderValue);
            lineFunction._exec(inBuf1, tmpBuf, lineLen, outBuf);
            shiftLine<T_in>(inBuf2, -dx, lineLen, tmpBuf, this->borderValue);
            lineFunction._exec(outBuf, tmpBuf, lineLen, outBuf);
        }
        inline void _exec_shifted_line_2ways(const lineInType inBuf, const int &dx, const int &lineLen, lineOutType outBuf, lineInType tmpBuf=NULL)
        {
            return _exec_shifted_line_2ways(inBuf, inBuf, dx, lineLen, outBuf, tmpBuf);
        }
    };

    template <class T_in, class lineFunction_T>
    class MorphImageFunction<T_in, lineFunction_T, T_in, true> 
      : public MorphImageFunction<T_in, lineFunction_T, T_in, false>
    {
      public:
        typedef MorphImageFunction<T_in, lineFunction_T, T_in, false> parentClass;
        
        typedef Image<T_in> imageType;
        typedef typename ImDtTypes<T_in>::lineType lineType;
        typedef typename ImDtTypes<T_in>::sliceType sliceType;
        typedef typename ImDtTypes<T_in>::volType volType;
        
        MorphImageFunction(T_in border=ImDtTypes<T_in>::min(), T_in initialValue = ImDtTypes<T_in>::min()) 
          : parentClass(border, initialValue),
          lineFunction(parentClass::lineFunction) // take ref from parent
        {
        }
        
        lineFunction_T &lineFunction;

        virtual RES_T _exec(const imageType &imIn, imageType &imOut, const StrElt &se);
        virtual RES_T _exec_single(const imageType &imIn, imageType &imOut, const StrElt &se);
        virtual RES_T _exec_single_generic(const imageType &imIn, imageType &imOut, const StrElt &se);                                 // Inplace unsafe !!
        
        virtual RES_T _exec_single_hexagonal_SE(const imageType &imIn, imageType &imOut);                                         // Inplace safe
        virtual RES_T _exec_single_square_SE(const imageType &imIn, imageType &imOut);                                                 // Inplace unsafe !!
        virtual RES_T _exec_single_cube_SE(const imageType &imIn, imageType &imOut);                                                 // Inplace unsafe !!
        virtual RES_T _exec_single_horizontal_2points(const imageType &imIn, int dx, imageType &imOut, bool oddLines=false);        // Inplace safe
        virtual RES_T _exec_single_vertical_2points(const imageType &imIn, int dx, imageType &imOut);                                // Inplace unsafe !!
        virtual RES_T _exec_single_horizontal_segment(const imageType &imIn, int xsize, imageType &imOut);                        // Inplace safe
        virtual RES_T _exec_single_vertical_segment(const imageType &imIn, imageType &imOut);                                        // Inplace unsafe !!
        virtual RES_T _exec_single_cross(const imageType &imIn, imageType &imOut);                                                // Inplace unsafe !!
        virtual RES_T _exec_single_cross_3d(const imageType &imIn, imageType &imOut);
        virtual RES_T _exec_single_depth_segment(const imageType &imIn, int zsize, imageType &imOut);                                 // Inplace safe
        virtual RES_T _exec_rhombicuboctahedron(const imageType &imIn, imageType &imOut, unsigned int size);                        // Inplace unsafe !!
    };
    
/** @} */

} // namespace smil

# endif // _MORPH_IMAGE_OPERATIONS_HPP
