/*
 * Copyright (c) 2011-2015, Matthieu FAESSEL and ARMINES
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


#ifndef _D_MORPHO_EXTREMA_HPP
#define _D_MORPHO_EXTREMA_HPP

#include "DMorphoGeodesic.hpp"
#include "DMorphoArrow.hpp"

namespace smil
{
    /**
    * \addtogroup Morpho
    * \{
    */

    // Extrema


    /**
    * Minima
    */
    template <class T>
    RES_T minima(const Image<T> &imIn, Image<T> &imOut, const StrElt &se=DEFAULT_SE)
    {
        ASSERT_ALLOCATED(&imIn, &imOut);
        ASSERT_SAME_SIZE(&imIn, &imOut);
        
        if (&imIn==&imOut)
        {
            Image<T> tmpIm(imIn);
            return minima(tmpIm, imOut, se);
        }
        
        ImageFreezer freeze(imOut);
        
        ASSERT((add(imIn, T(1), imOut)==RES_OK));
        ASSERT((dualBuild(imOut, imIn, imOut, se)==RES_OK));
        ASSERT((low(imIn, imOut, imOut)==RES_OK));
        
        return RES_OK;
    }

    /**
    * Maxima
    */
    template <class T>
    RES_T maxima(const Image<T> &imIn, Image<T> &imOut, const StrElt &se=DEFAULT_SE)
    {
        ASSERT_ALLOCATED(&imIn, &imOut);
        ASSERT_SAME_SIZE(&imIn, &imOut);
        
        if (&imIn==&imOut)
        {
            Image<T> tmpIm(imIn);
            return maxima(tmpIm, imOut, se);
        }
        
        ImageFreezer freeze(imOut);
        
        ASSERT((sub(imIn, T(1), imOut)==RES_OK));
        ASSERT((build(imOut, imIn, imOut, se)==RES_OK));
        ASSERT((grt(imIn, imOut, imOut)==RES_OK));
        
        return RES_OK;
    }

    /**
     * Calculate the minima and labelize them
     */
    template <class T1, class T2>
    RES_T minimaLabeled(const Image<T1> &imIn, Image<T2> &imOut, const StrElt &se=DEFAULT_SE)
    {
        ASSERT_ALLOCATED(&imIn, &imOut);
        ASSERT_SAME_SIZE(&imIn, &imOut);
        
        ImageFreezer freeze(imOut);
        
        Image<T1> imMinima(imIn);
        minima(imIn, imMinima, se);
        label(imMinima, imOut, se);
        
        return RES_OK;
    }
    
    /**
     * Calculate the maxima and labelize them
     */
    template <class T1, class T2>
    RES_T maximaLabeled(const Image<T1> &imIn, Image<T2> &imOut, const StrElt &se=DEFAULT_SE)
    {
        ASSERT_ALLOCATED(&imIn, &imOut);
        ASSERT_SAME_SIZE(&imIn, &imOut);
        
        ImageFreezer freeze(imOut);
        
        Image<T1> imMaxima(imIn);
        maxima(imIn, imMaxima, se);
        label(imMaxima, imOut, se);
        
        return RES_OK;
    }
    
    /**
    * h-Minima
    */
    template <class T>
    RES_T hMinima(const Image<T> &imIn, const T &height, Image<T> &imOut, const StrElt &se=DEFAULT_SE)
    {
        ASSERT_ALLOCATED(&imIn, &imOut);
        ASSERT_SAME_SIZE(&imIn, &imOut);
        
        ImageFreezer freeze(imOut);
        
        Image<T> tmpIm(imIn);
        
        ASSERT(hDualBuild(imIn, height, tmpIm, se)==RES_OK);
        ASSERT(minima(tmpIm, imOut, se)==RES_OK);
        
        return RES_OK;
    }

    /**
     * Calculate the h-minima and labelize them
     */
    template <class T1, class T2>
    RES_T hMinimaLabeled(const Image<T1> &imIn, const T1 &height, Image<T2> &imOut, const StrElt &se=DEFAULT_SE)
    {
        ASSERT_ALLOCATED(&imIn, &imOut);
        ASSERT_SAME_SIZE(&imIn, &imOut);
        
        ImageFreezer freeze(imOut);
        
        Image<T1> imMinima(imIn);
        hMinima(imIn, height, imMinima, se);
        label(imMinima, imOut, se);
        
        return RES_OK;
    }
    
    /**
    * h-Maxima
    */
    template <class T>
    RES_T hMaxima(const Image<T> &imIn, const T &height, Image<T> &imOut, const StrElt &se=DEFAULT_SE)
    {
        ASSERT_ALLOCATED(&imIn, &imOut);
        ASSERT_SAME_SIZE(&imIn, &imOut);
        
        Image<T> tmpIm(imIn);
        
        ImageFreezer freeze(imOut);
        
        ASSERT(hBuild(imIn, height, tmpIm, se)==RES_OK);
        ASSERT(maxima(tmpIm, imOut, se)==RES_OK);
        
        return RES_OK;
    }
    
    /**
     * Calculate the h-maxima and labelize them
     */
    template <class T1, class T2>
    RES_T hMaximaLabeled(const Image<T1> &imIn, const T1 &height, Image<T2> &imOut, const StrElt &se=DEFAULT_SE)
    {
        ASSERT_ALLOCATED(&imIn, &imOut);
        ASSERT_SAME_SIZE(&imIn, &imOut);
        
        ImageFreezer freeze(imOut);
        
        Image<T1> imMaxima(imIn);
        hMaxima(imIn, height, imMaxima, se);
        label(imMaxima, imOut, se);
        
        return RES_OK;
    }

    template <class T>
    RES_T fastExtrema (const Image<T> &imIn, Image<T> &imOut, const StrElt &se, const char *operation, const T& border_value) {

        // Typedefs
        typedef Image<T> inT;
        typedef Image<T> outT;
        typedef Image<T> arrowT;
        typedef typename outT::lineType outLineT;
        typedef typename arrowT::lineType arrowLineT;
        typedef typename inT::volType inVolT;
        typedef typename outT::volType outVolT;
        typedef typename arrowT::volType arrowVolT;

        // Initialisation.
        arrowT arrows(imIn);
        StrElt cpSe = se.noCenter();
        fill(imOut, T(0));

        // Processing vars.
        size_t size[3]; imIn.getSize(size);
        UINT sePtsNumber = cpSe.points.size();
        if (sePtsNumber == 0) return RES_OK;
            // Images related.
        inVolT inSlices = imIn.getSlices();
        outVolT outSlices = imOut.getSlices();
        arrowVolT arrowSlices = arrows.getSlices();
        outLineT* outLines;
        arrowLineT* arrowLines; 
        outLineT outP = imOut.getPixels();
        arrowLineT arrowP = arrows.getPixels();
            // Buffers.
        arrowLineT cstBuf = ImDtTypes<T>::createLine(size[0]);
        fillLine<T>(cstBuf, size[0], T(0)); 
        outLineT cstBuf2 = ImDtTypes<T>::createLine(size[0]);
        fillLine<T>(cstBuf2, size[0], ImDtTypes<T>::max());

        equLine<T> equOp;
        rightShiftLine<T> shiftOp;
        testLine<T, T> testOp;

        // Storing steep in imOut.
    #ifdef USE_OPEN_MP
        #pragma omp parallel 
    #endif // USE_OPEN_MP
        {
            size_t offset;
	    arrowPropagate <T, T, T> funcPropagation;
	    funcPropagation.propagationValue = T(1);

	    // Storing greater in out.
            arrow (imIn, operation, arrows, cpSe, border_value);

            for (size_t s=0; s<size[2]; ++s)
            {
                arrowLines = arrowSlices[s];
                outLines = outSlices[s];
                
            #ifdef USE_OPEN_MP
                #pragma omp for
            #endif // USE_OPEN_MP
                for (size_t l=0; l<size[1]; ++l)
                {
                    equOp._exec(arrowLines[l], cstBuf, size[0], outLines[l]);
                }
            }

            // Detecting plateaus and 1-pixel minimas.
            arrowEqu (imIn, arrows, cpSe);
            for (size_t s=0; s<size[2]; ++s)
            {
            #ifdef USE_OPEN_MP
                #pragma omp for
            #endif // USE_OPEN_MP
                for (size_t l=0; l<size[1]; ++l)
                {
                    for (size_t p=0; p<size[0]; ++p) 
                    {
                        offset = p+l*size[0]+s*size[1]*size[0];
                        if (outP[offset] == 0 && arrowP[offset] > 0)
                        {
                            funcPropagation (arrows, imOut, cpSe, offset);
                        }
                    }
                }
            }
            // Values of minimas back to max value.
            for (size_t s=0; s<size[2]; ++s)
            {
                outLines = outSlices[s];
            #ifdef USE_OPEN_MP
                #pragma omp for
            #endif // USE_OPEN_MP
                for (size_t l=0; l<size[1]; ++l)
                {
                    shiftOp._exec (outLines[l], 1, size[0], outLines[l]) ;
                    testOp._exec (outLines[l], cstBuf2, outLines[l], size[0], outLines[l]) ;
                }
            }

        }

        return RES_OK ;
    }

    template <class T>
    inline RES_T fastMinima (const Image<T> &imIn, Image<T> &imOut, const StrElt &se) 
    {
        return fastExtrema (imIn, imOut, se, ">", numeric_limits<T>::max()) ; 
    }

    template <class T>
    inline RES_T fastMaxima (const Image<T> &imIn, Image<T> &imOut, const StrElt &se)
    {
        return fastExtrema (imIn, imOut, se, "<", numeric_limits<T>::min()) ;
    }
    
/** \} */

} // namespace smil


#endif // _D_MORPHO_EXTREMA_HPP

