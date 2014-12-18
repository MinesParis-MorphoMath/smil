#ifndef _DMINIMA_HPP_
#define _DMINIMA_HPP_

#include "Morpho/include/DMorpho.h"
#include "DArrow.hpp"
#include "DPropagate.hpp"

namespace smil
{


    template <class T>
    RES_T fastMinima (const Image<T> &imIn, Image<T> &imOut, const StrElt &se) 
    {

        // Typedefs
        typedef Image<T> inT;
        typedef Image<T> outT;
        typedef Image<T> arrowT;
        typedef typename inT::lineType inLineT;
        typedef typename outT::lineType outLineT;
        typedef typename arrowT::lineType arrowLineT;
        typedef typename inT::sliceType inSliceT;
        typedef typename outT::sliceType outSliceT;
        typedef typename arrowT::sliceType arrowSliceT;
        typedef typename inT::volType inVolT;
        typedef typename outT::volType outVolT;
        typedef typename arrowT::volType arrowVolT;

        ImageFreezer freeze(imOut);
        
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
        inLineT* inLines;
        outLineT* outLines;
        arrowLineT* arrowLines; 
        outLineT outP = imOut.getPixels();
        arrowLineT arrowP = arrows.getPixels();
            // Buffers.
        arrowLineT cstBuf = ImDtTypes<T>::createLine(size[0]);
        fillLine<T>(cstBuf, size[0], T(0)); 
        outLineT cstBuf2 = ImDtTypes<T>::createLine(size[0]);
        fillLine<T>(cstBuf2, size[0], ImDtTypes<T>::max());
            // Line operators.
        equLine<T> equOp;
        rightShiftLine<T> shiftOp;
        testLine<T, T> testOp;

        #pragma omp parallel
        {
            size_t offset;
 
            // Storing greater in out. 
            arrowGrt (imIn, arrows, cpSe, numeric_limits<T>::max());
            for (size_t s=0; s<size[2]; ++s)
            {
                arrowLines = arrowSlices[s];
                outLines = outSlices[s];
                
                #pragma omp for
                for (size_t l=0; l<size[1]; ++l)
                {
                    equOp._exec(arrowLines[l], cstBuf, size[0], outLines[l]);
                }
            }

            // Detecting plateaus and propagating.
            arrowEqu (imIn, arrows, cpSe);
            for (size_t s=0; s<size[2]; ++s)
            {
                #pragma omp for
                for (size_t l=0; l<size[1]; ++l)
                {
                    for (size_t p=0; p<size[0]; ++p) 
                    {
                        offset = p+l*size[0]+s*size[1]*size[0];
                        if (outP[offset] == 0 && arrowP[offset] > 0)
                        {
                            bredthFirst (arrows, imOut, cpSe, offset, T(1));
                        }
                    }
                }
            }
            // Normalizing values.
            for (size_t s=0; s<size[2]; ++s)
            {
                outLines = outSlices[s];
                #pragma omp for
                for (size_t l=0; l<size[1]; ++l)
                {
                    shiftOp._exec (outLines[l], 1, size[0], outLines[l]) ;
                    testOp._exec (outLines[l], cstBuf2, outLines[l], size[0], outLines[l]) ;
                }
            }
        }
    }

}

#endif // _DMINIMA_HPP_
