#ifndef _MINIMA_H_
#define _MINIMA_H_

#include "DArrow.h"

namespace smil
{
    template <class T>
    RES_T fastMinima (const Image<T> &imIn, Image<T> &imOut, const StrElt &se) {

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

        // Initialisation.
        arrowT arrows(imIn);
        StrElt cpSe = se.noCenter();
        fill(imOut, T(0));

        // Processing vars.
        size_t size[3]; imIn.getSize(size);
        UINT sePtsNumber = cpSe.points.size();
        if (sePtsNumber == 0) return RES_OK;
        bool oddLine;
        vector <size_t> outlets;
        size_t offset, nb_offset;
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
        outLineT cstBuf2 = ImDtTypes<T>::createLine(size[0]);

        // Storing steep in imOut.
        arrowGrt (imIn, arrows, cpSe, numeric_limits<T>::max());
        grtLine<T> grtOp;
        fillLine<T>(cstBuf, size[0], T(0)); 
        for (size_t s=0; s<size[2]; ++s)
        {
            arrowLines = arrowSlices[s];
            outLines = outSlices[s];
            
            #pragma omp parallel for
            for (size_t l=0; l<size[1]; ++l)
            {
                grtOp._exec(arrowLines[l], cstBuf, size[0], outLines[l]);
            }
        }

        // Detecting plateaus and 1-pixel minimas.
        arrowEqu (imIn, arrows, cpSe);
        #pragma omp parallel private (offset)
        {
        vector<size_t> outlets_private;
        for (size_t s=0; s<size[2]; ++s)
        {
            #pragma omp for
            for (size_t l=0; l<size[1]; ++l)
            {
                for (size_t p=0; p<size[0]; ++p) {
                    offset = p+l*size[0]+s*size[1]*size[0];
                    if (arrowP[offset] > 0)
                    {
                        if (arrowP[offset] > 0 && outP[offset] > 0)
                        {
                            outlets_private.push_back(offset); // outlet pixels of plateaus.
                        }
                        outP[offset] = ImDtTypes<T>::max();
                    } else if (outP[offset] == 0)
                    {
                        outP[offset] = ImDtTypes<T>::max()/2; // 1-pixel minimas.
                    } else {
                        outP[offset] = 0;
                    }
                }
            }
        }
        #pragma omp critical
        outlets.insert(outlets.end(), outlets_private.begin(), outlets_private.end());
        }
    
        // Removing plateaus with outlets.
        size_t x, x0, y, y0, z, z0;
        UINT arrow;
        queue <size_t> breadth;
        #pragma omp parallel for private (breadth,offset,x0,y0,z0,x,y,z,oddLine,arrow,nb_offset)
        for (int i=0; i<outlets.size(); ++i)
//        for (vector<size_t>::iterator i=outlets.begin(); i!=outlets.end(); ++i) 
        {
            outP[outlets[i]] = 0;
            breadth.push (outlets[i]) ;

            do 
            {
                offset = breadth.front();
                breadth.pop();
                z0 = offset / (size[1] * size[0]);
                y0 = (offset % (size[1]*size[0])) / size[0];
                x0 = offset % size[0];
                oddLine = se.odd && y0%2;

                for (UINT p=0; p<sePtsNumber; ++p)
                {
                    arrow = (1UL << p);
                    if ((arrowP[offset] & arrow)) 
                    {
                        
                        x = x0 + cpSe.points[p].x;
                        y = y0 + cpSe.points[p].y;
                        z = z0 + cpSe.points[p].z;
                        if (oddLine)
                            x += (y+1)%2;
                        nb_offset = x + y*size[0] + z*size[1]*size[0];
                        if (outP[nb_offset] == ImDtTypes<T>::max())
                        {
                            outP[nb_offset] = 0;
                            breadth.push (nb_offset);
                        }
                    }
                }
            } while (!breadth.empty());

        }

        // Values of minimas back to max value.
        testLine<T, T> testOp;
        fillLine<T>(cstBuf2, size[0], ImDtTypes<T>::max()); 
        for (size_t s=0; s<size[2]; ++s)
        {
            outLines = outSlices[s];
            #pragma omp parallel for
            for (size_t l=0; l<size[1]; ++l)
            {
                testOp._exec(outLines[l], cstBuf2, outLines[l], size[0], outLines[l]);
            }
        }
    }
}

#endif // _MINIMA_H_
