#ifndef _MINIMA_H_
#define _MINIMA_H_

#include "DArrow.h"

namespace smil
{
    template <class T>
    RES_T fastMinima (const Image<T> &imIn, Image<T> &imOut, const StrElt &se) {

        Image <T> imFlag (imIn);
        Image <T> tmp (imIn);

        typedef Image<T> inT;
        typedef Image<T> outT;
        typedef typename inT::lineType inLineT;
        typedef typename outT::lineType outLineT;
        typedef typename Image<T>::lineType flagLineT;
        typedef typename inT::sliceType inSliceT;
        typedef typename outT::sliceType outSliceT;
        typedef typename Image<T>::sliceType flagSliceT;
        typedef typename inT::volType inVolT;
        typedef typename outT::volType outVolT;
        typedef typename Image<T>::volType flagVolT;

        fill (imOut, T(0));

        StrElt cpSE = se.noCenter ();

        bool oddSe = cpSE.odd, oddLine;
        size_t x, y, z;

        UINT sePtsNumber = cpSE.points.size ();
        if (sePtsNumber == 0) return RES_OK;

        size_t lineLen = imIn.getWidth ();
        size_t nSlices = imIn.getSliceCount ();
        size_t nLines = imIn.getHeight ();

        inVolT inSlices = imIn.getSlices();
        outVolT outSlices = imOut.getSlices ();
        outVolT tmpSlices = tmp.getSlices ();
        flagVolT flagSlices = imFlag.getSlices ();
        inLineT* inLines; outLineT* outLines; outLineT* tmpLines; flagLineT* flagLines;
        inLineT lineIn; outLineT lineOut; outLineT lineTmp; flagLineT lineFlag;

        outLineT nullBuf = ImDtTypes < T >:: createLine (lineLen);
        fillLine <T> ( nullBuf, lineLen, T (0) ); 

        arrowEqu (imIn, imFlag, cpSE);

        testLine<T, T> test;
        for ( size_t s=0; s<nSlices; ++s) 
        {
            inLines = inSlices [s];
            tmpLines = tmpSlices [s];
            flagLines = flagSlices [s];
            for (size_t l=0; l<nLines; ++l)
            {
                lineIn = inLines[l]; lineTmp = tmpLines[l]; lineFlag = flagLines[l];

                for ( UINT p=0; p<sePtsNumber; ++p)
                {
                    test._exec (lineFlag, lineIn, nullBuf, lineLen, lineTmp) ;
                }
                if (oddSe) oddLine = !oddLine;
            }
        }

        vector <size_t> outlets;

        flagLineT flagP = imFlag.getPixels ();
        outLineT outP = imOut.getPixels ();
        outLineT tmpP = tmp.getPixels ();

        arrowMin (imIn, imFlag, cpSE);
        size_t offset, nb_offset;

        for ( size_t k=0; k<nSlices; ++k) 
            for (size_t j=0; j<nLines; ++j)
                for (size_t i=0; i<lineLen; ++i)
                {
                    offset = i+j*lineLen+k*lineLen*nLines;            
                    if (flagP[offset] > 0 && tmpP[offset] > 0)
                    {
                        outlets.push_back (offset);
                    }
                    if (flagP[offset] == 0 && tmpP[offset] == 0)
                        outP[offset] = 255;
                }      

        size_t x0, y0, z0;
        UINT arrow;

        arrowEqu (imIn, imFlag, cpSE);

        queue <size_t> breadth;
        for (vector<size_t>::iterator i=outlets.begin(); i!=outlets.end(); ++i) 
        {

            breadth.push (*i) ;

            do {
                offset = breadth.front();
                breadth.pop();
                for (UINT p=0; p<sePtsNumber; ++p)
                {
                    arrow = (1UL << p);
                    if ((flagP[offset] & arrow) != 0) {
                        z0 = offset / (nLines * lineLen);
                        y0 = (offset - z0*nLines*lineLen) / lineLen;
                        x0 = offset - y0*lineLen - z0*nLines*lineLen;
                        if (oddSe)
                            oddLine = z0%2 != 0;

                        x = x0 + cpSE.points[p].x + oddLine;
                        y = y0 + cpSE.points[p].y;
                        z = z0 + cpSE.points[p].z;
                        nb_offset = x + y*lineLen + z*lineLen*nLines;
                        if (tmpP[nb_offset] > 0)
                        {
                            tmpP[nb_offset] = 0;
                            breadth.push (nb_offset);
                        }
                    }
                 }
            } while (!breadth.empty());

        }

        for ( size_t z=0; z<nSlices; ++z) 
            for (size_t y=0; y<nLines; ++y)
                for (size_t x=0; x<lineLen; ++x)
                {
                    offset = x+y*lineLen+z*lineLen*nLines;
                    if (outP[offset] != 0)
                        tmpP[offset] = outP[offset];
                }
        label (tmp, imOut, cpSE);

/*        T nbr_lbl = label (imOut, tmp, cpSE);
        vector<bool> plateau (nbr_lbl+1, true) ;

        flagLineT flagP = imFlag.getPixels ();
        outLineT outP = imOut.getPixels ();
        outLineT tmpP = tmp.getPixels ();

        size_t offset;

        fill (imFlag, T(0));
        arrowMin (imIn, imFlag, cpSE) ;
        
        for ( size_t z=0; z<nSlices; ++z) 
            for (size_t y=0; y<nLines; ++y)
                for (size_t x=0; x<lineLen; ++x)
                {
                    offset = x+y*lineLen+z*lineLen*nLines;
                    if (flagP[offset] > 0)
                        plateau[tmpP[offset]] = false;
                }

        for ( size_t z=0; z<nSlices; ++z) 
            for (size_t y=0; y<nLines; ++y)
                for (size_t x=0; x<lineLen; ++x)
                {
                    offset = x+y*lineLen+z*lineLen*nLines;
                    if (!plateau[tmp[offset]])
                        tmpP[offset] = T(0);
                }
        label (tmp, imOut, cpSE);
*/
    }

}

#endif // _MINIMA_H_
