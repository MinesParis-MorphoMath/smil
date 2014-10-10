#ifndef _MINIMA_H_
#define _MINIMA_H_

#include "DArrow.h"

namespace smil
{
    template <class T, class markerT>
    RES_T fastMinima (const Image<T> &imIn, Image<markerT> &imOut, const StrElt &se) {

        Image <UINT8> imFlag (imIn);
        Image <markerT> tmp (imIn);

        typedef Image<T> inT;
        typedef Image<markerT> outT;
        typedef typename inT::lineType inLineT;
        typedef typename outT::lineType outLineT;
        typedef typename Image<UINT8>::lineType flagLineT;
        typedef typename inT::sliceType inSliceT;
        typedef typename outT::sliceType outSliceT;
        typedef typename Image<UINT8>::sliceType flagSliceT;
        typedef typename inT::volType inVolT;
        typedef typename outT::volType outVolT;
        typedef typename Image<UINT8>::volType flagVolT;


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
        flagVolT flagSlices = imFlag.getSlices ();
        inLineT* inLines; outLineT* outLines; flagLineT* flagLines;
        inLineT lineIn; outLineT lineOut; flagLineT lineFlag;

        outLineT nullBuf = ImDtTypes < markerT >:: createLine (lineLen);
        fillLine <markerT> ( nullBuf, lineLen, markerT (0) );    

        arrowEqu (imIn, imFlag, cpSE);

        testLine<T, markerT> test;
        for ( size_t s=0; s<nSlices; ++s) 
        {
            inLines = inSlices [s];
            outLines = outSlices [s];
            flagLines = flagSlices [s];
            if (oddSe)
                oddLine = s%2 != 0;
            for (size_t l=0; l<nLines; ++l)
            {
                lineIn = inLines[l]; lineOut = outLines[l]; lineFlag = flagLines[l];

                for ( UINT p=0; p<sePtsNumber; ++p)
                {
                    x = -cpSE.points[p].x + oddLine;
                    y = l + cpSE.points[p].y;
                    z = s + cpSE.points[p].z;

                    test._exec (lineFlag, lineIn, nullBuf, lineLen, lineOut) ;
                }
                if (oddSe) oddLine = !oddLine;
            }
        }

        markerT nbr_lbl = label (imOut, tmp, cpSE);
        vector<bool> plateau (nbr_lbl+1, true) ;

        flagLineT flagP = imFlag.getPixels ();
        outLineT outP = imOut.getPixels ();
        outLineT tmpP = tmp.getPixels ();

        size_t offset;

        fill (imFlag, UINT8(0));
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
                        tmpP[offset] = markerT(0);
                }
        label (tmp, imOut, cpSE);
    }
}

#endif // _MINIMA_H_
