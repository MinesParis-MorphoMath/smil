#ifndef _DWATERSHED_HPP_
#define _DWATERSHED_HPP_

#include "DArrow.hpp"
#include "DMinima.hpp"

namespace smil 
{

    template <class T, class labelT>
    RES_T fastWatershed (const Image<T> &imIn, const Image<labelT> &imMarkers, Image<T> &imOut, Image<labelT> &imBasinsOut, const StrElt &se=DEFAULT_SE)
    {
        // Typedefs
        typedef Image<T> inT;
        typedef Image<T> outT;
        typedef Image<T> arrowT;
        typedef Image<labelT> basinsT;
        typedef typename inT::lineType inLineT;
        typedef typename outT::lineType outLineT;
        typedef typename arrowT::lineType arrowLineT;
        typedef typename basinsT::lineType basinsLineT;
        typedef typename inT::sliceType inSliceT;
        typedef typename outT::sliceType outSliceT;
        typedef typename arrowT::sliceType arrowSliceT;
        typedef typename basinsT::sliceType basinsSliceT;
        typedef typename inT::volType inVolT;
        typedef typename outT::volType outVolT;
        typedef typename arrowT::volType arrowVolT;
        typedef typename basinsT::volType basinsVolT;

        // Initialisation.
        arrowT arrows(imIn);
        StrElt cpSe = se.noCenter();
        fill(imOut, T(0));
 
        // Processing vars
        labelT conflict_value = T(1);
        size_t size[3]; imIn.getSize(size);
        UINT sePtsNumber = cpSe.points.size();
        if (sePtsNumber == 0) return RES_OK;
            // Images related.
        inVolT inSlices = imIn.getSlices();
        outVolT outSlices = imOut.getSlices();
        arrowVolT arrowSlices = arrows.getSlices();
        basinsVolT basinsSlices = imBasinsOut.getSlices();
        inLineT* inLines;
        outLineT* outLines;
        arrowLineT* arrowLines;
        basinsLineT* basinsLines;
        outLineT outP = imOut.getPixels();
        arrowLineT arrowP = arrows.getPixels();
        basinsLineT basinsP = imBasinsOut.getPixels();

        fastMinima (imIn, arrows, se) ;
        T nbr_label = fastLabel (arrows, imBasinsOut, se) ;
        #pragma omp parallel
        {
            size_t offset;
            arrowPropagateWithConflicts< T, T, labelT> funcBredthFirst(conflict_value);
            arrowPropagate<T,T,labelT, STD_Stack<size_t> > funcDepthFirst;
            // Rising basins independently.
            arrowLowOrEqu (imIn, arrows, cpSe, numeric_limits<T>::min());
            for (size_t s=0; s<size[2]; ++s)
            {
                #pragma omp for
                for (size_t l=0; l<size[1]; ++l)
                {
                    for (size_t p=0; p<size[0]; ++p) 
                    {
                        offset = p+l*size[0]+s*size[1]*size[0];
                        
                        if (basinsP[offset] != 0 && outP[offset] != conflict_value)
                        {

                            funcBredthFirst.propagationValue = basinsP[offset];
                            funcBredthFirst (arrows, imOut, imBasinsOut, cpSe, offset) ;
                        }
                    }
                }
                
            }
            #pragma omp single
            {
            imOut.printSelf (1);
            imBasinsOut.printSelf (1);
            }
            // Detecting non-biased watershed lines.
            size_t x,y,z, nb_offset;
            UINT n;
            bool oddLine;
            labelT redondant;
            bool all_neighbor_ws;
            for (size_t s=0; s<size[2]; ++s)
            {
                #pragma omp for
                for (size_t l=0; l<size[1]; ++l)
                {
                    oddLine  = se.odd && l%2;
                    for (size_t p=0; p<size[0]; ++p)
                    {
                        offset = p+l*size[0]+s*size[1]*size[0];
                        if (basinsP[offset] == 0)
                        {
                            n=0;
                            redondant = 0;
                            all_neighbor_ws = true;
                            while (n<sePtsNumber && outP[offset] != numeric_limits<T>::max()) 
                            {
                                y = l + cpSe.points[n].y;
                                x = p + cpSe.points[n].x + (oddLine && (y+1)%2);
                                z = s + cpSe.points[n].z;
                                if (x>=0 && x<size[0] &&
                                    y>=0 && y<size[1] &&
                                    z>=0 && z<size[2])
                                {
                                    nb_offset = x+y*size[0]+z*size[0]*size[1];

                                    if (basinsP[nb_offset] != 0) 
                                    {
                                        if (redondant != basinsP[nb_offset] && redondant != 0)
                                        {
                                            outP[offset] = numeric_limits<T>::max();
                                        }
                                        else
                                            redondant = basinsP[nb_offset];
                                    }
                                }

                                ++n;
                            }
                        }
                    }
                }
            }

            // Labelling the remaining unlabelled pixels with a depth first approach. 
            /*
            arrowGrtOrEqu (imIn, arrows, cpSe, numeric_limits<T>::max());    
            for (size_t s=0; s<size[2]; ++s) 
            {
                #pragma omp for
                for (size_t l=0; l<size[1]; ++l) 
                {
                    for (size_t p=0; p<size[0]; ++p)
                    {
                        offset = p+l*size[0]+s*size[1]*size[0];
                        if (basinsP[offset] == 0)
                        {
                            func
                        }
                    }
                }
            }*/
        } 
        imOut.printSelf (1);
        imBasinsOut.printSelf (1);
    }

}

#endif // _DWATERSHED_HPP_
