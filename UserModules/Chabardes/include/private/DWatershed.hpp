#ifndef _DWATERSHED_HPP_
#define _DWATERSHED_HPP_

#include "DArrow.hpp"
#include "DMinima.hpp"
#include "DPropagate.hpp"

namespace smil 
{

    template <class T>
    RES_T fastWatershed (const Image<T> &imIn, const Image<T> &imMarkers, Image<T> &imOut, Image<T> &imBasinsOut, const StrElt &se=DEFAULT_SE)
    {
        // Typedefs
        typedef Image<T> inT;
        typedef Image<T> outT;
        typedef Image<T> arrowT;
        typedef Image<T> basinsT;
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
        inLineT inP = imIn.getPixels();
        outLineT outP = imOut.getPixels();
        arrowLineT arrowP = arrows.getPixels();
        basinsLineT basinsP = imBasinsOut.getPixels();

        fastMinima (imIn, arrows, cpSe) ;
        T nbr_label = fastLabel (arrows, imBasinsOut, cpSe) ;

        int nthreads = Core::getInstance()->getNumberOfThreads();
        HierarchicalQueue<T> hq[nthreads];
        bool hqs_emptied;
/*
        #pragma omp parallel
        {
            size_t o;
            // Beginning. 
                // Arrowing used.
            arrowLowOrEqu (imIn, arrows, cpSe, numeric_limits<T>::min());
                // Initializing the hq.
            for (size_t s=0; s<size[2]; ++s)
                #pragma omp for 
                for (size_t l=0; l<size[1]; ++l)
                    for (size_t p=0; p<size[0]; ++p)
                    {
                        o = p + l * size[0] + s * size[0] * size[1]; 
                        if (basinsP[o] != 0)
                            hq [basinsP[o]%nthreads].push (o, imP[o]);
                    } 

            propagation1 ();

            propagation2 ();

            solveConflicts ();

            labeling ();

            hq ();



            // Processing.
            do
            {
                PropagationWS (arrows, imBasinsOut, imOut, cpSE, numeric_limits<T>::max (), hq[omp_get_thread_num()]);

                // Need a multi-threaded, in-place-safe version of labelling.
                #pragma omp single
                {
                    T nbr_label = fastLabel (imBasinsOut, imBasinsOut, cpSE);
                }

                // Refiling the hq.
                for (size_t s=0; s<size[2]; ++s)
                    #pragma omp for 
                    for (size_t l=0; l<size[1]; ++l)
                        for (size_t p=0; p<size[0]; ++p)
                        {
                            o = p + l * size[0] + s * size[0] * size[1]; 
                            if (basinsP[o] != 0)
                                hq [basinsP[o]%nthreads].push (o, imP[o]);
                        } 
                hqs_emptied = true;
                #pragma omp for reduction(&, hqs_emptied)
                for (int i=0; i<nthreads; ++i)
                    hqs_emptied = hqs_emptied & hq[i].empty();
            } while (!hqs_emptied);
            // Ending.
        }
*/    }

}

#endif // _DWATERSHED_HPP_
