#ifndef __DPROPAGATE_HPP_
#define __DPROPAGATE_HPP_

#include "DUtils.hpp"

namespace smil
{
    template <class arrowT, class valT>
    RES_T bredthFirst (const Image<arrowT>& arrowing, Image<valT>& im, const StrElt& se, const size_t& offset, const valT& val)
    {
        typedef Image<arrowT> arrowIT;
        typedef Image<valT>  valIT;
        typedef typename arrowIT::lineType arrowLT;
        typedef typename valIT::lineType valLT;

        // Processing vars.
            // Images related.
        arrowLT arrowP = arrowing.getPixels ();
        valLT imP = im.getPixels ();
        size_t size[3]; im.getSize (size);
            // Iteration related.
        bool oddLine;
        size_t x, x0, y, y0, z, z0, o, nb_o;
        UINT sePtsNumber = se.points.size ();
            // Processing related.
        arrowT arrow;
        STD_Queue<size_t> q;
        valT atomic_read;

        // Init
        q.push (offset);

        do
        {
            o = q.front (); q.pop();

            z0 = o / (size[1] * size[0]);
            y0 = (o % (size[1]*size[0])) / size[0];
            x0 = o % size[0];
            oddLine = se.odd && y0%2;

            for (UINT p=0; p<sePtsNumber; ++p)
            {
                arrow = (1UL << p);
                if (arrowP[o] & arrow) 
                {
                    x = x0 + se.points[p].x;
                    y = y0 + se.points[p].y;
                    z = z0 + se.points[p].z;

                    if (oddLine)
                        x += (y+1)%2;
                    nb_o = x + y*size[0] + z*size[1]*size[0];

                    #pragma omp atomic read
                    atomic_read = imP[nb_o];

                    if (atomic_read != val)
                    {
                        q.push (nb_o);
                    }

                    #pragma omp atomic write
                    imP[nb_o] = val;
                }
            }

        } while (!q.empty());
        return RES_OK;
    }

    template <class inT, class arrowT, class pT, class cT>
    RES_T bredthFirstConflict (const Image<inT>& imIn, const Image<arrowT>& arrowing, Image<pT>& imBasins, Image<cT>& imWatershed, const StrElt& se, const cT& cVal, HierarchicalQueue<cT> &pq)
    {
        typedef Image<inT> inIT;
        typedef Image<arrowT> arrowIT;
        typedef Image<pT> pIT;
        typedef Image<cT> cIT;
        typedef typename inIT::lineType inLT;
        typedef typename arrowIT::lineType arrowLT;
        typedef typename pIT::lineType pLT;
        typedef typename cIT::lineType cLT;

        // Processing vars
            // Images related.
        arrowLT arrowP = arrowing.getPixels ();
        pLT imB = imBasins.getPixels ();
        cLT imWS = imWatershed.getPixels ();

        size_t size[3]; imBasins.getSize (size);
            // Iteration related.
        bool oddLine;
        size_t x, x0, y, y0, z, z0, o, o2, nb_o;
        UINT sePtsNumber = se.points.size ();
            //Processing related.
        arrowT arrow;
        STD_Queue<size_t> cq;
        pT atomic_p;
        cT atomic_c;

        pT pVal = 0;

        do 
        {

            o = pq.pop ();

            #pragma omp atomic read
            atomic_p = imB[o];

            imB[o] = pVal;
            
            if (atomic_p != pVal && atomic_p > pT(0))
            {
                #pragma omp atomic read
                atomic_c = imWS[o];

                if (atomic_c == cT(0))
                {
                    cq.push (o);
                    do 
                    {
                        o2 = cq.front(); cq.pop();

                        #pragma omp atomic write
                        imWS[o2] = cVal;

                        z0 = o2 / (size[1] * size[0]);
                        y0 = (o2 % (size[1]*size[0])) / size[0];
                        x0 = o2 % size[0];
                        oddLine = se.odd && y0%2;

                        for (UINT p=0; p<sePtsNumber; ++p)
                        {
                            arrow = (1UL << p);
                            if (arrowP[o2] & arrow) 
                            {
                                x = x0 + se.points[p].x;
                                y = y0 + se.points[p].y;
                                z = z0 + se.points[p].z;

                                if (oddLine)
                                    x += (y+1)%2;
                                nb_o = x + y*size[0] + z*size[1]*size[0];

                                #pragma omp atomic read
                                atomic_c = imWS[nb_o];
                                
                                if (atomic_c != cVal)
                                    cq.push (nb_o);
                            }
                        }
                    } while (!cq.empty());
                }
            } else {
                z0 = o / (size[1] * size[0]);
                y0 = (o % (size[1]*size[0])) / size[0];
                x0 = o % size[0];
                oddLine = se.odd && y0%2;

                for (UINT p=0; p<sePtsNumber; ++p)
                {
                    arrow = (1UL << p);
                    if (arrowP[o] & arrow) 
                    {
                        x = x0 + se.points[p].x;
                        y = y0 + se.points[p].y;
                        z = z0 + se.points[p].z;

                        if (oddLine)
                            x += (y+1)%2;
                        nb_o = x + y*size[0] + z*size[1]*size[0];

                        #pragma omp atomic read
                        atomic_p = imB[nb_o];

                        if (atomic_p != pVal) {
                            pq.push (nb_o);
                        }
                    }
                }

            }
        } while (!pq.empty());
    }

}

#endif // _DPROPAGATE_HPP_
