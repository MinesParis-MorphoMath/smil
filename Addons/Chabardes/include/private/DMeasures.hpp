#ifndef _D_CHABARDES_MEASURES_HPP_
#define _D_CHABARDES_MEASURES_HPP_

#include "Core/include/private/DImage.hpp"
#include "Base/include/private/DBaseMeasureOperations.hpp"
#include "Base/include/private/DLineArith.hpp"

namespace smil
{
    template <class T>
    vector<double> measHaralickFeatures (Image<T> &imIn, const StrElt &s) 
    {
        map<T, UINT> hist = histogram(imIn);
        map<T,T> equivalence;

        T nbr_components = 0;
        for (typename map<T, UINT>::iterator it=hist.begin(); it!=hist.end(); ++it)
        {
                if (it->second != 0) {
                       equivalence.insert (pair<T,T>(it->first,nbr_components));                
                       nbr_components ++;
                }
        }

        size_t S[3];
        imIn.getSize (S);
        size_t nbrPixelsInSlice = S[0]*S[1];
        size_t nbrPixels = nbrPixelsInSlice*S[2];
        T *in = imIn.getPixels ();
        StrElt se = s.noCenter();
        UINT sePtsNumber = se.points.size();
        UINT nthreads = Core::getInstance()->getNumberOfThreads ();

        vector<double> vec = vector<double> (nbr_components*nbr_components, 0.0);


        //# pragma omp parallel num_threads(nthreads) 
        {
                index p, q;
                UINT pts;
                vector<double> vec_local = vector<double> (nbr_components*nbr_components, 0.0);
                T* counts = new T[nbr_components]; 
                T max = 0;

                #pragma omp for
                ForEachPixel (p)
                {
                        max = 0;
                        for (int i=0; i<nbr_components; ++i)
                        {
                                counts[i] = 0;
                        }

                        ForEachNeighborOf (p,q)
                        {
                                if (in[q.o] != in[p.o])
                                {
                                        counts[equivalence[in[q.o]]]++;
                                }
                        }
                        ENDForEachNeighborOf

                        for (int i=0; i<nbr_components; ++i)
                        {
                                max = (counts[i] > counts[max] || (counts[i] == counts[max] && vec_local[i] < vec_local[max])) ? i : max;
                        }
                        if (counts[max] != 0)
                          vec_local[equivalence[in[p.o]]*nbr_components+max]++;
                }
                ENDForEachPixel
                
                #pragma omp for ordered schedule (static,1)
                for (int t=0; t<omp_get_num_threads(); ++t)
                {
                        #pragma omp ordered
                        {
                                for (int i=0; i<vec.size(); ++i)
                                {
                                        vec[i] += vec_local[i];
                                }
                        }
                }
                delete counts;
        }

        return vec;
    }

    /**
     * CrossCorrelation between two phases
     * 
     * The direction is given by \b dx, \b dy and \b dz.
     * The lenght corresponds to the max number of steps \b maxSteps
     */
    template <class T>
    vector<double> measCrossCorrelation(const Image<T> &imIn, const T &val1, const T &val2, size_t dx, size_t dy, size_t dz, UINT maxSteps=0, bool normalize=false)
    {
        vector<double> vec;
        ASSERT(areAllocated(&imIn, NULL), vec);
        
        size_t s[3];
        imIn.getSize(s);
        if (maxSteps==0)
          maxSteps = max(max(s[0], s[1]), s[2]) - 1;
        vec.clear();
        
        typename ImDtTypes<T>::volType slicesIn = imIn.getSlices();
        typename ImDtTypes<T>::sliceType curSliceIn1;
        typename ImDtTypes<T>::sliceType curSliceIn2;
        typename ImDtTypes<T>::lineType lineIn1;
        typename ImDtTypes<T>::lineType lineIn2;
        typename ImDtTypes<T>::lineType bufLine1 = ImDtTypes<T>::createLine(s[0]);
        typename ImDtTypes<T>::lineType bufLine2 = ImDtTypes<T>::createLine(s[0]);
        typename ImDtTypes<T>::lineType val1L = ImDtTypes<T>::createLine(s[0]);
        fillLine<T> (val1L, s[0], val1);
        typename ImDtTypes<T>::lineType val2L = ImDtTypes<T>::createLine(s[0]);
        fillLine<T> (val2L, s[0], val2);
        equLine<T> eqOp;
        
         for (UINT len=0;len<=maxSteps;len++)
        {
            double prod = 0;
            size_t xLen = s[0] - dx*len;
            size_t yLen = s[1] - dy*len;
            size_t zLen = s[2] - dz*len;
            
            for (size_t z=0;z<zLen;z++)
            {
                curSliceIn1 = slicesIn[z];
                curSliceIn2 = slicesIn[z+len*dz];
                for (UINT y=0;y<yLen;y++)
                {
                    lineIn1 = curSliceIn1[y];
                    lineIn2 = curSliceIn2[y+len*dy];
                    eqOp (lineIn1, val1L, xLen, bufLine1);
                    eqOp (lineIn2, val2L, xLen, bufLine2);
                    for (size_t x=0;x<xLen;x++) // Vectorized loop
                      prod += bufLine1[x] * bufLine2[x];
                }
            }
            if (xLen*yLen*zLen != 0)
              prod /= (xLen*yLen*zLen);
            vec.push_back(prod);
        }
        
        if (normalize)
        {
          double orig = vec[0];
          for (vector<double>::iterator it=vec.begin();it!=vec.end();it++)
            *it /= orig;
        }
        
        ImDtTypes<T>::deleteLine(bufLine1);
        ImDtTypes<T>::deleteLine(bufLine2);

        
        return vec;
    }

}

#endif // _D_CHABARDES_MEASURES_HPP_

