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


#ifndef _D_MORPHO_LABEL_HPP
#define _D_MORPHO_LABEL_HPP

#include "Base/include/private/DImageArith.hpp"
#include "Core/include/DImage.h"
#include "DMorphImageOperations.hpp"
#include "Base/include/private/DBlobMeasures.hpp"


#include <set>
#include <map>
#include <functional>

namespace smil
{
   /**
    * \ingroup Morpho
    * \defgroup Labelling
    * @{
    */
  

#ifndef SWIG

    template <class T1, class T2, class compOperatorT=std::equal_to<T1> >
    class labelFunctGeneric : public MorphImageFunctionBase<T1, T2>
    {
      public:
        typedef MorphImageFunctionBase<T1, T2> parentClass;
        typedef typename parentClass::imageInType imageInType;
        typedef typename parentClass::imageOutType imageOutType;
        
        T2 getLabelNbr() { return labels; }

        virtual RES_T initialize(const imageInType &imIn, imageOutType &imOut, const StrElt &se)
        {
            parentClass::initialize(imIn, imOut, se);
            fill(imOut, T2(0));
            labels = T2(0);
            return RES_OK;
        }

        virtual RES_T processImage(const imageInType &imIn, imageOutType &imOut, const StrElt &se)
        {
            this->pixelsIn = imIn.getPixels();
            for (size_t i=0; i<this->imSize[2]*this->imSize[1]*this->imSize[0]; i++) 
            {
                if (this->pixelsOut[i]==T2(0))
                    processPixel(i);
            }
            return RES_OK;
        }
        
        virtual void processPixel(size_t pointOffset)
        {

            T1 pVal = this->pixelsIn[pointOffset];

            if (pVal == T1(0) || this->pixelsOut[pointOffset] != T2(0))
                return;

            queue<size_t> propagation;
            int x, y, z, n_x, n_y, n_z;
            IntPoint p;

            ++labels;
            this->pixelsOut[pointOffset] = labels;
            propagation.push (pointOffset); 

            bool oddLine = 0;
            size_t curOffset, nbOffset;

            while (!propagation.empty ()) 
            {
                curOffset = propagation.front();
                pVal = this->pixelsIn[curOffset];
                
                z = curOffset / (this->imSize[1]*this->imSize[0]);
                y = (curOffset - z*this->imSize[1]*this->imSize[0])/this->imSize[0];
                x = curOffset - y*this->imSize[0] - z*this->imSize[1]*this->imSize[0];

                oddLine = this->oddSe && (y%2);

                for (UINT i=0; i<this->sePointNbr; ++i) 
                {
                     p = this->sePoints[i];
                     n_x = x+p.x;
                     n_y = y+p.y;
                     n_x += (oddLine && ((n_y+1)%2) != 0) ;
                     n_z = z+p.z; 
                     nbOffset = n_x+(n_y)*this->imSize[0]+(n_z)*this->imSize[1]*this->imSize[0];
                     if (nbOffset!=curOffset && 
                         n_x >= 0 && n_x < (int)this->imSize[0] &&
                         n_y >= 0 && n_y < (int)this->imSize[1] &&
                         n_z >= 0 && n_z < (int)this->imSize[2] &&
                         this->pixelsOut[nbOffset] != labels &&
                         compareFunc(this->pixelsIn[nbOffset], pVal))
                    {
                        this->pixelsOut[nbOffset] = labels;
                        propagation.push (nbOffset);
                    }
                }
                propagation.pop();
            } 
        }
        
        compOperatorT compareFunc;
    protected:
        T2 labels;
    };

    template <class T1, class T2, class compOperatorT=std::equal_to<T1> >
    class labelFunctFast : public MorphImageFunctionBase <T1, T2>
    {
    public:
        typedef MorphImageFunctionBase<T1, T2> parentClass;
        typedef typename parentClass::imageInType imageInType;
        typedef typename parentClass::imageOutType imageOutType;
        typedef typename imageInType::lineType lineInType;
        typedef typename imageInType::sliceType sliceInType;
        typedef typename imageOutType::lineType lineOutType;
        typedef typename imageOutType::sliceType sliceOutType;

        T2 getLabelNbr() { return labels; }

        virtual RES_T initialize (const imageInType &imIn, imageOutType &imOut, const StrElt &se) 
        {
            parentClass::initialize(imIn, imOut, se);
            fill(imOut, T2(0));
            labels = T2(0);
            return RES_OK;
        }

        virtual RES_T processImage (const imageInType &imIn, imageOutType &imOut, const StrElt &se) {
            Image<T1> tmp(imIn);
            Image<T1> tmp2(imIn);
            ASSERT(clone(imIn, tmp)==RES_OK);
            if (this->imSize[2] == 1) {
                 ASSERT(erode (tmp, tmp2, SquSE())==RES_OK); 
            } else {
                 ASSERT(erode (tmp, tmp2, CubeSE())==RES_OK);           
            }
            ASSERT(sub(tmp, tmp2, tmp)==RES_OK);
        
            lineInType pixelsTmp = tmp.getPixels () ;
         
            // Adding the first point of each line to tmp.
            #pragma omp parallel
            {
                #pragma omp for
                for (size_t i=0; i<this->imSize[2]*this->imSize[1]; ++i) {
                    pixelsTmp[i*this->imSize[0]] = this->pixelsIn[i*this->imSize[0]];
                }
            }           
           
              queue <size_t> propagation;
            int x,y,z, n_x, n_y, n_z;
            IntPoint p;

            T2 current_label = labels;
            bool is_not_a_gap = false;
            bool process_labeling = false;
            bool oddLine = 0;

            // First PASS to label the boundaries. //
            for (size_t i=0; i<this->imSize[2]*this->imSize[1]*this->imSize[0]; ++i) {
                if (i%(this->imSize[0]) == 0) {
                    is_not_a_gap=false;
                }
                if (pixelsTmp[i] != T1(0)) {
                    if (this->pixelsOut[i] == T2(0)) {
                        if (!is_not_a_gap) {
                            current_label = ++labels;
                        }
                        this->pixelsOut[i] = current_label;
                        process_labeling = true;
                    } else {
                        current_label = this->pixelsOut[i];
                    }

                    is_not_a_gap = true;
                } 
                if (this->pixelsIn[i] == T1(0)) {
                    is_not_a_gap = false;
                }

                if (process_labeling) {
                    propagation.push (i);                   

                    while (!propagation.empty ()) {
                        z = propagation.front() / (this->imSize[1]*this->imSize[0]);
                        y = (propagation.front() - z*this->imSize[1]*this->imSize[0])/this->imSize[0];
                        x = propagation.front() - y*this->imSize[0] - z*this->imSize[1]*this->imSize[0];

                        oddLine = this->oddSe && (y%2);
                        size_t nbOffset;

                       for (UINT i=0; i<this->sePointNbr; ++i) { 
                            p = this->sePoints[i]; 
                             n_x = x+p.x;
                             n_y = y+p.y;
                             n_x += (oddLine && ((n_y+1)%2) != 0) ;
                             n_z = z+p.z; 
                             nbOffset = n_x+(n_y)*this->imSize[0]+(n_z)*this->imSize[1]*this->imSize[0];
                             if (n_x >= 0 && n_x < (int)this->imSize[0] &&
                                 n_y >= 0 && n_y < (int)this->imSize[1] &&
                                 n_z >= 0 && n_z < (int)this->imSize[2] &&
                                compareFunc(this->pixelsIn[nbOffset], pixelsTmp[propagation.front ()]) &&
                                 this->pixelsOut[nbOffset] != current_label)
                             {
                                 this->pixelsOut[nbOffset] = current_label;
                                 propagation.push (nbOffset);
                             }
         
                        }

                        propagation.pop();
                    } 
                    process_labeling = false;
                }
            }
            // Propagate labels inside the borders //

            size_t nSlices = imIn.getDepth () ;
            size_t nLines = imIn.getHeight () ;
            size_t nPixels = imIn.getWidth () ;
            size_t l, v;
            T1 previous_value;
            T2 previous_label;

            sliceInType srcLines = imIn.getLines () ;
            sliceOutType desLines = imOut.getLines () ;
            lineInType lineIn;
            lineOutType lineOut;

            for (size_t s=0; s<nSlices; ++s) {
                #pragma omp parallel private(lineIn,lineOut,l,v,previous_value,previous_label)
                {
                    #pragma omp for
                    for (l=0; l<nLines; ++l) {
                        lineIn = srcLines[l+s*nSlices];
                        lineOut = desLines[l+s*nSlices];
                        previous_value = lineIn[0];
                        previous_label = lineOut[0];
                        for (v=1; v<nPixels; ++v) {
                            if (compareFunc (lineIn[v], previous_value)) {
                                lineOut[v] = previous_label;
                            } else {
                                previous_value = lineIn[v];
                                previous_label = lineOut[v];
                            }
                        } 
                    }
                }
            }
        return RES_OK;  
        }
    protected :
            compOperatorT compareFunc;
        T2 labels;
    };
     
    
    template <class T>
    struct lambdaEqualOperator
    {
        inline bool operator()(T &a, T&b) 
    { 
        bool retVal = a>b ? (a-b)<=lambda : (b-a)<=lambda;
        return retVal;
        
    }
        T lambda;
    };
  
#endif // SWIG
 
    template <class T1, class T2 >
    size_t labelWithoutFunctor(const Image<T1> &imIn, Image<T2> &imOut, const StrElt &se=DEFAULT_SE)
    {
        // Checks
        ASSERT_ALLOCATED (&imIn, &imOut) ;
        ASSERT_SAME_SIZE (&imIn, &imOut) ;

        // Typedefs
        typedef Image<T1> inT;
        typedef Image<T2> outT;
        typedef typename inT::lineType inLineT;
        typedef typename outT::lineType outLineT;

        // Initialisation.
        StrElt cpSe = se.noCenter () ;
        fill (imOut, T2(0));

        // Processing vars.
        T2 lblNbr = 0;
        size_t size[3]; imIn.getSize (size) ;
        UINT sePtsNumber = cpSe.points.size();
        if (sePtsNumber == 0) return 0;
        queue<size_t> propagation;
        size_t o, nb_o;
        size_t x,x0,y,y0,z,z0;
        bool oddLine;
            // Image related.
        inLineT inP = imIn.getPixels () ;
        outLineT outP = imOut.getPixels () ;
 
        for (size_t s=0; s<size[2]; ++s)
        {
            for (size_t l=0; l<size[1]; ++l)
            {
                for (size_t p=0; p<size[0]; ++p)
                {
                    o = p + l*size[0] + s*size[0]*size[1];
                    if (inP[o] != T1(0) && outP[o] == T2(0)) 
                    {
                        ++lblNbr ;
                        outP [o] = lblNbr;
                        propagation.push (o);
                        do 
                        {
                            o = propagation.front () ;
                            propagation.pop () ;

                            x0 = o % size[0];
                            y0 = (o % (size[1]*size[0])) / size[0];
                            z0 = o / (size[0]*size[1]);
                            oddLine = cpSe.odd && y0 %2;
                            for (UINT pSE=0; pSE<sePtsNumber; ++pSE)
                            {
                                x = x0 + cpSe.points[pSE].x;
                                y = y0 + cpSe.points[pSE].y;
                                z = z0 + cpSe.points[pSE].z;
                                
                                if (oddLine)
                                    x += (y+1)%2;

                                nb_o = x + y*size[0] + z*size[0]*size[1];
                                if (x >= 0 && x < size[0] && y >= 0 && y < size[1] && z >= 0 && z<size[2] && outP [nb_o] != lblNbr && inP [nb_o] == inP[o])
                                {
                                    outP[nb_o] = lblNbr;
                                    propagation.push (nb_o);
                                }
                            }
                        } while (!propagation.empty()) ;
                    }
                }
            }
        }

        return lblNbr;
    }
 
    /**
    * Image labelization
    * 
    * Return the number of labels (or 0 if error).
    */
    template<class T1, class T2>
    size_t label(const Image<T1> &imIn, Image<T2> &imOut, const StrElt &se=DEFAULT_SE)
    {
        if ((void*)&imIn==(void*)&imOut)
        {
            Image<T1> tmpIm(imIn, true); // clone
            return label(tmpIm, imOut);
        }
        
        ASSERT_ALLOCATED(&imIn, &imOut);
        ASSERT_SAME_SIZE(&imIn, &imOut);
        
        labelFunctGeneric<T1,T2> f;
        
        ASSERT((f._exec(imIn, imOut, se)==RES_OK), 0);
        
        size_t lblNbr = f.getLabelNbr();
        
        ASSERT((lblNbr < size_t(ImDtTypes<T2>::max())), "Label number exceeds data type max!", 0);
        
        return lblNbr;
    }

    /**
    * Lambda-flat zones labelization
    * 
    * Return the number of labels (or 0 if error).
    */
    template<class T1, class T2>
    size_t lambdaLabel(const Image<T1> &imIn, const T1 &lambdaVal, Image<T2> &imOut, const StrElt &se=DEFAULT_SE)
    {
        ASSERT_ALLOCATED(&imIn, &imOut);
        ASSERT_SAME_SIZE(&imIn, &imOut);
        
        labelFunctGeneric<T1,T2,lambdaEqualOperator<T1> > f;
        f.compareFunc.lambda = lambdaVal;
        
        ASSERT((f._exec(imIn, imOut, se)==RES_OK), 0);
        
        size_t lblNbr = f.getLabelNbr();
        
        ASSERT((lblNbr < size_t(ImDtTypes<T2>::max())), "Label number exceeds data type max!", 0);
        
        return lblNbr;
    }

    /**
    * Image labelization
    * 
    * Return the number of labels (or 0 if error).
    */
    template<class T1, class T2>
    size_t fastLabel(const Image<T1> &imIn, Image<T2> &imOut, const StrElt &se=DEFAULT_SE)
    {
        ASSERT_ALLOCATED(&imIn, &imOut);
        ASSERT_SAME_SIZE(&imIn, &imOut);
        
        labelFunctFast<T1,T2> f;
        
        ASSERT((f._exec(imIn, imOut, se)==RES_OK), 0);
        
        size_t lblNbr = f.getLabelNbr();
        
        ASSERT((lblNbr < size_t(ImDtTypes<T2>::max())), "Label number exceeds data type max!", 0);

        return lblNbr;
    }

    
    /**
    * Image labelization with the size of each connected components
    * 
    */
    template<class T1, class T2>
    size_t labelWithArea(const Image<T1> &imIn, Image<T2> &imOut, const StrElt &se=DEFAULT_SE)
    {
        ASSERT_ALLOCATED(&imIn, &imOut);
        ASSERT_SAME_SIZE(&imIn, &imOut);
        
        ImageFreezer freezer(imOut);
        
        Image<T2> imLabel(imIn);
        
        ASSERT(label(imIn, imLabel, se)!=0);
         map<T2, double> areas = measAreas(imLabel);
        ASSERT(!areas.empty());
        
        ASSERT(applyLookup(imLabel, areas, imOut)==RES_OK);
        
        return RES_OK;
    }


    template <class T1, class T2>
    class neighborsFunct : public MorphImageFunctionBase<T1, T2>
    {
    public:
        typedef MorphImageFunctionBase<T1, T2> parentClass;
        
        virtual inline void processPixel(size_t pointOffset, vector<int> &dOffsetList)
        {
            vector<T1> vals;
            UINT nbrValues = 0;
            vector<int>::iterator dOffset = dOffsetList.begin();
            while(dOffset!=dOffsetList.end())
            {
                T1 val = parentClass::pixelsIn[pointOffset + *dOffset];
                if (find(vals.begin(), vals.end(), val)==vals.end())
                {
                  vals.push_back(val);
                  nbrValues++;
                }
                dOffset++;
            }
            parentClass::pixelsOut[pointOffset] = T2(nbrValues);
        }
    };
    
    /**
    * Neighbors
    * 
    * Return for each pixel the number of different values in the neighborhoud.
    * Usefull in order to find interfaces or multiple points between basins.
    * 
    * \not_vectorized
    * \not_parallelized
    */ 
    template <class T1, class T2>
    RES_T neighbors(const Image<T1> &imIn, Image<T2> &imOut, const StrElt &se=DEFAULT_SE)
    {
        ASSERT_ALLOCATED(&imIn, &imOut);
        ASSERT_SAME_SIZE(&imIn, &imOut);
        
        neighborsFunct<T1, T2> f;
        
        ASSERT((f._exec(imIn, imOut, se)==RES_OK));
        
        return RES_OK;
        
    }

    
/** \} */

} // namespace smil

#endif // _D_MORPHO_LABEL_HPP

