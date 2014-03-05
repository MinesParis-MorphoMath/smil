/*
 * Copyright (c) 2011, Matthieu FAESSEL and ARMINES
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
#include "DImage.h"
#include "DMorphImageOperations.hpp"
#include "Base/include/private/DBlobMeasures.hpp"


#include <set>
#include <map>

namespace smil
{
   /**
    * \ingroup Morpho
    * \defgroup Labelling
    * @{
    */
  
    template <class T1, class T2>
    class labelFunct : public unaryMorphImageFunctionBase<T1, T2>
    {
    public:
	typedef unaryMorphImageFunctionBase<T1, T2> parentClass;
	typedef typename parentClass::imageInType imageInType;
	typedef typename parentClass::imageOutType imageOutType;
	
	size_t getLabelNbr() { return labels; }
	
	virtual RES_T initialize(const imageInType &imIn, imageOutType &imOut, const StrElt &se)
	{
	    parentClass::initialize(imIn, imOut, se);
	    fill(imOut, T2(0));
	    labels = 0;
	    pairs.clear();
	    lut.clear();
	    return RES_OK;
	}
	
	// The generic way
	virtual inline void processPixel(size_t &pointOffset, vector<int>::iterator dOffset, vector<int>::iterator dOffsetEnd)
	{
	    T1 pVal = this->pixelsIn[pointOffset];
	    
	    if (pVal==0)
	      return;

	    T2 curLabel = this->pixelsOut[pointOffset];
	    
	    if (curLabel==0)
	    {
	      curLabel = ++labels;
	      this->pixelsOut[pointOffset] = curLabel;
	    }
	    
	    while(dOffset!=dOffsetEnd)
	    {
		size_t curDOffset = pointOffset + *dOffset;
		
		if (this->pixelsIn[curDOffset] == pVal)
		{
		  T2 outPixVal = this->pixelsOut[curDOffset];
		  if (outPixVal==0)
		    this->pixelsOut[curDOffset] = curLabel;
		  else if (outPixVal != curLabel)
		    pairs.insert(make_pair(max(curLabel, outPixVal), min(curLabel, outPixVal)));
		}
		dOffset++;
	    }
	}
	virtual RES_T finalize(const imageInType &imIn, imageOutType &imOut, const StrElt &se)
	{
	    this->pixelsOut = imOut.getPixels();
	    
	    set<pair<size_t, size_t> >::iterator pair_it = pairs.begin();
	    
	    vector< set<size_t> > stacks;
	    
	    lut.clear();
	    
	    vector< set<size_t> >::iterator stack_it = stacks.begin(), sf1, sf2;
	    
    // 	for (pair_it = pairs.begin();pair_it!=pairs.end();pair_it++)
    // 	    cout << (*pair_it).first << " -> " << (*pair_it).second << endl;
	    
	    
	    pair_it = pairs.begin();
	    while(pair_it!=pairs.end())
	    {
		size_t val1 = (*pair_it).first;
		size_t val2 = (*pair_it).second;
		// find in the stack a set conaining one of the pair values
		stack_it = stacks.begin();
		sf1 = stacks.end();
		sf2 = stacks.end();
		while(stack_it!=stacks.end())
		{
		    if (find((*stack_it).begin(), (*stack_it).end(), val1)!=(*stack_it).end())
		      sf1 = stack_it;
		    if (find((*stack_it).begin(), (*stack_it).end(), val2)!=(*stack_it).end())
		      sf2 = stack_it;
		    
		    stack_it++;
		}
		if (sf1==stacks.end() && sf2==stacks.end()) // not found
		{
		  set<size_t> newSet;
		  newSet.insert(val1);
		  newSet.insert(val2);
		  lut[val1] = 1;
		  lut[val2] = 1;
		  stacks.push_back(newSet);
		}
		else if (sf1!=stacks.end() && sf2!=stacks.end() && sf1!=sf2)
		{
		  (*sf1).insert((*sf2).begin(), (*sf2).end());
		  stacks.erase(sf2);
		}
		else if (sf1!=stacks.end())
		  (*sf1).insert(val2);
		else if (sf2!=stacks.end())
		  (*sf2).insert(val1);
		  
	      pair_it++;
	    }
	    
    // 	cout << "----------" << endl;
    //       
    // 	for (stack_it = stacks.begin();stack_it!=stacks.end();stack_it++)
    // 	{
    // 	  for (set<size_t>::iterator it=(*stack_it).begin();it!=(*stack_it).end();it++)
    // 	    cout << int(*it) << " ";
    // 	  cout << endl;
    // 	}
		
	      
	    map<size_t, set<size_t> *> stackMap;
	    
	    typedef vector< set<size_t> >::iterator stackIterT;
	    typedef set<size_t>::iterator setIterT;
	    
	    for(stack_it=stacks.begin() ; stack_it!=stacks.end() ; stack_it++)
	      stackMap[*(*stack_it).begin()] = &(*stack_it);
	    
	    
	    size_t index = 1;
	    
	    for(size_t i=index;i<=labels;i++)
	    {
		if (lut[i]==0)
		  lut[i] = index++;
		else
		{
		  set<size_t> *curStack = stackMap[i];
		  if (curStack)
		  {
		    for(setIterT set_it=(*curStack).begin() ; set_it!=(*curStack).end() ; set_it++)
		      lut[*set_it] = index;
		    index++;
		  }
		    
		}
	    }
		
    // 	cout << "----------" << endl;
    //       
    // 	for (map<size_t, size_t>::iterator it = lut.begin();it!=lut.end();it++)
    // 	    cout << int((*it).first) << " " << int((*it).second) << endl;
	      
	    for (size_t i=0;i<imOut.getPixelCount();i++)
	      if (this->pixelsOut[i]!=0)
		this->pixelsOut[i] = lut[this->pixelsOut[i]];
	      
	    labels = index-1;
	    
	    return RES_OK;
	}
    protected:
      size_t labels;
      map<size_t, size_t> lut;
      set<pair<size_t, size_t> > pairs;
    };

    template <class T1, class T2>
    class labelFunct_v2 : public unaryMorphImageFunctionBase<T1, T2>
    {
    public:
	typedef unaryMorphImageFunctionBase<T1, T2> parentClass;
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

        virtual void processPixel (size_t &pointOffset, vector<int>::iterator dOffset, vector<int>::iterator dOffsetEnd) 
        {

            T1 pVal = this->pixelsIn[pointOffset];

            if (pVal == T1(0) || this->pixelsOut[pointOffset] != T2(0))
                return;

            queue <size_t> propagation;
            int x, y, z;
            IntPoint p;

            ++labels;
            this->pixelsOut[pointOffset] = labels;
            propagation.push (pointOffset); 
          

            while (!propagation.empty ()) {
                z = propagation.front() / (this->imSize[1]*this->imSize[0]);
                y = (propagation.front() - z*this->imSize[1]*this->imSize[0])/this->imSize[0];
                x = propagation.front() - y*this->imSize[0] - z*this->imSize[1]*this->imSize[0];

                for (UINT i=0; i<this->sePointNbr; ++i) {
                     p = this->sePoints[i];
                     if (x+p.x >= 0 && x+p.x < this->imSize[0] &&
                         y+p.y >= 0 && y+p.y < this->imSize[1] &&
                         z+p.z >= 0 && z+p.z < this->imSize[2] &&
                         this->pixelsIn[x+p.x+(y+p.y)*this->imSize[0]+(z+p.z)*this->imSize[1]*this->imSize[0]] == pVal &&
                         this->pixelsOut[x+p.x+(y+p.y)*this->imSize[0]+(z+p.z)*this->imSize[1]*this->imSize[0]] != labels)
                     {
                         this->pixelsOut[x+p.x+(y+p.y)*this->imSize[0]+(z+p.z)*this->imSize[1]*this->imSize[0]] = labels;
                         propagation.push (x+p.x+(y+p.y)*this->imSize[0]+(z+p.z)*this->imSize[1]*this->imSize[0]);
                     }
                }

                propagation.pop();
            } 


        }

    protected:
        T2 labels;
    };

    template <class T1, class T2>
    class labelFunct_v3 : public unaryMorphImageFunctionBase <T1, T2>
    {
    public:
	typedef unaryMorphImageFunctionBase<T1, T2> parentClass;
	typedef typename parentClass::imageInType imageInType;
	typedef typename parentClass::imageOutType imageOutType;
	typedef typename imageInType::lineType lineInType;
	typedef typename imageInType::sliceType sliceInType;
	typedef typename imageOutType::lineType lineOutType;
	typedef typename imageOutType::sliceType sliceOutType;

        T2 getLabelNbr() { return labels; }

        virtual RES_T initialize (const imageInType &imIn, imageOutType &imOut, const StrElt &se) {
            parentClass::initialize(imIn, imOut, se);
	    fill(imOut, T2(0));
	    labels = T2(0);
	    return RES_OK;
        }

        virtual RES_T processImage (const imageInType &imIn, imageOutType &imOut, const StrElt &se) {
            Image<T1> tmp(imIn);
            Image<T1> tmp2(imIn);
            ASSERT(clone(imIn, tmp)==RES_OK);
            ASSERT(erode (tmp, tmp2, se)==RES_OK); 
            ASSERT(sub(tmp, tmp2, tmp)==RES_OK);
        
            lineInType pixelsTmp = tmp.getPixels () ;
         
            // Adding the first point of each line to tmp.
            #pragma omp parallel
            {
                #pragma omp for
                for (int i=0; i<this->imSize[2]*this->imSize[1]; ++i) {
                    pixelsTmp[i*this->imSize[0]] = this->pixelsIn[i*this->imSize[0]];
                }
            }           
           
              queue <size_t> propagation;
            int x,y,z;
            IntPoint p;

            T2 current_label = labels;
            bool is_not_a_gap = false;
            bool process_labeling = false;

            // First PASS to label the boundaries. //
            for (int i=0; i<this->imSize[2]*this->imSize[1]*this->imSize[0]; ++i) {
                if (pixelsTmp[i] != T1(0)) {
                    if (this->pixelsOut[i] == T2(0)) {
                        if (!is_not_a_gap) {
                            current_label = ++labels;
                        }
                        this->pixelsOut[i] = current_label;
                        process_labeling = true;
                    } else {
                        current_label = this->pixelsOut[i];
                        is_not_a_gap = true;
                    }
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

                       for (UINT i=0; i<this->sePointNbr; ++i) {
                            p = this->sePoints[i]; 
                            if (x+p.x >= 0 && x+p.x < this->imSize[0] &&
                                 y+p.y >= 0 && y+p.y < this->imSize[1] &&
                                 z+p.z >= 0 && z+p.z < this->imSize[2] &&
                                 pixelsTmp[x+p.x + (y+p.y)*this->imSize[0] + (z+p.z)*this->imSize[1]*this->imSize[0]] == pixelsTmp[propagation.front ()] &&
                                 this->pixelsOut[x+p.x + (y+p.y)*this->imSize[0] + (z+p.z)*this->imSize[1]*this->imSize[0]] != current_label)
                             {
                                 this->pixelsOut[x+p.x + (y+p.y)*this->imSize[0] + (z+p.z)*this->imSize[1]*this->imSize[0]] = current_label;
                                 propagation.push (x+p.x + (y+p.y)*this->imSize[0] + (z+p.z)*this->imSize[1]*this->imSize[0]);
                             }
                             
                        }

                        propagation.pop();
                    } 
                    process_labeling = false;
                    is_not_a_gap = false;
                }
            }

            // Propagate labels inside the borders //

            size_t nSlices = imIn.getDepth () ;
            size_t nLines = imIn.getHeight () ;
            size_t nPixels = imIn.getWidth () ;
            int l, v;
            T1 previous_value = this->pixelsIn[0];
            T2 previous_label = this->pixelsOut[0];

            sliceInType srcLines = imIn.getLines () ;
            sliceOutType desLines = imOut.getLines () ;
            lineInType lineIn;
            lineOutType lineOut;

            for (int s=0; s<nSlices; ++s) {
                #pragma omp parallel private(lineIn,lineOut,l,v,previous_value,previous_label)
                {
                    #pragma omp for
                    for (l=0; l<nLines; ++l) {
                        lineIn = srcLines[l];
                        lineOut = desLines[l];
                        previous_value = lineIn[0];
                        previous_label = lineOut[0];
                        for (v=1; v<nPixels; ++v) {
                            if (lineIn[v] == previous_value) {
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
            T2 labels;
    };
     
    /**
    * Image labelization
    * 
    * Return the number of labels (or 0 if error).
    */
    template<class T1, class T2>
    size_t label(const Image<T1> &imIn, Image<T2> &imOut, const StrElt &se=DEFAULT_SE)
    {
	ASSERT_ALLOCATED(&imIn, &imOut);
	ASSERT_SAME_SIZE(&imIn, &imOut);
	
	labelFunct_v2<T1,T2> f;
	
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
    size_t label_v3(const Image<T1> &imIn, Image<T2> &imOut, const StrElt &se=DEFAULT_SE)
    {
	ASSERT_ALLOCATED(&imIn, &imOut);
	ASSERT_SAME_SIZE(&imIn, &imOut);
	
	labelFunct_v3<T1,T2> f;
	
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
    size_t label_v0(const Image<T1> &imIn, Image<T2> &imOut, const StrElt &se=DEFAULT_SE)
    {
	ASSERT_ALLOCATED(&imIn, &imOut);
	ASSERT_SAME_SIZE(&imIn, &imOut);
	
	labelFunct<T1,T2> f;
	
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
    size_t labelWithArea_v0(const Image<T1> &imIn, Image<T2> &imOut, const StrElt &se=DEFAULT_SE)
    {
	ASSERT_ALLOCATED(&imIn, &imOut);
	ASSERT_SAME_SIZE(&imIn, &imOut);
	
	ImageFreezer freezer(imOut);
	
	Image<T2> imLabel(imIn);
	
	ASSERT(label_v0(imIn, imLabel, se)!=0);
 	map<UINT, double> areas = measAreas(imLabel);
	ASSERT(!areas.empty());
	
	ASSERT(applyLookup<T2>(imLabel, areas, imOut)==RES_OK);
	
	return RES_OK;
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
 	map<UINT, double> areas = measAreas(imLabel);
	ASSERT(!areas.empty());
	
	ASSERT(applyLookup<T2>(imLabel, areas, imOut)==RES_OK);
	
	return RES_OK;
    }

    /**
    * Area opening
    * 
    * Remove from image all connected components of size less than \a size pixels
    */
    template<class T>
    size_t areaOpen(const Image<T> &imIn, size_t size, Image<T> &imOut, const StrElt &se=DEFAULT_SE)
    {
	if (&imIn==&imOut)
	{
	    Image<T> tmpIm(imIn, true); // clone
	    return areaOpen(tmpIm, size, imOut, se);
	}
	
	ImageFreezer freezer(imOut);
	
	ASSERT_ALLOCATED(&imIn, &imOut);
	ASSERT_SAME_SIZE(&imIn, &imOut);
	
	Image<size_t> imLabel(imIn);
	
	ASSERT((labelWithArea(imIn, imLabel, se)!=0));
	ASSERT((threshold(imLabel, size, imLabel)==RES_OK));
	ASSERT((copy(imLabel, imOut)==RES_OK));
	ASSERT((inf(imIn, imOut, imOut)==RES_OK));
	
	return RES_OK;
    }

    template <class T1, class T2>
    class neighborsFunct : public unaryMorphImageFunctionBase<T1, T2>
    {
    public:
	typedef unaryMorphImageFunctionBase<T1, T2> parentClass;
	
	virtual inline void processPixel(size_t &pointOffset, vector<int>::iterator dOffset, vector<int>::iterator dOffsetEnd)
	{
	    vector<T1> vals;
	    UINT nbrValues = 0;
	    while(dOffset!=dOffsetEnd)
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

