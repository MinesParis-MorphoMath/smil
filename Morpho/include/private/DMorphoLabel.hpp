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
	
	size_t getLabelNbr() { return labels; }

	virtual RES_T initialize(const imageInType &imIn, imageOutType &imOut, const StrElt &se)
	{
	    parentClass::initialize(imIn, imOut, se);
	    fill(imOut, T2(0));
	    labels = 1;
	    return RES_OK;
	}

        virtual void processPixel (size_t &pointOffset, vector<int>::iterator dOffset, vector<int>::iterator dOffsetEnd) 
        {
            T1 pVal = this->pixelsIn[pointOffset];
            vector<int>::iterator dOffsetStart = dOffset;
            
            if (pVal==0)
                return;

            size_t candidates[this->sePointNbr];
            int nbr_candidates = 0;
            size_t curDOffset;

            // Populating canditates, forward scan.
            while (dOffset != dOffsetEnd) {
                 curDOffset = pointOffset + *dOffset; 
                 if (this->pixelsIn[curDOffset] == pVal && this->pixelsOut[curDOffset] != T2(0)) {
                        candidates[nbr_candidates++] = curDOffset;
                 }
                 ++dOffset;
            }

            // No label assigned around the current pixel.
            if (nbr_candidates == 0) {
                this->pixelsOut[pointOffset] = T2(labels);
                ++labels;
            }
            else {
                T2 labelTmp = this->pixelsOut[candidates[0]];
                if (nbr_candidates > 1) {
                    // Keeping the smallest label.
                    for (int i=0; i<nbr_candidates; ++i) {
                        if (this->pixelsOut[candidates[i]] < labelTmp)
                            labelTmp = this->pixelsOut[candidates[i]];
                    }
                }
                // Associating the current pixel to the label.
                this->pixelsOut[pointOffset] = labelTmp;

                if (nbr_candidates > 1) {
                    queue <size_t> propagation;
                    int x, y, z;
                    IntPoint p;

                    // Backward scan.
                    for (int i=0; i<nbr_candidates; ++i) {
                        if (this->pixelsOut[candidates[i]] != labelTmp) {
                            this->pixelsOut[candidates[i]] = labelTmp;
                            propagation.push (candidates[i]);
                        }
                    }

                    // Depth First Search: keep the queue the smallest possible.
                    while (!propagation.empty ()) {
                        z = propagation.back() / (this->imSize[1]*this->imSize[0]);
                        y = (propagation.back() - z*this->imSize[1]*this->imSize[0])/this->imSize[0];
                        x = propagation.back() - y*this->imSize[0] - z*this->imSize[1]*this->imSize[0];

                        for (UINT i=0; i<this->sePointNbr; ++i) {
                             p = this->sePoints[i];
                             if (x+p.x >= 0 && x+p.x < this->imSize[0] &&
                                 y+p.y >= 0 && y+p.y < this->imSize[1] &&
                                 z+p.z >= 0 && z+p.z < this->imSize[2] &&
                                 this->pixelsOut[x+p.x+(y+p.y)*this->imSize[0]+(z+p.z)*this->imSize[1]*this->imSize[0]] > T2(0) &&
                                 this->pixelsOut[x+p.x+(y+p.y)*this->imSize[0]+(z+p.z)*this->imSize[1]*this->imSize[0]] != labelTmp)
                             {
                                 this->pixelsOut[x+p.x+(y+p.y)*this->imSize[0]+(z+p.z)*this->imSize[1]*this->imSize[0]] = labelTmp;
                                 propagation.push (x+p.x+(y+p.y)*this->imSize[0]+(z+p.z)*this->imSize[1]*this->imSize[0]);
                             }
                        }
                        propagation.pop();
                    }
                }
            }
        }

        virtual RES_T finalize (const imageInType &imIn, imageOutType &imOut, const StrElt &se)
        {
 /*           labels=0;
            map <T2, T2> equivalence;
            for (size_t i=0; i<this->imSize[0]*this->imSize[1]*this->imSize[2]; ++i) {
                if (this->pixelsOut[i] != T2(0))
                {
                    if (equivalence.count (this->pixelsOut[i])) {
                        this->pixelsOut[i] = equivalence [this->pixelsOut[i]];
                    } else {
                            ++labels;
                            equivalence[this->pixelsOut[i]] = labels;
                            this->pixelsOut[i] = labels;
                    }
                }
            }
            ++labels;
*/
            return RES_OK;
        }

    protected:
        T2 labels;
    };   
    
    /**
    * Image labelization
    * 
    * Return the number of labels (or 0 if error).
    */
    template<class T1, class T2>
    size_t label_v2(const Image<T1> &imIn, Image<T2> &imOut, const StrElt &se=DEFAULT_SE)
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
    size_t label(const Image<T1> &imIn, Image<T2> &imOut, const StrElt &se=DEFAULT_SE)
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
    * Image labelization with the size of each connected components
    * 
    */
    template<class T1, class T2>
    size_t labelWithArea_v2(const Image<T1> &imIn, Image<T2> &imOut, const StrElt &se=DEFAULT_SE)
    {
	ASSERT_ALLOCATED(&imIn, &imOut);
	ASSERT_SAME_SIZE(&imIn, &imOut);
	
	ImageFreezer freezer(imOut);
	
	Image<T2> imLabel(imIn);
	
	ASSERT(label_v2(imIn, imLabel, se)!=0);
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

