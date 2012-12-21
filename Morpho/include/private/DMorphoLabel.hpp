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

#include "DImage.h"
#include "DMorphImageOperations.hpp"
#include "Base/include/private/DLabelMeasures.hpp"


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
    class labelFunct : public unaryMorphImageFunctionGeneric<T1, T2>
    {
    public:
	typedef unaryMorphImageFunctionGeneric<T1, T2> parentClass;
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
		// find in the stack a set containing one of the pair values
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
	
	ASSERT((label(imIn, imLabel, se)!=0));
 	map<T2, size_t> areas = measAreas(imLabel);
	ASSERT(!areas.empty());
	
	// Verify that the max(areas) doesn't exceed the T2 type max
	typename map<T2,size_t>::iterator max_it = std::max_element(areas.begin(), areas.end());
	ASSERT(( (*max_it).second < double(ImDtTypes<T2>::max()) ), "Area max exceeds data type max!", RES_ERR);

	// Convert areas map into a lookup
	map<T2, T2> lookup;
	for (typename map<T2,size_t>::iterator it = areas.begin();it!=areas.end();it++)
	  lookup[(*it).first] = T2((*it).second);
	
	ASSERT((fill(imOut, T2(0))==RES_OK));
	ASSERT((applyLookup<T2>(imLabel, lookup, imOut)==RES_OK));
	
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
	    return areaOpen(tmpIm, size, imOut);
	}
	
	ASSERT_ALLOCATED(&imIn, &imOut);
	ASSERT_SAME_SIZE(&imIn, &imOut);
	
	Image<size_t> imLabel(imIn);
	
	ASSERT((labelWithArea(imIn, imLabel, se)!=0));
	ASSERT((threshold(imLabel, size, imLabel)==RES_OK));
	ASSERT((copy(imLabel, imOut)==RES_OK));
	ASSERT((inf(imIn, imOut, imOut)==RES_OK));
	
	return RES_OK;
    }
    
/** \} */

} // namespace smil

#endif // _D_MORPHO_LABEL_HPP

