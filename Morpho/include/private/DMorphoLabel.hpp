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
#include "DImageArith.h"
#include "DMorphImageOperations.hpp"


#include <set>
#include <map>

template <class T1, class T2>
class labelFunct : public unaryMorphImageFunctionGeneric<T1, T2>
{
public:
    typedef unaryMorphImageFunctionGeneric<T1, T2> parentClass;
    virtual RES_T initialize(const typename parentClass::imageInType &imIn, typename parentClass::imageOutType &imOut, const StrElt &se)
    {
	parentClass::initialize(imIn, imOut, se);
	fill(imOut, T2(0));
	labels = 0;
	pairs.clear();
	lut.clear();
	return RES_OK;
    }
    
    // The generic way
    virtual inline void processPixel(UINT &pointOffset, vector<UINT>::iterator dOffset, vector<UINT>::iterator dOffsetEnd)
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
	    UINT curDOffset = pointOffset + *dOffset;
	    
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
    virtual RES_T finalize(const typename parentClass::imageInType &imIn, typename parentClass::imageOutType &imOut, StrElt &se)
    {
	this->pixelsOut = imOut.getPixels();
	
	set<pair<UINT, UINT> >::iterator pair_it = pairs.begin();
	
	vector< set<UINT> > stacks;
	
	lut.clear();
	
	vector< set<UINT> >::iterator stack_it = stacks.begin(), sf1, sf2;
	
// 	for (pair_it = pairs.begin();pair_it!=pairs.end();pair_it++)
// 	    cout << (*pair_it).first << " -> " << (*pair_it).second << endl;
	
	
	pair_it = pairs.begin();
	while(pair_it!=pairs.end())
	{
	    UINT val1 = (*pair_it).first;
	    UINT val2 = (*pair_it).second;
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
	      set<UINT> newSet;
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
// 	  for (set<UINT>::iterator it=(*stack_it).begin();it!=(*stack_it).end();it++)
// 	    cout << int(*it) << " ";
// 	  cout << endl;
// 	}
	    
	  
	map<UINT, set<UINT> *> stackMap;
	
	typedef vector< set<UINT> >::iterator stackIterT;
	typedef set<UINT>::iterator setIterT;
	
	for(stack_it=stacks.begin() ; stack_it!=stacks.end() ; stack_it++)
	  stackMap[*(*stack_it).begin()] = &(*stack_it);
	
	
	UINT index = 1;
	
	for(UINT i=index;i<=labels;i++)
	{
	    if (lut[i]==0)
	      lut[i] = index++;
	    else
	    {
	      set<UINT> *curStack = stackMap[i];
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
// 	for (map<UINT, UINT>::iterator it = lut.begin();it!=lut.end();it++)
// 	    cout << int((*it).first) << " " << int((*it).second) << endl;
	  
	for (UINT i=0;i<imOut.getPixelCount();i++,this->pixelsOut++)
	  if (*this->pixelsOut!=0)
	    *this->pixelsOut = lut[*this->pixelsOut];
	
	return RES_OK;
    }
protected:
  UINT labels;
  map<UINT, UINT> lut;
  set<pair<UINT, UINT> > pairs;
};


template<class T1, class T2>
RES_T label(const Image<T1> &imIn, Image<T2> &imOut, const StrElt &se=DEFAULT_SE)
{
    labelFunct<T1,T2> f;
    return f._exec(imIn, imOut, se);
}


#endif // _D_MORPHO_LABEL_HPP

