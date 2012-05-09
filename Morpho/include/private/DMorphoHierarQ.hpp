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
 *     * Neither the name of the University of California, Berkeley nor the
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


#ifndef _D_MORPHO_HIERARQ_HPP
#define _D_MORPHO_HIERARQ_HPP

/**
 * \ingroup Morpho
 * \defgroup HierarQ
 * @{
 */

#include <queue>
#include <deque>

#include "DTypes.hpp"
#include <DStructuringElement.h>



enum HQ_STATUS
{
  HQ_CANDIDATE,
  HQ_QUEUED,
  HQ_LABELED,
  HQ_WS_LINE
};

template <class T>
class HQToken
{
public:
    HQToken(T _value, UINT _offset, UINT _index)
      : value(_value), offset(_offset), index(_index)
    {
    }
    T value;
    UINT offset;
    bool operator > (const HQToken<T> &s ) const 
    {
	T sVal = s.value;
	if (value!=sVal)
	  return value > sVal;
	else return index > s.index;
    }
protected:
    UINT index;
};


template <class T>
class HierarchicalQueue
{
public:
//     typedef typename std::pair<T, UINT> elementType;
    typedef HQToken<T> elementType;
    typedef typename std::vector< elementType > containerType;
    typedef typename std::greater<typename containerType::value_type > compareType;
    
    HierarchicalQueue()
    {
      reset();
    }
    
    void reset()
    {
      while(!priorityQueue.empty())
	priorityQueue.pop();
      index = 0;
    }
    
    inline bool empty()
    {
      return priorityQueue.empty();
    }
    
    inline void push(T value, UINT offset)
    {
      priorityQueue.push(HQToken<T>(value, offset, index++));
    }
    
    inline const elementType& top()
    {
      return priorityQueue.top();
    }
    
    inline void pop()
    {
      priorityQueue.pop();
    }
    
    inline UINT size()
    {
      return priorityQueue.size();
    }
    
    void printSelf()
    {
	while(!priorityQueue.empty())
	{
	    cout << (int)(priorityQueue.top().value) << ", " << (int)(priorityQueue.top().offset) << endl;
	    priorityQueue.pop();
	}
    }
protected:
    priority_queue<elementType, containerType, compareType > priorityQueue;
    UINT index;
};

template <class T, class labelT>
RES_T initHierarchicalQueue(Image<T> &imIn, Image<labelT> &imLbl, Image<UINT8> &imStatus, HierarchicalQueue<T> &hq)
{
    // Empty the priority queue
    hq.reset();
    
    typename ImDtTypes<T>::lineType inPixels = imIn.getPixels();
    typename ImDtTypes<labelT>::lineType lblPixels = imLbl.getPixels();
    typename ImDtTypes<UINT8>::lineType statPixels = imStatus.getPixels();
    
    UINT x, y, z;
    UINT s[3];
    
    imIn.getSize(s);
    UINT offset = 0;
    
    for (UINT k=0;k<s[2];k++)
      for (UINT j=0;j<s[1];j++)
	for (UINT i=0;i<s[0];i++)
	{
	  if (*lblPixels!=0)
	  {
	      hq.push(*inPixels, offset);
	      *statPixels = HQ_LABELED;
	  }
	  else 
	  {
	      *statPixels = HQ_CANDIDATE;
	  }
	  inPixels++;
	  lblPixels++;
	  statPixels++;
	  offset++;
	}
    
}

/** @}*/

#endif // _D_MORPHO_HIERARQ_HPP

