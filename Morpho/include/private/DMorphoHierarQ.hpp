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


#ifndef _D_MORPHO_HIERARQ_HPP
#define _D_MORPHO_HIERARQ_HPP


#include <queue>
#include <deque>

#include "Core/include/private/DTypes.hpp"
#include <DStructuringElement.h>


namespace smil
{
    /**
    * \ingroup Morpho
    * \defgroup HierarQ
    * @{
    */


    enum HQ_STATUS
    {
      HQ_CANDIDATE,
      HQ_QUEUED,
      HQ_LABELED,
      HQ_WS_LINE,
      HQ_FINAL
    };

    template <class T>
    class FIFO_Queue
    {
      public:
	FIFO_Queue(size_t newSize=0)
	{
	    data = NULL;
	    if (newSize)
	      initialize(newSize);
	}
	void reset()
	{
	    if (data)
	      delete[] data;
	}
	void initialize(size_t newSize)
	{
	    reset();
	    realSize = newSize+1;
	    data = new T[realSize];
	    _size = 0;
	    first = 0;
	    last = 0;
	}
	inline size_t size()
	{
	    return _size;
	}
	void swap()
	{
	    memcpy(data, data+first, (last-first)*sizeof(T));
	    first = 0;
	    last = _size;
	}
	void push(T val)
	{
	    if (last>=realSize-1)
	      swap();
	    data[last++] = val;
	    _size++;
	}
	inline T pop()
	{
	    _size--;
	    return data[first++];
	}
      protected:
	size_t _size;
	size_t realSize;
	size_t first;
	size_t last;
	T* data;
    };

    template <class T, class TokenType=UINT>
    class HierarchicalQueue
    {
    private:
// 	typedef FIFO_Queue<TokenType> StackType;
	typedef queue<TokenType> StackType;
	
	size_t GRAY_LEVEL_NBR;
	size_t TYPE_FLOOR;
	StackType **stacks;
	size_t *tokenNbr;
	size_t size;
	size_t referenceLevel;
	
	bool initialized;
	const bool reverseOrder;
	size_t h[256];
	
    public:
	HierarchicalQueue(bool rOrder=false)
	  : reverseOrder(rOrder)
	{
	    GRAY_LEVEL_NBR = ImDtTypes<T>::max()-ImDtTypes<T>::min()+1;
	    TYPE_FLOOR = -ImDtTypes<T>::min();
	    
	    stacks = new StackType*[GRAY_LEVEL_NBR]();
	    tokenNbr = new size_t[GRAY_LEVEL_NBR];
	    initialized = false;
	}
	~HierarchicalQueue()
	{
	    reset();
	    delete[] stacks;
	    delete[] tokenNbr;
	}
	
	void reset()
	{
	    if (!initialized)
	      return;
	    
	    for(size_t i=0;i<GRAY_LEVEL_NBR;i++)
	    {
		if (stacks[i])
		    delete stacks[i];
		stacks[i] = NULL;
	    }
	    
	    initialized = false;
	}
	
	void initialize(const Image<T> &img)
	{
	    if (initialized)
	      reset();
	    
// 	    size_t *h = new size_t[GRAY_LEVEL_NBR];
	    histogram(img, h);

	    for(size_t i=0;i<GRAY_LEVEL_NBR;i++)
// 		if (h[i]!=0)
// 		  stacks[i] = new StackType(h[i]);
		  stacks[i] = new StackType();
		
// 	    delete[] h;
	    memset(tokenNbr, 0, GRAY_LEVEL_NBR*sizeof(size_t));
	    size = 0;
	    
	    if (reverseOrder)
	      referenceLevel = 0;
	    else
	      referenceLevel = ImDtTypes<T>::max();
	    
	    initialized = true;
	}
	
	inline size_t getSize()
	{
	    return size;
	}
	
	inline bool isEmpty()
	{
	    return size==0;
	}
	
	inline size_t getHigherLevel()
	{
	    return referenceLevel;
	}
	
	inline void push(T value, TokenType dOffset)
	{
	    size_t level = TYPE_FLOOR + value;
	    if (reverseOrder)
	    {
		if (level>referenceLevel)
		  referenceLevel = level;
	    }
	    else
	    {
		if (level<referenceLevel)
		  referenceLevel = level;
	    }
	    stacks[level]->push(dOffset);
	    tokenNbr[level]++;
	    size++;
	}
	inline void findNewReferenceLevel()
	{
	    if (reverseOrder)
	    {
		for (size_t i=referenceLevel-1;i>=0;i--)
		  if (tokenNbr[i]>0)
		  {
		      referenceLevel = i;
		      break;
		  }
	    }
	    else
	    {
		for (size_t i=referenceLevel+1;i<GRAY_LEVEL_NBR;i++)
		  if (tokenNbr[i]>0)
		  {
		      referenceLevel = i;
		      break;
		  }
	    }
	}
	
	inline TokenType pop()
	{
	    size_t hlSize = tokenNbr[referenceLevel];
	    TokenType dOffset = stacks[referenceLevel]->front();
	    stacks[referenceLevel]->pop();
	    if (hlSize>1)
	      tokenNbr[referenceLevel]--;
	    else if (size>1) // Find new ref level (non empty stack)
	    {
		tokenNbr[referenceLevel] = 0;
		findNewReferenceLevel();
	    }
	    size--;
	    
	    return dOffset;
	}
      
    };


/** @}*/

} // namespace smil



#endif // _D_MORPHO_HIERARQ_HPP

