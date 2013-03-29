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


    template <class T, class compareType=std::less<T> >
    class HierarchicalQueue
    {
    public:
	
	typedef size_t elementType;
	typedef typename std::queue< elementType > containerType;
	typedef map< T, containerType, compareType > pqueueType;
	
	
	HierarchicalQueue()
	{
	}
	
	void reset()
	{
	  while(!priorityQueue.empty())
	    pop();
	}
	
	
	inline bool empty()
	{
	  return priorityQueue.empty();
	}
	
	inline void push(T value, size_t offset)
	{
	    priorityQueue[value].push(offset);
	}
	
	inline const containerType& topStack()
	{
	    return priorityQueue.begin()->second;
	}
	
	inline const elementType& top()
	{
	    if (!priorityQueue.empty())
	      return topStack().front();
	}
	
	inline void popStack()
	{
	    priorityQueue.erase(priorityQueue.begin());
	}
	
	inline void pop()
	{
	    priorityQueue.begin()->second.pop();
	    if (priorityQueue.begin()->second.empty())
	      popStack();
	}
	
// 	inline size_t size()
// 	{
// 	  return priorityQueue.size();
// 	}
	
	inline void printSelf()
	{
	    HierarchicalQueue tmpQueue(*this);
	    
	    while(!tmpQueue.empty())
	    {
		cout << (int)tmpQueue.top() << endl;
		tmpQueue.pop();
	    }
	}
    protected:
	pqueueType priorityQueue;
    };

/** @}*/

} // namespace smil



#endif // _D_MORPHO_HIERARQ_HPP

