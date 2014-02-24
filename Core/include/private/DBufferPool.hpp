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


#ifndef _DBUFFER_POO_HPP
#define _DBUFFER_POO_HPP

#include <limits>
#include <stack>

#include "DMemory.hpp"
#include "DTypes.hpp"
#include "DErrors.h"

namespace smil
{
    
    template <class T>
    class BufferPool
    {
    public:
      BufferPool(size_t bufSize=0)
	: mutexLock(false),
	  bufferSize(bufSize),
	  numberOfBuffers(0),
	  maxNumberOfBuffers(std::numeric_limits<size_t>::max())
      {}
      ~BufferPool()
      {}

      bool mutexLock;
      
      typedef typename ImDtTypes<T>::lineType bufferType;
      
      RES_T initialize(size_t bufSize, size_t nbr=0)
      {
	  if (buffers.size()!=0)
	    clear();
	  this->bufferSize = bufSize;
      }
      void clear()
      {
	  while(mutexLock); 
	  mutexLock = true;
	  
	  for (typename vector<bufferType>::iterator it=buffers.begin();it!=buffers.end();it++)
	    ImDtTypes<T>::deleteLine(*it);
	  buffers.clear();
	  while (!availableBuffers.empty())
	    availableBuffers.pop();
	  
	  mutexLock = false;
      }
      void setMaxNumberOfBuffers(size_t nbr)
      {
	  maxNumberOfBuffers = nbr;
      }
      size_t getMaxNumberOfBuffers()
      {
	  return maxNumberOfBuffers;
      }
      inline bufferType getBuffer()
      {
	  while(mutexLock); 
	  mutexLock = true;
	  
	  if (availableBuffers.empty())
	    createBuffer();
	  
	  while (availableBuffers.empty()); // wait
	  
	  bufferType buf = availableBuffers.top();
	  availableBuffers.pop();
	  
	  mutexLock = false;

	  return buf;
      }
      inline void releaseBuffer(bufferType &buf)
      {
	  while(mutexLock); 
	  mutexLock = true;
	  
	  availableBuffers.push(buf);
	  buf = NULL;
	  
	  mutexLock = false;
      }
      void releaseAllBuffers()
      {
	  while(mutexLock); 
	  mutexLock = true;
	  
	  while (!availableBuffers.empty())
	    availableBuffers.pop();
	  for (typename vector<bufferType>::iterator it=buffers.begin();it!=buffers.end();it++)
	    availableBuffers.push(*it);
	  
	  mutexLock = false;
      }
    protected:
      bool createBuffer()
      {
	  if (buffers.size()>=maxNumberOfBuffers)
	    return false;
	    
	  bufferType buf = ImDtTypes<T>::createLine(this->bufferSize);
	  buffers.push_back(buf);
	  availableBuffers.push(buf);
	  
      }
      stack<bufferType> availableBuffers;
      vector<bufferType> buffers;
      size_t bufferSize;
      size_t maxNumberOfBuffers;
      size_t numberOfBuffers;
    };

} // namespace smil

#endif // _DBUFFER_POO_HPP

