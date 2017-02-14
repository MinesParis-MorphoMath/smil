/*
 * Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES
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


#ifndef _DBUFFER_POOL_HPP
#define _DBUFFER_POOL_HPP

#include <limits>
#include <stack>

#include "Core/include/private/DMemory.hpp"
#include "Core/include/private/DTypes.hpp"
#include "Core/include/DErrors.h"

namespace smil
{
    
    template <class T>
    class BufferPool
    {
    public:
      BufferPool(size_t bufSize=0)
        : bufferSize(bufSize),
          numberOfBuffers(0),
          maxNumberOfBuffers(std::numeric_limits<size_t>::max())
      {}
      ~BufferPool()
      {}

      typedef typename ImDtTypes<T>::lineType bufferType;
      
      RES_T initialize(size_t bufSize, __attribute__((__unused__))size_t nbr=0)
      {
          if (buffers.size()!=0)
            clear();
          this->bufferSize = bufSize;
          return RES_OK;
      }
      void clear()
      {
          #ifdef USE_OPEN_MP
            #pragma omp critical
          #endif // USE_OPEN_MP
          {
              while (!availableBuffers.empty())
              {
                  ImDtTypes<T>::deleteLine(availableBuffers.front());
                  availableBuffers.pop();
              }
          }
      }
      void setMaxNumberOfBuffers(size_t nbr)
      {
          maxNumberOfBuffers = nbr;
      }
      size_t getMaxNumberOfBuffers()
      {
          return maxNumberOfBuffers;
      }
      bufferType getBuffer()
      {
        bufferType buf;
        
        #ifdef USE_OPEN_MP
          #pragma omp critical
        #endif // USE_OPEN_MP
        {
            if (availableBuffers.empty())
            {
                if (!createBuffer())
                  while (availableBuffers.empty()); // wait
            }
            
            buf = availableBuffers.top();
            availableBuffers.pop();
          
        }
        return buf;
      }
      vector<bufferType> getBuffers(size_t nbr)
      {
        vector<bufferType> buffVect;
        
        for (int i=0;i<nbr;i++)
          buffVect.push_back(this->getBuffer());

        return buffVect;
      }
      
      void releaseBuffer(bufferType &buf)
      {
          availableBuffers.push(buf);
          buf = NULL;
      }
      void releaseBuffers(vector<bufferType> &bufs)
      {
          for (int i=0;i<bufs.size();i++)
            availableBuffers.push(bufs[i]);
          bufs.clear();
      }
      void releaseAllBuffers()
      {
          while (!availableBuffers.empty())
            availableBuffers.pop();
          for (typename vector<bufferType>::iterator it=buffers.begin();it!=buffers.end();it++)
            availableBuffers.push(*it);
      }
    protected:
      bool createBuffer()
      {
          if (buffers.size()>=maxNumberOfBuffers)
            return false;
            
          bufferType buf = ImDtTypes<T>::createLine(this->bufferSize);
          buffers.push_back(buf);
          availableBuffers.push(buf);
        
          return true;
      }
      stack<bufferType> availableBuffers;
      vector<bufferType> buffers;
      size_t bufferSize;
      size_t numberOfBuffers;
      size_t maxNumberOfBuffers;
    };

} // namespace smil

#endif // _DBUFFER_POOL_HPP

