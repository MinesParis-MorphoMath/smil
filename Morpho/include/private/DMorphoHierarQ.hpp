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


#ifndef _D_MORPHO_HIERARQ_HPP
#define _D_MORPHO_HIERARQ_HPP


#include <queue>
#include <deque>

#include "Core/include/private/DTypes.hpp"
#include "Morpho/include/DStructuringElement.h"


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

    /**
     * Preallocated FIFO Queue
     */
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
        virtual ~FIFO_Queue()
        {
            if (data)
              delete[] data;
        }
        
        void reset()
        {
            if (data)
              delete[] data;
            data = NULL;
            _size = realSize = 0;
        }
        void initialize(size_t newSize)
        {
            reset();
            realSize = max( newSize+1, size_t(8) ); // Avoid to have to small buffer
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
            memmove(data, data+first, (last-first)*sizeof(T));
            first = 0;
            last = _size;
        }
        void push(T val)
        {
            if (last==realSize)
              swap();
            data[last++] = val;
            _size++;
        }
        inline T front()
        {
            return data[first];
        }
        inline void pop()
        {
            _size--;
            first++;
        }
        
      static const bool preallocate = true;
      
      protected:
        size_t _size;
        size_t realSize;
        size_t first;
        size_t last;
        T* data;
    };

    template <class TokenType=size_t>
    class STD_Queue : public queue<TokenType>
    {
    public:
      // Dummy constructor for compatibilty with FIFO_Queue one
      STD_Queue(size_t /*newSize*/=0)
        : queue<TokenType>()
        {
        }
      static const bool preallocate = false;
    };

    template <class TokenType=size_t>
    class STD_Stack : public stack<TokenType>
    {
    public:
        // Dummy operator for compatibility with other containers from smil
        inline TokenType front () 
        {
            return this->top();
        }
    };
    
    template <class T, class TokenType=size_t, class StackType=STD_Queue<TokenType> >
    class HierarchicalQueue
    {
    private:
      
        size_t GRAY_LEVEL_NBR;
        size_t GRAY_LEVEL_MIN;
        StackType **stacks;
        size_t *tokenNbr;
        size_t size;
        size_t higherLevel;
        
        bool initialized;
        const bool reverseOrder;
        
    public:
        HierarchicalQueue(bool rOrder=false)
          : reverseOrder(rOrder)
        {
            stacks = NULL;
            tokenNbr = NULL;
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
                {
                  delete stacks[i];
                  stacks[i] = NULL;
                }
            }
            
            initialized = false;
        }
        
        void initialize(const Image<T> &img)
        {
            if (initialized)
              reset();
            
//             vector<T> rVals = rangeVal(img);
            
            GRAY_LEVEL_MIN = ImDtTypes<T>::min();
            GRAY_LEVEL_NBR = ImDtTypes<T>::cardinal();
            
            stacks = new StackType*[GRAY_LEVEL_NBR]();
            tokenNbr = new size_t[GRAY_LEVEL_NBR];
            
            if (StackType::preallocate)
            {
                size_t *h = new size_t[GRAY_LEVEL_NBR];
                histogram(img, h);

                for(size_t i=0;i<GRAY_LEVEL_NBR;i++)
                {
                  if (h[i]!=0)
                    stacks[i] = new StackType(h[i]);
                  else stacks[i] = NULL;
                }
                    
                delete[] h;
            }
            else
            {
                for(size_t i=0;i<GRAY_LEVEL_NBR;i++)
                  stacks[i] = new StackType();
            }
            
            memset(tokenNbr, 0, GRAY_LEVEL_NBR*sizeof(size_t));
            size = 0;
            
            if (reverseOrder)
              higherLevel = 0;
            else
              higherLevel = ImDtTypes<T>::max();
            
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
            return GRAY_LEVEL_MIN + higherLevel;
        }
        
        inline void push(T value, TokenType dOffset)
        {
            size_t level = size_t(value) - GRAY_LEVEL_MIN;
            if (reverseOrder)
            {
                if (level>higherLevel)
                  higherLevel = level;
            }
            else
            {
                if (level<higherLevel)
                  higherLevel = level;
            }
            stacks[level]->push(dOffset);
            tokenNbr[level]++;
            size++;
        }
        inline void findNewReferenceLevel()
        {
            if (reverseOrder)
            {
                for (size_t i=higherLevel-1;i!=numeric_limits<size_t>::max();i--)
                  if (tokenNbr[i]>0)
                  {
                      higherLevel = i;
                      break;
                  }
            }
            else
            {
                for (size_t i=higherLevel+1;i<GRAY_LEVEL_NBR;i++)
                  if (tokenNbr[i]>0)
                  {
                      higherLevel = i;
                      break;
                  }
            }
        }
        
        inline TokenType pop()
        {
            size_t hlSize = tokenNbr[higherLevel];
            TokenType dOffset = stacks[higherLevel]->front();
            stacks[higherLevel]->pop();
            size--;
          
            if (hlSize>1)
              tokenNbr[higherLevel]--;
            else if (size>0) // Find new ref level (non empty stack)
            {
                tokenNbr[higherLevel] = 0;
                findNewReferenceLevel();
            }
            else // Empty -> re-initilize
            {            
                tokenNbr[higherLevel] = 0;
                if (reverseOrder)
                    higherLevel = 0;
                else
                    higherLevel = ImDtTypes<size_t>::max();
          }
          return dOffset;
        }      
    };


/** @}*/

} // namespace smil



#endif // _D_MORPHO_HIERARQ_HPP

