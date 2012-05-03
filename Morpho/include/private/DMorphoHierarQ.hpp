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
RES_T initHierarchicalQueue(Image<T> &imIn, Image<labelT> &imLbl, Image<UINT8> &imStatus, HierarchicalQueue<T> *hq)
{
    // Empty the priority queue
    hq->reset();
    
    typename ImDtTypes<T>::lineType inPixels = imIn.getPixels();
    typename ImDtTypes<labelT>::lineType lblPixels = imLbl.getPixels();
    typename ImDtTypes<labelT>::lineType statPixels = imStatus.getPixels();
    
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
	      hq->push(*inPixels, offset);
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

template <class T, class labelT>
RES_T processHierarchicalQueue(Image<T> &imIn, Image<labelT> &imLbl, Image<UINT8> &imStatus, HierarchicalQueue<T> *hq, StrElt *e)
{
    typename ImDtTypes<T>::lineType inPixels = imIn.getPixels();
    typename ImDtTypes<labelT>::lineType lblPixels = imLbl.getPixels();
    typename ImDtTypes<labelT>::lineType statPixels = imStatus.getPixels();
    
    vector<int> dOffsets;
    
    vector<Point>::iterator it_start = e->points.begin();
    vector<Point>::iterator it_end = e->points.end();
    vector<Point>::iterator it;
    
    vector<UINT> tmpOffsets;
    
    UINT s[3];
    imIn.getSize(s);
    
    // set an offset distance for each se point
    for(it=it_start;it!=it_end;it++)
    {
	dOffsets.push_back(it->x - it->y*s[0] + it->z*s[0]*s[1]);
    }
    
    vector<int>::iterator it_off_start = dOffsets.begin();
    vector<int>::iterator it_off;
    
    
    while(!hq->empty())
    {
	
	HQToken<T> token = hq->top();
	UINT x0, y0, z0;
	
	UINT curOffset = token.offset;
	
	
	imIn.getCoordsFromOffset(curOffset, x0, y0, z0);
	
	int x, y, z;
	UINT nbOffset;
	UINT8 nbStat;
	
	if (token.value > 3)
	  break;
	
	int oddLine = e->odd * y0%2;
	
	for(it=it_start,it_off=it_off_start;it!=it_end;it++,it_off++)
	    if (it->x!=0 || it->y!=0 || it->z!=0) // useless if x=0 & y=0 & z=0
	{
	    
	    x = x0 + it->x;
	    y = y0 - it->y;
	    z = z0 + it->z;
	    
	    if (oddLine)
	      x += (y+1)%2;
	  
	    if (x>=0 && x<s[0] && y>=0 && y<s[1] && z>=0 && z<s[2])
	    {
		nbOffset = curOffset + *it_off;
		
		if (oddLine)
		  nbOffset += (y+1)%2;
		
		nbStat = statPixels[nbOffset];
		
		if (nbStat==HQ_CANDIDATE) // Add it to the tmp offsets queue
		    tmpOffsets.push_back(nbOffset);
		else if (nbStat==HQ_LABELED)
		{
		    if (statPixels[curOffset]==HQ_LABELED)
		    {
			if (lblPixels[curOffset]!=lblPixels[nbOffset])
			    statPixels[curOffset] = HQ_WS_LINE;
		    }
		    else if (statPixels[curOffset]!=HQ_WS_LINE)
		    {
		      statPixels[curOffset] = HQ_LABELED;
		      lblPixels[curOffset] = lblPixels[nbOffset];
		    }
		}
		
	    }
	}

	if (statPixels[curOffset]==HQ_LABELED && !tmpOffsets.empty())
	{
	    typename vector<UINT>::iterator t_it = tmpOffsets.begin();
	    while (t_it!=tmpOffsets.end())
	    {
		hq->push(inPixels[*t_it], *t_it);
		statPixels[*t_it] = HQ_QUEUED;
		
		t_it++;
	    }
	    
	    tmpOffsets.clear();
	}
	hq->pop();
    }
    imLbl.printSelf(1);
    imStatus.printSelf(1);
    hq->printSelf();
}

// /**
//  * Initializes the hierarchical list with the marker image
//  * \param local_ctx pointer to the structure holding all the information needed 
//  * by the algorithm
//  */
// static INLINE void MB_HierarchyInit(MB_Watershed_Ctx *local_ctx)
// {
//     Uint32 i,j;
//     PIX32 *p;
//     
//     /*All the control are reset */
//     for(i=0;i<256;i++) {
//         local_ctx->HierarchicalList[i].firstx = local_ctx->HierarchicalList[i].lastx = MB_LIST_END;
//         local_ctx->HierarchicalList[i].firsty = local_ctx->HierarchicalList[i].lasty = MB_LIST_END;
//     }
//      
//     /* The first marker are inserted inside the hierarchical list */
//     local_ctx->current_water_level = 0;
//     for(i=0; i<local_ctx->height; i++) {
//         for(j=0; j<local_ctx->bytes_marker; j+=4) {
//              p = (PIX32 *) (local_ctx->plines_marker[i] + local_ctx->linoff_marker + j);
//              if ((*p)!=0) {
//                  MB_InsertInHierarchicalList(local_ctx,j/4,i,0);
//              }
//         }
//     }
// }
// 

#endif // _D_MORPHO_HIERARQ_HPP

