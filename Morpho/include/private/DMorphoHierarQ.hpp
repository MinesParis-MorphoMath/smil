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

#include "DTypes.hpp"


template <class T>
class PriorityQueueToken
{
public:
  
};

template <class T, class labelT>
RES_T initPriorityQueue(Image<T> &imIn, Image<labelT> &imLbl, priority_queue<T> &pq)
{
    // Empty the priority queue
    pq.empty();
    
    typename ImDtTypes<T>::lineType pixels = imIn.getPixels();
    
    for (int i=0;i<imIn.getPixelCount();i++)
    {
    }
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

