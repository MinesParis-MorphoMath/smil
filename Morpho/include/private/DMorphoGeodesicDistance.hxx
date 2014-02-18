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


#ifndef _BASE_IMAGE_OPERATIONS_HXX
#define _BASE_IMAGE_OPERATIONS_HXX

#include "Core/include/private/DImage.hpp"

namespace smil
{
    enum _STATE {_UNREACHED, _SCANNED};

    template <class T>
    struct {
        size_t x,y,z;
        // labels.
        T key; 
        _STATE state; 
        T dist;
        T* potential;

        // extract the node with mininum key label.
        bool operator()(const _node &n1, const _node &n2) {
            return n1.key < n2.key;
        }
    } _node;
    
    template <class T>
    struct {
        size_t length;
        struct _node *n;
    } _Partition;

    /**
     *  Partition the set of pixels of imIn into p Partitions such as |V_i| <= B for each V_i.
     *
     *  \param imIn Input Image
     *  \param B number of source vertices simultaneously computed.
     *  \param V _Partition array unallocated.
     *  \param p number of partition so that |V_i| <= B
     */
    template <class T>
    RES_T _createPartitions (const Image<T> &imIn, size_t B, struct _Partition *V, int &p) {
        return RES_OK; 
    }

/*    template <class T>
    RES_T _MSLC (_Partition* V, const StrElt &se=DEFAULT_SE) {
    
    }
*/
} // namespace smil


#endif
