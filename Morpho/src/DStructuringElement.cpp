/*
 * Copyright (c) 2011-2015, Matthieu FAESSEL and ARMINES
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

#include <algorithm>

#include "Morpho/include/DStructuringElement.h"

using namespace smil;

IntPoint SE_SquIndices[] = { IntPoint(0,0,0), IntPoint(1,0,0), IntPoint(1,-1,0), 
      IntPoint(0,-1,0), IntPoint(-1,-1,0), IntPoint(-1,0,0), 
      IntPoint(-1,1,0), IntPoint(0,1,0), IntPoint(1,1,0) };
IntPoint SE_HexIndices[] = { IntPoint(0,0,0), IntPoint(1,0,0), IntPoint(0,-1,0), 
      IntPoint(-1,-1,0), IntPoint(-1,0,0), IntPoint(-1,1,0), IntPoint(0,1,0) };



StrElt& StrElt::operator=(const StrElt &rhs)
{
    this->clone(rhs);
    return *this;
}

void StrElt::clone(const StrElt &rhs)
{
    this->seT = rhs.seT;
    this->size = rhs.size;
    this->odd = rhs.odd;
    this->points = rhs.points;
}

void StrElt::addPoint(const UINT index)
{
    if (odd)
      addPoint(SE_HexIndices[index]);
    else
      addPoint(SE_SquIndices[index]);
}

void StrElt::addPoint(int x, int y, int z)
{
    IntPoint p(x,y,z);
    if (find(points.begin(), points.end(), p)==points.end())
      this->points.push_back(p);
}

void StrElt::addPoint(const IntPoint &pt)
{
    if (find(points.begin(), points.end(), pt)==points.end())
      points.push_back(pt);
}

StrElt StrElt::homothety(const UINT s) const
{
    StrElt newSE;;
    newSE.points = this->points;
    newSE.odd = odd;
    int oddLine = 0;
    for (UINT i=0;i<s-1;i++)
    {
        vector<IntPoint> pts = newSE.points;
        for(vector<IntPoint>::iterator it = pts.begin();it!=pts.end();it++)
        {
          const IntPoint &p = *it;
          for(vector<IntPoint>::const_iterator it2 = points.begin();it2!=points.end();it2++)
          {
              const IntPoint &p2 = *it2;
              if (odd)
                oddLine = (p2.z+1)%2 && p2.y%2 && p.y%2;
              newSE.addPoint(p2.x+p.x+oddLine, p2.y+p.y, p2.z+p.z);
          }
        }
    }
    return newSE;
}

const StrElt StrElt::operator()(int s) const
{
    StrElt se(*this);
    se.size = s;
    return se;
}

// Transpose points
StrElt StrElt::transpose() const
{
    StrElt se;
    se.seT = this->seT;
    se.size = this->size;
    se.odd = this->odd;

    for (vector<IntPoint>::const_iterator it=this->points.begin();it!=this->points.end();it++)
    {
        const IntPoint &p = *it;
        se.addPoint(-p.x - (this->odd && p.y%2), -p.y, -p.z);
    }
    
    return se;
}

// Remove central pixel
StrElt StrElt::noCenter() const
{
    StrElt se;

    se.odd = this->odd;
    se.seT = this->seT;
    se.size = this->size;

    vector < IntPoint >::const_iterator it_start = this->points.begin () ;
    vector < IntPoint >::const_iterator it_end = this->points.end () ;
    vector < IntPoint >::const_iterator it;

    for ( it=it_start; it!=it_end; ++it )
        if (it->x != 0 || it->y != 0 || it->z != 0)
        {
            se.addPoint (*it);
        }
    return se;
}

void StrElt::printSelf(ostream &os, string indent) const
{
    os << indent << "Structuring Element" << endl;
    os << indent << "Type: " << seT << endl;
    os << indent << "Size: " << size << endl;
    size_t ptNbr = points.size();
    os << indent << "Point Nbr: " << ptNbr << endl;
    if (!ptNbr)
      return;
    
      for (UINT i=0;i<ptNbr;i++)
        os << indent << "#" << i+1 << ": (" << points[i].x << "," << points[i].y << "," << points[i].z << ")" << endl;
      
}

