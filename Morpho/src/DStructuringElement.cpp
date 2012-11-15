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


#include "DStructuringElement.h"

IntPoint SE_SquIndices[] = { IntPoint(0,0,0), IntPoint(1,0,0), IntPoint(1,-1,0), 
      IntPoint(0,-1,0), IntPoint(-1,-1,0), IntPoint(-1,0,0), 
      IntPoint(-1,1,0), IntPoint(0,1,0), IntPoint(1,1,0) };
IntPoint SE_HexIndices[] = { IntPoint(0,0,0), IntPoint(1,0,0), IntPoint(0,-1,0), 
      IntPoint(-1,-1,0), IntPoint(-1,0,0), IntPoint(-1,1,0), IntPoint(0,1,0) };


StrElt::StrElt(UINT s)
  : BaseObject("StrElt"),
    seT(SE_Generic), 
    size(s), 
    odd(false)
{
}

StrElt::StrElt(bool oddSE, UINT nbrPts, ...)
  : BaseObject("StrElt"),
    seT(SE_Generic), 
    size(1), 
    odd(oddSE)
{
    UINT indice;
    va_list vl;
    va_start(vl, nbrPts);
    
    for (UINT i=0;i<nbrPts;i++)
    {
	indice = va_arg(vl, UINT);
	if (odd)
	  addPoint(SE_HexIndices[indice]);
	else
	  addPoint(SE_SquIndices[indice]);
    }
}

StrElt::StrElt(const StrElt &rhs)
  : BaseObject(rhs)
{
    this->clone(rhs);
}

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

void StrElt::addPoint(int x, int y, int z)
{
    IntPoint p;
    p.x = x;
    p.y = y;
    p.z = z;
    points.push_back(p);
}

void StrElt::addPoint(const IntPoint &pt)
{
    points.push_back(pt);
}

const StrElt StrElt::operator()(int s)
{
    StrElt se(*this);
    se.size = s;
    return se;
}

void StrElt::printSelf(ostream &os, string indent)
{
    os << indent << "Structuring Element" << endl;
    os << indent << "Size: " << size << endl;
    int ptNbr = points.size();
    os << indent << "Point Nbr: " << ptNbr << endl;
    if (ptNbr)
      for (int i=0;i<ptNbr;i++)
	os << indent << "#" << i+1 << ": (" << points[i].x << "," << points[i].y << "," << points[i].z << ")" << endl;
      
}

