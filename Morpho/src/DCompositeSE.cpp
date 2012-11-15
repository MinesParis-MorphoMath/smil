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

#include "DCompositeSE.h"

extern IntPoint SE_SquIndices[];
extern IntPoint SE_HexIndices[];


int getSEPointIndice(IntPoint &pt, bool oddSE)
{
    if (oddSE)
    {
      for (UINT i=0;i<7;i++)
	  if (SE_HexIndices[i].x==pt.x && SE_HexIndices[i].y==pt.y)
	    return i;
    }
    else
    {
      for (UINT i=0;i<9;i++)
	  if (SE_SquIndices[i].x==pt.x && SE_SquIndices[i].y==pt.y)
	    return i;
    }
    return -1;
}

IntPoint rotatePoint(IntPoint &pt, int steps, bool oddSE)
{
    IntPoint newPt;
    int ind = getSEPointIndice(pt, oddSE);
    if (ind==0)
      return newPt;
    if (oddSE)
	return SE_HexIndices[(ind-1+steps)%6 + 1];
    else
	return SE_SquIndices[(ind-1+steps)%8 + 1];
    
    
}

CompStrElt::CompStrElt(const CompStrElt &rhs)
  : BaseObject("CompStrElt")
{
    fgSE = rhs.fgSE;
    bgSE = rhs.bgSE;
}

CompStrElt::CompStrElt(const StrElt &fg, const StrElt &bg)
  : BaseObject("CompStrElt")
{
    fgSE = fg;
    bgSE = bg;
}

//! Switch foreground/background SE
CompStrElt CompStrElt::operator~()
{
    CompStrElt cSE;
    cSE.fgSE = bgSE;
    cSE.bgSE = fgSE;
    return cSE;
}

//! Counterclockwise rotate SE points
CompStrElt &CompStrElt::rotate(int steps)
{
    bool odd = fgSE.odd;
    for (vector<IntPoint>::iterator it=fgSE.points.begin();it!=fgSE.points.end();it++)
      (*it) = rotatePoint((*it), steps, odd);
    for (vector<IntPoint>::iterator it=bgSE.points.begin();it!=bgSE.points.end();it++)
      (*it) = rotatePoint((*it), steps, odd);
    return *this;	
}

CompStrEltList CompStrElt::operator | (const CompStrElt &rhs)
{
    CompStrEltList cseList(*this);
    cseList.add(rhs);
    return cseList;
}

CompStrEltList CompStrElt::operator () (UINT nrot)
{
    return CompStrEltList(*this, nrot);
}

void CompStrElt::printSelf(ostream &os, string indent)
{
    os << indent << "Composite Structuring Element" << endl;
    os << indent << "Foreground SE:" << endl;
    fgSE.printSelf(os, indent + "\t");
    os << indent << "Background SE:" << endl;
    bgSE.printSelf(os, indent + "\t");
}



CompStrEltList::CompStrEltList(const CompStrEltList &rhs) 
{
    compSeList = rhs.compSeList;
}

CompStrEltList::CompStrEltList(const CompStrElt &compSe) 
{
    compSeList.push_back(compSe);
}

CompStrEltList::CompStrEltList(const CompStrElt &compSe, UINT nrot)
{
    this->add(compSe, nrot);
}

CompStrEltList CompStrEltList::operator~()
{
    CompStrEltList hmtSE;
    for (std::vector<CompStrElt>::const_iterator it=compSeList.begin();it!=compSeList.end();it++)
      hmtSE.add((*it).bgSE, (*it).fgSE);
    return hmtSE;
}

CompStrEltList CompStrEltList::operator | (const CompStrEltList &rhs) 
{
    CompStrEltList hmtSE(*this);
    for (std::vector<CompStrElt>::const_iterator it=rhs.compSeList.begin();it!=rhs.compSeList.end();it++)
      hmtSE.add((*it).fgSE, (*it).bgSE);
    return hmtSE;
}

void CompStrEltList::add(const CompStrElt &cse)
{
    compSeList.push_back(cse);
}

void CompStrEltList::add(const StrElt &fgse, const StrElt &bgse)
{
    compSeList.push_back(CompStrElt(fgse, bgse));
}

void CompStrEltList::add(const StrElt &fgse, const StrElt &bgse, UINT nrot)
{
    CompStrElt compSE(fgse, bgse);
    int steps = fgse.odd ? 6/nrot : 8/nrot;
    compSeList.push_back(compSE);
    for (UINT n=1;n<nrot;n++)
	compSeList.push_back(compSE.rotate(steps));
}

void CompStrEltList::add(const CompStrElt &cse, UINT nrot)
{
    this->add(cse.fgSE, cse.bgSE, nrot);
}

CompStrEltList &CompStrEltList::rotate(int nrot)
{
    CompStrEltList sel;
    for (std::vector<CompStrElt>::iterator it=compSeList.begin();it!=compSeList.end();it++)
      (*it).rotate(nrot);
    return *this;
}

void CompStrEltList::printSelf(ostream &os, string indent)
{
    os << indent << "HitOrMiss SE (composite structuring element list)" << endl;
    int i=0;
    for (std::vector<CompStrElt>::iterator it=compSeList.begin();it!=compSeList.end();it++,i++)
    {
	os << indent << "CompSE #" << i << ":" << endl;
	(*it).printSelf(os, indent + "\t");
    }
}

