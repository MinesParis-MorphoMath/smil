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

CompStrElt::CompStrElt()
  : BaseObject("CompStrElt")
{
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
CompStrElt CompStrElt::rotate(int deg)
{
    StrElt fg, bg;
    bool odd = fgSE.odd;
    fg.odd = bg.odd = odd;
    for (vector<IntPoint>::iterator it=fgSE.points.begin();it!=fgSE.points.end();it++)
      fg.addPoint(rotatePoint((*it), -deg, odd));
    for (vector<IntPoint>::iterator it=bgSE.points.begin();it!=bgSE.points.end();it++)
      bg.addPoint(rotatePoint((*it), -deg, odd));
    return CompStrElt(fg, bg);	
}

void CompStrElt::printSelf(ostream &os, string indent)
{
    os << indent << "Composite Structuring Element" << endl;
    os << indent << "Foreground SE:" << endl;
    fgSE.printSelf(os, indent + "\t");
    os << indent << "Background SE:" << endl;
    bgSE.printSelf(os, indent + "\t");
}



CompStrEltList::CompStrEltList() 
{
}

CompStrEltList::CompStrEltList(const CompStrEltList &rhs) 
{
    compSeList = rhs.compSeList;
}

CompStrEltList::CompStrEltList(const CompStrElt &compSe) 
{
    compSeList.push_back(compSe);
}

CompStrEltList CompStrEltList::operator~()
{
    CompStrEltList hmtSE;
    for (std::list<CompStrElt>::const_iterator it=compSeList.begin();it!=compSeList.end();it++)
      hmtSE.add((*it).bgSE, (*it).fgSE);
    return hmtSE;
}

CompStrEltList CompStrEltList::operator | (const CompStrEltList &rhs) 
{
    CompStrEltList hmtSE(*this);
    for (std::list<CompStrElt>::const_iterator it=rhs.compSeList.begin();it!=rhs.compSeList.end();it++)
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
    int angle = 360 / (nrot+1);
    compSeList.push_back(compSE);
    for (UINT n=0;n<nrot;n++)
    {
	compSeList.push_back(compSE.rotate((n+1)*angle));
    }
}

void CompStrEltList::printSelf(ostream &os, string indent)
{
    os << indent << "HitOrMiss SE (composite structuring element list)" << endl;
    int i=0;
    for (std::list<CompStrElt>::iterator it=compSeList.begin();it!=compSeList.end();it++,i++)
    {
	os << indent << "CompSE #" << i << ":" << endl;
	(*it).printSelf(os, indent + "\t");
    }
}

HMT_hL_SE::HMT_hL_SE()
{
    this->add(StrElt(true, 1, 2, 1,3), StrElt(true, 1, 2, 5,6), 5);
}
