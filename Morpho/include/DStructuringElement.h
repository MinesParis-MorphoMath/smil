/*
 * Smil
 * Copyright (c) 2010 Matthieu Faessel
 *
 * This file is part of Smil.
 *
 * Smil is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * Smil is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Smil.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 */


#ifndef _STR_ELT_HPP
#define _STR_ELT_HPP

struct Point
{
  int x;
  int y;
  int z;
};

enum seType { stGeneric, stHexSE, stSquSE };

class StrElt
{
  public:
    StrElt(UINT s=1) : seT(stGeneric), size(s) 
    {
    }
    vector<Point> points;
    inline void addPoint(int x, int y, int z=0)
    {
	Point p;
	p.x = x;
	p.y = y;
	p.z = z;
	points.push_back(p);
    }
    inline StrElt& operator()(int s=1)
    {
	this->size = s;
	return *this;
    }
    bool odd;
    UINT size;
    seType seT;
    virtual seType getType() { return seT; }
    
    void printSelf(ostream &os=std::cout)
    {
	os << "Structuring Element" << endl;
	os << "Size: " << size << endl;
	int ptNbr = points.size();
	os << "Point Nbr: " << ptNbr << endl;
	if (ptNbr)
	  for (int i=0;i<ptNbr;i++)
	    os << "#" << i+1 << ": (" << points[i].x << "," << points[i].y << "," << points[i].z << ")" << endl;
	  
    }
};

inline void operator << (ostream &os, StrElt &se)
{
    se.printSelf(os);
}

class hSE : public StrElt
{
  public:
    hSE(UINT s=1) 
    {
	seT = stHexSE;
	size = s;
	odd = true;
	addPoint(0,0);
	addPoint(1,0);
	addPoint(-1,0);
	addPoint(-1,1);
	addPoint(0,1);
	addPoint(-1,-1);
	addPoint(0,-1);
    }
};

class sSE : public StrElt
{
  public:
    sSE(UINT s=1) : StrElt(s)
    {
// 	seT = stSquSE;
	odd = false;
	addPoint(0,0);
	addPoint(1,0);
	addPoint(1,1);
	addPoint(0,1);
	addPoint(-1,1);
	addPoint(-1,0);
	addPoint(-1,-1);
	addPoint(0,-1);
	addPoint(1,-1);
    }
};

#define DEFAULT_SE sSE()

#endif
