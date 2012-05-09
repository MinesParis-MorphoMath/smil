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


#ifndef _STR_ELT_HPP
#define _STR_ELT_HPP

/**
 * \defgroup StrElt Structuring Elements
 * \ingroup Morpho
 * @{
 */

struct Point
{
  int x;
  int y;
  int z;
};

enum seType { stGeneric, stHexSE, stHexSE0, stSquSE, stSquSE0 };

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
    virtual inline StrElt& operator()(int s=1)
    {
	static StrElt clone = *this;
	clone.size = s;
	return clone;
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

/**
 * Hexagonal structuring element.
 * 
 * Points :
 * 
 * <table>
 *   <tr>  <th>4</th> <th>3</th> <th>x</th>  </tr>
 *   <tr>  <th>5</th> <th>1</th> <th>2</th>  </tr>
 *   <tr>  <th>6</th> <th>7</th> <th>x</th>  </tr>
 * </table>
 * 
 */

class HexagonalSE : public StrElt
{
  public:
    HexagonalSE(UINT s=1) 
    {
	seT = stHexSE;
	size = s;
	odd = true;
	addPoint(0,0);	// 1
	addPoint(1,0);	// 2
	addPoint(0,1);	// 3
	addPoint(-1,1);	// 4
	addPoint(-1,0);	// 5
	addPoint(-1,-1);// 6
	addPoint(0,-1);	// 7
    }
    virtual inline HexagonalSE& operator()(int s=1)
    {
	static HexagonalSE clone = *this;
	clone.size = s;
	return clone;
    }
};

/**
 * Hexagonal structuring element without center point.
 * 
 * Points :
 * 
 * <table>
 *   <tr>  <th>3</th> <th>2</th> <th>x</th>  </tr>
 *   <tr>  <th>4</th> <th>x</th> <th>1</th>  </tr>
 *   <tr>  <th>5</th> <th>6</th> <th>x</th>  </tr>
 * </table>
 * 
 */

class Hexagonal0SE : public StrElt
{
  public:
    Hexagonal0SE(UINT s=1) 
    {
	seT = stHexSE0;
	size = s;
	odd = true;
	addPoint(1,0);	// 1
	addPoint(0,1);	// 2
	addPoint(-1,1);	// 3
	addPoint(-1,0);	// 4
	addPoint(-1,-1);// 5
	addPoint(0,-1);	// 6
    }
    virtual inline Hexagonal0SE& operator()(int s=1)
    {
	static Hexagonal0SE clone = *this;
	clone.size = s;
	return clone;
    }
};


/**
 * Square structuring element.
 * 
 * Points :
 * 
 * <table>
 *   <tr>  <th>5</th> <th>4</th> <th>3</th>  </tr>
 *   <tr>  <th>6</th> <th>1</th> <th>2</th>  </tr>
 *   <tr>  <th>7</th> <th>8</th> <th>9</th>  </tr>
 * </table>
 * 
 */

class SquareSE : public StrElt
{
  public:
    SquareSE(UINT s=1) : StrElt(s)
    {
	seT = stSquSE;
	odd = false;
	addPoint(0,0); 	// 1
	addPoint(1,0);	// 2
	addPoint(1,1);	// 3
	addPoint(0,1);	// 4
	addPoint(-1,1);	// 5
	addPoint(-1,0);	// 6
	addPoint(-1,-1);// 7
	addPoint(0,-1);	// 8
	addPoint(1,-1);	// 9
    }
    virtual inline SquareSE& operator()(int s=1)
    {
	static SquareSE clone = *this;
	clone.size = s;
	return clone;
    }
};

/**
 * Square structuring element without center point.
 * 
 * Points :
 * 
 * <table>
 *   <tr>  <th>4</th> <th>3</th> <th>2</th>  </tr>
 *   <tr>  <th>5</th> <th>x</th> <th>1</th>  </tr>
 *   <tr>  <th>6</th> <th>7</th> <th>8</th>  </tr>
 * </table>
 * 4 3 2
 * 5 x 1
 * 6 7 8
 * 
 */

class Square0SE : public StrElt
{
  public:
    Square0SE(UINT s=1) : StrElt(s)
    {
	seT = stSquSE0;
	odd = false;
	addPoint(1,0);	// 1
	addPoint(1,1);	// 2
	addPoint(0,1);	// 3
	addPoint(-1,1);	// 4
	addPoint(-1,0);	// 5
	addPoint(-1,-1);// 6
	addPoint(0,-1);	// 7
	addPoint(1,-1);	// 8
    }
    virtual inline Square0SE& operator()(int s=1)
    {
	static Square0SE clone = *this;
	clone.size = s;
	return clone;
    }
};

// #define DEFAULT_SE sSE
SquareSE sSE;
Square0SE sSE0;
HexagonalSE hSE;
Hexagonal0SE hSE0;

StrElt DEFAULT_SE = sSE;

/** @} */

#endif
