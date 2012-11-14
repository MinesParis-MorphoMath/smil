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


#ifndef _COMPOSITE_SE_HPP
#define _COMPOSITE_SE_HPP

#include "DStructuringElement.h"
#include <list>

/**
 * \defgroup CompSE Composite Structuring Elements
 * \ingroup StrElt
 * @{
 */


class testStrElt /*: public BaseObject*/
{
public:
  testStrElt();
};

/**
 * Composite structuring element
 */
class CompStrElt : public BaseObject
{
public:
    StrElt fgSE;
    StrElt bgSE;
    
    CompStrElt() {};
    ~CompStrElt() {}
    CompStrElt(const CompStrElt &rhs);
    CompStrElt(const StrElt &fg, const StrElt &bg);
    //! Switch foreground/background SE
    CompStrElt operator~();
    //! Counterclockwise rotate SE points
    CompStrElt rotate(int deg);
    virtual void printSelf(ostream &os=std::cout, string indent="");
};

class CompStrEltList : public BaseObject
{
public:
    std::list<CompStrElt> compSeList;
    CompStrEltList();
    CompStrEltList(const CompStrEltList &rhs);
    CompStrEltList(const CompStrElt &compSe);
    CompStrEltList operator~();
    
    //! Append composite SEs to current list
    CompStrEltList operator | (const CompStrEltList &rhs);
    
    void add(const CompStrElt &cse);
    void add(const StrElt &fgse, const StrElt &bgse);
    
    //! Add as StrElt pair and their nrot rotations
    //! The rotation angle is 360/(nrot+1) counterclockwise
    //! (for ex. 90° for n=3, 45° for n=7)
    void add(const StrElt &fgse, const StrElt &bgse, UINT nrot);
    
    virtual void printSelf(ostream &os=std::cout, string indent="");
};


//! Square edge SE ([8,1,2], [4,5,6])
class HMT_sEdge_SE : public CompStrEltList
{
public:
    HMT_sEdge_SE()
    {
	this->add(StrElt(false, 1, 3, 8,1,2), StrElt(false, 1, 3, 4,5,6), 3);
    }
};

//! Square diagonal edge SE ([1,2,3], [5,6,7])
class HMT_sDiagEdge_SE : public CompStrEltList
{
public:
    HMT_sDiagEdge_SE()
    {
	add(StrElt(false, 1, 3, 1,2,3), StrElt(false, 1, 3, 5,6,7), 3);
    }
};

//! Square corner SE ([1,0,7], [3,4,5])
class HMT_sCorner_SE : public CompStrEltList
{
public:
    HMT_sCorner_SE()
    {
	add(StrElt(false, 1, 3, 1,0,7), StrElt(false, 1, 3, 3,4,5), 3);
    }
};

//! Square diagonal SE ([1,2,7], [3,4,5,6]) and ([1,6,7], [2,3,4,5])
class HMT_sDiagonal_SE : public CompStrEltList
{
public:
    HMT_sDiagonal_SE()
    {
	add(StrElt(false, 1, 3, 1,2,7), StrElt(false, 1, 4, 3,4,5,6), 3);
	add(StrElt(false, 1, 3, 1,6,7), StrElt(false, 1, 4, 2,3,4,5), 3);
    }
};

//! Square convex hull SE ([3,4,5,6], [0,8]) and ([2,3,4,5], [0,8])
class HMT_sConvHull_SE : public CompStrEltList
{
public:
    HMT_sConvHull_SE()
    {
	add(StrElt(false, 1, 4, 3,4,5,6), StrElt(false, 1, 2, 0,8), 3);
	add(StrElt(false, 1, 4, 2,3,4,5), StrElt(false, 1, 2, 0,8), 3);
    }
};

//! Square line-end SE ([0,1], [3,4,5,6,7] and [0,2], [3,4,5,6,7,8])
class HMT_sLineEnd_SE : public CompStrEltList
{
public:
    HMT_sLineEnd_SE()
    {
	add(StrElt(false, 1, 2, 0,1), StrElt(false, 1, 5, 3,4,5,6,7), 3);
    }
};

//! Square diagonal line-end SE ([0,2], [3,4,5,6,7,8,1])
class HMT_sLineEndD_SE : public CompStrEltList
{
public:
    HMT_sLineEndD_SE()
    {
	add(StrElt(false, 1, 2, 0,2), StrElt(false, 1, 7, 3,4,5,6,7,8,1), 3);
    }
};






//! Square L ([8,1,2], [4,5,6])
class HMT_sL_SE : public CompStrEltList
{
public:
    HMT_sL_SE()
    {
	this->add(StrElt(false, 1, 3, 8,1,2), StrElt(false, 1, 3, 4,5,6), 3);
    }
};

//! Hexagonal L ([1,3], [5,6])
class HMT_hL_SE : public CompStrEltList
{
public:
    HMT_hL_SE();
};





//! hexagonalM ([1], [4,5,6])
class HMT_hM_SE : public CompStrEltList
{
public:
    HMT_hM_SE()
    {
	add(StrElt(true, 1, 1, 1), StrElt(true, 1, 3, 4,5,6), 3);
    }
};

//! squareD ([3,4,5,6], [1,8])
class HMT_sD_SE : public CompStrEltList
{
public:
    HMT_sD_SE()
    {
	add(StrElt(false, 1, 4, 3,4,5,6), StrElt(false, 1, 2, 1,8), 3);
    }
};

//! hexagonalD ([4,5,6], [1])
class HMT_hD_SE : public CompStrEltList
{
public:
    HMT_hD_SE()
    {
	add(StrElt(true, 1, 3, 4,5,6), StrElt(true, 1, 1, 1));
    }
};

//! squareE ([3,4,5,6,7], [0])
class HMT_sE_SE : public CompStrEltList
{
public:
    HMT_sE_SE()
    {
	add(StrElt(false, 1, 5, 3,4,5,6,7), StrElt(false, 1, 1, 0));
    }
};

//! hexagonalE ([4,5,6,7], [0])
class HMT_hE_SE : public CompStrEltList
{
public:
    HMT_hE_SE()
    {
	add(StrElt(true, 1, 4, 4,5,6,7), StrElt(true, 1, 1, 0));
    }
};


// # Some other specific structuring elements used for multiple points extraction
//! squareS1 ([3,7], [0,1,5])
class HMT_sS1_SE : public CompStrEltList
{
public:
    HMT_sS1_SE()
    {
	add(StrElt(false, 1, 2, 3,7), StrElt(false, 1, 3, 0,1,5));
    }
};

//! hexagonalS1 ([3,4,6,7], [0,1,5])
class HMT_hS1_SE : public CompStrEltList
{
public:
    HMT_hS1_SE()
    {
	add(StrElt(true, 1, 4, 3,4,6,7), StrElt(true, 1, 3, 0,1,5));
    }
};

//! squareS2 ([2,5,6,7], [0,1,3])
class HMT_sS2_SE : public CompStrEltList
{
public:
    HMT_sS2_SE()
    {
	add(StrElt(false, 1, 4, 2,5,6,7), StrElt(false, 1, 3, 0,1,3));
    }
};

//! hexagonalS2 ([3,5,6,7], [0,1,4])
class HMT_hS2_SE : public CompStrEltList
{
public:
    HMT_hS2_SE()
    {
	add(StrElt(true, 1, 4, 3,5,6,7), StrElt(false, 1, 3, 0,1,4));
    }
};

// # Special pattern used to perform SKIZ
//! squareS3 ([3,4,5,6,7], [1])
class HMT_sS3_SE : public CompStrEltList
{
public:
    HMT_sS3_SE()
    {
	add(StrElt(false, 1, 5, 3,4,5,6,7), StrElt(false, 1, 1, 1));
    }
};

//! hexagonalS3 ([], [])
// class HMT_hS3_SE : public CompStrEltList
// {
// public:
//     HMT_sS3_SE()
//     {
// 	add(StrElt(false, 1, 5, 3,4,5,6,7), StrElt(false, 1, 1, 1));
//     }
// };


// Isolated points detection
//! squareI ([0], [1,2,3,4,5,6,7,8])
class HMT_sI_SE : public CompStrEltList
{
public:
    HMT_sI_SE()
    {
	add(StrElt(false, 1, 1, 0), StrElt(false, 1, 8, 1,2,3,4,5,6,7,8));
    }
};

//! hexagonalI ([0], [1,2,3,4,5,6])
class HMT_hI_SE : public CompStrEltList
{
public:
    HMT_hI_SE()
    {
	add(StrElt(true, 1, 1, 0), StrElt(true, 1, 6, 1,2,3,4,5,6));
    }
};

/** @} */

#endif // _COMPOSITE_SE_HPP

