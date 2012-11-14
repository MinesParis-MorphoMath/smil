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
    CompStrElt &rotate(int steps=1);
    virtual void printSelf(ostream &os=std::cout, string indent="");
};

class CompStrEltList : public BaseObject
{
public:
    std::vector<CompStrElt> compSeList;
    CompStrEltList() {}
    CompStrEltList(const CompStrEltList &rhs);
    CompStrEltList(const CompStrElt &compSe);
    CompStrEltList operator~();
    
    //! Append composite SEs to current list
    CompStrEltList operator | (const CompStrEltList &rhs);
    
    void add(const CompStrElt &cse);
    void add(const StrElt &fgse, const StrElt &bgse);
    
    //! Add as the nrot rotations of the StrElt pair
    //! The rotation is 6/nrot counterclockwise for Hex
    //! and 8/nrot for Squ
    void add(const StrElt &fgse, const StrElt &bgse, UINT nrot);
    
    CompStrEltList &rotate(int steps=1);
    
    virtual void printSelf(ostream &os=std::cout, string indent="");
};


//! Square L ([8,1,2], [4,5,6])
class HMT_sL_SE : public CompStrEltList
{
public:
    HMT_sL_SE(UINT nrot=1)
    {
	this->add(StrElt(false, 1, 3, 8,1,2), StrElt(false, 1, 3, 4,5,6), nrot);
    }
};

//! Hexagonal L ([1,2], [4,5])
class HMT_hL_SE : public CompStrEltList
{
public:
    HMT_hL_SE(UINT nrot=1)
    {
      this->add(StrElt(true, 1, 2, 1,2), StrElt(true, 1, 2, 4,5), nrot);
    }
};


//! Square M ([1,8], [3,4,5,6])
class HMT_sM_SE : public CompStrEltList
{
public:
    HMT_sM_SE(UINT nrot=1)
    {
	add(StrElt(false, 1, 2, 1,8), StrElt(false, 1, 4, 3,4,5,6), nrot);
    }
};

//! Hexagonal M ([1], [3,4,5])
class HMT_hM_SE : public CompStrEltList
{
public:
    HMT_hM_SE(UINT nrot=1)
    {
	add(StrElt(true, 1, 1, 1), StrElt(true, 1, 3, 4,5,6), nrot);
    }
};

//! Square D ([3,4,5,6], [1,8])
class HMT_sD_SE : public CompStrEltList
{
public:
    HMT_sD_SE(UINT nrot=1)
    {
	add(StrElt(false, 1, 4, 3,4,5,6), StrElt(false, 1, 2, 1,8), nrot);
    }
};

//! Hexagonal D ([3,4,5], [1])
class HMT_hD_SE : public CompStrEltList
{
public:
    HMT_hD_SE(UINT nrot=1)
    {
	add(StrElt(true, 1, 3, 3,4,5), StrElt(true, 1, 1, 1), nrot);
    }
};

//! Square E ([3,4,5,6,7], [0])
class HMT_sE_SE : public CompStrEltList
{
public:
    HMT_sE_SE(UINT nrot=1)
    {
	add(StrElt(false, 1, 5, 3,4,5,6,7), StrElt(false, 1, 1, 0), nrot);
    }
};

//! Hexagonal E ([3,4,5,6], [0])
class HMT_hE_SE : public CompStrEltList
{
public:
    HMT_hE_SE(UINT nrot=1)
    {
	add(StrElt(true, 1, 4, 3,4,5,6), StrElt(true, 1, 1, 0), nrot);
    }
};


// # Some other specific structuring elements used for multiple points extraction
//! Square S1 ([3,7], [0,1,5])
class HMT_sS1_SE : public CompStrEltList
{
public:
    HMT_sS1_SE(UINT nrot=1)
    {
	add(StrElt(false, 1, 2, 3,7), StrElt(false, 1, 3, 0,1,5), nrot);
    }
};

//! Hexagonal S1 ([2,3,5,6], [0,1,4])
class HMT_hS1_SE : public CompStrEltList
{
public:
    HMT_hS1_SE(UINT nrot=1)
    {
	add(StrElt(true, 1, 4, 2,3,5,6), StrElt(true, 1, 3, 0,1,4), nrot);
    }
};

//! Square S2 ([2,5,6,7], [0,1,3])
class HMT_sS2_SE : public CompStrEltList
{
public:
    HMT_sS2_SE(UINT nrot=1)
    {
	add(StrElt(false, 1, 4, 2,5,6,7), StrElt(false, 1, 3, 0,1,3), nrot);
    }
};

//! Hexagonal S2 ([2,4,5,6], [0,1,3])
class HMT_hS2_SE : public CompStrEltList
{
public:
    HMT_hS2_SE(UINT nrot=1)
    {
	add(StrElt(true, 1, 4, 2,4,5,6), StrElt(false, 1, 3, 0,1,3), nrot);
    }
};

// # Special pattern used to perform SKIZ
//! Square S3 ([3,4,5,6,7], [1])
class HMT_sS3_SE : public CompStrEltList
{
public:
    HMT_sS3_SE(UINT nrot=1)
    {
	add(StrElt(false, 1, 5, 3,4,5,6,7), StrElt(false, 1, 1, 1), nrot);
    }
};

// Isolated points detection
//! Square I: ([0], [1,2,3,4,5,6,7,8])
class HMT_sI_SE : public CompStrEltList
{
public:
    HMT_sI_SE()
    {
	add(StrElt(false, 1, 1, 0), StrElt(false, 1, 8, 1,2,3,4,5,6,7,8));
    }
};

//! Hexagonal I: ([0], [1,2,3,4,5,6])
class HMT_hI_SE : public CompStrEltList
{
public:
    HMT_hI_SE()
    {
	add(StrElt(true, 1, 1, 0), StrElt(true, 1, 6, 1,2,3,4,5,6));
    }
};

//! Hexagonal I: ([3,4,5,6,7], [0,1])
class HMT_hLineEnd_SE : public CompStrEltList
{
public:
    HMT_hLineEnd_SE(UINT nrot=1)
    {
	add(StrElt(false, 1, 5, 3,4,5,6,7), StrElt(false, 1, 2, 0,1), nrot);
    }
};

/** @} */

#endif // _COMPOSITE_SE_HPP

