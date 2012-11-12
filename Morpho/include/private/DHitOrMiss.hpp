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


#ifndef _D_THINNING_HPP
#define _D_THINNING_HPP

#include "DMorphoBase.hpp"

/**
 * \ingroup Morpho
 * \defgroup Thinning
 * \{
 */

class CompositeStrElt : public BaseObject
{
public:
    StrElt fgSE;
    StrElt bgSE;
    CompositeStrElt()
      : BaseObject("CompositeStrElt")
    {
    }
    CompositeStrElt(const CompositeStrElt &rhs)
      : BaseObject("CompositeStrElt")
    {
	fgSE = rhs.fgSE;
	bgSE = rhs.bgSE;
    }
    CompositeStrElt(const StrElt &fg, const StrElt &bg)
      : BaseObject("CompositeStrElt")
    {
	fgSE = fg;
	bgSE = bg;
    }
    //! Invert foreground/background
    CompositeStrElt operator~()
    {
	CompositeStrElt cSE;
	cSE.fgSE = bgSE;
	cSE.bgSE = fgSE;
	return cSE;
    }
    virtual void printSelf(ostream &os=std::cout, string indent="")
    {
	os << indent << "Composite Structuring Element" << endl;
	os << indent << "Foreground SE:" << endl;
	fgSE.printSelf(os, indent + "\t");
	os << indent << "Background SE:" << endl;
	bgSE.printSelf(os, indent + "\t");
    }
};

class HMTStrElt : public BaseObject
{
public:
    std::list<CompositeStrElt> compSeList;
    HMTStrElt() {}
    HMTStrElt(const HMTStrElt &rhs) 
    {
	compSeList = rhs.compSeList;
    }
    HMTStrElt operator~()
    {
	HMTStrElt hmtSE;
	for (std::list<CompositeStrElt>::const_iterator it=compSeList.begin();it!=compSeList.end();it++)
	  hmtSE.add((*it).bgSE, (*it).fgSE);
	return hmtSE;
    }
    //! Append composite SEs to current list
    HMTStrElt operator | (const HMTStrElt &rhs) 
    {
	HMTStrElt hmtSE(*this);
	for (std::list<CompositeStrElt>::const_iterator it=rhs.compSeList.begin();it!=rhs.compSeList.end();it++)
	  hmtSE.add((*it).fgSE, (*it).bgSE);
	return hmtSE;
    }
    void add(const CompositeStrElt &cse)
    {
	compSeList.push_back(cse);
    }
    void add(const StrElt &fgse, const StrElt &bgse)
    {
	compSeList.push_back(CompositeStrElt(fgse, bgse));
    }
    //! Add as StrElt pair and their nrot rotations (90°, counterclockwise)
    void add(const StrElt &fgse, const StrElt &bgse, UINT nrot)
    {
	StrElt fg=fgse, bg=bgse;
	compSeList.push_back(CompositeStrElt(fg, bg));
	for (UINT n=0;n<nrot;n++)
	{
	    rotateSE(fg);
	    rotateSE(bg);
	    compSeList.push_back(CompositeStrElt(fg, bg));
	}
    }
    // Rotate SE points (counterclockwise 90°)
    void rotateSE(StrElt &se)
    {
	int x;
	for (vector<IntPoint>::iterator it=se.points.begin();it!=se.points.end();it++)
	{
	    x = (*it).x;
	    (*it).x = (*it).y;
	    (*it).y = -x;
	}
    }
    virtual void printSelf(ostream &os=std::cout, string indent="")
    {
	os << indent << "HitOrMiss SE (composite structuring element list)" << endl;
	int i=0;
	for (std::list<CompositeStrElt>::iterator it=compSeList.begin();it!=compSeList.end();it++,i++)
	{
	    os << indent << "CompSE #" << i << ":" << endl;
	    (*it).printSelf(os, indent + "\t");
	}
    }
};

inline UINT rInd(UINT ind, UINT rot=2)
{
    if (ind==0)
      return 0;
    return (ind+rot)%8;
}

//! Square edge SE ([8,1,2], [4,5,6])
class HMT_sEdge_SE : public HMTStrElt
{
public:
    HMT_sEdge_SE()
    {
	add(StrElt(false, 1, 3, 8,1,2), StrElt(false, 1, 3, 4,5,6), 3);
    }
};

//! Square diagonal edge SE ([1,2,3], [5,6,7])
class HMT_sDiagEdge_SE : public HMTStrElt
{
public:
    HMT_sDiagEdge_SE()
    {
	add(StrElt(false, 1, 3, 1,2,3), StrElt(false, 1, 3, 5,6,7), 3);
    }
};

//! Square corner SE ([1,0,7], [3,4,5])
class HMT_sCorner_SE : public HMTStrElt
{
public:
    HMT_sCorner_SE()
    {
	add(StrElt(false, 1, 3, 1,0,7), StrElt(false, 1, 3, 3,4,5), 3);
    }
};

//! Square diagonal SE ([1,2,7], [3,4,5,6]) and ([1,6,7], [2,3,4,5])
class HMT_sDiagonal_SE : public HMTStrElt
{
public:
    HMT_sDiagonal_SE()
    {
	add(StrElt(false, 1, 3, 1,2,7), StrElt(false, 1, 4, 3,4,5,6), 3);
	add(StrElt(false, 1, 3, 1,6,7), StrElt(false, 1, 4, 2,3,4,5), 3);
    }
};

//! Square convex hull SE ([3,4,5,6], [0,8]) and ([2,3,4,5], [0,8])
class HMT_sConvHull_SE : public HMTStrElt
{
public:
    HMT_sConvHull_SE()
    {
	add(StrElt(false, 1, 4, 3,4,5,6), StrElt(false, 1, 2, 0,8), 3);
	add(StrElt(false, 1, 4, 2,3,4,5), StrElt(false, 1, 2, 0,8), 3);
    }
};

// //! Hex corner SE ([1,7], [4,5])
// class HMT_hL_SE : public HMTStrElt
// {
// public:
//     HMT_hL_SE()
//     {
// 	add(StrElt(true, 1, 2, 1,7), StrElt(true, 1, 2, 4,5));
//     }
// };

//! Square line-end SE ([0,1], [3,4,5,6,7] and [0,2], [3,4,5,6,7,8])
class HMT_sLineEnd_SE : public HMTStrElt
{
public:
    HMT_sLineEnd_SE()
    {
	add(StrElt(false, 1, 2, 0,1), StrElt(false, 1, 5, 3,4,5,6,7), 3);
    }
};

//! Square diagonal line-end SE ([0,2], [3,4,5,6,7,8,1])
class HMT_sLineEndD_SE : public HMTStrElt
{
public:
    HMT_sLineEndD_SE()
    {
	add(StrElt(false, 1, 2, 0,2), StrElt(false, 1, 7, 3,4,5,6,7,8,1), 3);
    }
};

//! hexagonalM ([1], [4,5,6])
class HMT_hM_SE : public HMTStrElt
{
public:
    HMT_hM_SE()
    {
	add(StrElt(true, 1, 1, 1), StrElt(true, 1, 3, 4,5,6), 3);
    }
};

//! squareD ([3,4,5,6], [1,8])
class HMT_sD_SE : public HMTStrElt
{
public:
    HMT_sD_SE()
    {
	add(StrElt(false, 1, 4, 3,4,5,6), StrElt(false, 1, 2, 1,8), 3);
    }
};

//! hexagonalD ([4,5,6], [1])
class HMT_hD_SE : public HMTStrElt
{
public:
    HMT_hD_SE()
    {
	add(StrElt(true, 1, 3, 4,5,6), StrElt(true, 1, 1, 1));
    }
};

//! squareE ([3,4,5,6,7], [0])
class HMT_sE_SE : public HMTStrElt
{
public:
    HMT_sE_SE()
    {
	add(StrElt(false, 1, 5, 3,4,5,6,7), StrElt(false, 1, 1, 0));
    }
};

//! hexagonalE ([4,5,6,7], [0])
class HMT_hE_SE : public HMTStrElt
{
public:
    HMT_hE_SE()
    {
	add(StrElt(true, 1, 4, 4,5,6,7), StrElt(true, 1, 1, 0));
    }
};


// # Some other specific structuring elements used for multiple points extraction
//! squareS1 ([3,7], [0,1,5])
class HMT_sS1_SE : public HMTStrElt
{
public:
    HMT_sS1_SE()
    {
	add(StrElt(false, 1, 2, 3,7), StrElt(false, 1, 3, 0,1,5));
    }
};

//! hexagonalS1 ([3,4,6,7], [0,1,5])
class HMT_hS1_SE : public HMTStrElt
{
public:
    HMT_hS1_SE()
    {
	add(StrElt(true, 1, 4, 3,4,6,7), StrElt(true, 1, 3, 0,1,5));
    }
};

//! squareS2 ([2,5,6,7], [0,1,3])
class HMT_sS2_SE : public HMTStrElt
{
public:
    HMT_sS2_SE()
    {
	add(StrElt(false, 1, 4, 2,5,6,7), StrElt(false, 1, 3, 0,1,3));
    }
};

//! hexagonalS2 ([3,5,6,7], [0,1,4])
class HMT_hS2_SE : public HMTStrElt
{
public:
    HMT_hS2_SE()
    {
	add(StrElt(true, 1, 4, 3,5,6,7), StrElt(false, 1, 3, 0,1,4));
    }
};

// # Special pattern used to perform SKIZ
//! squareS3 ([3,4,5,6,7], [1])
class HMT_sS3_SE : public HMTStrElt
{
public:
    HMT_sS3_SE()
    {
	add(StrElt(false, 1, 5, 3,4,5,6,7), StrElt(false, 1, 1, 1));
    }
};

//! hexagonalS3 ([], [])
// class HMT_hS3_SE : public HMTStrElt
// {
// public:
//     HMT_sS3_SE()
//     {
// 	add(StrElt(false, 1, 5, 3,4,5,6,7), StrElt(false, 1, 1, 1));
//     }
// };


// Isolated points detection
//! squareI ([0], [1,2,3,4,5,6,7,8])
class HMT_sI_SE : public HMTStrElt
{
public:
    HMT_sI_SE()
    {
	add(StrElt(false, 1, 1, 0), StrElt(false, 1, 8, 1,2,3,4,5,6,7,8));
    }
};

//! hexagonalI ([0], [1,2,3,4,5,6])
class HMT_hI_SE : public HMTStrElt
{
public:
    HMT_hI_SE()
    {
	add(StrElt(true, 1, 1, 0), StrElt(true, 1, 6, 1,2,3,4,5,6));
    }
};


template <class T>
RES_T hitOrMiss(const Image<T> &imIn, const StrElt &foreSE, const StrElt &backSE, Image<T> &imOut)
{
    SLEEP(imOut);
    Image<T> tmpIm(imIn);
    erode<T>(imIn, imOut, foreSE);
    inv<T>(imIn, tmpIm);
    erode(tmpIm, tmpIm, backSE);
    inf(tmpIm, imOut, imOut);
    WAKE_UP(imOut);
    
    imOut.modified();
    
    return RES_OK;
}

template <class T>
RES_T hitOrMiss(const Image<T> &imIn, const HMTStrElt &mhtSE, Image<T> &imOut)
{
    Image<T> tmpIm(imIn);
    SLEEP(imOut);
    fill(imOut, ImDtTypes<T>::min());
    for (std::list<CompositeStrElt>::const_iterator it=mhtSE.compSeList.begin();it!=mhtSE.compSeList.end();it++)
    {
	hitOrMiss<T>(imIn, (*it).fgSE, (*it).bgSE, tmpIm);
	sup(imOut, tmpIm, imOut);
    }
    WAKE_UP(imOut);
    imOut.modified();
    return RES_OK;
}

template <class T>
RES_T thin(const Image<T> &imIn, const HMTStrElt &mhtSE, Image<T> &imOut)
{
    SLEEP(imOut);
    Image<T> tmpIm(imIn);
    hitOrMiss<T>(imIn, mhtSE, tmpIm);
    inv(tmpIm, tmpIm);
    WAKE_UP(imOut);
    inf(imIn, tmpIm, imOut);
    
    return RES_OK;
}


template <class T>
RES_T fullThin(const Image<T> &imIn, const HMTStrElt &mhtSE, Image<T> &imOut)
{
    SLEEP(imOut);
    double v1, v2;
    thin<T>(imIn, mhtSE, imOut);
    v1 = vol(imOut);
    while(true)
    {
	thin<T>(imOut, mhtSE, imOut);
	v2 = vol(imOut);
	if (v2==v1)
	  break;
	v1 = v2;
    }
    WAKE_UP(imOut);
    imOut.modified();
    
    return RES_OK;
}


/** \} */

#endif // _D_THINNING_HPP

