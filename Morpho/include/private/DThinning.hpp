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
 * \addtogroup Morpho Mathematical Morphology
 * \{
 */

class HMTStrElt : public BaseObject
{
public:
    StrElt fgSE;
    StrElt bgSE;
    
    void rotate()
    {
	_rotate(fgSE);
	_rotate(bgSE);
    }
protected:
  
void _rotate(StrElt &se)
{
    if (se.odd)
    {
    }
    else
    {
	int x, y;
	for (vector<IntPoint>::iterator it = se.points.begin();it!=se.points.end();it++)
	{
	    x = (*it).x;
	    y = (*it).y;
	    
	    if (x==1 && y==0) { x=1; y=1; }
	    else if (x==1 && y==1) { x=0; y=1; }
	    else if (x==0 && y==1) { x=-1; y=1; }
	    else if (x==-1 && y==1) { x=-1; y=0; }
	    else if (x==-1 && y==0) { x=-1; y=-1; }
	    else if (x==-1 && y==-1) { x=0; y=-1; }
	    else if (x==0 && y==-1) { x=1; y=-1; }
	    else if (x==1 && y==-1) { x=1; y=0; }
	    
	    (*it).x = x;
	    (*it).y = y;
	}
    }
}
};


// squareL = doubleStructuringElement([1,2,8], [4,5,6], mamba.SQUARE)
class HMT_L_SE : public HMTStrElt
{
public:
    HMT_L_SE()
    {
	fgSE.addPoint(1,1);
	fgSE.addPoint(1,0);
	fgSE.addPoint(1,-1);
	
	bgSE.addPoint(-1,-1);
	bgSE.addPoint(-1,0);
	bgSE.addPoint(-1,1);
    }
};

// squareM = doubleStructuringElement([1,8], [3,4,5,6], mamba.SQUARE)
class HMT_M_SE : public HMTStrElt
{
public:
    HMT_M_SE()
    {
	fgSE.addPoint(1,1);
	fgSE.addPoint(1,0);
	
	bgSE.addPoint(0,-1);
	bgSE.addPoint(-1,-1);
	bgSE.addPoint(-1,0);
	bgSE.addPoint(-1,1);
    }
};

// squareD = doubleStructuringElement([3,4,5,6], [1,8], mamba.SQUARE)
class HMT_D_SE : public HMTStrElt
{
public:
    HMT_D_SE()
    {
	fgSE.addPoint(0,-1);
	fgSE.addPoint(-1,-1);
	fgSE.addPoint(-1,0);
	fgSE.addPoint(-1,1);
	
	bgSE.addPoint(1,1);
	bgSE.addPoint(1,0);
    }
};

// squareE = doubleStructuringElement([3,4,5,6,7], [0], mamba.SQUARE)
class HMT_E_SE : public HMTStrElt
{
public:
    HMT_E_SE()
    {
	fgSE.addPoint(0,-1);
	fgSE.addPoint(-1,-1);
	fgSE.addPoint(-1,0);
	fgSE.addPoint(-1,1);
	fgSE.addPoint(0,1);
	
	bgSE.addPoint(0,0);
    }
};


// # Some other specific structuring elements used for multiple points extraction
// squareS1 = doubleStructuringElement([3,7], [0,1,5], mamba.SQUARE)
class HMT_S1_SE : public HMTStrElt
{
public:
    HMT_S1_SE()
    {
	fgSE.addPoint(0,-1);
	fgSE.addPoint(0,1);
	
	bgSE.addPoint(0,0);
	bgSE.addPoint(1,0);
	bgSE.addPoint(-1,0);
    }
};

// squareS2 = doubleStructuringElement([2,5,6,7], [0,1,3], mamba.SQUARE)
class HMT_S2_SE : public HMTStrElt
{
public:
    HMT_S2_SE()
    {
	fgSE.addPoint(1,-1);
	fgSE.addPoint(-1,0);
	fgSE.addPoint(-1,1);
	fgSE.addPoint(0,1);
	
	bgSE.addPoint(0,0);
	bgSE.addPoint(1,0);
	bgSE.addPoint(0,-1);
    }
};

// # Special pattern used to perform SKIZ
// squareS3 = doubleStructuringElement([3,4,5,6,7], [1], mamba.SQUARE)
class HMT_S3_SE : public HMTStrElt
{
public:
    HMT_S3_SE()
    {
	fgSE.addPoint(0,-1);
	fgSE.addPoint(-1,-1);
	fgSE.addPoint(-1,0);
	fgSE.addPoint(-1,1);
	fgSE.addPoint(0,1);
	
	bgSE.addPoint(1,0);
    }
};

// 
// # Isolated points detection
// squareI = doubleStructuringElement([1,2,3,4,5,6,7,8], [0], mamba.SQUARE)
class HMT_I_SE : public HMTStrElt
{
public:
    HMT_I_SE()
    {
	fgSE.addPoint(1,0);
	fgSE.addPoint(1,-1);
	fgSE.addPoint(0,-1);
	fgSE.addPoint(-1,-1);
	fgSE.addPoint(-1,0);
	fgSE.addPoint(-1,1);
	fgSE.addPoint(0,1);
	fgSE.addPoint(1,1);
	
	bgSE.addPoint(0,0);
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
    return hitOrMiss<T>(imIn, mhtSE.fgSE, mhtSE.bgSE, imOut);
}

/** \} */

#endif // _D_THINNING_HPP

