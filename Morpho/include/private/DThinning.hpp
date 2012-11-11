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
 * \defgroup Thinning
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
    virtual void printSelf(ostream &os=std::cout)
    {
	os << "HMT Structuring Element" << endl;
	os << "Foreground SE:" << endl;
	fgSE.printSelf(os);
	os << "Background SE:" << endl;
	bgSE.printSelf(os);
    }
protected:
  
void _rotate(StrElt &se)
{
    if (se.odd)
    {
	int x, y;
	for (vector<IntPoint>::iterator it = se.points.begin();it!=se.points.end();it++)
	{
	    x = (*it).x;
	    y = (*it).y;
	    
	    if (x==1 && y==0) { x=0; y=-1; }
	    else if (x==0 && y==-1) { x=-1; y=-1; }
	    else if (x==-1 && y==-1) { x=-1; y=0; }
	    else if (x==-1 && y==0) { x=-1; y=1; }
	    else if (x==-1 && y==1) { x=0; y=1; }
	    else if (x==0 && y==1) { x=1; y=0; }
	    
	    (*it).x = x;
	    (*it).y = y;
	}
    }
    else
    {
	int x, y;
	for (vector<IntPoint>::iterator it = se.points.begin();it!=se.points.end();it++)
	{
	    x = (*it).x;
	    y = (*it).y;
	    
	    if (x==1 && y==0) { x=1; y=-1; }
	    else if (x==1 && y==-1) { x=0; y=-1; }
	    else if (x==0 && y==-1) { x=-1; y=-1; }
	    else if (x==-1 && y==-1) { x=-1; y=0; }
	    else if (x==-1 && y==0) { x=-1; y=1; }
	    else if (x==-1 && y==1) { x=0; y=1; }
	    else if (x==0 && y==1) { x=1; y=1; }
	    else if (x==1 && y==1) { x=1; y=0; }
	    
	    (*it).x = x;
	    (*it).y = y;
	}
    }
}
};


// squareL = ([2,3,9], [5,6,7])
class HMT_sL_SE : public HMTStrElt
{
public:
    HMT_sL_SE()
    {
	fgSE.odd = false;
	bgSE.odd = false;
	fgSE.addPoint(1,1);
	fgSE.addPoint(1,0);
	fgSE.addPoint(1,-1);
	
	bgSE.addPoint(-1,-1);
	bgSE.addPoint(-1,0);
	bgSE.addPoint(-1,1);
    }
};

// hexagonalL ([2,7], [4,5])
class HMT_hL_SE : public HMTStrElt
{
public:
    HMT_hL_SE()
    {
	fgSE.odd = true;
	bgSE.odd = true;
	fgSE.addPoint(1,0);
	fgSE.addPoint(0,1);
	
	bgSE.addPoint(-1,-1);
	bgSE.addPoint(-1,0);
    }
};

// squareM ([2,9], [4,5,6,7])
class HMT_sM_SE : public HMTStrElt
{
public:
    HMT_sM_SE()
    {
	fgSE.odd = false;
	bgSE.odd = false;
	fgSE.addPoint(1,1);
	fgSE.addPoint(1,0);
	
	bgSE.addPoint(0,-1);
	bgSE.addPoint(-1,-1);
	bgSE.addPoint(-1,0);
	bgSE.addPoint(-1,1);
    }
};

// hexagonalM ([2], [4,5,6])
class HMT_hM_SE : public HMTStrElt
{
public:
    HMT_hM_SE()
    {
	fgSE.odd = true;
	bgSE.odd = true;
	fgSE.addPoint(1,0);
	
	bgSE.addPoint(-1,-1);
	bgSE.addPoint(-1,0);
	bgSE.addPoint(-1,1);
    }
};

// squareD ([4,5,6,7], [2,9])
class HMT_sD_SE : public HMTStrElt
{
public:
    HMT_sD_SE()
    {
	fgSE.odd = false;
	bgSE.odd = false;
	fgSE.addPoint(0,-1);
	fgSE.addPoint(-1,-1);
	fgSE.addPoint(-1,0);
	fgSE.addPoint(-1,1);
	
	bgSE.addPoint(1,1);
	bgSE.addPoint(1,0);
    }
};

// hexagonalD ([4,5,6], [2])
class HMT_hD_SE : public HMTStrElt
{
public:
    HMT_hD_SE()
    {
	fgSE.odd = true;
	bgSE.odd = true;
	fgSE.addPoint(-1,-1);
	fgSE.addPoint(-1,0);
	fgSE.addPoint(-1,1);
	
	bgSE.addPoint(1,0);
    }
};

// squareE ([4,5,6,7,8], [1])
class HMT_sE_SE : public HMTStrElt
{
public:
    HMT_sE_SE()
    {
	fgSE.odd = false;
	bgSE.odd = false;
	fgSE.addPoint(0,-1);
	fgSE.addPoint(-1,-1);
	fgSE.addPoint(-1,0);
	fgSE.addPoint(-1,1);
	fgSE.addPoint(0,1);
	
	bgSE.addPoint(0,0);
    }
};

// hexagonalE ([4,5,6,7], [1])
class HMT_hE_SE : public HMTStrElt
{
public:
    HMT_hE_SE()
    {
	fgSE.odd = true;
	bgSE.odd = true;
	fgSE.addPoint(-1,-1);
	fgSE.addPoint(-1,0);
	fgSE.addPoint(-1,1);
	fgSE.addPoint(0,1);
	
	bgSE.addPoint(0,0);
    }
};


// # Some other specific structuring elements used for multiple points extraction
// squareS1 ([4,8], [1,2,6])
class HMT_sS1_SE : public HMTStrElt
{
public:
    HMT_sS1_SE()
    {
	fgSE.odd = false;
	bgSE.odd = false;
	fgSE.addPoint(0,-1);
	fgSE.addPoint(0,1);
	
	bgSE.addPoint(0,0);
	bgSE.addPoint(1,0);
	bgSE.addPoint(-1,0);
    }
};

// hexagonalS1 ([3,4,6,7], [1,2,5])
class HMT_hS1_SE : public HMTStrElt
{
public:
    HMT_hS1_SE()
    {
	fgSE.odd = true;
	bgSE.odd = true;
	fgSE.addPoint(0,-1);
	fgSE.addPoint(-1,-1);
	fgSE.addPoint(-1,1);
	fgSE.addPoint(0,1);
	
	bgSE.addPoint(0,0);
	bgSE.addPoint(1,0);
	bgSE.addPoint(-1,0);
    }
};

// squareS2 ([3,6,7,8], [1,2,4])
class HMT_sS2_SE : public HMTStrElt
{
public:
    HMT_sS2_SE()
    {
	fgSE.odd = false;
	bgSE.odd = false;
	fgSE.addPoint(1,-1);
	fgSE.addPoint(-1,0);
	fgSE.addPoint(-1,1);
	fgSE.addPoint(0,1);
	
	bgSE.addPoint(0,0);
	bgSE.addPoint(1,0);
	bgSE.addPoint(0,-1);
    }
};

// hexagonalS2 ([3,5,6,7], [1,2,4])
class HMT_hS2_SE : public HMTStrElt
{
public:
    HMT_hS2_SE()
    {
	fgSE.odd = true;
	bgSE.odd = true;
	fgSE.addPoint(0,-1);
	fgSE.addPoint(-1,0);
	fgSE.addPoint(-1,1);
	fgSE.addPoint(0,1);
	
	bgSE.addPoint(0,0);
	bgSE.addPoint(1,0);
	bgSE.addPoint(-1,-1);
    }
};

// # Special pattern used to perform SKIZ
// squareS3 ([4,5,6,7,8], [2])
class HMT_sS3_SE : public HMTStrElt
{
public:
    HMT_sS3_SE()
    {
	fgSE.odd = false;
	bgSE.odd = false;
	fgSE.addPoint(0,-1);
	fgSE.addPoint(-1,-1);
	fgSE.addPoint(-1,0);
	fgSE.addPoint(-1,1);
	fgSE.addPoint(0,1);
	
	bgSE.addPoint(1,0);
    }
};

// hexagonalS3 ([3,4,5,6,7], [1])
// class HMT_hS3_SE : public HMTStrElt
// {
// public:
//     HMT_hS3_SE()
//     {
// 	fgSE.odd = true;
// 	bgSE.odd = true;
//     }
// };

// 
// # Isolated points detection
// squareI ([2,3,4,5,6,7,8,9], [1])
class HMT_sI_SE : public HMTStrElt
{
public:
    HMT_sI_SE()
    {
	fgSE.odd = false;
	bgSE.odd = false;
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

// hexagonalI ([2,3,4,5,6,7], [1])
class HMT_I_SE : public HMTStrElt
{
public:
    HMT_I_SE()
    {
	fgSE.odd = true;
	bgSE.odd = true;
	fgSE.addPoint(1,0);
	fgSE.addPoint(0,-1);
	fgSE.addPoint(-1,-1);
	fgSE.addPoint(-1,0);
	fgSE.addPoint(-1,1);
	fgSE.addPoint(0,1);
	
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
RES_T rotThin(const Image<T> &imIn, const HMTStrElt &mhtSE, Image<T> &imOut)
{
    SLEEP(imOut);
    Image<T> tmpIm(imIn);
    Image<T> tmpIm2(imIn);
    HMTStrElt cpSe = mhtSE;
    hitOrMiss<T>(imIn, mhtSE, tmpIm2);
    int nrot = 3;
    for (int i=0;i<nrot;i++)
    {
	cpSe.rotate();
// 	cpSe.rotate(); // (90 deg)
	hitOrMiss<T>(imIn, cpSe, tmpIm);
	sup(tmpIm, tmpIm2, tmpIm2);
    }
    
    inv(tmpIm2, tmpIm);
    WAKE_UP(imOut);
    inf(imIn, tmpIm, imOut);
    
    return RES_OK;
}

template <class T>
RES_T fullThin(const Image<T> &imIn, const HMTStrElt &mhtSE, Image<T> &imOut)
{
    SLEEP(imOut);
    copy(imIn, imOut);
    double v = vol(imIn), v2;
    while(true)
    {
	rotThin<T>(imOut, mhtSE, imOut);
	v2 = vol(imOut);
	if (v2==v)
	  break;
	v = v2;
    }
    WAKE_UP(imOut);
    imOut.modified();
    
    return RES_OK;
}

/**
 * Zhang skeleton
 * 
 * Implementation corresponding to the algorithm described in \cite khanyile_comparative_2011.
 */
template <class T>
RES_T zhangSkeleton(const Image<T> &imIn, Image<T> &imOut)
{
	size_t w = imIn.getWidth();
	size_t h = imIn.getHeight();
	
	// Create a copy image with a border to avoid border checks
	size_t width=w+2, height=h+2;
	Image<T> tmpIm(width, height);
	fill(tmpIm, ImDtTypes<T>::min());
	copy(imIn, tmpIm, 1, 1);
	
	typedef typename Image<T>::sliceType sliceType;
	typedef typename Image<T>::lineType lineType;
	
	lineType tab = tmpIm.getPixels();
	sliceType lines = tmpIm.getLines();
	lineType curLine;
	lineType curPix;
	
	bool ptsDeleted;
	bool goOn;
	bool oddIter = false;
	UINT nbrTrans, nbrNonZero;
	
	int ngbOffsets[9] = { -width-1, -width, -width+1, 1, width+1, width, width-1, -1, -width-1 };
	
	do
	{
	    oddIter = !oddIter;
	    ptsDeleted = false;
	    
	    for (size_t y=1;y<height-1;y++)
	    {
	      curLine = lines[y];
	      curPix  = curLine + 1;
	      for (size_t x=1;x<width;x++,curPix++)
	      {
		if (*curPix!=0)
		{
		    goOn = false;
		    
		    // Calculate the number of non-zero neighbors
		    nbrNonZero = 0;
		    for (int n=0;n<8;n++)
		      if (*(curPix+ngbOffsets[n])!=0)
			nbrNonZero++;
		      
		    if (nbrNonZero>=2 && nbrNonZero<=6)
		      goOn = true;
		    
		    if (goOn)
		    {
			// Calculate the number of transitions in clockwise direction 
			// from point (-1,-1) back to itself
			nbrTrans = 0;
			for (int n=0;n<8;n++)
			  if (*(curPix+ngbOffsets[n])==0)
			    if (*(curPix+ngbOffsets[n+1])!=0)
			      nbrTrans++;
			if (nbrTrans==1)
			  goOn = true;
			else goOn = false;
		    }
		    
		    if (goOn)
		    {
			if (oddIter && (*(curPix+ngbOffsets[1]) * *(curPix+ngbOffsets[3]) * *(curPix+ngbOffsets[5])!=0))
			      goOn = false;
			else if (oddIter && (*(curPix+ngbOffsets[1]) * *(curPix+ngbOffsets[3]) * *(curPix+ngbOffsets[7])!=0))
			      goOn = false;
		    }
		    if (goOn)
		    {
			if (oddIter && (*(curPix+ngbOffsets[3]) * *(curPix+ngbOffsets[5]) * *(curPix+ngbOffsets[7])!=0))
			      goOn = false;
			else if (oddIter && (*(curPix+ngbOffsets[1]) * *(curPix+ngbOffsets[5]) * *(curPix+ngbOffsets[7])!=0))
			      goOn = false;
		    }
		    if (goOn)
		    {
			*curPix = 0;
			ptsDeleted = true;
		    }
		}
	      }
	    }
		
	} while(ptsDeleted);
	
	copy(tmpIm, 1, 1, imOut);

	return RES_OK;
}

/** \} */

#endif // _D_THINNING_HPP

