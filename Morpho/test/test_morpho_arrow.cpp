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



#include "DMorphoArrow.hpp"

class TestArrow : public TestCase
{
  virtual void run()
  {
      Image_UINT8 im1(5,5);
      Image_UINT8 im2(5,5);
      Image_UINT8 imTruth(5,5);
      
      UINT8 vec1[] = { 
	1, 3, 10, 2, 9, 
	5, 5, 5, 9, 3, 
	3, 5, 7, 5, 5, 
	8, 7, 4, 1, 1, 
	4, 10, 1, 6, 0
      };
      
      im1 << vec1;
      arrowGrt(im1, im2, sSE0(), UINT8(255));
      
      UINT8 vecTruth[] = { 
	0,  16, 241,  0, 80, 	// 16 = 0b00010000
	70, 44, 10, 245, 8, 
	0, 144, 221, 226, 100, 
	71, 173, 65, 128, 64, 
	0, 31, 0, 31, 0, 
      };
      
//       im2.printSelf(1);
      imTruth << vecTruth;
      TEST_ASSERT(im2==imTruth);
  }
};

struct A
{
  A(const char* _name=NULL)
  {
    name = _name;
    if (name)
      cout << name << " created" << endl;
  }
  ~A()
  {
    if (name)
      cout << name << " destroyed" << endl;
  }
  const char *name;
  virtual void func()
  {
    cout << "A" << endl;
  }
  virtual inline A& operator()(int i=0)
  {
    clone.reset(new A("clone"));
    return *(clone.get());
  }
    auto_ptr<A> clone;
protected:
};

struct B : public A
{
  virtual void func()
  {
    cout << "B" << endl;
  }
  virtual inline B& operator()(int i=0)
  {
    static B clone = *this;
    return clone;
  }
};

B defA;
void expose(A &a=defA)
{
  a.func();
}

void expose2(A &a=defA)
{
  expose(a);
}

struct C
{
  virtual void func(A &a)
  {
    expose(a);
  }
};



struct D : public C
{
  virtual void func(A &a)
  {
    C::func(a);
  }
  virtual void func(B &b)
  {
    cout << "oké" << endl;
    expose(b);
  }
};

template <class T>
class unaryMorphImageFunctionGeneric : public imageFunctionBase<T>
{
public:
    unaryMorphImageFunctionGeneric(T _borderValue = numeric_limits<T>::min())
      : borderValue(_borderValue),
	initialValue(_borderValue)
    {
    }
    
    unaryMorphImageFunctionGeneric(T _borderValue, T _initialValue = numeric_limits<T>::min())
      : borderValue(_borderValue),
	initialValue(_initialValue)
    {
    }
    
    typedef Image<T> imageType;
    typedef typename imageType::lineType lineType;
    typedef typename imageType::sliceType sliceType;
    typedef typename imageType::volType volType;
    
    virtual RES_T initialize(imageType &imIn, imageType &imOut, StrElt &se)
    {
	imIn.getSize(imSize);
	
	slicesIn = imIn.getSlices();
	slicesOut = imOut.getSlices();
	pixelsIn = imIn.getPixels();
	pixelsOut = imOut.getPixels();
	
	sePoints = se.points;
	Point p0 = sePoints[0];
	if (p0.x==0 && p0.y==0 && p0.z==0)
	{
	    copy(imIn, imOut);
	    sePoints.erase(sePoints.begin());
	}
	else fill(imOut, initialValue);
	
	sePointNbr = sePoints.size();
	relativeOffsets.clear();
	vector<Point>::iterator pt = sePoints.begin();
	se_xmin = numeric_limits<int>::max();
	se_xmax = numeric_limits<int>::min();
	se_ymin = numeric_limits<int>::max();
	se_ymax = numeric_limits<int>::min();
	se_zmin = numeric_limits<int>::max();
	se_zmax = numeric_limits<int>::min();
	while(pt!=sePoints.end())
	{
	    if(pt->x < se_xmin) se_xmin = pt->x;
	    if(pt->x > se_xmax) se_xmax = pt->x;
	    if(pt->y < se_ymin) se_ymin = pt->y;
	    if(pt->y > se_ymax) se_ymax = pt->y;
	    if(pt->z < se_zmin) se_zmin = pt->z;
	    if(pt->z > se_zmax) se_zmax = pt->z;
	    
	    relativeOffsets.push_back(pt->x - pt->y*imSize[0] + pt->z*imSize[0]*imSize[1]);
	    pt++;
	}
	
    }
    
    virtual RES_T _exec(Image<T> &imIn, Image<T> &imOut, StrElt &se)
    {
	initialize(imIn, imOut, se);
	
	seType st = se.getType();
	
	switch(st)
	{
	  case stGeneric:
	    return processImage(imIn, imOut, se);
	  case stHexSE:
	    return processImage(imIn, imOut, *static_cast<hSE*>(&se));
	  case stSquSE:
	    return processImage(imIn, imOut, *static_cast<sSE*>(&se));
	}
	
	return RES_NOT_IMPLEMENTED;
	
    }
    virtual RES_T processImage(Image<T> &imIn, Image<T> &imOut, StrElt &se)
    {
	for(curSlice=0;curSlice<imSize[2];curSlice++)
	{
	    curLine = 0;
	    processSlice(*slicesIn, *slicesOut, imSize[1], se);
	    slicesIn++;
	    slicesOut++;
	}
	    
    }
    virtual RES_T processImage(Image<T> &imIn, Image<T> &imOut, hSE &se)
    {
    }
    virtual inline void processSlice(sliceType linesIn, sliceType linesOut, UINT &lineNbr, StrElt &se)
    {
	while(curLine<lineNbr)
	{
	    curPixel = 0;
	    processLine(*linesIn, *linesOut, imSize[0], se);
	    curLine++;
	    linesIn++;
	    linesOut++;
	}
    }
    virtual inline void processLine(lineType pixIn, lineType pixOut, UINT &pixNbr, StrElt &se)
    {
	int x, y, z;
	Point p;
	UINT offset = pixIn - pixelsIn;
	vector<Point> ptList;
	vector<UINT> relOffsetList;
	vector<UINT> offsetList;
	
	// Remove points wich are outside image
	for (UINT i=0;i<sePointNbr;i++)
	{
	    p = sePoints[i];
	    y = curLine - p.y;
	    z = curSlice + p.z;
	    if (y>=0 && y<imSize[1] && z>=0 && z<imSize[2])
	    {
	      ptList.push_back(p);
	      relOffsetList.push_back(relativeOffsets[i]);
	    }
	}
	UINT ptNbr = ptList.size();
	
	// Left border
	while(curPixel < -se_xmin)
	{
	    offsetList.clear();
	    for (UINT i=0;i<ptNbr;i++)
	    {
		x = curPixel + ptList[i].x;
		
		if (x>=0 && x<imSize[0])
		  offsetList.push_back(relOffsetList[i]);
	    }
	    processPixel(offset, offsetList.begin(), offsetList.end());
	    curPixel++;
	    offset++;
	}
	
	// Midle
	offsetList.clear();
	for (UINT i=0;i<ptNbr;i++)
	  offsetList.push_back(relOffsetList[i]);
	while(curPixel < pixNbr-se_xmax)
	{
	    processPixel(offset, offsetList.begin(), offsetList.end());
	    curPixel++;
	    offset++;
	}
	
	// Right border
	while(curPixel<pixNbr)
	{
	    offsetList.clear();
	    for (UINT i=0;i<ptNbr;i++)
	    {
		x = curPixel + ptList[i].x;
		
		if (x>=0 && x<imSize[0])
		  offsetList.push_back(relOffsetList[i]);
	    }
	    processPixel(offset, offsetList.begin(), offsetList.end());
	    curPixel++;
	    offset++;
	}
    }
    virtual inline void processPixel(UINT &pointOffset, vector<UINT>::iterator dOffset, vector<UINT>::iterator dOffsetEnd)
    {
	while(dOffset!=dOffsetEnd)
	{
	    pixelsOut[pointOffset] = max(pixelsOut[pointOffset], pixelsIn[pointOffset + *dOffset]);
	    dOffset++;
	}
    }
protected:
      UINT imSize[3];
      volType slicesIn;
      volType slicesOut;
      lineType pixelsIn;
      lineType pixelsOut;
      
      UINT curSlice;
      UINT curLine;
      UINT curPixel;
      
      vector<Point> sePoints;
      UINT sePointNbr;
      vector<int> relativeOffsets;
      
      int se_xmin;
      int se_xmax;
      int se_ymin;
      int se_ymax;
      int se_zmin;
      int se_zmax;
public:
    T initialValue;
    T borderValue;
};

template<class T>
RES_T testDil(Image<T> &imIn, Image<T> &imOut, StrElt se)
{
  unaryMorphImageFunctionGeneric<T> f;
  f._exec(imIn, imOut, se());

}

#include "DCore.h"
#include "DMorphoBase.hpp"

int main(int argc, char *argv[])
{
//     B b;
//      expose2(b);
    
	
      Image_UINT8 im1(5,5);
      im1 << "/home/mat/src/morphee/trunk/utilities/Images/Gray/akiyo_y.png";
      Image_UINT8 im2(im1);
      testDil(im1, im2, sSE());
      
      int BENCH_NRUNS = 1E2;
      BENCH_IMG(testDil, im1, im2, sSE());
//       BENCH_IMG(dilate, im1, im2, sSE);
      
      im2.show();
      Core::execLoop();
      
      TestSuite ts;
      ADD_TEST(ts, TestArrow);
      
      
      return ts.run();
      
}

