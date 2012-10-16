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


#include "DCore.h"
#include "DMorpho.h"
#include "DGui.h"
#include "DIO.h"

class Test_Dilate_Hex : public TestCase
{
  virtual void run()
  {
      typedef UINT8 dataType;
      typedef Image<dataType> imType;
      
      imType im1(7,7);
      imType im2(im1);
      imType im3(im1);
      
      dataType vec1[] = {
	114, 133,  74, 160,  57,  25,  37,
	 23,  73,   9, 196, 118,  23, 110,
	154, 248, 165, 159, 210,  47,  58,
	213,  74,   8, 163,   3, 240, 213,
	158,  67,  52, 103, 163, 158,   9,
	 85,  36, 124,  12,   7,  56, 253,
	214, 148,  20, 200,  53,  10,  58
      };
      
      im1 << vec1;
      
      dataType dilateHexVec[] = {
	133, 133, 160, 196, 196, 118, 110, 
	248, 248, 196, 210, 210, 118, 110, 
	248, 248, 248, 210, 210, 240, 240, 
	248, 248, 165, 210, 240, 240, 240, 
	213, 213, 124, 163, 163, 240, 253, 
	214, 148, 200, 200, 163, 253, 253, 
	214, 214, 200, 200, 200, 58, 253
      };
      im3 << dilateHexVec;
      
      // The specialized way
      dilate(im1, im2, hSE());
      TEST_ASSERT(im2==im3);      
      
      // The generic way
      StrElt se;
      se.points = hSE().points;
      se.odd = true;
      dilate(im1, im2, se);
      TEST_ASSERT(im2==im3);      
      
//       im1.printSelf(1);
//       im2.printSelf(1);
//       im3.printSelf(1);
  }
};

template <class T, class lineFunction_T>
class testClass : public unaryMorphImageFunction<T, lineFunction_T > 
{
public:
    typedef imageFunctionBase<T> parentClass;
    typedef Image<T> imageType;
    typedef typename imageType::lineType lineType;
    typedef typename imageType::sliceType sliceType;
    typedef typename imageType::volType volType;
    virtual RES_T _exec_single_squSE(const imageType &imIn, imageType &imOut)
    {
	_exec_single_V1_segment(imIn, imOut);
	_exec_single_H_segment(imOut, 1, imOut);
	
	return RES_OK;
    }  
    
    virtual RES_T _exec_single_H_segment(const imageType &imIn, int xsize, imageType &imOut)
    {
	  int lineCount = imIn.getLineCount();
	  
	  int nthreads = Core::getInstance()->getNumberOfThreads();
	  lineType *_bufs = this->createAlignedBuffers(2*nthreads, this->lineLen);
	  lineType buf1 = _bufs[0];
	  lineType buf2 = _bufs[nthreads];
	  
	  sliceType srcLines = imIn.getLines();
	  sliceType destLines = imOut.getLines();
	  
	  lineType lineIn;
	  
	  int l, tid, dx = xsize;

	  #pragma omp parallel private(tid,buf1,buf2,lineIn)
	  {
	      #ifdef USE_OPEN_MP
		  tid = omp_get_thread_num();
		  buf1 = _bufs[tid];
		  buf2 = _bufs[tid+nthreads];
	      #endif
	      #pragma omp for schedule(dynamic,nthreads) nowait
	      for (l=0;l<lineCount;l++)
	      {
		// Todo: if oddLines...
		  lineIn = srcLines[l];
		  shiftLine<T>(lineIn, dx, this->lineLen, buf1, this->borderValue);
		  this->lineFunction._exec(buf1, lineIn, this->lineLen, buf2);
		  shiftLine<T>(lineIn, -dx, this->lineLen, buf1, this->borderValue);
		  this->lineFunction._exec(buf1, buf2, this->lineLen, destLines[l]);
	      }
	  }
	  
	  return RES_OK;
    }

    virtual RES_T _exec_single_V1_segment(const imageType &imIn, imageType &imOut)
    {
	int imHeight = imIn.getHeight();
	volType srcSlices = imIn.getSlices();
	volType destSlices = imOut.getSlices();
	sliceType srcLines;
	sliceType destLines;

	int nthreads = MIN(Core::getInstance()->getNumberOfThreads(), imHeight-1);
	lineType *_bufs = this->createAlignedBuffers(2*nthreads, this->lineLen);
	lineType buf1 = _bufs[0];
	lineType buf2 = _bufs[nthreads];
	
	int l, tid, i, b;
	int nblocks = imHeight / nthreads;

	for (int s=0;s<imIn.getDepth();s++)
	{
	    srcLines = srcSlices[s];
	    destLines = destSlices[s];

	    // First line
	    this->lineFunction(srcLines[0], this->borderBuf, this->lineLen, destLines[0]);
	    this->lineFunction(srcLines[0], srcLines[1], this->lineLen, buf1);
	    copyLine<T>(buf1, this->lineLen, destLines[0]);
	    
	    l = 1;
	    
	    #pragma omp parallel private(tid,buf1,buf2,i,l,b) num_threads(nthreads)
	    {
		#ifdef USE_OPEN_MP
		    tid = omp_get_thread_num();
		    buf1 = _bufs[tid];
		    buf2 = _bufs[tid+nthreads];
		#endif
		    
 		#pragma omp for schedule(static, 1)
		for (b=0;b<nblocks;b++)
		{
		    l = b*nthreads;
		    if (l==0)
		      this->lineFunction(this->borderBuf, srcLines[l], this->lineLen, buf1);
		    else
		      this->lineFunction(srcLines[l-1], srcLines[l], this->lineLen, buf1);
    
		    for (i=0;i<nthreads-1;i++)
		    {
			this->lineFunction(srcLines[l], srcLines[l+1], this->lineLen, buf2);
			this->lineFunction(buf1, buf2, this->lineLen, destLines[l]);
			swap(buf1, buf2);
			l++;
		    }
		    if (l==imHeight-1)
			this->lineFunction(srcLines[l], this->borderBuf, this->lineLen, buf2);
		    else
			this->lineFunction(srcLines[l], srcLines[l+1], this->lineLen, buf2);
		    this->lineFunction(buf1, buf2, this->lineLen, destLines[l]);
		}
		
	    }	  
	    // Remaining lines
	    l = nblocks*nthreads;
	    if (l<imHeight)
	    {
		this->lineFunction(srcLines[l-1], srcLines[l], this->lineLen, buf1);
		while (l<imHeight-1)
		{
		    this->lineFunction(srcLines[l], srcLines[l+1], this->lineLen, buf2);
		    this->lineFunction(buf1, buf2, this->lineLen, destLines[l]);
		    swap(buf1, buf2);
		    l++;
		}
		// Last line
		this->lineFunction(srcLines[l], this->borderBuf, this->lineLen, buf2);
		this->lineFunction(buf1, buf2, this->lineLen, destLines[l]);
	    }
	}
	return RES_OK;
    }

  
};

testClass<UINT8, supLine<UINT8> > tc;

class Test_Dilate_Squ : public TestCase
{
  virtual void run()
  {
      typedef UINT8 dataType;
      typedef Image<dataType> imType;
      
      imType im1(7,7);
      imType im2(im1);
      imType im3(im1);
      
      dataType vec1[] = {
	114, 133,  74, 160,  57,  25,  37,
	 23,  73,   9, 196, 118,  23, 110,
	154, 248, 165, 159, 210,  47,  58,
	213,  74,   8, 163,   3, 240, 213,
	158,  67,  52, 103, 163, 158,   9,
	 85,  36, 124,  12,   7,  56, 253,
	214, 148,  20, 200,  53,  10,  58
      };
      
      im1 << vec1;
      
      dataType dilateSquVec[] = {
	133, 133, 196, 196, 196, 118, 110, 
	248, 248, 248, 210, 210, 210, 110, 
	248, 248, 248, 210, 240, 240, 240, 
	248, 248, 248, 210, 240, 240, 240, 
	213, 213, 163, 163, 240, 253, 253, 
	214, 214, 200, 200, 200, 253, 253, 
	214, 214, 200, 200, 200, 253, 253
      };
      im3 << dilateSquVec;

      // The specialized way
      dilate(im1, im2, sSE());
      TEST_ASSERT(im2==im3);      
      
      // The generic way
      StrElt se;
      se.points = sSE().points;
      dilate(im1, im2, se);
      TEST_ASSERT(im2==im3);      
//       im1.printSelf(1);
//       im2.printSelf(1);
//       im3.printSelf(1);
  }
};



int main(int argc, char *argv[])
{
      TestSuite ts;
      ADD_TEST(ts, Test_Dilate_Hex);
      ADD_TEST(ts, Test_Dilate_Squ);
      
      UINT BENCH_NRUNS = 5E3;
      Image_UINT8 im1(1024, 1024), im2(im1);
//       BENCH_IMG_STR(dilate, "hSE", im1, im2, hSE());
//       BENCH_IMG_STR(dilate, "sSE", im1, im2, sSE());
// cout << endl;
//       tc(im1, im2, sSE());
      return ts.run();
  
}

