/*
 * Copyright (c) 2011-2014, Matthieu FAESSEL and ARMINES
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


#include "Core/include/DCore.h"
#include "DMorpho.h"
#include "DMorphoWatershed.hpp"
#include "DMorphoWatershedExtinction.hpp"

using namespace smil;

class Test_Basins : public TestCase
{
  virtual void run()
  {
      typedef UINT8 dtType;
      
      dtType vecIn[] = { 
	2, 2, 2, 2, 2, 2,
	 7, 7, 7, 7, 7, 7,
	2, 7, 5, 6, 2, 2,
	 2, 6, 5, 6, 2, 2,
	2, 2, 6, 4, 3, 2,
	 2, 2, 3, 4, 2, 2,
	2, 2, 2, 2, 4, 2
      };
      
      dtType vecMark[] = { 
	1, 1, 1, 1, 1, 1,
	 0, 0, 0, 0, 0, 0,
	2, 0, 0, 0, 3, 3,
	 2, 0, 0, 0, 3, 3,
	2, 2, 0, 0, 0, 3,
	 2, 2, 0, 0, 3, 3,
	2, 2, 2, 2, 0, 3
      };
      
      Image<dtType> imIn(6,7);
      Image<dtType> imMark(imIn);
      Image<dtType> imLbl(imIn);

      imIn << vecIn;
      imMark << vecMark;
      
      StrElt se = hSE();
      
      basins(imIn, imMark, imLbl, se);
      
      dtType vecLblTruth[] = { 
	1,    1,    1,    1,    1,    1,
	  1,    1,    1,    1,    1,    1,
	2,    2,    3,    3,    3,    3,
	  2,    2,    3,    3,    3,    3,
	2,    2,    2,    3,    3,    3,
	  2,    2,    2,    3,    3,    3,
	2,    2,    2,    2,    3,    3,
      };
      
      Image<dtType> imLblTruth(imIn);
      
      imLblTruth << vecLblTruth;
      
      TEST_ASSERT(imLbl==imLblTruth);
      
      if (retVal!=RES_OK)
	imLbl.printSelf(1, true);
  }
};


class Test_ProcessWatershedHierarchicalQueue : public TestCase
{
  virtual void run()
  {
      UINT8 vecIn[] = { 
	2, 2, 2, 2, 2, 2,
	7, 7, 7, 7, 7, 7,
	2, 7, 5, 6, 2, 2,
	2, 6, 5, 6, 2, 2,
	2, 2, 6, 4, 3, 2,
	2, 2, 3, 4, 2, 2,
	2, 2, 2, 2, 4, 2
      };
      
      UINT8 vecLbl[] = { 
	1, 1, 1, 1, 1, 1,
	0, 0, 0, 0, 0, 0,
	2, 0, 0, 0, 3, 3,
	2, 0, 0, 0, 3, 3,
	2, 2, 0, 0, 0, 3,
	2, 2, 0, 0, 3, 3,
	2, 2, 2, 2, 0, 3
      };
      
      Image_UINT8 imIn(6,7);
      Image_UINT8 imLbl(imIn);
      Image_UINT8 imStatus(imIn);

      imIn << vecIn;
      imLbl << vecLbl;
      
      HierarchicalQueue<UINT8> pq;
      StrElt se = hSE();
      
      initWatershedHierarchicalQueue(imIn, imLbl, imStatus, pq);
      processWatershedHierarchicalQueue(imIn, imLbl, imStatus, pq, se);

      UINT8 vecLblTruth[] = { 
	1,    1,    1,    1,    1,    1,
	  1,    1,    1,    1,    1,    1,
	2,    3,    3,    3,    3,    3,
	  2,    3,    3,    3,    3,    3,
	2,    2,    3,    3,    3,    3,
	  2,    2,    2,    3,    3,    3,
	2,    2,    2,    2,    3,    3,
      };
      
      UINT8 vecStatusTruth[] = { 
	2, 2, 2, 2, 2, 2,
	3, 3, 3, 3, 3, 3,
	2, 3, 2, 2, 2, 2,
	2, 3, 2, 2, 2, 2,
	2, 2, 3, 3, 2, 2,
	2, 2, 2, 3, 2, 2,
	2, 2, 2, 2, 3, 2
      };
      
      Image_UINT8 imLblTruth(imIn);
      Image_UINT8 imStatusTruth(imIn);
      
      imLblTruth << vecLblTruth;
      imStatusTruth << vecStatusTruth;
      
      TEST_ASSERT(imLbl==imLblTruth);
      TEST_ASSERT(imStatus==imStatusTruth);
      
      if (retVal!=RES_OK)
      {
	imLbl.printSelf(1, true);
	imStatus.printSelf(1, true);
      }
  }
};




class Test_Watershed : public TestCase
{
  virtual void run()
  {
      UINT8 vecIn[] = { 
	2, 2, 2, 2, 2, 2,
	 7, 7, 7, 7, 7, 7,
	2, 7, 5, 6, 2, 2,
	 2, 6, 5, 6, 2, 2,
	2, 2, 6, 4, 3, 2,
	 2, 2, 3, 4, 2, 2,
	2, 2, 2, 2, 4, 2
      };
      
      UINT8 vecMark[] = { 
	1, 1, 1, 1, 1, 1,
	 0, 0, 0, 0, 0, 0,
	2, 0, 0, 0, 3, 3,
	 2, 0, 0, 0, 3, 3,
	2, 2, 0, 0, 0, 3,
	 2, 2, 0, 0, 3, 3,
	2, 2, 2, 2, 0, 3
      };
      
      Image_UINT8 imIn(6,7);
      Image_UINT8 imMark(imIn);
      Image_UINT8 imWs(imIn);
      Image_UINT8 imLbl(imIn);

      imIn << vecIn;
      imMark << vecMark;
      
      StrElt se = hSE();
      
      watershed(imIn, imMark, imWs, imLbl, se);
      
      UINT8 vecLblTruth[] = { 
	1,    1,    1,    1,    1,    1,
	  1,    1,    1,    1,    1,    1,
	2,    3,    3,    3,    3,    3,
	  2,    3,    3,    3,    3,    3,
	2,    2,    3,    3,    3,    3,
	  2,    2,    2,    3,    3,    3,
	2,    2,    2,    2,    3,    3,
      };
      
      UINT8 vecWsTruth[] = { 
	0,    0,    0,    0,    0,    0,
	255,  255,  255,  255,  255,  255,
	0,  255,    0,    0,    0,    0,
	  0,  255,    0,    0,    0,    0,
	0,    0,  255,  255,    0,    0,
	  0,    0,    0,  255,    0,    0,
	0,    0,    0,    0,  255,    0,
      };
      
      Image_UINT8 imLblTruth(imIn);
      Image_UINT8 imWsTruth(imIn);
      
      imLblTruth << vecLblTruth;
      imWsTruth << vecWsTruth;
      
      TEST_ASSERT(imLbl==imLblTruth);
      TEST_ASSERT(imWs==imWsTruth);
      
      if (retVal!=RES_OK)
      {
	imLbl.printSelf(1, true);
	imWs.printSelf(1, true);
      }
      
      // Test idempotence
      
      Image_UINT8 imWs2(imIn);
      
      watershed(imWs, imMark, imWs2, imLbl, se);
      TEST_ASSERT(imWs2==imWs);
  }
};

class Test_Watershed_Indempotence : public TestCase
{
  virtual void run()
  {
      UINT8 vecIn[] = { 
	  98,   81,   45,  233,  166,  112,  100,   20,  176,   79,
	      4,   11,   57,  246,  137,   90,   69,  212,   16,  219,
	  131,  165,   20,    4,  201,  100,  166,   57,  144,  104,
	    143,  242,  185,  188,  221,   97,   46,   66,  117,  222,
	  146,  121,  234,  204,  113,  116,   40,  183,   74,   56,
	    147,  205,  221,  168,  210,  168,   14,  122,  226,  158,
	  226,  114,  146,  157,   48,  112,  254,   94,  179,  117,
	      61,   71,  238,   40,   20,   97,  157,   60,   25,  231,
	  116,  173,  181,   83,   86,  137,  252,  100,    4,  223,
	      4,  231,   83,  150,  133,  131,    8,  133,  226,  187,
      };
      
      UINT8 vecMark[] = { 
	  1,    1,    0,    0,    0,    2,    0,    0,    0,    0,
	    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
	  0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
	    0,    0,    0,    0,    0,    0,    0,    3,    0,    0,
	  4,    0,    0,    0,    5,    0,    0,    0,    0,    0,
	    0,    0,    6,    0,    0,    0,    0,    0,    0,    0,
	  0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
	    0,    0,    0,    0,    7,    0,    0,    8,    0,    0,
	  0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
	    0,    0,    0,    0,    9,    0,    0,    0,    0,   10,
      };
      
      Image_UINT8 imIn(10,10);
      Image_UINT8 imMark(imIn);
      Image_UINT8 imWs(imIn);
      Image_UINT8 imWs2(imIn);

      imIn << vecIn;
      imMark << vecMark;

      watershed(imIn, imMark, imWs, hSE());
      watershed(imWs, imMark, imWs2, hSE());
      
      TEST_ASSERT(imWs==imWs2);
      
      if (retVal!=RES_OK)
      {
	  imWs.printSelf(1, true);
	  imWs2.printSelf(1, true);
      }
  }
};


class Test_Watershed_Extinction : public TestCase 
{
	virtual void run () 
	{
		UINT8 vecIn[] = {
		    2,    2,    2,    2,    2,
		      3,    2,    5,    9,    5,
		    3,    3,    9,    0,    0,
		      1,    1,    9,    0,    0,
		    1,    1,    9,    0,    0,
		};
		UINT8 vecMark[] = {
		    0,    1,    0,    0,    0,
		      0,    0,    0,    0,    0,
		    0,    0,    0,    0,    0,
		      0,    2,    0,    3,    0,
		    0,    2,    0,    0,    3,
		};
		StrElt se = sSE();

		Image_UINT8 imIn (5,5) ;
		Image_UINT8 imMark (imIn) ;
		Image_UINT8 imTruth (imIn) ;
		Image_UINT8 imResult (imIn) ;

		imIn << vecIn;
		imMark << vecMark;

		// Volume
		UINT8 volTruth[] = {
		    0,    3,    0,    0,    0,
		      0,    0,    0,    0,    0,
		    0,    0,    0,    0,    0,
		      0,    1,    0,    2,    0,
		    0,    1,    0,    0,    2,
		};
		imTruth << volTruth;

		watershedExtinction (imIn, imMark, imResult, "v", se) ;
		TEST_ASSERT(imResult==imTruth);
		if (retVal!=RES_OK)
		{
		    cout << endl << "in watershedExtinction volumic" << endl;
		    imResult.printSelf (1,true);
		    imTruth.printSelf (1,true);
		}
		
		// Area
		UINT8 areaTruth[] = {
		  0,    1,    0,    0,    0,
		    0,    0,    0,    0,    0,
		  0,    0,    0,    0,    0,
		    0,    3,    0,    2,    0,
		  0,    3,    0,    0,    2,
		};
		imTruth << areaTruth;

		watershedExtinction (imIn, imMark, imResult, "a", se) ;
		TEST_ASSERT(imResult==imTruth);
		if (retVal!=RES_OK)
		{
		    cout << endl << "in watershedExtinction area" << endl;
		    imResult.printSelf (1,true);
		    imTruth.printSelf (1,true);
		}
		
		// Dynamic
		UINT8 dynTruth[] = {
		    0,    3,    0,    0,    0,
		      0,    0,    0,    0,    0,
		    0,    0,    0,    0,    0,
		      0,    2,    0,    1,    0,
		    0,    2,    0,    0,    1,
		};
		imTruth << dynTruth;

		watershedExtinction (imIn, imMark, imResult, "d", se) ;
		TEST_ASSERT(imResult==imTruth);
		if (retVal!=RES_OK)
		{
		    cout << endl << "in watershedExtinction dynamic" << endl;
		    imResult.printSelf (1,true);
		    imTruth.printSelf (1,true);
		}
		
	}
};

class Test_Watershed_Extinction_Graph : public TestCase 
{
	virtual void run () 
	{
		UINT8 vecIn[] = {
			2, 2, 2, 2, 2,
			3, 2, 3, 9, 5,
			3, 3, 9, 0, 0,
			1, 1, 9, 0, 0,
			1, 1, 9, 0, 0
		};
		UINT8 vecMark[] = {
			0, 1, 0, 0, 0,
			0, 0, 0, 0, 0,
			0, 0, 0, 0, 0,
			0, 2, 0, 3, 0,
			0, 2, 0, 0, 3
		};
		UINT8 vecTruth[] = {
		  0,    2,    0,    0,    0,
		    0,    0,    0,    0,    0,
		  0,    0,    0,    0,    0,
		    0,    1,    0,    3,    0,
		  0,    1,    0,    0,    3,
		};

		StrElt se = sSE();

		Image_UINT8 imIn (5,5) ;
		Image_UINT8 imMark (imIn) ;
		Image_UINT8 imTruth (imIn) ;
		Image_UINT8 imResult (imIn) ;

		Graph<UINT8,UINT8> g;

		imIn << vecIn;
		imMark << vecMark;
		imTruth << vecTruth;

		watershedExtinctionGraph (imIn, imMark, imResult, g, se) ;
//		g.printSelf();
		TEST_ASSERT(imResult==imTruth);
		if (retVal!=RES_OK)
		{
		    imResult.printSelf (1,true);
		    imTruth.printSelf (1,true);
		}
	}
};


class Test_Build : public TestCase
{
  virtual void run()
  {
      UINT8 vecIn[] = { 
	1, 2, 0, 5, 5, 5, 3, 3, 3, 1, 1
      };
      
      UINT8 vecMark[] = { 
	0, 0, 0, 0, 4, 1, 1, 2, 0, 0, 0
      };
      
      Image_UINT8 imIn(11,1);
      Image_UINT8 imMark(imIn);
      Image_UINT8 imBuild(imIn);

      imIn << vecIn;
      imMark << vecMark;
      
      dualBuild(imIn, imMark, imBuild, sSE());
      
      UINT8 vecTruth[] = { 
	0, 0, 0, 0, 4, 2, 2, 2, 1, 1, 1
      };
      
      Image_UINT8 imTruth(imIn);
      
      imTruth << vecTruth;
      
      TEST_ASSERT(imBuild==imTruth);
      
      if (retVal!=RES_OK)
	imBuild.printSelf(1);
  }
};

// template <class T>
// class FloodingBasin
// {
//   public:
//     virtual void test() { cout << "FloodingBasin" << endl; }
//     virtual FloodingBasin *clone() = 0;
//     virtual void initValue(const T &val) {}
//     virtual void increment(const T &val=1) {}
//     virtual void raise(const T &val) {}
//     virtual void merge(FloodingBasin<T> *other) {}
//     virtual void finalize() {}
//     UINT label;
// };
// 
// template <class T>
// class VolumeFloodingBasin : public FloodingBasin<T>
// {
//   public:
//     virtual void test() { cout << "Volume FloodingBasin" << endl; }
//     virtual VolumeFloodingBasin *clone() { return new VolumeFloodingBasin; }
//     virtual void initValue(const T &val) {}
//     virtual void increment(const T &val=1) {}
//     virtual void raise(const T &val) {}
//     virtual void merge(FloodingBasin<T> *other) {}
//     virtual void finalize() {}
//     UINT label;
// };

template <class T>
class Flooding
{
  public:
    Flooding()
      : basinNbr(0)
    {
    }
    virtual ~Flooding()
    {
	deleteBasins();
	delete aType;
    }
    void setBasinType(FloodingBasin<T> *a)
    {
	aType = a->clone();
    }
    
    virtual UINT merge(UINT &lbl1, UINT &lbl2) {}
    
    template <class labelT>
    RES_T flood(const Image<T> &imIn, Image<labelT> &imLbl, const StrElt & se)
    {
	ASSERT_ALLOCATED (&imIn, &imLbl);
	ASSERT_SAME_SIZE (&imIn, &imLbl);
// 	ImageFreezer freezer (imOut);
// 	ImageFreezer freezer2 (imBasinsOut);

// 	copy (imMarkers, imBasinsOut);

	UINT nbr_label = maxVal (imLbl);
	createBasins(nbr_label + 1);

/*
	HierarchicalQueue < T > pq;
	initWatershedHierarchicalQueueExtinction(imIn, imBasinsOut, pq, e, nbr_label);
	processWatershedExtinctionHierarchicalQueue<T,labelT,HierarchicalQueue<T>,outT,basinT> (imIn, imBasinsOut, pq, se, e, nbr_label);
	
	typename ImDtTypes < outT >::lineType pixOut = imOut.getPixels ();
	typename ImDtTypes < labelT >::lineType pixMarkers = imMarkers.getPixels ();

	fill (imOut, outT(0));
	// Create the image containing the ws lines
	T wsVal = ImDtTypes < outT >::max ();
	size_t max_ext=0;
	for (int i=0; i<nbr_label; ++i) {
		if (max_ext<e[i].extinctionValue) max_ext = e[i].extinctionValue ;
	}
	basinT *e_cpy = new basinT[nbr_label+1]; 
	memcpy (e_cpy, e, (nbr_label+1)*sizeof(basinT));
	for (int i=1; i<nbr_label+1; ++i)
		e_cpy[i].label = i;
	vector<basinT> ve (e_cpy+1, e_cpy+nbr_label+1);
	// Sorting the vol_ev_val.
	sort (ve.begin(), ve.end());
	for (int i=0; i<nbr_label; ++i){
		e[ve[i].label].extinctionValue = nbr_label - i;
	}
	delete[]e_cpy;
	for (size_t i=0; i<imIn.getPixelCount (); i++, *pixMarkers++, pixOut++) {
		if(*pixMarkers != labelT(0))
			*pixOut = e[*pixMarkers].extinctionValue ;
	}

	delete[]e;
	return RES_OK;
	
	*/
	initHierarchicalQueue(imIn, imLbl);
	processHierarchicalQueue(imIn, imLbl, se);
    }
    
  protected:
    FloodingBasin<T> *aType;
    vector<FloodingBasin<T>*> basins;
    UINT basinNbr;
    HierarchicalQueue < T > hq;
    T currentLevel;
    
    void createBasins(UINT nbr)
    {
	if (nbr==basinNbr)
	  return;
	
	if (basinNbr!=0)
	  deleteBasins();
	
	for (UINT i=0;i<nbr;i++)
	{
	    FloodingBasin<T> *_basin = aType->clone();
	    basins.push_back(aType->clone());
	}
    }
    void deleteBasins()
    {
	if (basinNbr==0)
	  return;
	
	for (UINT i=0;i<basinNbr;i++)
	  delete basins[i];
	basins.clear();
	basinNbr = 0;
    }
    
    template <class labelT>
    void initHierarchicalQueue(const Image<T> &imIn, Image<labelT> &imLbl)
    {
	for (UINT i = 0; i < basinNbr + 1; ++i)
	  basins[i]->label = i;

	// Empty the priority queue
	hq.initialize (imIn);
	typename ImDtTypes < T >::lineType inPixels = imIn.getPixels ();
	typename ImDtTypes < labelT >::lineType lblPixels = imLbl.getPixels ();
	size_t s[3];

	imIn.getSize (s);
	size_t offset = 0;

	for (size_t k = 0; k < s[2]; k++)
	    for (size_t j = 0; j < s[1]; j++)
		for (size_t i = 0; i < s[0]; i++) {
		    if (*lblPixels != 0) {
			hq.push (T (*inPixels), offset);
			basins[*lblPixels]->initValue(*inPixels);  
			basins[*lblPixels]->increment(*inPixels);  
		    }

		    inPixels++;
		    lblPixels++;
		    offset++;
		}
    }
    
    template < class labelT >
    RES_T processHierarchicalQueue (const Image < T > &imIn,
							   Image < labelT > &imLbl,
							   const StrElt & se)
    {
	typename ImDtTypes < T >::lineType inPixels = imIn.getPixels ();
	typename ImDtTypes < labelT >::lineType lblPixels =
	    imLbl.getPixels ();
	vector < int >dOffsets;

	vector < IntPoint >::const_iterator it_start = se.points.begin ();
	vector < IntPoint >::const_iterator it_end = se.points.end ();
	vector < IntPoint >::const_iterator it;
	vector < UINT > tmpOffsets;
	size_t s[3];

	imIn.getSize (s);

	T curPixVal;
	currentLevel = 0;
	size_t curOffset;

	// set an offset distance for each se point
	for (it = it_start; it != it_end; it++) 
	{
	    dOffsets.push_back (it->x + it->y * s[0] + it->z * s[0] * s[1]);
	}
	vector < int >::iterator it_off_start = dOffsets.begin ();
	vector < int >::iterator it_off;

	while (!hq.isEmpty ()) 
	{
	    curOffset = hq.pop ();
	    curPixVal = inPixels[curOffset];

	    // Rising the elevation of the basins
	    if (curPixVal > currentLevel) 
	    {
		currentLevel = curPixVal;
		for (labelT i = 1; i < basinNbr + 1; ++i) 
			basins[i]->raise(currentLevel);
	    }

	    size_t x0, y0, z0;

	    imIn.getCoordsFromOffset (curOffset, x0, y0, z0);
	    bool oddLine = se.odd && ((y0) % 2);
	    int x, y, z;
	    size_t nbOffset;
	    UINT l1 = lblPixels[curOffset], l2;

	    for (it = it_start, it_off = it_off_start; it != it_end; it++, it_off++)
		if (it->x != 0 || it->y != 0 || it->z != 0)	// useless if x=0 & y=0 & z=0
		{
		    x = x0 + it->x;
		    y = y0 + it->y;
		    z = z0 + it->z;
		    if (oddLine)
			x += (((y + 1) % 2) != 0);
		    if (x >= 0 && x < (int) s[0] && y >= 0 && y < (int) s[1]
			&& z >= 0 && z < (int) s[2]) 
		    {
			nbOffset = curOffset + *it_off;
			if (oddLine)
			    nbOffset += (((y + 1) % 2) != 0);
			l2 = lblPixels[nbOffset];
			if (l2 > 0 && l2 != basinNbr + 1) 
			{
			    while (l2 != basins[l2]->label) 
			    {
				l2 = basins[l2]->label;
			    }

			    if (l1 == 0 || l1 == basinNbr + 1) 
			    {
				l1 = l2; // current pixel takes the label of its first labelled ngb  found
				lblPixels[curOffset] = l1;
				basins[l1]->increment();
			    }
			    else if (l1 != l2) 
			    {
				while (l1 != basins[l1]->label) 
				{
				    l1 = basins[l1]->label;
				}
				if (l1 != l2) 
				  merge(l1, l2);
			    }
			}
			else if (l2 == 0) 	// Add it to the tmp offsets queue
			{
			    tmpOffsets.push_back (nbOffset);
			    lblPixels[nbOffset] = basinNbr + 1;
			}
		    }
		}
	    if (!tmpOffsets.empty ()) 
	    {
		typename vector < UINT >::iterator t_it = tmpOffsets.begin ();

		while (t_it != tmpOffsets.end ()) 
		{
		    hq.push (inPixels[*t_it], *t_it);
		    t_it++;
		}
	    }
	    tmpOffsets.clear ();
	}

	// Update Last level of flooding.
	basins[lblPixels[curOffset]]->finalize();

	return RES_OK;
    }
};


template <class T>
struct Flooding_B : public Flooding<T>
{
};



int main(int argc, char *argv[])
{
      TestSuite ts;
      ADD_TEST(ts, Test_Basins);
      ADD_TEST(ts, Test_ProcessWatershedHierarchicalQueue);
      ADD_TEST(ts, Test_Watershed);
      ADD_TEST(ts, Test_Watershed_Indempotence);
      ADD_TEST(ts, Test_Watershed_Extinction);
//       ADD_TEST(ts, Test_Watershed_Extinction_Graph);
      ADD_TEST(ts, Test_Build);
      
      Flooding_B<UINT8> fb;
      VolumeFloodingBasin<UINT8> b;
      fb.setBasinType(&b);
      
      UINT8 vecIn[] = {
	  2,    2,    2,    2,    2,
	    3,    2,    5,    9,    5,
	  3,    3,    9,    0,    0,
	    1,    1,    9,    0,    0,
	  1,    1,    9,    0,    0,
      };
      UINT8 vecMark[] = {
	  0,    1,    0,    0,    0,
	    0,    0,    0,    0,    0,
	  0,    0,    0,    0,    0,
	    0,    2,    0,    3,    0,
	  0,    2,    0,    0,    3,
      };
      StrElt se = sSE();

      Image_UINT8 imIn (5,5) ;
      Image_UINT8 imMark (imIn) ;
      
      fb.flood(imIn, imMark, hSE());
      fb[0]->test();
      
      return ts.run();
      
}


