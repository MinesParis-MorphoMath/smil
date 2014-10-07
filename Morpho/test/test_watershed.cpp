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
#include <boost/concept_check.hpp>

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

      imIn << vecIn;
      imLbl << vecLbl;
      
      HierarchicalQueue<UINT8> pq;
      StrElt se = hSE();
      
      watershedFlooding<UINT8,UINT8> flooding;
      flooding.initialize(imIn, imLbl, se);
      flooding.processImage(imIn, imLbl, se);

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
      TEST_ASSERT(flooding.imStatus==imStatusTruth);
      
      if (retVal!=RES_OK)
      {
	imLbl.printSelf(1, true);
	flooding.imStatus.printSelf(1, true);
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

		Graph<UINT8,UINT8> graph;

		imIn << vecIn;
		imMark << vecMark;

		vector<Edge<UINT8> > trueEdges;
		trueEdges.push_back(Edge<UINT8>(1,2, 6));
		trueEdges.push_back(Edge<UINT8>(3,2, 30));
      
		watershedExtinctionGraph (imIn, imMark, imResult, graph, "v", se) ;
		
		TEST_ASSERT(trueEdges==graph.getEdges());
		if (retVal!=RES_OK)
		    graph.printSelf();
		
		vector<Edge<UINT8> > trueEdges2;
		trueEdges2.push_back(Edge<UINT8>(3,2, 1));
		trueEdges2.push_back(Edge<UINT8>(1,2, 2));
		
		Graph<UINT8,UINT8> rankGraph = watershedExtinctionGraph (imIn, imMark, imResult, "v", se) ;
		
		TEST_ASSERT(trueEdges2==rankGraph.getEdges());
		if (retVal!=RES_OK)
		    rankGraph.printSelf();
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


template <class T, class labelT, class extValType=UINT, class HQ_Type=HierarchicalQueue<T> >
class ExtinctionFlooding : public BaseFlooding<T, labelT, HQ_Type>
{
  protected:
      UINT labelNbr, basinNbr;
      T currentLevel;
      vector<labelT> equivalentLabels;
      vector<extValType> extinctionValues;
      size_t lastOffset;

      inline virtual void insertPixel(const size_t &offset, const labelT &lbl) {}
      inline virtual void raiseLevel(const labelT &lbl) {}
      inline virtual UINT mergeBasins(const labelT &lbl1, const labelT &lbl2) { return 0; };
      inline virtual void finalize(const labelT &lbl) {}

  public:
    template <class outT>
    RES_T floodWithExtValues(const Image<T> &imIn, const Image<labelT> &imMarkers, Image<outT> &imExtValOut, Image<labelT> &imBasinsOut, const StrElt & se=DEFAULT_SE)
    {
	ASSERT_ALLOCATED (&imExtValOut);
	ASSERT_SAME_SIZE (&imIn, &imExtValOut);
	
	ASSERT(this->flood(imIn, imMarkers, imBasinsOut, se)==RES_OK);
	
	ImageFreezer freezer (imExtValOut);
	
	typename ImDtTypes < outT >::lineType pixOut = imExtValOut.getPixels ();
	typename ImDtTypes < labelT >::lineType pixMarkers = imMarkers.getPixels ();

	fill(imExtValOut, outT(0));
	
	for (size_t i=0; i<imIn.getPixelCount (); i++, pixMarkers++, pixOut++) 
	{
	    if(*pixMarkers != labelT(0))
	      *pixOut = extinctionValues[*pixMarkers] ;
	}

	return RES_OK;
    }
    template <class outT>
    RES_T floodWithExtValues(const Image<T> &imIn, const Image<labelT> &imMarkers, Image<outT> &imExtValOut, const StrElt & se=DEFAULT_SE)
    {
	Image<labelT> imBasinsOut(imMarkers);
	return floodWithExtValues(imIn, imMarkers, imExtValOut, imBasinsOut, se);
    }
    
    RES_T floodWithExtRank(const Image<T> &imIn, const Image<labelT> &imMarkers, Image<labelT> &imExtRankOut, Image<labelT> &imBasinsOut, const StrElt & se=DEFAULT_SE)
    {
	ASSERT_ALLOCATED (&imExtRankOut);
	ASSERT_SAME_SIZE (&imIn, &imExtRankOut);
	
	ASSERT(this->flood(imIn, imMarkers, imBasinsOut, se)==RES_OK);

	typename ImDtTypes < labelT >::lineType pixOut = imExtRankOut.getPixels ();
	typename ImDtTypes < labelT >::lineType pixMarkers = imMarkers.getPixels ();

	ImageFreezer freezer (imExtRankOut);
	
	// Sort by extinctionValues
	vector<UINT> rank(labelNbr);
	for (UINT i=0;i<labelNbr;i++)
	  rank[i] = i+1;
	extinctionValues.size();
// 	sort(rank.begin(), rank.end(), extinctionValues);
	for (UINT i=0;i<labelNbr;i++)
	  extinctionValues[rank[i]] = i+1;
	
	fill(imExtRankOut, labelT(0));
	
	for (size_t i=0; i<imIn.getPixelCount (); i++, pixMarkers++, pixOut++) 
	{
	    if(*pixMarkers != labelT(0))
	      *pixOut = extinctionValues[*pixMarkers] ;
	}

	return RES_OK;
    }
    RES_T floodWithExtRank(const Image<T> &imIn, const Image<labelT> &imMarkers, Image<labelT> &imExtRankOut, const StrElt & se=DEFAULT_SE)
    {
	Image<labelT> imBasinsOut(imMarkers);
	return floodWithExtRank(imIn, imMarkers, imExtRankOut, imBasinsOut, se);
    }
    
  protected:
    virtual RES_T initialize(const Image<T> &imIn, Image<labelT> &imLbl, const StrElt &se)
    {
	BaseFlooding<T, labelT, HQ_Type>::initialize(imIn, imLbl, se);
	
	labelNbr = maxVal(imLbl);
	createBasins(labelNbr+1);
	currentLevel = ImDtTypes<T>::min();
    }
    
    virtual void createBasins(const UINT nbr)
    {
	equivalentLabels.resize(nbr);
	extinctionValues.resize(nbr, 0);
	
	for (UINT i=0;i<nbr;i++)
	  equivalentLabels[i] = i;
	
	basinNbr = nbr;
    }
    virtual void deleteBasins()
    {
	equivalentLabels.clear();
	extinctionValues.clear();
	
	basinNbr = 0;
    }
	
    virtual RES_T processImage(const Image<T> &imIn, Image<labelT> &imLbl, const StrElt &se)
    {
	BaseFlooding<T, labelT, HQ_Type>::processImage(imIn, imLbl, se);
	
	// Update last level of flooding.
	labelT l2 = this->lblPixels[lastOffset];
	while (l2 != equivalentLabels[l2]) 
	    l2 = equivalentLabels[l2];
	finalize(l2);
    }
    inline virtual void processPixel(const size_t &curOffset)
    {
	if (this->inPixels[curOffset] > currentLevel) 
	{
	    currentLevel = this->inPixels[curOffset];
	    for (labelT i = 1; i < labelNbr + 1 ; ++i) 
		    raiseLevel(i);
	}

	BaseFlooding<T, labelT, HQ_Type>::processPixel(curOffset);
	
	lastOffset = curOffset;
    }
    inline virtual void processNeighbor(const size_t &curOffset, const size_t &nbOffset)
    {
	UINT8 nbStat = this->statPixels[nbOffset];
	labelT l1 = this->lblPixels[curOffset];
	
	if (nbStat==HQ_CANDIDATE) 
	{
	    this->lblPixels[nbOffset] = l1;
	    this->statPixels[nbOffset] = HQ_QUEUED;
	    this->hq.push(this->inPixels[nbOffset], nbOffset);
	    this->insertPixel(nbOffset, l1);
	}
	else if (nbStat==HQ_LABELED)
	{
	    labelT l2 = this->lblPixels[nbOffset];
	    
	    if (l1==0)
	    {
		this->lblPixels[curOffset] = l2;
		this->insertPixel(curOffset, l2);
	    }
	    else if (l1!=l2)
	    {
		while (l1!=equivalentLabels[l1])
		  l1 = equivalentLabels[l1];
		while (l2!=equivalentLabels[l2])
		  l2 = equivalentLabels[l2];
		
		if (l1 != l2)
		    mergeBasins(l1, l2);
	    }
	}
    }

    
};


template <class T, class labelT, class extValType=UINT, class HQ_Type=HierarchicalQueue<T> >
class AreaExtinctionFlooding : public ExtinctionFlooding<T, labelT, extValType, HQ_Type>
{
  public:
    vector<UINT> areas;
    vector<T> minValues;
    
    virtual void createBasins(const UINT nbr)
    {
	areas.resize(nbr, 0);
	minValues.resize(nbr, ImDtTypes<T>::max());
	
	ExtinctionFlooding<T, labelT, extValType, HQ_Type>::createBasins(nbr);
    }
    
    virtual void deleteBasins()
    {
	areas.clear();
	minValues.clear();
	
	ExtinctionFlooding<T, labelT, extValType, HQ_Type>::deleteBasins();
    }
    
    inline virtual void insertPixel(const size_t &offset, const labelT &lbl)
    {
	if (this->inPixels[offset] < minValues[lbl])
	  minValues[lbl] = this->inPixels[offset];
	
	areas[lbl]++;
    }
    virtual UINT mergeBasins(const labelT &lbl1, const labelT &lbl2)
    {
	UINT eater, eaten;
	
	if (areas[lbl1] > areas[lbl2] || (areas[lbl1] == areas[lbl2] && minValues[lbl1] < minValues[lbl2]))
	{
	    eater = lbl1;
	    eaten = lbl2;
	}
	else 
	{
	    eater = lbl2;
	    eaten = lbl1;
	}
	
	this->extinctionValues[eaten] = areas[eaten];
	areas[eater] += areas[eaten];
	this->equivalentLabels[eaten] = eater;
	
	return eater;
    }
    virtual void finalize(const labelT &lbl)
    {
	this->extinctionValues[lbl] += areas[lbl];
    }
};



class Test_Watershed_Extinction_new : public TestCase 
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
		Image_UINT8 imBasins (imIn) ;
		Image_UINT8 imTruth (imIn) ;
		Image_UINT8 imResult (imIn) ;

		imIn << vecIn;
		imMark << vecMark;

		UINT8 basinsTruth[] = {
		  1,    1,    1,    1,    1,
		    1,    1,    3,    3,    3,
		  2,    2,    3,    3,    3,
		    2,    2,    3,    3,    3,
		  2,    2,    3,    3,    3,
		};
		imTruth << basinsTruth;
		  
		AreaExtinctionFlooding<UINT8,UINT8> areaFlood;
		areaFlood.floodWithExtValues(imIn, imMark, imResult, imBasins, se);

		TEST_ASSERT(imBasins==imTruth);
		if (retVal!=RES_OK)
		{
		    imBasins.printSelf (1);
		    imTruth.printSelf(1);
		}
		
		// Area
		UINT8 areaTruth[] = {
		  0,   22,    0,    0,    0,
		    0,    0,    0,    0,    0,
		  0,    0,    0,    0,    0,
		    0,    4,    0,    6,    0,
		  0,    4,    0,    0,    6,

		};
		imTruth << areaTruth;
		
		TEST_ASSERT(imResult==imTruth);
		if (retVal!=RES_OK)
		{
		    imResult.printSelf (1);
		    imTruth.printSelf(1);
		}
		
		areaFlood.floodWithExtRank(imIn, imMark, imResult, imBasins, se);
		
		// Area-rank
		UINT8 areaRankTruth[] = {
		  0,   22,    0,    0,    0,
		    0,    0,    0,    0,    0,
		  0,    0,    0,    0,    0,
		    0,    4,    0,    6,    0,
		  0,    4,    0,    0,    6,

		};
		imTruth << areaRankTruth;
		
		TEST_ASSERT(imResult==imTruth);
		if (retVal!=RES_OK)
		{
		    imResult.printSelf (1);
		    imTruth.printSelf(1);
		}
	}
};

int main(int argc, char *argv[])
{
      TestSuite ts;
      
      ADD_TEST(ts, Test_Basins);
      ADD_TEST(ts, Test_ProcessWatershedHierarchicalQueue);
      ADD_TEST(ts, Test_Watershed);
      ADD_TEST(ts, Test_Watershed_Indempotence);
      ADD_TEST(ts, Test_Watershed_Extinction);
      ADD_TEST(ts, Test_Watershed_Extinction_Graph);
      ADD_TEST(ts, Test_Build);
      
      ADD_TEST(ts, Test_Watershed_Extinction_new);
      
      return ts.run();
      
}


