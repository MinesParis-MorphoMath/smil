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



class Test_Extinction_Flooding : public TestCase 
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
	    1,    1,    1,    1,    1,
	  1,    1,    3,    3,    3,
	    2,    2,    3,    3,    3,
	  2,    2,    3,    3,    3,
	};
	imTruth << basinsTruth;
	  
	AreaExtinctionFlooding<UINT8,UINT8> areaFlood;
	areaFlood.floodWithExtValues(imIn, imMark, imResult, imBasins, se);

	TEST_ASSERT(imBasins==imTruth);
	if (retVal!=RES_OK)
	{
	    imBasins.printSelf(1, true);
	    imTruth.printSelf(1, true);
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
    }
};

class Test_Area_Extinction : public TestCase 
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
	UINT16 vecMark[] = {
	    0,    1,    0,    0,    0,
	      0,    0,    0,    0,    0,
	    0,    0,    0,    0,    0,
	      0,    2,    0,    3,    0,
	    0,    2,    0,    0,    3,
	};
	StrElt se = sSE();

	Image_UINT8 imIn (5,5) ;
	Image_UINT16 imMark (imIn) ;
	Image_UINT16 imTruth (imIn) ;
	Image_UINT16 imResult (imIn) ;

	imIn << vecIn;
	imMark << vecMark;

	watershedExtinction(imIn, imMark, imResult, "a", se, false);
	
	// Area
	UINT16 areaTruth[] = {
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
	
	watershedExtinction(imIn, imMark, imResult, "a", se, true);
	
	// Area-rank
	UINT16 areaRankTruth[] = {
	  0,   1,    0,    0,    0,
	    0,    0,    0,    0,    0,
	  0,    0,    0,    0,    0,
	    0,    3,    0,    2,    0,
	  0,    3,    0,    0,    2,

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

class Test_Volumic_Extinction : public TestCase 
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

	watershedExtinction(imIn, imMark, imResult, "v", se, false);
	
	// Volume
	UINT8 volumeTruth[] = {
	  0,   6,   0,   0,   0,
	  0,   0,   0,   0,   0,
	  0,   0,   0,   0,   0,
	  0, 151,   0,  30,   0,
	  0, 151,   0,   0,  30,
	};
	imTruth << volumeTruth;
	
	TEST_ASSERT(imResult==imTruth);
	if (retVal!=RES_OK)
	{
	    imResult.printSelf (1);
	    imTruth.printSelf(1);
	}
	
	watershedExtinction(imIn, imMark, imResult, "v", se, true);
	
	// Volume-rank
	UINT8 volumeRankTruth[] = {
	    0,    3,    0,    0,    0,
	      0,    0,    0,    0,    0,
	    0,    0,    0,    0,    0,
	      0,    1,    0,    2,    0,
	    0,    1,    0,    0,    2,

	};
	imTruth << volumeRankTruth;
	
	TEST_ASSERT(imResult==imTruth);
	if (retVal!=RES_OK)
	{
	    imResult.printSelf (1);
	    imTruth.printSelf(1);
	}
    }
};

class Test_Dynamic_Extinction : public TestCase 
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

	watershedExtinction(imIn, imMark, imResult, "d", se, false);
	
	// Dynamic
	UINT8 dynamicTruth[] = {
	  0,   1,   0,   0,   0,
	  0,   0,   0,   0,   0,
	  0,   0,   0,   0,   0,
	  0,   4,   0,  20,   0,
	  0,   4,   0,   0,  20,
	};
	imTruth << dynamicTruth;
	
	TEST_ASSERT(imResult==imTruth);
	if (retVal!=RES_OK)
	{
	    imResult.printSelf (1);
	    imTruth.printSelf(1);
	}
	
	watershedExtinction(imIn, imMark, imResult, "d", se, true);
	
	// Dynamic-rank
	UINT8 dynamicRankTruth[] = {
	    0,    3,    0,    0,    0,
	      0,    0,    0,    0,    0,
	    0,    0,    0,    0,    0,
	      0,    2,    0,    1,    0,
	    0,    2,    0,    0,    1,

	};
	imTruth << dynamicRankTruth;
	
	TEST_ASSERT(imResult==imTruth);
	if (retVal!=RES_OK)
	{
	    imResult.printSelf (1);
	    imTruth.printSelf(1);
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
	trueEdges.push_back(Edge<UINT8>(3,1, 30));

	watershedExtinctionGraph (imIn, imMark, imResult, graph, "v", se) ;
	
	TEST_ASSERT(trueEdges==graph.getEdges());
	if (retVal!=RES_OK)
	    graph.printSelf();
	
	vector<Edge<UINT8> > trueEdges2;
	trueEdges2.push_back(Edge<UINT8>(3,1, 1));
	trueEdges2.push_back(Edge<UINT8>(1,2, 2));
	
	Graph<UINT8,UINT8> rankGraph = watershedExtinctionGraph (imIn, imMark, imResult, "v", se) ;
	
	TEST_ASSERT(trueEdges2==rankGraph.getEdges());
	if (retVal!=RES_OK)
	    rankGraph.printSelf();
    }
};


int main(int argc, char *argv[])
{
    TestSuite ts;

    ADD_TEST(ts, Test_Extinction_Flooding);
    ADD_TEST(ts, Test_Area_Extinction);
    ADD_TEST(ts, Test_Volumic_Extinction);
    ADD_TEST(ts, Test_Dynamic_Extinction);
    ADD_TEST(ts, Test_Watershed_Extinction_Graph);

//     Image<UINT8> imIn("http://cmm.ensmp.fr/~faessel/smil/images/mosaic.png");
//     Image<UINT8> imGrad(imIn);
//     Image<UINT16> imMark(imIn);
//     Image<UINT16> imBasins(imIn);
//     Image<UINT16> imOut(imIn);
//     
//     gradient(imIn, imGrad);
//     minimaLabeled(imGrad, imMark);
//     
//     watershedExtinction(imGrad, imMark, imOut,imBasins);
//     imBasins.showLabel();
//     
//     Gui::execLoop();
    
    return ts.run();

}


