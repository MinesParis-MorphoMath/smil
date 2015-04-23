/*
 * Copyright (c) 2011-2015, Matthieu FAESSEL and ARMINES
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



class Test_Extinction_Flooding : public TestCase 
{
    virtual void run () 
    {
        UINT8 vecIn[] = {
            2,    2,    2,    2,    2,
              3,    2,    5,    9,    5,
            3,    3,    9,    0,    0,
              1,    1,    9,    0,    0,
            9,    1,    9,    0,    0,
              5,    4,    5,    9,    5,
            2,    2,    1,    3,    0,
              1,    1,    9,    0,    0,
        };
        UINT8 vecMark[] = {
            0,    1,    0,    0,    0,
              0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,
              0,    2,    0,    3,    0,
            0,    0,    0,    0,    3,
              0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,
              0,    4,    0,    5,    0,
        };
        StrElt se = hSE();

        Image_UINT8 imIn (5,8) ;
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
       2,    2,    3,    3,    3,
    4,    4,    4,    5,    5,
       4,    4,    5,    5,    5,
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
    0,   40,    0,    0,    0,
       0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,
       0,    3,    0,    8,    0,
    0,    0,    0,    0,    8,
       0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,
       0,    9,    0,    4,    0,
        };
        imTruth << areaTruth;
        
        TEST_ASSERT(imResult==imTruth);
        if (retVal!=RES_OK)
        {
            imResult.printSelf (1, true);
            imTruth.printSelf(1, true);
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
          0,    25,     0,     0,     0,
          0,     0,     0,     0,     0,
          0,     0,     0,     0,     0,
          0,     4,     0,     6,     0,
          0,     4,     0,     0,     6,

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
        StrElt se = hSE();

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
          0, 179,   0,  30,   0,
          0, 179,   0,   0,  30,
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
          0,   4,   0,   9,   0,
          0,   4,   0,   0,   9,
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
            9,    1,    9,    0,    0,
              5,    4,    5,    9,    5,
            2,    2,    1,    3,    0,
              1,    1,    9,    0,    0,
        };
        UINT8 vecMark[] = {
            0,    1,    0,    0,    0,
              0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,
              0,    2,    0,    3,    0,
            0,    0,    0,    0,    3,
              0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,
              0,    4,    0,    5,    0,
        };

        StrElt se = hSE();

        Image_UINT8 imIn (5,8) ;
        Image_UINT8 imMark (imIn) ;
        Image_UINT8 imTruth (imIn) ;
        Image_UINT8 imResult (imIn) ;

        typedef Graph<UINT8,UINT8> GraphT;
        typedef Graph<UINT8,UINT8>::EdgeType EdgeT;

        GraphT graph;
        
        imIn << vecIn;
        imMark << vecMark;

        vector< EdgeT > trueEdges;
        trueEdges.push_back(EdgeT(4,5, 4));
        trueEdges.push_back(EdgeT(1,2, 3));
        trueEdges.push_back(EdgeT(2,4, 9));
        trueEdges.push_back(EdgeT(5,3, 8));

        watershedExtinctionGraph (imIn, imMark, imResult, graph, "a", se) ;
        
//         imResult.printSelf(1);
        TEST_ASSERT(trueEdges==graph.getEdges());
        if (retVal!=RES_OK)
            graph.printSelf();
        
        vector<EdgeT > trueEdges2;
        trueEdges2.push_back(EdgeT(2,4, 1));
        trueEdges2.push_back(EdgeT(5,3, 2));
        trueEdges2.push_back(EdgeT(4,5, 3));
        trueEdges2.push_back(EdgeT(1,2, 4));
        
        GraphT rankGraph = watershedExtinctionGraph (imIn, imMark, imResult, "a", se) ;
        
        TEST_ASSERT(trueEdges2==rankGraph.getEdges());
        if (retVal!=RES_OK)
            rankGraph.printSelf();
    }
};

class Test_Watershed_Extinction_Compare : public TestCase 
{
    virtual void run () 
    {
        typedef UINT8 T1;
        typedef UINT16 T2;
        Image<T1> imGrad (5,8) ;
        Image<T2> imMark (imGrad) ;
        Image<T2> imBasins (imGrad) ;
        Image<T2> imOut (imGrad) ;
        Image<T2> imOut2 (imGrad) ;

        T1 vecGrad[] = {
            2,    2,    2,    2,    2,
              3,    2,    5,    9,    5,
            3,    3,    9,    0,    0,
              1,    1,    9,    0,    0,
            9,    1,    9,    0,    0,
              5,    4,    5,    9,    5,
            2,    2,    1,    3,    0,
              1,    1,    9,    0,    0,
        };
        T2 vecMark[] = {
            0,    1,    0,    0,    0,
              0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,
              0,    2,    0,    3,    0,
            0,    0,    0,    0,    3,
              0,    0,    0,    0,    0,
            0,    0,    0,    0,    0,
              0,    4,    0,    5,    0,
        };

        imGrad << vecGrad;
        imMark << vecMark;
        
//         Image_UINT8 imIn("http://cmm.ensmp.fr/~faessel/smil/images/lena.png");
//         gradient(imIn, imGrad);
//         hMinimaLabeled(imGrad, (UINT8)5, imMark);
                
        StrElt se = hSE();
        int nRegions = 30;
        
        Graph<UINT16,UINT16> g2 = watershedExtinctionGraph(imGrad, imMark, imBasins, "v", se);
        g2.removeLowEdges(nRegions);
        graphToMosaic(imBasins, g2, imOut);
      
        watershedExtinction(imGrad, imMark, imOut2, imBasins, "v", se);
        compare(imOut2, ">", (UINT16)nRegions, (UINT16)0, imMark, imMark);
        basins(imGrad, imMark, imOut2, se);
        
        label(imOut, imOut);
        label(imOut2, imOut2);
        
        TEST_ASSERT(imOut==imOut2);
        
//         diff(imOut, imOut2, imBasins);
//         
//         imBasins.showLabel("Basins");
//         imOut.showLabel("imOut");
//         imOut2.showLabel("imOut2");
//         
//         Gui::execLoop();
    }
};


int main()
{
    TestSuite ts;

    ADD_TEST(ts, Test_Extinction_Flooding);
    ADD_TEST(ts, Test_Area_Extinction);
    ADD_TEST(ts, Test_Volumic_Extinction);
    ADD_TEST(ts, Test_Dynamic_Extinction);
    ADD_TEST(ts, Test_Watershed_Extinction_Graph);
    ADD_TEST(ts, Test_Watershed_Extinction_Compare);

    
    return ts.run();

}


