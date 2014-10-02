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
//     virtual FloodingBasin *clone() { return new FloodingBasin; };
//     UINT label;
// };
// 
// template <class T>
// class VolumeFloodingBasin : public FloodingBasin<T>
// {
//   public:
//     virtual void test() { cout << "Volume FloodingBasin" << endl; }
//     virtual VolumeFloodingBasin *clone() { return new VolumeFloodingBasin; }
// };

template <class T, class extValType> class Flooding;

template <class T, class extValType>
struct extinctionValuesComp
{
    extinctionValuesComp(Flooding<T,extValType> *_flooding)
    {
	flooding = _flooding;
    }
    inline bool operator() (const UINT &i, const UINT &j)
    {
	return (flooding->extinctionValues[i] > flooding->extinctionValues[j] );
    }
    Flooding<T,extValType> *flooding;
};

template <class T, class extValType=UINT>
class Flooding : public BaseObject
{
  public:
    Flooding()
      : BaseObject("Flooding"),
	basinNbr(0)
    {
    }
    virtual ~Flooding()
    {
	if (basinNbr!=0)
	  deleteBasins();
    }
    
    
    UINT *equivalents;
    extValType *extinctionValues;
    
    UINT getCurrentLevel() { return currentLevel; }
  protected:
    UINT labelNbr, basinNbr;
    HierarchicalQueue < T > hq;
    T currentLevel, currentPixVal;
    size_t currentOffset;
    
    virtual void initValue(const UINT &lbl) {}
    virtual void increment(const UINT &lbl) {}
    virtual void raise(const UINT &lbl) {}
    virtual UINT merge(UINT &lbl1, UINT &lbl2) {}
    virtual void finalize(const UINT &lbl) {}
    
    virtual void createBasins(const UINT nbr)
    {
	equivalents = new UINT[nbr];
	extinctionValues = new UINT[nbr];
	
	for (UINT i=0;i<nbr;i++)
	{
	    equivalents[i] = i;
	    extinctionValues[i] = 0;
	}
	
	basinNbr = nbr;
    }
    virtual void deleteBasins()
    {
	delete[] equivalents;
	delete[] extinctionValues;
	basinNbr = 0;
    }
    
    template <class labelT>
    void initHierarchicalQueue(const Image<T> &imIn, Image<labelT> &imLbl)
    {
	// Empty the priority queue
	hq.initialize (imIn);
	typename ImDtTypes < T >::lineType inPixels = imIn.getPixels ();
	typename ImDtTypes < labelT >::lineType lblPixels = imLbl.getPixels ();
	size_t s[3];

	imIn.getSize (s);
	currentOffset = 0;

	for (size_t k = 0; k < s[2]; k++)
	    for (size_t j = 0; j < s[1]; j++)
		for (size_t i = 0; i < s[0]; i++) {
		    if (*lblPixels != 0) {
			currentPixVal = *inPixels;
			hq.push (T (*inPixels), currentOffset);
			initValue(*lblPixels);
		    }

		    inPixels++;
		    lblPixels++;
		    currentOffset++;
		}
    }
    
    template < class labelT >
    RES_T processHierarchicalQueue (const Image < T > &imIn, 
				    Image < labelT > &imLbl,
				    const StrElt & se, Graph<labelT,extValType> *graph=NULL)
    {
	typename ImDtTypes < T >::lineType inPixels = imIn.getPixels ();
	typename ImDtTypes < labelT >::lineType lblPixels = imLbl.getPixels ();
	vector < int >dOffsets;

	vector < IntPoint >::const_iterator it_start = se.points.begin ();
	vector < IntPoint >::const_iterator it_end = se.points.end ();
	vector < IntPoint >::const_iterator it;
	vector < UINT > tmpOffsets;
	size_t s[3];

	imIn.getSize (s);

	currentLevel = 0;
	currentOffset = 0;

	// set an offset distance for each se point
	for (it = it_start; it != it_end; it++) 
	{
	    dOffsets.push_back (it->x + it->y * s[0] + it->z * s[0] * s[1]);
	}
	vector < int >::iterator it_off_start = dOffsets.begin ();
	vector < int >::iterator it_off;

	while (!hq.isEmpty ()) 
	{
	    currentOffset = hq.pop ();
	    currentPixVal = inPixels[currentOffset];

	    // Rising the elevation of the basins
	    if (currentPixVal > currentLevel) 
	    {
		currentLevel = currentPixVal;
		for (labelT i = 1; i < labelNbr + 1 ; ++i) 
			raise(i);
	    }

	    size_t x0, y0, z0;

	    imIn.getCoordsFromOffset (currentOffset, x0, y0, z0);
	    bool oddLine = se.odd && ((y0) % 2);
	    int x, y, z;
	    size_t nbOffset;
	    UINT l1 = lblPixels[currentOffset], l2;

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
			nbOffset = currentOffset + *it_off;
			if (oddLine)
			    nbOffset += (((y + 1) % 2) != 0);
			l2 = lblPixels[nbOffset];
			if (l2 > 0 && l2 != labelNbr + 1 ) 
			{
			    while (l2 != equivalents[l2]) 
			    {
				l2 = equivalents[l2];
			    }

			    if (l1 == 0 || l1 == labelNbr + 1 ) 
			    {
				l1 = l2; // current pixel takes the label of its first labelled ngb  found
				lblPixels[currentOffset] = l1;
				increment(l1);
			    }
			    else if (l1 != l2) 
			    {
				while (l1 != equivalents[l1]) 
				{
				    l1 = equivalents[l1];
				}
				if (l1 != l2)
				{
				    // merge basins
				    UINT eater = merge(l1, l2);
				    UINT eaten = (eater==l1) ? l2 : l1;
				    
				    if (graph)
				      graph->addEdge(eaten, eater, extinctionValues[eaten]);
				    
				    if (eater==l2)
				      l1 = l2;
				}
				      
			    }
			}
			else if (l2 == 0) 	// Add it to the tmp offsets queue
			{
			    tmpOffsets.push_back (nbOffset);
			    lblPixels[nbOffset] = labelNbr + 1 ;
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
	finalize(lblPixels[currentOffset]);

	return RES_OK;
    }
    
  public:
    template <class labelT>
    RES_T flood(const Image<T> &imIn, Image<labelT> &imMarkers, Image<labelT> &imBasinsOut, Graph<labelT,extValType> *graph, const StrElt & se)
    {
	ASSERT_ALLOCATED (&imIn, &imMarkers, &imBasinsOut);
	ASSERT_SAME_SIZE (&imIn, &imMarkers, &imBasinsOut);
	ImageFreezer freezer2 (imBasinsOut);

	copy (imMarkers, imBasinsOut);

	labelNbr = maxVal (imMarkers);
	
	if (basinNbr!=0)
	  deleteBasins();
	
	createBasins(labelNbr + 1);

	initHierarchicalQueue(imIn, imBasinsOut);
	processHierarchicalQueue(imIn, imBasinsOut, se, graph);
	
	return RES_OK;
    }
    template <class labelT>
    RES_T flood(const Image<T> &imIn, Image<labelT> &imMarkers, Image<labelT> &imBasinsOut, const StrElt & se)
    {
	Graph<labelT,extValType> *nullGraph = NULL;
	return flood(imIn, imMarkers, imBasinsOut, nullGraph, se);
    }
    
    template <class labelT, class outT>
    RES_T floodWithValues(const Image<T> &imIn, Image<labelT> &imMarkers, Image<outT> &imExtValOut, Image<labelT> &imBasinsOut, const StrElt & se)
    {
	ASSERT_ALLOCATED (&imExtValOut);
	ASSERT_SAME_SIZE (&imIn, &imExtValOut);
	
	ASSERT(flood(imIn, imMarkers, imBasinsOut, se)==RES_OK);

	ImageFreezer freezer (imExtValOut);
	
	typename ImDtTypes < outT >::lineType pixOut = imExtValOut.getPixels ();
	typename ImDtTypes < labelT >::lineType pixMarkers = imMarkers.getPixels ();

	fill(imExtValOut, outT(0));
	
	for (size_t i=0; i<imIn.getPixelCount (); i++, *pixMarkers++, pixOut++) 
	{
	    if(*pixMarkers != labelT(0))
	      *pixOut = extinctionValues[*pixMarkers] ;
	}

	return RES_OK;
    }
    template <class labelT, class outT>
    RES_T floodWithValues(const Image<T> &imIn, Image<labelT> &imMarkers, Image<outT> &imExtValOut, const StrElt & se)
    {
	Image<labelT> imBasinsOut(imMarkers);
	return floodWithValues(imIn, imMarkers, imExtValOut, se);
    }
    
    template <class labelT>
    RES_T floodWithRank(const Image<T> &imIn, Image<labelT> &imMarkers, Image<labelT> &imExtRankOut, Image<labelT> &imBasinsOut, const StrElt & se)
    {
	ASSERT_ALLOCATED (&imExtRankOut);
	ASSERT_SAME_SIZE (&imIn, &imExtRankOut);
	
	ASSERT(flood(imIn, imMarkers, imBasinsOut, se)==RES_OK);

	typename ImDtTypes < labelT >::lineType pixOut = imExtRankOut.getPixels ();
	typename ImDtTypes < labelT >::lineType pixMarkers = imMarkers.getPixels ();

	ImageFreezer freezer (imExtRankOut);
	
	// Sort by extinctionValues
	vector<UINT> rank(labelNbr);
	for (UINT i=0;i<labelNbr;i++)
	  rank[i] = i+1;
	
	extinctionValuesComp<T,extValType> comp(this);
	sort(rank.begin(), rank.end(), comp);
	for (UINT i=0;i<labelNbr;i++)
	  extinctionValues[rank[i]] = i+1;
	
	fill(imExtRankOut, labelT(0));
	
	for (size_t i=0; i<imIn.getPixelCount (); i++, *pixMarkers++, pixOut++) 
	{
	    if(*pixMarkers != labelT(0))
	      *pixOut = extinctionValues[*pixMarkers] ;
	}

	return RES_OK;
    }
    template <class labelT, class outT>
    RES_T floodWithRank(const Image<T> &imIn, Image<labelT> &imMarkers, Image<outT> &imExtRankOut, const StrElt & se)
    {
	Image<labelT> imBasinsOut(imMarkers);
	return floodWithValues(imIn, imMarkers, imExtRankOut, se);
    }
};


template <class T>
struct VolumeFlooding : public Flooding<T,UINT>
{
    UINT *areas, *volumes;
    T *floodLevels;
    
    virtual void createBasins(const UINT nbr)
    {
	areas = new UINT[nbr];
	volumes = new UINT[nbr];
	floodLevels = new T[nbr];
	
	for (UINT i=0;i<nbr;i++)
	{
	    areas[i] = 0;
	    volumes[i] = 0;
	    floodLevels[i] = 0;
	}
	
	Flooding<T>::createBasins(nbr);
    }
    
    virtual void deleteBasins()
    {
	delete[] areas;
	delete[] volumes;
	delete[] floodLevels;
	
	Flooding<T>::deleteBasins();
    }
    
    
    virtual void initValue(const UINT &lbl)
    {
	floodLevels[lbl] = this->currentPixVal;
	areas[lbl]++;
    }
    virtual void increment(const UINT &lbl)
    {
	areas[lbl]++;
    }
    virtual void raise(const UINT &lbl)
    {
	if (floodLevels[lbl] < this->currentLevel) 
	{
	    volumes[lbl] += areas[lbl] * (this->currentLevel - floodLevels[lbl]);
	    floodLevels[lbl] = this->currentLevel;
	}
    }
    virtual UINT merge(UINT &lbl1, UINT &lbl2)
    {
	UINT eater, eaten;
	
	if (volumes[lbl1] > volumes[lbl2] || (volumes[lbl1] == volumes[lbl2] && floodLevels[lbl1] < floodLevels[lbl2]))
	{
	    eater = lbl1;
	    eaten = lbl2;
	}
	else 
	{
	    eater = lbl2;
	    eaten = lbl1;
	}
	
	this->extinctionValues[eaten] = volumes[eaten];
	volumes[eater] += volumes[eaten];
	areas[eater] += areas[eaten];
	this->equivalents[eaten] = eater;
	
	return eater;
    }
    virtual void finalize(const UINT &lbl)
    {
	volumes[lbl] += areas[lbl];
	this->extinctionValues[lbl] += volumes[lbl];
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
//       ADD_TEST(ts, Test_Watershed_Extinction_Graph);
      ADD_TEST(ts, Test_Build);
      
      VolumeFlooding<UINT8> vf;
      
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
      Image_UINT8 imOut (imIn) ;
      Image_UINT8 imBasins (imIn) ;
      
      imIn << vecIn;
      imMark << vecMark;
      
      vf.floodWithRank(imIn, imMark, imOut, imBasins, hSE());
      Graph<UINT8,UINT> graph;
      vf.flood(imIn, imMark, imBasins, &graph, hSE());
      
      graph.printSelf();
      
//     0,    6,    0,    0,    0,
//        0,    0,    0,    0,    0,
//     0,    0,    0,    0,    0,
//        0,  179,    0,   30,    0,
//     0,  179,    0,    0,   30,
      imOut.printSelf(1);
      
      return ts.run();
      
}


