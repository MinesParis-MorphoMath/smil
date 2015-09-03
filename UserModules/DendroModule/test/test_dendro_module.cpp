/*
 * Smil
 * Copyright (c) 2011-2015 Matthieu Faessel
 *
 * This file is part of Smil.
 *
 * Smil is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * Smil is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Smil.  If not, see
 * <http://www.gnu.org/licenses/>.
 * 
 * last modification by Amin Fehri
 *
 */


#include <map>
#include <iostream>
#include "../../../Core/include/DBinary.h"
#include "DendroModule.hpp"

#include "Morpho/include/private/DMorphoGraph.hpp"
#include <include/private/DMorphoLabel.hpp>

using namespace smil;

class Test_DendroModule : public TestCase
{
  virtual void run()
  {	
      // Construction dendrogramme exemple
// 	Dendrogram<UINT16,UINT16,UINT16> dendroTest(11);
// 	DendroNode<UINT16> dendroNodeA,dendroNodeB,dendroNodeC,dendroNodeD,dendroNodeE,
// 			dendroNodeF,dendroNodeG,dendroNodeH,dendroNodeI,dendroNodeJ,dendroNodeK;
// 	
// 	// Nodes A, B, C, D, E, F are leafs
// 	// NodeA
// 	dendroNodeA.setValuation(3);
// 	dendroNodeA.setMarker(0); //already default value, just to make things clearer
// 	dendroNodeA.setFather(&dendroNodeG);
// 	// NodeB
// 	dendroNodeB.setValuation(1);
// 	dendroNodeB.setMarker(1);
// 	dendroNodeB.setFather(&dendroNodeG);
// 	// NodeC
// 	dendroNodeC.setValuation(4);
// 	dendroNodeC.setMarker(0);
// 	dendroNodeC.setFather(&dendroNodeH);
// 	// NodeD
// 	dendroNodeD.setValuation(2);
// 	dendroNodeD.setMarker(2);
// 	dendroNodeD.setFather(&dendroNodeH);
// 	// NodeE
// 	dendroNodeE.setValuation(5);
// 	dendroNodeE.setMarker(3);
// 	dendroNodeE.setFather(&dendroNodeI);
// 	// NodeF
// 	dendroNodeF.setValuation(2);
// 	dendroNodeF.setMarker(0);
// 	dendroNodeF.setFather(&dendroNodeI);
// 	
// 	// Nodes G, H, I, J, K are internal nodes
// 	// NodeG	
// 	dendroNodeG.setChildLeft(&dendroNodeA);
// 	dendroNodeG.setChildRight(&dendroNodeB);
// 	dendroNodeG.setNeighborLeft(&dendroNodeA);
// 	dendroNodeG.setNeighborRight(&dendroNodeB);
// 	dendroNodeG.setInternalNodeValuationInitial(1);
// 	dendroNodeG.setFather(&dendroNodeK);
// 	// NodeH	
// 	dendroNodeH.setChildLeft(&dendroNodeC);
// 	dendroNodeH.setChildRight(&dendroNodeD);
// 	dendroNodeH.setNeighborLeft(&dendroNodeC);
// 	dendroNodeH.setNeighborRight(&dendroNodeD);
// 	dendroNodeH.setInternalNodeValuationInitial(2);
// 	dendroNodeH.setFather(&dendroNodeJ);
// 	// NodeI	
// 	dendroNodeI.setChildLeft(&dendroNodeE);
// 	dendroNodeI.setChildRight(&dendroNodeF);
// 	dendroNodeI.setNeighborLeft(&dendroNodeE);
// 	dendroNodeI.setNeighborRight(&dendroNodeF);
// 	dendroNodeI.setInternalNodeValuationInitial(1);
// 	dendroNodeI.setFather(&dendroNodeJ);
// 	// NodeJ	
// 	dendroNodeJ.setChildLeft(&dendroNodeH);
// 	dendroNodeJ.setChildRight(&dendroNodeI);
// 	dendroNodeJ.setNeighborLeft(&dendroNodeD);
// 	dendroNodeJ.setNeighborRight(&dendroNodeE);
// 	dendroNodeJ.setInternalNodeValuationInitial(7);
// 	dendroNodeJ.setFather(&dendroNodeK);
// 	// NodeK	
// 	dendroNodeK.setChildLeft(&dendroNodeG);
// 	dendroNodeK.setChildRight(&dendroNodeJ);
// 	dendroNodeK.setNeighborLeft(&dendroNodeB);
// 	dendroNodeK.setNeighborRight(&dendroNodeC);
// 	dendroNodeK.setInternalNodeValuationInitial(8);
// 	dendroNodeK.setFather(&dendroNodeK);
// 
// 
// 	// remplissage de dendroTest avec les noeuds créés
// 	dendroTest.addDendroNodes(&dendroNodeA);
// 	dendroTest.addDendroNodes(&dendroNodeB);
// 	dendroTest.addDendroNodes(&dendroNodeC);
// 	dendroTest.addDendroNodes(&dendroNodeD);
// 	dendroTest.addDendroNodes(&dendroNodeE);
// 	dendroTest.addDendroNodes(&dendroNodeF);
// 	dendroTest.addDendroNodes(&dendroNodeG);
// 	dendroTest.addDendroNodes(&dendroNodeH);
// 	dendroTest.addDendroNodes(&dendroNodeI);
// 	dendroTest.addDendroNodes(&dendroNodeJ);
// 	dendroTest.addDendroNodes(&dendroNodeK);
	

// TEST sortNode   
//     cout << " \n *********TEST_SORT_NODES********" << endl;
//     dendroTest.sortNodes();
//     std::vector<DendroNode<UINT16>*> dendroTestNodes = dendroTest.getDendroNodes();
//     for (int i=0;i<dendroTestNodes.size();i++){
// 		cout << "dendroTestNodes["<<i
// 		<<"].getValuation= " 
// 		<< dendroTestNodes[i]->getValuation() << endl;
// 	}

// TEST Hierarchy    
//     cout << "***********TEST_HIERARCHY**********" << endl;
//     Dendrogram<UINT16,UINT16,UINT16>::HierarchicalDendrogramConstruction(dendroTest,"marker");
//     std::vector<DendroNode<UINT16>*> dendroTestNodesAfterHierarchy = dendroTest.getDendroNodes();
//     for (int i=0;i<dendroTestNodesAfterHierarchy.size();i++){
// 		cout << "dendroTestNodesAfterHierarchy["<<i
// 		<<"].getValuation = " 
// 		<< dendroTestNodesAfterHierarchy[i]->getInternalNodeValuationFinal() << endl;
// 	}
//     
    
	
// TEST MST from Graph

// STEP 1 : Creation of the graph...
// ... either by hand...
       smil::Graph<UINT16,UINT16> g1;
       g1.addNode(1,3);
       g1.addNode(2,1);
       g1.addNode(3,4);
       g1.addNode(4,2);
       g1.addNode(5,5);
       g1.addNode(6,2);
       
       g1.addEdge(1,2, 1);
       g1.addEdge(1,3, 9);
       g1.addEdge(3,4, 8);
       g1.addEdge(3,4, 2);
       g1.addEdge(3,5, 8);
       g1.addEdge(4,5, 7);
       g1.addEdge(2,5, 9);
       g1.addEdge(2,6, 10);
       g1.addEdge(5,6, 1);
       
// ... or by using the mosaicToGraph function
       
       Graph<UINT32,UINT32> g2;
       Image<UINT8>  imIn = smil::Image<UINT8>("/home/afehri/Dropbox/Images/imTestPerso/cameramanMosaic.png");     
       Image<UINT16> imMosaic(imIn);
       Image<UINT16> imNodeValues(imIn);
       Image<UINT16> imEdgeValues(imIn);
       
       Image<UINT8> imMosaic8(imIn);
       Image<UINT8> imMosaicEnd(imIn);
       
       
       label(imIn, imMosaic);
//        imMosaic.show("imMosaic");
//        copy(imMosaic,imMosaic8);
//        imMosaic8.show("imMosa8");
//        Gui::execLoop();
       copy(imIn, imEdgeValues);
       labelWithArea(imIn,imNodeValues);
       mosaicToGraph(imMosaic, imEdgeValues,imNodeValues, g2); //imEdgeValues, imNodeValues,
       cout << "STEP 1 complete : graph creation" << endl;

// STEP 2 : Computation of the associated Minimum Spanning Tree
//        Graph<UINT16,UINT16> MST1 = graphMST(g1);
       Graph<UINT32,UINT32> MST2 = graphMST(g2);
//        MST1.sortReverseEdges(); // we sort edges by growing valuation
//        MST2.sortReverseEdges();
       cout << "STEP 2 complete : MST computation" << endl;
//        
// STEP 3 : We compute dendrograms associated with these MST
// 	Dendrogram<UINT16,UINT16,UINT16> dendro1 = Dendrogram<UINT16,UINT16,UINT16>(MST1);
	Dendrogram<UINT32,UINT32,UINT32> dendro2 = Dendrogram<UINT32,UINT32,UINT32>(MST2);

//     cout << " \n *********TEST_SORT_NODES********" << endl;
/*    dendro1.sortNodes(true);
    std::vector<DendroNode<UINT16>*> dendro1Nodes = dendro1.getDendroNodes();
    for (int i=0;i<dendro1Nodes.size();i++){
		cout << "dendroTestNodes["<<i
		<<"].getValuation= " 
		<< dendro1Nodes[i]->getInternalNodeValuationInitial() << endl;
	}*/	
	
//     dendro2.sortNodes(true);
//     std::vector<DendroNode<UINT16>*> dendro2Nodes = dendro2.getDendroNodes();
//     for (int i=0;i<dendro2Nodes.size();i++){
// 		cout << "dendroTestNodes["<<i
// 		<<"].getInternalNodeValuationInitial= " 
// 		<< dendro2Nodes[i]->getInternalNodeValuationInitial() << endl;
// 	}
	
//       Dendrogram<UINT16,UINT16,UINT16>::HierarchicalDendrogramConstruction(dendro2,"surfacic");
//       dendro2.sortReverseNodes();
//       for (int i = 0;i<dendro2.getNbrNodes();i++){
// 	cout << dendro2.getNodeValue(i,"internalNodeValuationInitial") << endl; 
//       }
       cout << "STEP 3 complete : Dendrograms computation" << endl;
       
	
// STEP 4 : Modifications of the dendrogram 
// 	Dendrogram<UINT16,UINT16,UINT16>::HierarchicalDendrogramConstruction(dendro1,"stochasticSurfacic");
// 	Dendrogram<UINT32,UINT32,UINT32>::HierarchicalDendrogramConstruction(dendro2,"surfacic",2,imMosaic);
	cout << "STEP 4 complete : Modifications of the dendrogram " << endl;
	
// 	dendro2.sortNodes(true);//sort Nodes of dendro2 by decreasing internalNodeValuationInitial
	
// 	Dendrogram<UINT16,UINT16,UINT16>::removeMSTEdgesDendrogram(dendro2,MST2,10);
// 	graphToMosaic(imMosaic8, MST2, imMosaicEnd);
// 	Image<UINT8> imFinal(imIn);
// 	smil::copy(imMosaicEnd,imFinal);
// 	imMosaicEnd.show();
// 	imFinal.show();
// 	Gui::execLoop();
	
// STEP 5 : obtaining segmentations hierarchy
	dendro2.sortNodes(true);//sort Nodes of dendro2 by decreasing internalNodeValuationInitial
	float lambda = 100000000;
	string name_image = "/home/afehri/ownCloud/Images/imTestPerso/cameramanMosaic.png"; //"/home/afehri/Dev/Results/toolsHierarchy/volumic/images/tools";	
	std::vector<DendroNode<UINT32>*>& dendroNodes2 = dendro2.getDendroNodes();
	
	Dendrogram<UINT32,UINT32,UINT32>::removeMSTEdgesDendrogram(dendro2,MST2,lambda);	
	graphToMosaic(imIn, MST2, imMosaicEnd);
	Image<UINT8> imFinal(imIn);
	smil::copy(imMosaicEnd,imFinal);
	imMosaicEnd.show("imMosaicEnd");
	imFinal.show("imFinal");
	imIn.show("imIn");
	Gui::execLoop();


// 	Dendrogram<UINT16,UINT16,UINT16>::removeMSTEdgesDendrogram(dendro2,MST2,10);
// 	graphToMosaic(imMosaic8, MST2, imMosaicEnd);
// 	Image<UINT8> imFinal(imIn);
// 	smil::copy(imMosaicEnd,imFinal);
// 	imMosaicEnd.show();
// 	imFinal.show();
// 	Gui::execLoop();
	
// 	for (int i=0;i<dendroNodes2.size();i++){
// 		DendroNode<UINT16> &curNode = *dendroNodes2[i];
// 		if (lambda!=curNode.getInternalNodeValuationInitial()){
// 			lambda = curNode.getInternalNodeValuationInitial();
// 			if (lambda!=0){// we verify that it's an internal node
// 				Dendrogram<UINT16,UINT16,UINT16>::removeMSTEdgesDendrogram(dendro2,MST2,lambda);
// 				graphToMosaic(imMosaic, MST2, imMosaicEnd);
// 				stringstream ss;
// 				int nbr_image = (dendroNodes2.size()-1)/2-i;
// 				cout << nbr_image << endl;
// 				if (nbr_image<10){
// 					ss << name_image <<"00"<< nbr_image << ".png";
// 					string name=ss.str();
// 					const char *cname = name.c_str();
// 					cout << name << endl;
// 					imMosaicEnd.save(cname);
// 				}
// 				else if (nbr_image<100){
// 					ss << name_image <<"0"<< nbr_image << ".png";
// 					string name=ss.str();
// 					const char *cname = name.c_str();
// 					cout << name << endl;
// 					imMosaicEnd.save(cname);
// 				}
// 				else if (nbr_image<1000){
// 					ss << name_image << nbr_image << ".png";
// 					string name=ss.str();
// 					const char *cname = name.c_str();
// 					cout << name << endl;
// 					imMosaicEnd.save(cname);
// 				}
// 
// 			}
// 		}
// 	}
// 	imMosaicEnd.show();
// 	Gui::execLoop();
// 	int numberone = 1;
// 	int a = 10;
// 	string name_image = "blob";
// 	stringstream ss;
// 	ss << name_image << 1;
// 	string name=ss.str();
// 	string name = "choop" + std::to_string(numberone);
// 	cout << name << endl;
// 	Dendrogram::removeMSTEdgesDendrogram(dendro2,MST2,lambda);
// 	graphToMosaic(imMosaic, MST2, imMosaicEnd);
// 	imIn.show();
// 	imMosaicEnd.show();
// 	Gui::execLoop();
// 	vector<short unsigned int> imNewSegValues = valueList(imMosaicEnd);
// 	for (int i=0;i<imNewSegValues.size();i++){
// 		short unsigned int label = imNewSegValues[i];
// 		
// 	}

	
// 	std::vector<DendroNode*>&dendroNodes1 = dendro1.getDendroNodes();
// 	for (int i=0;i<dendroNodes1.size();i++){
// 		DendroNode &curNode = *dendroNodes1[i];
// 		cout << curNode.getInternalNodeValuationInitial() << endl;
// 	}
	
	
	
	
// 	dendro2.sortReverseNodes();
// 	for (int lambda = 8000; lambda>-1 ; lambda = lambda-250){
// 			Dendrogram::removeMSTEdgesDendrogram(dendro2,MST2,lambda);
// 			graphToMosaic(imMosaic, MST2, imMosaicEnd); 
// 			imMosaicEnd.show();
// 			Gui::execLoop();
// 	}
	
// 	int lambda = 10;
// 	Dendrogram::removeMSTEdgesDendrogram(dendro2,MST2,lambda);
// 	graphToMosaic(imMosaic, MST2, imMosaicEnd); 
	
	//entry : imMosaic,imSeg
	//out : imOut with imMosaic values averaged on each imSeg region
	
// 	Image<UINT16> imThrTemp(imIn);
// 	Image<UINT16> imMosaicAveraged(imIn);
// 	Image<UINT16> imTruncated(imIn);
// 	Image<UINT16> imOnes(imIn);
// 	short unsigned int one = 1;
// 	fill(imOnes,one);
// 	Image<UINT16> imZeros(imIn);
	
	
// 	for (int i = 1; i<maxVal(imIn)+1;i++){
// 		Image<UINT16> imI(imIn);
// 		short unsigned int ip = i;
// 		fill(imI,ip); 
// 		compare(imMosaicEnd,"==",imI,imOnes,imZeros,imThrTemp);
// 		
// 		Image<UINT16> imTruncated(imIn);
// 		mul(imInp,imThrTemp,imTruncated);
// 		
// 		
// 		int nb_pix = vol(imThrTemp);
// 		int sum_pix = vol(imTruncated);
// 		
// // 		if(nb_pix!=0){
// 			short unsigned int mean_value = sum_pix/nb_pix;
// 			Image<UINT16> imToAddToResult(imIn);
// 			fill(imToAddToResult,mean_value);
// 			mul(imToAddToResult,imThrTemp,imToAddToResult);
// 			add(imMosaicAveraged,imToAddToResult,imMosaicAveraged);
// // 		}
// 		
// 	}
	
// 	for i in range(1,maxVal(imMosa)): # parcours des labels
// 		compare(imMosa,"==",i,1,0,imLab)
// 		mul(imLab,imIn,imTruncated) # imTruncated = imThr*imIn1	
// 		sum_pix = int(vol(imTruncated))
// 		nb_pix = int(vol(imLab))
// 		if (nb_pix !=0):
// 			mean = sum_pix/nb_pix
// 			mul(imLab,mean,imLab)
// 			add(imOut,imLab,imOut) 
// 	
// 	
// 	lambda = 100;
// 	Dendrogram::removeMSTEdgesDendrogram(dendro2,MST2,lambda);
// 	graphToMosaic(imMosaic, MST2, imMosaicEnd);
// 	imMosaicEnd.show();
// 	Gui::execLoop();
	
	
// 	imIn.show();
// 	imMosaic.show();
// 	imNodeValues.show();
// 	imMosaicEnd.show();
// 	Gui::execLoop();


// TEST 	
// 	map<long unsigned int, long unsigned int > &G1Nodes = g1.getNodeValues();
// 	map<long unsigned int, long unsigned int > &G2Nodes = g2.getNodeValues();
// 	map<long unsigned int, long unsigned int > &MST2Nodes = MST2.getNodeValues();
// 	cout << " g2Nodes nbr = " << g2.getNodeNbr() << endl;
// 	cout << "MST2Nodes size = " << MST2Nodes.size()<< endl;
// 	cout << "MST2 Node number = " << MST2.getNodeNbr() << endl;
//        cout << "MST2Nodes 0 = " << MST2Nodes.find(1)->second << endl;
//        cout << "G2Nodes size = " << G2Nodes.size()<< endl; 
       
//        typedef map<long unsigned int, long unsigned int > MapIterator;
//        MapIterator my_map;
//        for( MapIterator::iterator it = MST2Nodes.begin(); it != MST2Nodes.end(); ++it ) //traversal of the MST edges by growing values of edges
// 	    {
// 	      int key = it->first;
// 	      int value = it->second;
// 	      cout << "key = " << key << " ; value = " << value << endl;
// 	    }
       
//        MST2.sortReverseEdges();
//        MST2.printSelf();
      
       
//        measAreas<UINT16>(imLabel,imValues);
//       mst.printSelf();       


       
//        Graph<> mst = g.computeMST();
//        cout << "mst.getEdgeNbr() = " << mst.getEdgeNbr() << endl; 
//        cout << "mst.getNodeNbr() = " << mst.getNodeNbr() << endl;
//        Dendrogram dendroFromMST = Dendrogram::initFromMST(mst);
//        Dendrogram::HierarchicalDendrogramConstruction(dendroFromMST,"volumic");
//        Graph<> graphAfterHierarchy = Dendrogram::dendroToGraph(dendroFromMST);


      
// 	std::vector<DendroNode*>&dendro1Nodes = dendro1.getDendroNodes();
// 	std::vector<DendroNode*>&dendro2Nodes = dendro2.getDendroNodes();
// 	int leavesNbr1 = MST2.getNodeNbr();
// 	int NodesNbr1 = MST2.getEdgeNbr();
// 	for (int i=leavesNbr1 ; i<leavesNbr1+NodesNbr1; i++){
// 	  DendroNode &curNode = *dendro2Nodes[i];
// 	  cout << "******************NODE " << i << "****************** \n" 
// 	  << "own label = " << curNode.getLabel() << "\n" 
// 	  << "childLeft label = " << curNode.getChildLeft()->getLabel() << "\n"
// 	  << "childRight label = " << curNode.getChildRight()->getLabel() << "\n"
// 	  << "neighborLeft label = " << curNode.getNeighborLeft()->getLabel() << "\n"
// 	  << "neighborRight label = " << curNode.getNeighborRight()->getLabel() << "\n"
// 	  << "father label = " << curNode.getFather()->getLabel() << endl; 
// 	}
// 	
// 	vector< Edge<long unsigned int>,allocator<Edge<long unsigned int> > > MST1edges = MST1.getEdges();
// 	for (int i = 0; i<MST1edges.size() ; i++){
// 	  cout << "******************EDGE " << i << "****************** \n"
// 	  << " MST1edges["<<i<<"].target = " << MST1edges[i].target << "\n"
// 	  << " MST1edges["<<i<<"].source = " << MST1edges[i].source << "\n" 
// 	  << "min = " << min(MST1edges[i].source,MST1edges[i].target) << "\n"
// 	  << "max = " << max(MST1edges[i].source,MST1edges[i].target) << endl; 
// 	}       
       
       
       
       
//        int nNodeNbr = mst.getNodeNbr();
//        int leavesNbr = mst.getEdgeNbr();
//        cout << " nNodeNbr = " << nNodeNbr << endl;
//        cout << " leavesNbr = " << leavesNbr << endl;
//        Dendrogram dendroFromMST(nNodeNbr+leavesNbr);
//        vector< Edge<long unsigned int>,allocator<Edge<long unsigned int> > > mstEdges = mst.getEdges();
//        std::vector<DendroNode*>&dendroFromMSTNodes = dendroFromMST.getDendroNodes();
//        cout << dendroFromMST.getDendroNodes().size() << endl;
//        DendroNode &curNode = *dendroFromMSTNodes[5];//leavesNbr+i
// 
//        
//        map<long unsigned int,vector<long unsigned int> >mstNodeEdges = mst.getNodeEdges();
//        typedef map<long unsigned int, vector<long unsigned int> > MapIterator;
//        int counter=0;
//        MapIterator my_map;
//        for( MapIterator::iterator it = mstNodeEdges.begin(); it != mstNodeEdges.end(); ++it ) //traversal of the MST edges by growing values of edges
// 	    {
// 	      int key = it->first;
// 	      vector<long unsigned int> value = it->second;
// 	      cout << key << endl;
// 	      DendroNode dendroNodeFromEdge;
// 	      dendroFromMST.addDendroNodes(&dendroNodeFromEdge);
// 	      counter++;
// 	    }
       
       
//        mst.printSelf();
//        
//        
//        Image<UINT16> imMosaic;
//        Image<UINT8> imValues;
//        Image<UINT16> imMosaicEnd;
//        // blabla...
//        
//        mosaicToGraph(imMosaic, imValues, g);
//        // blbal sur graph
//        
//        graphToMosaic(imMosaic, g, imMosaicEnd);
       
      //~ UINT8 vec1[16] = {
        //~ 1,   2,   3,   4,
        //~ 5,   6,   7,   8,
        //~ 9,  10,  11,  12,
        //~ 13,  14,  15,  16,
      //~ };
      //~ 
      //~ Image_UINT8 im1(4,4), im2(im1);
      //~ im1 << vec1;
      //~ 
      //~ UINT8 vecTruth[16] = {
        //~ 254, 253, 252, 251,
        //~ 250, 249, 248, 247,
        //~ 246, 245, 244, 243,
        //~ 242, 241, 240, 239,
      //~ };
      //~ 
      //~ Image_UINT8 imTruth(im1);
      //~ imTruth << vecTruth;
      //~ 
      //~ samplePixelFunction(im1, im2);
      //~ 
      TEST_ASSERT(1)
      
      if (retVal!=RES_OK)
      {
          //~ im2.printSelf(1);
          //~ imTruth.printSelf(1);
          cout << "Test DendroModule OK." << endl;
      }
      
  }
};

int main(int argc, char *argv[])
{
      TestSuite ts;

      ADD_TEST(ts, Test_DendroModule);
      
      return ts.run();
}

