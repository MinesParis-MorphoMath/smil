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
 * 
 * last modification by Amin Fehri
 * 
 */


#ifndef _DENDRO_MODULE_HPP
#define _DENDRO_MODULE_HPP

#include "Core/include/DCore.h"
#include "Morpho/include/DMorpho.h"

#include <unistd.h> // For usleep

#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>
#include <cmath>
#include "Morpho/include/private/DMorphoGraph.hpp"

namespace smil
{ 
  /**
   * DendroNode : node of a dendrogram
   */
  template <class WeightT = size_t, class MarkerLabelT = size_t>
  class DendroNode 
  {
  protected: 
    WeightT valuation;
    WeightT internalNodeValuationInitial;
    WeightT internalNodeValuationFinal;
    MarkerLabelT marker;
    MarkerLabelT label;
    bool isInternalNode;
    DendroNode* father;
    DendroNode* childLeft; 
    DendroNode* childRight;
    DendroNode* neighborLeft;
    DendroNode* neighborRight;
  public: 
    typedef DendroNode<WeightT, MarkerLabelT> DendroNodeType;
    //! Default constructor
    DendroNode():
    valuation(0),internalNodeValuationInitial(0),internalNodeValuationFinal(0),marker(0),label(0),isInternalNode(0),
    father(0),childLeft(0),childRight(0),neighborLeft(0),neighborRight(0)
    {
    }
    //! Copy constructor
    DendroNode(const DendroNode &dendroNodeToCopy){
      internalNodeValuationInitial = dendroNodeToCopy.internalNodeValuationInitial;
      internalNodeValuationFinal = dendroNodeToCopy.internalNodeValuationFinal;
      valuation = dendroNodeToCopy.valuation;
      marker = dendroNodeToCopy.marker;
      label=dendroNodeToCopy.label;
      isInternalNode = dendroNodeToCopy.isInternalNode;
      if (dendroNodeToCopy.father != NULL){
	father = new DendroNode(*(dendroNodeToCopy.father));
      }
      else{
	father = NULL;
      }
      if (dendroNodeToCopy.childLeft != NULL){
	childLeft = new DendroNode(*(dendroNodeToCopy.childLeft));
      }
      else{
	childLeft = NULL;
      }
      if (dendroNodeToCopy.childRight != NULL){
	childRight = new DendroNode(*(dendroNodeToCopy.childRight));
      }
      else{
	childRight = NULL;
      }
      if (dendroNodeToCopy.neighborLeft != NULL){
	neighborLeft = new DendroNode(*(dendroNodeToCopy.neighborLeft));
      }
      else{
	neighborLeft = NULL;
      }
      if (dendroNodeToCopy.neighborRight != NULL){
	neighborRight = new DendroNode(*(dendroNodeToCopy.neighborRight));
      }
      else{
	neighborRight = NULL;
      }
    }
    //! Assignment operator
    DendroNode& operator=(DendroNode const& dendroNodeToCopy){
      if(this != &dendroNodeToCopy){
	internalNodeValuationInitial = dendroNodeToCopy.internalNodeValuationInitial;
	internalNodeValuationFinal = dendroNodeToCopy.internalNodeValuationFinal;
	valuation = dendroNodeToCopy.valuation;
	marker = dendroNodeToCopy.marker;
	label=dendroNodeToCopy.label;
	isInternalNode = dendroNodeToCopy.isInternalNode;
	if (dendroNodeToCopy.father != NULL){
	  delete father;
	  father = new DendroNode(*(dendroNodeToCopy.father));
	}
	else{
	  father = NULL;
	}
	if (dendroNodeToCopy.childLeft != NULL){
	  delete childLeft;
	  childLeft = new DendroNode(*(dendroNodeToCopy.childLeft));
	}
	else{
	  childLeft = NULL;
	}
	if (dendroNodeToCopy.childRight != NULL){
	  delete childRight;
	  childRight = new DendroNode(*(dendroNodeToCopy.childRight));
	}
	else{
	  childRight = NULL;
	}
	if (dendroNodeToCopy.neighborLeft != NULL){
	  delete neighborLeft;
	  neighborLeft = new DendroNode(*(dendroNodeToCopy.neighborLeft));
	}
	else{
	  neighborLeft = NULL;
	}
	if (dendroNodeToCopy.neighborRight != NULL){
	  delete neighborRight;
	  neighborRight = new DendroNode(*(dendroNodeToCopy.neighborRight));
	}
	else{
	  neighborRight = NULL;
	}
      }
      return *this;
    }
    //! Destructor
    virtual ~DendroNode()
    {
    }
    //! Setters and getters
    WeightT getInternalNodeValuationInitial(){return internalNodeValuationInitial;};
    void setInternalNodeValuationInitial(WeightT nValuation){internalNodeValuationInitial = nValuation;};
    WeightT getInternalNodeValuationFinal(){return internalNodeValuationFinal;};
    void setInternalNodeValuationFinal(WeightT nValuation){internalNodeValuationFinal = nValuation;};
    WeightT getValuation(){return valuation;};
    void setValuation(WeightT nValuation){valuation = nValuation;};
    MarkerLabelT getMarker(){return marker;};
    void setMarker(WeightT nMarker){marker=nMarker;};
    MarkerLabelT getLabel(){return label;};
    void setLabel(WeightT nLabel){label=nLabel;};
    bool getIsInternalNode(){return isInternalNode;};
    void setIsInternalNode(bool nIsInternalNode){isInternalNode = nIsInternalNode;};
    DendroNode * getFather() {return father;};
    void setFather(DendroNode *nFather){ father = nFather;};
    DendroNode * getChildLeft() {return childLeft;};
    void setChildLeft(DendroNode *nChildLeft){ childLeft = nChildLeft;};
    DendroNode * getChildRight(){return childRight;};
    void setChildRight(DendroNode *nChildRight){childRight = nChildRight;};
    DendroNode * getNeighborLeft(){return neighborLeft;};
    void setNeighborLeft(DendroNode *nNeighborLeft){neighborLeft = nNeighborLeft;};
    DendroNode * getNeighborRight(){return neighborRight;};
    void setNeighborRight(DendroNode *nNeighborRight){neighborRight = nNeighborRight;};
    DendroNode * getAncestor(){
      DendroNode* refToReturn = this;
      if (refToReturn->getFather() != NULL){
	while (refToReturn != refToReturn->getFather() && refToReturn->getFather() != NULL){
	  refToReturn = refToReturn->getFather();
	}
      }
      return refToReturn;
    };
    DendroNode * getSelf(){
      return this;
    };
    //! Comparison functions
    static  bool isInferior (DendroNode* dendroNodeL,DendroNode* dendroNodeR){
      return dendroNodeL->getInternalNodeValuationInitial()<dendroNodeR->getInternalNodeValuationInitial();
    };
    static bool isSuperior (DendroNode* dendroNodeL,DendroNode* dendroNodeR){
      return dendroNodeL->getInternalNodeValuationInitial()>dendroNodeR->getInternalNodeValuationInitial();
    };
    //! Operators overriding
    bool operator<(const DendroNode &nDendroNode)
    {
      return internalNodeValuationInitial < nDendroNode.internalNodeValuationInitial;
    };
    bool operator>(const DendroNode &nDendroNode)
    {
      return internalNodeValuationInitial > nDendroNode.internalNodeValuationInitial;
    };
    bool operator==(const DendroNode &nDendroNode)
    {
      return internalNodeValuationInitial == nDendroNode.internalNodeValuationInitial;
    };
  }; // end DendroNode
  
  
  /**
   * Dendrogram
   */
  template <class WeightT = size_t, class MarkerLabelT = size_t, class NodeT = size_t>
  class Dendrogram 
  {
  public:
    typedef DendroNode<WeightT,MarkerLabelT> DendroNodeType;
    typedef std::map<NodeT, WeightT> NodeValuesType;
    typedef std::vector< Edge<NodeT,WeightT> > EdgeListType;
    typedef Graph<NodeT, WeightT> GraphType;
    //! Default constructor
    Dendrogram(){};
    //! Copy constructor
    Dendrogram (const Dendrogram & dendrogramToCopy){
      nbrNodes = dendrogramToCopy.nbrNodes; // N leafs and (N-1) internal nodes = (2N-1)
      dendroNodes = dendrogramToCopy.dendroNodes;
    };
//     bool operator==(const DendroNode &nDendroNode)
//     {
//       return nbrNodes == nDendroNode.internalNodeValuationInitial;
//       
//     };
  
    //! Constructor from a MST graph
    Dendrogram(GraphType& mst){
      mst.sortEdges(true);//sort Edges of the MST by increasing weights of edges
      // Extract informations from the MST
      size_t leavesNbr = mst.getNodeNbr();
      size_t internalNodesNbr = mst.getEdgeNbr();
      NodeValuesType &mstNodes = mst.getNodeValues();
      EdgeListType &mstEdges = mst.getEdges();
      
      // Set the number of required nodes and creates them in dendroNodes
      nbrNodes = leavesNbr+internalNodesNbr;
      for (size_t i = 0;i<nbrNodes;i++){
	DendroNodeType* nNode = new DendroNodeType;
	dendroNodes.push_back(nNode);
      }
    // Filling the leaves
    for (size_t i=0; i<leavesNbr; i++){
      DendroNodeType &curNode = *dendroNodes[i];//dendroNodes is filled with leavesNbr leaves and then with internalNodesNbr internal nodes
      curNode.setLabel(mstNodes.find(i)->first); // so Leaves have labels from 0 to leavesNbr-1
      curNode.setValuation(mstNodes.find(i)->second);
//       cout << "mstNodes.find(" << i << ")->second = " << mstNodes.find(i+1)->first << endl; 
    }

    // Filling the internal nodes
    for (size_t i=0;i<internalNodesNbr;i++){
      DendroNodeType &curNode = *dendroNodes[leavesNbr+i];//dendroFromMSTNodes is filled with leavesNbr leaves and then with internalNodesNbr internal nodes
      curNode.setIsInternalNode(1);
      curNode.setLabel(leavesNbr+i);
      curNode.setInternalNodeValuationInitial(mstEdges[i].weight);

      WeightT NeighborLeftLabel = min(mstEdges[i].source,mstEdges[i].target); //min(mstEdges[i].source,mstEdges[i].target);
      WeightT NeighborRightLabel = max(mstEdges[i].source,mstEdges[i].target); //max(mstEdges[i].source,mstEdges[i].target);
// //       cout << "mstEdges[i].source = " << mstEdges[i].source << " ; mstEdges[i].target = " << mstEdges[i].target << 
// //       " ; mstEdges[i].weight = " << mstEdges[i].weight <<endl;

      curNode.setNeighborLeft(dendroNodes[NeighborLeftLabel]);// dendroNodesLabels[NeighborLeftLabel] to modify
      curNode.setNeighborRight(dendroNodes[NeighborRightLabel]);// dendroNodesLabels[NeighborRightLabel]to modify

      curNode.setChildLeft(curNode.getNeighborLeft()->getAncestor());
      curNode.setChildRight(curNode.getNeighborRight()->getAncestor());
      
      curNode.getChildLeft()->setFather(&curNode);
      curNode.getChildRight()->setFather(&curNode);
    }
    //last node parameters
    DendroNodeType &lastNode = *dendroNodes[leavesNbr+internalNodesNbr-1];
    lastNode.setFather(&lastNode); //the last internal node is its own father 
    }; //end Dendrogram(mst)
    
    //! Constructor with a given number of nodes
    Dendrogram(size_t nNbrNodes)
    {	
      nbrNodes = nNbrNodes;
      for (size_t i = 0;i<nNbrNodes;i++){
	DendroNodeType* nNode = new DendroNodeType;
	dendroNodes.push_back(nNode);
      }
    }
    //! Destructor
    ~Dendrogram(){
      for (size_t i = 0 ;i<nbrNodes; i++){
	delete dendroNodes[i];
      }
    };
    //! Reorganizes dendroNodes... 
    void sortNodes(bool reverse=false){
      if (!reverse) // ... by growing valuationInitial
	std::stable_sort(dendroNodes.begin(),dendroNodes.end(),DendroNodeType::isInferior);
      else // ... by decreasing valuationInitial
	std::stable_sort(dendroNodes.begin(),dendroNodes.end(),DendroNodeType::isSuperior);
    };
    //! Reorganizes dendroNodes by decreasing valuationInitial
    void sortReverseNodes(){
      std::stable_sort(dendroNodes.begin(),dendroNodes.end(),DendroNodeType::isSuperior);
    };
    void addDendroNodes(DendroNodeType *dendroNode){
      dendroNodes.push_back(dendroNode);
    };
#ifndef SWIG
    //! Put the internalNodeValuationFinal in internalNodeValuationInitial and then set internalNodeValuationFinal = 0
    void putValuationsFinalInInitial(){
      for (size_t i = 0; i<nbrNodes ; i++){
	DendroNodeType &curNode = *dendroNodes[i];
	if (curNode.getIsInternalNode() == 1){
	  WeightT temp = curNode.getInternalNodeValuationFinal();
	  curNode.setInternalNodeValuationInitial(temp);
	  curNode.setInternalNodeValuationFinal(0);
	}
      }
    }; 
#endif
    //! Given an internal node index lambda of the dendrogram, remove corresponding edge in the associated MST 
    static void removeMSTEdgesDendrogram(Dendrogram& dendrogram,GraphType& associated_mst,WeightT lambda){
      std::vector<DendroNodeType*>&dendroNodes = dendrogram.getDendroNodes();
      size_t leavesNbr = associated_mst.getNodeNbr();
      size_t internalNodesNbr = associated_mst.getEdgeNbr();
      
      for (size_t i=0; i<internalNodesNbr+leavesNbr; i++){
	DendroNodeType &curNode = *dendroNodes[i];
	if (curNode.getInternalNodeValuationInitial()>lambda && curNode.getIsInternalNode()==true){
	  MarkerLabelT srcToRemove = curNode.getNeighborLeft()->getLabel();
	  MarkerLabelT targetToRemove = curNode.getNeighborRight()->getLabel();
	  associated_mst.removeEdge(srcToRemove,targetToRemove);
	  associated_mst.removeEdge(targetToRemove,srcToRemove);
      } 
    } 
    };
    
    //! Computes a new hierarchy from a given dendrogram hierarchy
    static void HierarchicalDendrogramConstruction(Dendrogram& dendrogram,std::string typeOfHierarchy,int nParam = 2){
      dendrogram.sortNodes(); // sort by decreasing values of internalNodeValuationInitial
      std::vector<DendroNodeType*>&dendroNodes = dendrogram.getDendroNodes();		
      
      if (typeOfHierarchy == "surfacic"){
	for (size_t i=0;i<dendroNodes.size();i++){// only one dendroNodes traversal, and only on internal nodes,
						    // because leaves already have surface as valuations
	  DendroNodeType &curNode = *dendroNodes[i];
	  if (curNode.getIsInternalNode() == 1){// we verify that it's an internal node
	    WeightT childLeftValuation = curNode.getChildLeft()->getValuation();
	    WeightT childRightValuation = curNode.getChildRight()->getValuation();
	    curNode.setValuation(childLeftValuation+childRightValuation);
	    curNode.setInternalNodeValuationFinal(min(childLeftValuation,childRightValuation));
	  }
	}
      dendrogram.putValuationsFinalInInitial();
      }
      else if (typeOfHierarchy == "stochasticSurfacic"){
	for (size_t i=0;i<dendroNodes.size();i++){// First dendroNodes traversal, to get the "surfacic" dendrogram
	  DendroNodeType &curNode = *dendroNodes[i];
	  if (curNode.getIsInternalNode() == 1){// we verify that it's an internal node
	    WeightT childLeftValuation = curNode.getChildLeft()->getValuation();
	    WeightT childRightValuation = curNode.getChildRight()->getValuation();
	    curNode.setValuation(childLeftValuation+childRightValuation);
// 	    cout << "childLeftValuation = " << childLeftValuation << " ; childRightValuation = " << childRightValuation << endl;
// 	    cout << "curNode.getInternalNodeValuationFinal = " << curNode.getInternalNodeValuationFinal() << endl;
	  }
	}
	
	DendroNodeType &curNodeTmp = *dendroNodes[1];
	WeightT totalSurface = curNodeTmp.getAncestor()->getValuation(); // get the total surface of the domain
	
	for (size_t i=0;i<dendroNodes.size();i++){// Second dendroNodes traversal, to get the stochasticSurfacic dendrogram
	  DendroNodeType &curNode = *dendroNodes[i];
	  if (curNode.getIsInternalNode() == 1){// we verify that it's an internal node
	    WeightT childLeftSurf = curNode.getChildLeft()->getValuation();
	    WeightT childRightSurf = curNode.getChildRight()->getValuation();
	    WeightT newVal = 1-pow(1-(childLeftSurf/totalSurface),nParam)
			    -pow(1-(childRightSurf/totalSurface),nParam)
			    +pow(1-((childRightSurf+childRightSurf)/totalSurface),nParam);
// 	    cout << i << endl;
// 	    cout << "newVal = " << newVal << endl;
// 	    cout << "childLeftVal + childRightVal = " << childLeftSurf+childRightSurf << endl;
// 	    cout << "surface = " << curNode.getValuation()<< endl;
// 	    cout << "totalSurface = " << totalSurface<< endl;
	    curNode.setInternalNodeValuationFinal(newVal);
	  }
	}
      dendrogram.putValuationsFinalInInitial();
      }
      else if (typeOfHierarchy == "volumic"){
	for (WeightT i=0;i<dendroNodes.size();i++){//First dendroNodes traversal, to get the "surfacic" dendrogram
	  DendroNodeType &curNode = *dendroNodes[i];
	  if (curNode.getIsInternalNode() == 1){// we verify that it's an internal node
	    WeightT childLeftValuation = curNode.getChildLeft()->getValuation();
	    WeightT childRightValuation = curNode.getChildRight()->getValuation();
	    curNode.setValuation(childLeftValuation+childRightValuation);
	  }
	}
	for (size_t i=0;i<dendroNodes.size();i++){//Second dendroNodes traversal, to get the "volumic" dendrogram 
	  DendroNodeType &curNode = *dendroNodes[i];
	  if (curNode.getIsInternalNode() == 1){// we verify that the node is not the ancestor
// 	    WeightT height = curNode.getFather()->getInternalNodeValuationInitial();
	    WeightT height = curNode.getInternalNodeValuationInitial();
	    WeightT surface = curNode.getValuation();
// 	    curNode.setValuation(surface);
	    curNode.setValuation(height*surface);
	  }
	}
	for (size_t i=0;i<dendroNodes.size();i++){//Third dendroNodes traversal, to get the final valuation of internal nodes
	  DendroNodeType &curNode = *dendroNodes[i];
	  if (curNode.getIsInternalNode() == 1){// we verify that it's an internal node
	    WeightT childLeftValuation = curNode.getChildLeft()->getValuation();
	    WeightT childRightValuation = curNode.getChildRight()->getValuation();
	    WeightT height = curNode.getInternalNodeValuationInitial();
	    curNode.setInternalNodeValuationFinal(height*min(childLeftValuation,childRightValuation));
	  }
	}
	dendrogram.putValuationsFinalInInitial();
      }
      else if (typeOfHierarchy == "stochasticVolumic"){
	WeightT totalSurface = 0;
	WeightT totalDepth = 0;
	
	for (WeightT i=0;i<dendroNodes.size();i++){//First dendroNodes traversal, to get the "surfacic" dendrogram
	  DendroNodeType &curNode = *dendroNodes[i];
	  if (curNode.getIsInternalNode() == 1){// we verify that it's an internal node
	    WeightT childLeftValuation = curNode.getChildLeft()->getValuation();
	    WeightT childRightValuation = curNode.getChildRight()->getValuation();
	    curNode.setValuation(childLeftValuation+childRightValuation);
	  }
	}
	
	DendroNodeType &curNodeTmp = *dendroNodes[1];
	totalSurface = curNodeTmp.getAncestor()->getValuation(); // get the total surface of the domain
	totalDepth = curNodeTmp.getAncestor()->getInternalNodeValuationInitial(); // get the total depth of the domain
	WeightT totalVolume = totalSurface*totalDepth;
	
	for (size_t i=0;i<dendroNodes.size();i++){//Second dendroNodes traversal, to get the "volumic" dendrogram 
	  DendroNodeType &curNode = *dendroNodes[i];
	  if (curNode.getIsInternalNode() == 1){// we verify that the node is not the ancestor
// 	    WeightT height = curNode.getFather()->getInternalNodeValuationInitial();
	    WeightT height = curNode.getInternalNodeValuationInitial();
	    WeightT surface = curNode.getValuation();
// 	    curNode.setValuation(surface);
	    curNode.setValuation(height*surface);
	  }
	}

	for (size_t i=0;i<dendroNodes.size();i++){// Third dendroNodes traversal, to get the stochasticVolumic dendrogram
	  DendroNodeType &curNode = *dendroNodes[i];
	  if (curNode.getIsInternalNode() == 1){// we verify that it's an internal node
	    WeightT childLeftVol = curNode.getChildLeft()->getValuation();
	    WeightT childRightVol = curNode.getChildRight()->getValuation();
	    
	    WeightT newVal = 1 - pow(1-(childLeftVol/totalVolume),nParam)-pow(1-(childRightVol/totalVolume),nParam)
			      +pow(1-(childLeftVol+childRightVol)/totalVolume,nParam);
	    curNode.setInternalNodeValuationFinal(newVal);
// 	    cout << i << endl;
// 	    cout << "newVal = " << newVal << endl;
// 	    cout << "childLeftVol = " << childLeftVol << endl;
// 	    cout << "childRightVol = " << childRightVol << endl;
// 	    cout << "totalVolume = " << totalVolume << endl;
	  }
	}
	dendrogram.putValuationsFinalInInitial();
      }
      else if (typeOfHierarchy == "marker"){
	for (size_t i=0;i<dendroNodes.size();i++){//First dendroNodes traversal, to get the "marker" dendrogram
	  DendroNodeType &curNode = *dendroNodes[i];
	  if (curNode.getIsInternalNode() == 1){// we verify that it's an internal node
	    WeightT markerLeft = curNode.getChildLeft()->getMarker();
	    WeightT markerRight = curNode.getChildRight()->getMarker();
	    curNode.setMarker(max(markerLeft,markerRight));    
	  }
	}
	for (size_t i=0;i<dendroNodes.size();i++){//Second dendroNodes traversal, to get the final valuation of internal nodes
	  DendroNodeType &curNode = *dendroNodes[i];
	  if (curNode.getIsInternalNode() == 1){
	    WeightT markerLeft = curNode.getChildLeft()->getMarker();
	    WeightT markerRight = curNode.getChildRight()->getMarker();
	    curNode.setInternalNodeValuationFinal(min(markerLeft,markerRight));
	  }
	}
	dendrogram.putValuationsFinalInInitial();
      }
      else {cout<< "void Dendrogram::HierarchicalDendrogramConstruction(Dendrogram& dendrogram,std::string typeOfHierarchy) \n"<<
	"Please choose one of the following hierarchies: surfacic, volumic, stochasticSurfacic, stochasticVolumic, marker" << endl;
      }
    };
    
    //! Setters and Getters
    std::vector<DendroNodeType*> &getDendroNodes(){return dendroNodes;};
    
//     DendroNodeType* &getDendroNodeLabel(std::vector<DendroNodeType*> dendroNodes, MarkerLabelT label){
//       DendroNodeType &curNode = *dendroNodes[0];
//       for (size_t i = 0 ; i<dendroNodes.size() ; i++){
// 	 &curNode = *dendroNodes[i];
// 	 if (curNode.getLabel() == label){
// 	   break; 
// 	}
//       }
//       return &curNode;
//     }
    size_t getNbrNodes(){return nbrNodes;}
    WeightT getNodeValue(size_t nodeIndex,string nameOfValueWanted){
      DendroNodeType &curNode = *dendroNodes[nodeIndex];
      if (nameOfValueWanted == "valuation"){
	return curNode.getValuation();
      }
      else if (nameOfValueWanted == "internalNodeValuationInitial"){
	return curNode.getInternalNodeValuationInitial();
      }
      else if (nameOfValueWanted == "internalNodeValuationFinal"){
	return curNode.getInternalNodeValuationFinal();
      }
      else if (nameOfValueWanted == "label"){
	return curNode.getLabel();
      }
      else if (nameOfValueWanted == "marker"){
	return curNode.getMarker();
      }
      else{
	cout << "int Dendrogram::getNodeValue(int nodeIndex,string nameOfValueWanted) -> nameOfValueWanted must be chosen in this list: valuation, internalNodeValuationInitial "<<
	 ", internalNodeValuationInitial, label, marker." << endl;
      }
    };
    
  protected:
    size_t nbrNodes;
    std::vector<DendroNodeType*> dendroNodes;
  }; // end Dendrogram
}

// EXEMPLE ACTION SUR PIXELS   
//     template <class T>
//     RES_T samplePixelFunction(const Image<T> &imIn, Image<T> &imOut)
//     {
//         ASSERT_ALLOCATED(&imIn)
//         ASSERT_SAME_SIZE(&imIn, &imOut)
//         
//         ImageFreezer freeze(imOut);
//         
//         typename Image<T>::lineType pixelsIn = imIn.getPixels();
//         typename Image<T>::lineType pixelsOut = imOut.getPixels();
//         
//         for (size_t i=0;i<imIn.getPixelCount();i++)
//           pixelsOut[i] = ImDtTypes<T>::max() - pixelsIn[i];
//         
//         return RES_OK;
//     }
        


#endif // _SAMPLE_MODULE_HPP 
