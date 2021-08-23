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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS AND CONTRIBUTORS BE
 * LIABLE FOR ANY
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

#ifndef _DENDROGRAM_HPP
#define _DENDROGRAM_HPP

#include "DendroNode.hpp"

using namespace std;

namespace smil
{
  /**
   * Dendrogram
   */
  template <class MarkerLabelT = size_t, class NodeT = size_t,
            class ValGraphT = size_t>
  class Dendrogram
  {
  protected:
    size_t                                  nbrNodes;
    size_t                                  nbrMarkers;
    std::vector<DendroNode<MarkerLabelT> *> dendroNodes;

  public:
    typedef Dendrogram<MarkerLabelT, NodeT, ValGraphT> DendrogramType;

    typedef DendroNode<MarkerLabelT>            DendroNodeType;
    typedef std::map<NodeT, ValGraphT>          NodeValuesType;
    typedef std::vector<Edge<NodeT, ValGraphT>> EdgeListType;

    //! Default constructor
    Dendrogram() : nbrNodes(0), nbrMarkers(0), dendroNodes(0){};
    //! Copy constructor
    Dendrogram(const Dendrogram &dendrogramToCopy)
    {
      nbrNodes = dendrogramToCopy
                     .nbrNodes; // N leafs and (N-1) internal nodes = (2N-1)
      nbrMarkers = dendrogramToCopy.nbrMarkers;
      std::vector<DendroNodeType *> dendroNodesToCopy =
          dendrogramToCopy.dendroNodes;
      for (size_t i = 0; i < dendrogramToCopy.dendroNodes.size(); i++) {
        dendroNodes.push_back(new DendroNodeType);
      }
      for (size_t i = 0; i < nbrNodes; i++) {
        DendroNodeType &curNode = *dendroNodesToCopy[i];
        DendroNodeType &newNode = *dendroNodes[i];

        newNode.setLabel(curNode.getLabel());
        newNode.setMarker(curNode.getMarker());
        newNode.setNbMarkersUnder(curNode.getNbMarkersUnder());
        newNode.setMarkersCount(curNode.getMarkersCount());
        newNode.setValuation(curNode.getValuation());
        newNode.setLeafValuation(curNode.getLeafValuation());
        newNode.setInternalNodeValuationInitial(
            curNode.getInternalNodeValuationInitial());
        newNode.setInternalNodeValuationFinal(
            curNode.getInternalNodeValuationFinal());
        newNode.setIsInternalNode(curNode.getIsInternalNode());
        newNode.setLookupProgeny(curNode.getLookupProgeny());
        newNode.setMoments(curNode.getMoments());
        newNode.setContoursCount(curNode.getContoursCount());
        newNode.setContoursSize(curNode.getContoursSize());
      }
      for (size_t i = 0; i < nbrNodes; i++) {
        DendroNodeType &newNode         = *dendroNodes[i];
        MarkerLabelT    researchedLabel = newNode.getLabel();

        DendroNodeType &correspondingNode =
            *dendrogramToCopy.researchLabel(researchedLabel);
        MarkerLabelT fatherLabel = correspondingNode.getFather()->getLabel();
        newNode.setFather(this->researchLabel(fatherLabel));

        if (correspondingNode.getIsInternalNode() == true) {
          MarkerLabelT childLeftLabel =
              correspondingNode.getChildLeft()->getLabel();
          MarkerLabelT childRightLabel =
              correspondingNode.getChildRight()->getLabel();
          MarkerLabelT neighborLeftLabel =
              correspondingNode.getNeighborLeft()->getLabel();
          MarkerLabelT neighborRightLabel =
              correspondingNode.getNeighborRight()->getLabel();
          newNode.setChildLeft(this->researchLabel(childLeftLabel));
          newNode.setChildRight(this->researchLabel(childRightLabel));
          newNode.setNeighborLeft(this->researchLabel(neighborLeftLabel));
          newNode.setNeighborRight(this->researchLabel(neighborRightLabel));
        }
      }
    };

    /**
     * Dendrogram constructor from a MST graph
     *
     * @param[in] mst Input mst (graph)
     */
    Dendrogram(Graph<NodeT, ValGraphT> &mst)
        : nbrNodes(0), nbrMarkers(0), dendroNodes(0)
    {
      mst.sortEdges(
          true); // sort Edges of the MST by increasing weights of edges
      // Extract informations from the MST
      size_t          leavesNbr        = mst.getNodeNbr();
      size_t          internalNodesNbr = mst.getEdgeNbr();
      NodeValuesType &mstNodes         = mst.getNodeValues();
      EdgeListType &  mstEdges         = mst.getEdges();

      // Set the number of required nodes and creates them in dendroNodes
      nbrNodes = leavesNbr + internalNodesNbr;
      //       nbrNodes = 2*leavesNbr - 1 ;
      for (size_t i = 0; i < nbrNodes; i++) {
        DendroNodeType *nNode = new DendroNodeType;
        dendroNodes.push_back(nNode);
      }
      // Filling the leaves
      for (size_t i = 0; i < leavesNbr; i++) {
        // dendroNodes is filled with leavesNbr
        // leaves and then with internalNodesNbr
        // internal nodes
        DendroNodeType &curNode = *dendroNodes[i];
        // so Leaves have labels from 0 to leavesNbr-1
        curNode.setLabel(mstNodes.find(i)->first);
        curNode.setLeafValuation(mstNodes.find(i)->second);
        curNode.setValuation(mstNodes.find(i)->second);
        // initialize the lookupProgeny for each leaf with 1 for the label of
        // the leaf, and 0 elsewhere
        std::vector<MarkerLabelT> nLookupProgeny(leavesNbr, 0);
        nLookupProgeny.at(mstNodes.find(i)->first) = 1;
        curNode.setLookupProgeny(nLookupProgeny);
        // initialize the contoursCount
        std::vector<MarkerLabelT> initContoursCount(leavesNbr, 0);
        curNode.setContoursCount(initContoursCount);
      }

      // Filling the internal nodes
      for (size_t i = 0; i < internalNodesNbr; i++) {
        // dendroFromMSTNodes is filled with leavesNbr leaves and then
        // with internalNodesNbr internal nodes
        DendroNodeType &curNode = *dendroNodes[leavesNbr + i];
        curNode.setIsInternalNode(1);
        curNode.setLabel(leavesNbr + i);
        curNode.setInternalNodeValuationInitial(mstEdges[i].weight);
        MarkerLabelT NeighborLeftLabel =
            min(mstEdges[i].source, mstEdges[i].target);
        MarkerLabelT NeighborRightLabel =
            max(mstEdges[i].source, mstEdges[i].target);

        // dendroNodesLabels[NeighborLeftLabel] to modify
        curNode.setNeighborLeft(dendroNodes[NeighborLeftLabel]);
        // dendroNodesLabels[NeighborRightLabel] to modify
        curNode.setNeighborRight(dendroNodes[NeighborRightLabel]);

        curNode.setChildLeft(curNode.getNeighborLeft()->getAncestor());
        curNode.setChildRight(curNode.getNeighborRight()->getAncestor());

        curNode.getChildLeft()->setFather(&curNode);
        curNode.getChildRight()->setFather(&curNode);

        std::vector<MarkerLabelT> lookupChildLeft =
            curNode.getChildLeft()->getLookupProgeny();
        std::vector<MarkerLabelT> lookupChildRight =
            curNode.getChildRight()->getLookupProgeny();
        std::vector<MarkerLabelT> nLookupProgeny(leavesNbr, 0);
        for (MarkerLabelT i = 0; i < lookupChildLeft.size(); i++) {
          nLookupProgeny.at(i) =
              max(lookupChildRight.at(i), lookupChildLeft.at(i));
        }
        // new lookupProgeny for each leaf, with 1 for the label of
        // the leaf, and 0 elsewhere
        curNode.setLookupProgeny(nLookupProgeny);
      }
      // last node parameters
      DendroNodeType &lastNode = *dendroNodes[leavesNbr + internalNodesNbr - 1];
      lastNode.setFather(&lastNode); // the last internal node is its own father

    }; // end Dendrogram(mst)

    //!
    /**
     * Constructor from a MST graph and labels/image to compute the moments
     *
     * @param[in] mst Input mst (graph)
     * @param[in] imLabels Labels image
     * @param[in] imIn Original image
     * @param[in] nbOfMoments (optional) Number of moments to compute (default =
     * 5)
     */
    Dendrogram(Graph<NodeT, ValGraphT> &  mst,
               smil::Image<MarkerLabelT> &imLabels,
               smil::Image<MarkerLabelT> &imIn, const size_t nbOfMoments = 5)
        : nbrNodes(0), nbrMarkers(0), dendroNodes(0)
    {
      // sort Edges of the MST by increasing weights of edges
      mst.sortEdges(true);
      // Extract informations from the MST
      size_t          leavesNbr        = mst.getNodeNbr();
      size_t          internalNodesNbr = mst.getEdgeNbr();
      NodeValuesType &mstNodes         = mst.getNodeValues();
      EdgeListType &  mstEdges         = mst.getEdges();

      // Set the number of required nodes and creates them in dendroNodes
      nbrNodes = leavesNbr + internalNodesNbr;
      //       nbrNodes = 2*leavesNbr - 1 ;
      for (size_t i = 0; i < nbrNodes; i++) {
        DendroNodeType *nNode = new DendroNodeType;
        dendroNodes.push_back(nNode);
      }
      // Filling the leaves
      for (size_t i = 0; i < leavesNbr; i++) {
        // dendroNodes is filled with leavesNbr leaves
        // and then with internalNodesNbr internal nodes
        DendroNodeType &curNode = *dendroNodes[i];
        // Leaves have labels from 0 to leavesNbr-1
        curNode.setLabel(mstNodes.find(i)->first);
        curNode.setLeafValuation(mstNodes.find(i)->second);
        curNode.setValuation(mstNodes.find(i)->second);
        // initialize the moments
        std::vector<double> initMoments(nbOfMoments);
        curNode.setMoments(initMoments);
        // initialize the lookupProgeny for each leaf
        // with 1 for the label of the leaf, and 0 elsewhere
        std::vector<MarkerLabelT> nLookupProgeny(leavesNbr, 0);
        nLookupProgeny.at(mstNodes.find(i)->first) = 1;
        curNode.setLookupProgeny(nLookupProgeny);
        // initialize the contoursCount
        std::vector<MarkerLabelT> initContoursCount(leavesNbr, 0);
        curNode.setContoursCount(initContoursCount);
      }

      // Filling the internal nodes
      for (size_t i = 0; i < internalNodesNbr; i++) {
        // dendroFromMSTNodes is filled with leavesNbr leaves and then
        // with internalNodesNbr internal nodes
        DendroNodeType &curNode = *dendroNodes[leavesNbr + i];
        curNode.setIsInternalNode(1);
        curNode.setLabel(leavesNbr + i);
        curNode.setInternalNodeValuationInitial(mstEdges[i].weight);
        MarkerLabelT NeighborLeftLabel =
            min(mstEdges[i].source, mstEdges[i].target);
        MarkerLabelT NeighborRightLabel =
            max(mstEdges[i].source, mstEdges[i].target);

        // dendroNodesLabels[NeighborLeftLabel] to modify
        curNode.setNeighborLeft(dendroNodes[NeighborLeftLabel]);
        // dendroNodesLabels[NeighborRightLabel] to modify
        curNode.setNeighborRight(dendroNodes[NeighborRightLabel]);

        curNode.setChildLeft(curNode.getNeighborLeft()->getAncestor());
        curNode.setChildRight(curNode.getNeighborRight()->getAncestor());

        curNode.getChildLeft()->setFather(&curNode);
        curNode.getChildRight()->setFather(&curNode);

        std::vector<MarkerLabelT> lookupChildLeft =
            curNode.getChildLeft()->getLookupProgeny();
        std::vector<MarkerLabelT> lookupChildRight =
            curNode.getChildRight()->getLookupProgeny();
        std::vector<MarkerLabelT> nLookupProgeny(leavesNbr, 0);
        for (MarkerLabelT i = 0; i < lookupChildLeft.size(); i++) {
          nLookupProgeny.at(i) =
              max(lookupChildRight.at(i), lookupChildLeft.at(i));
        }
        // new lookupProgeny for each leaf, with 1 for the label
        // of the leaf, and 0 elsewhere
        curNode.setLookupProgeny(nLookupProgeny);
      }
      // last node parameters
      DendroNodeType &lastNode = *dendroNodes[leavesNbr + internalNodesNbr - 1];
      // the last internal node is its own father
      lastNode.setFather(&lastNode);

      // Complete the moments values
      this->setMomentsContours(imLabels, imIn, nbOfMoments);
    }; // end Dendrogram(mst,imLabels,imMarkers)

    //! Destructor
    virtual ~Dendrogram()
    {
      for (size_t i = 0; i < nbrNodes; i++) {
        delete dendroNodes[i];
      }
    };
    //! Clone Dendrogram
    Dendrogram clone()
    {
      return Dendrogram(*this);
    };
    //! Reorganizes dendroNodes...
    void sortNodes(bool reverse = false)
    {
      if (!reverse) // ... by growing valuationInitial
        std::stable_sort(dendroNodes.begin(), dendroNodes.end(),
                         DendroNodeType::isInferior);
      else // ... by decreasing valuationInitial
        std::stable_sort(dendroNodes.begin(), dendroNodes.end(),
                         DendroNodeType::isSuperior);
    };
    //! Reorganizes dendroNodes by decreasing valuationInitial
    void sortReverseNodes()
    {
      std::stable_sort(dendroNodes.begin(), dendroNodes.end(),
                       DendroNodeType::isSuperior);
    };
    void addDendroNodes(DendroNodeType *dendroNode)
    {
      dendroNodes.push_back(dendroNode);
    };
    DendroNodeType *researchLabel(MarkerLabelT researchedLabel) const
    {
      DendroNodeType *dendronodeToReturn(0);
      for (size_t i = 0; i < nbrNodes; i++) {
        DendroNodeType &curNode = *dendroNodes[i];
        if (curNode.getLabel() == researchedLabel) {
          dendronodeToReturn = dendroNodes[i];
        }
      }
      return dendronodeToReturn;
    };
    //   TODO : check -1
    //   void setMarkers(Image<MarkerLabelT> &imLabels,
    //                   Image<MarkerLabelT> &imMarkers) {
    //     // get the Image makers regions, with the marker value instead of the
    //     label Image<MarkerLabelT> imMarkersbis =
    //     Image<MarkerLabelT>(imMarkers); std::map<MarkerLabelT, MarkerLabelT>
    //         lookupMapNewMarkers; // to get the markers 1,2,3,...
    //
    //     int sizeImMarkers[3];
    //     imMarkers.getSize(sizeImMarkers);
    //     // imMarkers traversal to fill the lookupMapNewMarkers
    //     MarkerLabelT nMarker = 1;
    //     lookupMapNewMarkers[0] = 0;
    //     for (int i = 0; i < sizeImMarkers[0]; i++) {
    //       for (int j = 0; j < sizeImMarkers[1]; j++) {
    //         MarkerLabelT markerValue = imMarkers.getPixel(i, j);
    //         if (markerValue != 0 && (lookupMapNewMarkers.find(markerValue) ==
    //                                  lookupMapNewMarkers.end())) {
    //           lookupMapNewMarkers[markerValue] = nMarker;
    //           nMarker++;
    //         }
    //       }
    //     }
    //     applyLookup(
    //         imMarkers, lookupMapNewMarkers,
    //         imMarkersbis); // imMarkersbis = new imMarkers with values
    //         1,2,3,...
    //     int nbMarkers = maxVal(imMarkersbis);
    //     //       cout << "minVal : " <<  minVal(imMarkersbis,true) <<  "
    //     maxVal  "
    //     //       << maxVal(imMarkersbis) << endl;
    //     //       imMarkersbis.show();
    //     //       Gui::execLoop();
    //     this->setNbrMarkers(nbMarkers);
    //
    //     // initialize the markersCount
    //     for (size_t i = 0; i < nbrNodes; i++) {
    //       DendroNodeType &curNode = *dendroNodes[i];
    //       std::vector<MarkerLabelT> nMarkersCount(nbrMarkers, 0);
    //       curNode.setMarkersCount(nMarkersCount);
    //     }
    //
    //     std::map<MarkerLabelT, MarkerLabelT> lookupMapMarkers; // to get the
    //     marker
    //                                                            // value if
    //                                                            it's the
    //                                                            // only one
    //                                                            for a
    //                                                            // given label
    //                                                            //
    //                                                            (label,marker)
    //     std::map<MarkerLabelT, std::vector<MarkerLabelT> >
    //         lookupMapMarkersCount; //(label,markerCount)
    //     int sizeIm[3];
    //     imLabels.getSize(sizeIm);
    //
    //     // imLabels traversal to initialize the lookupMapMarkers
    //     for (int i = 0; i < sizeIm[0]; i++) {
    //       for (int j = 0; j < sizeIm[1]; j++) {
    //         MarkerLabelT labelValue = imLabels.getPixel(i, j);
    //         lookupMapMarkers[labelValue] = 0;
    //         lookupMapMarkersCount[labelValue] =
    //             std::vector<MarkerLabelT>(nbrMarkers, 0);
    //       }
    //     }
    //
    //     // imMarkersbis traversal
    //     for (int i = 0; i < sizeIm[0]; i++) {
    //       for (int j = 0; j < sizeIm[1]; j++) {
    //         MarkerLabelT markerValue = imMarkersbis.getPixel(i, j);
    //         if (markerValue != 0) {
    //           MarkerLabelT labelValue = imLabels.getPixel(i, j);
    //           lookupMapMarkersCount[labelValue].at(markerValue - 1) =
    //               lookupMapMarkersCount[labelValue].at(markerValue - 1) + 1;
    //           if (lookupMapMarkers[labelValue] != 0 &&
    //               lookupMapMarkers[labelValue] != -1 &&
    //               lookupMapMarkers[labelValue] != markerValue) {
    //             lookupMapMarkers[labelValue] = -1;
    //           } else if (lookupMapMarkers[labelValue] == 0) {
    //             lookupMapMarkers[labelValue] = markerValue;
    //           }
    //         }
    //       }
    //     } // end imMarkers traversal
    //     // lookupMapMarkers traversal
    //     for (typename std::map<MarkerLabelT, MarkerLabelT>::iterator iter =
    //              lookupMapMarkers.begin();
    //          iter != lookupMapMarkers.end(); ++iter) {
    //       if (iter->second == -1) {
    //         iter->second = 0;
    //       }
    //     }
    //     for (typename std::map<MarkerLabelT, MarkerLabelT>::iterator iter =
    //              lookupMapMarkers.begin();
    //          iter != lookupMapMarkers.end(); ++iter) {
    //       if (iter->second != 0) {
    //         MarkerLabelT researchedLabel = iter->first;
    //         MarkerLabelT markerValue = iter->second;
    //         //    cout << "researchedLabel : " << researchedLabel << " ,
    //         // markerValue : " << markerValue << endl;
    //         this->researchLabel(researchedLabel)->setMarker(markerValue);
    //       }
    //     }
    //     // lookupMapMarkersCount traversal
    //     for (typename std::map<MarkerLabelT, std::vector<MarkerLabelT>
    //     >::iterator
    //              iter = lookupMapMarkersCount.begin();
    //          iter != lookupMapMarkersCount.end(); ++iter) {
    //       //    cout << "researchedLabel : " << iter->first << " ,
    //       // markersValues : " << iter->second[25] << endl;
    //       //   cout << "label : " << iter->first << endl;
    //       //   for (int i = 0;i<iter->second.size();i++){
    //       //     if (iter->second[i]!=0){
    //       //       cout << " marker = " << i << " ; nombre = " <<
    //       // iter->second[i] << endl;
    //       //     }
    //       //   }
    //       //    cout << std::accumulate((iter->second).begin(),
    //       // (iter->second).end(), 0) << endl;
    //       this->researchLabel(iter->first)->setMarkersCount(iter->second);
    //       this->researchLabel(iter->first)
    //           ->setNbMarkersUnder(
    //               std::accumulate((iter->second).begin(),
    //               (iter->second).end(), 0));
    //     }
    //
    //     this->reorganize(); // to get the new nbMarkersUnder and markersCount
    //     //       applyLookup(imLabels,lookupMapMarkers,imMarkersbis);
    //     //       imMarkersbis.show();
    //     //       Gui::execLoop();
    //   }

    void setMomentsContours(Image<MarkerLabelT> &imLabels,
                            Image<MarkerLabelT> &imIn, const size_t nbOfMoments)
    {
      size_t leavesNbr = nbrNodes / 2 + 1;

      std::map<MarkerLabelT, std::vector<double>>       lookupMapMoments;
      std::map<MarkerLabelT, std::vector<MarkerLabelT>> lookupMapContours;
      size_t                                            sizeIm[3];
      imLabels.getSize(sizeIm);

      // imLabels traversal to initialize the lookupMapMoments and
      // lookupMapContours
      for (size_t i = 0; i < sizeIm[0]; i++) {
        for (size_t j = 0; j < sizeIm[1]; j++) {
          MarkerLabelT labelValue = imLabels.getPixel(i, j);
          if (lookupMapMoments.count(labelValue) == 0) {
            std::vector<double> initMoments(nbOfMoments);
            lookupMapMoments[labelValue] = initMoments;
            std::vector<MarkerLabelT> initContours(leavesNbr, 0);
            lookupMapContours[labelValue] = initContours;
          }
        }
      }

      // imIn/imLabels traversal to complete lookupMapMoments
      for (size_t i = 0; i < sizeIm[0]; i++) {
        for (size_t j = 0; j < sizeIm[1]; j++) {
          MarkerLabelT imValue    = imIn.getPixel(i, j);
          MarkerLabelT labelValue = imLabels.getPixel(i, j);
          for (size_t k = 0; k < nbOfMoments; k++) {
            lookupMapMoments[labelValue][k] =
                lookupMapMoments[labelValue][k] + std::pow(imValue, k);
          }
        }
      } // end imIn traversal

      // imLabels traversal to complete lookupMapContours
      // we avoid borders
      for (size_t i = 1; i < sizeIm[0] - 1; i++) {
        for (size_t j = 1; j < sizeIm[1] - 1; j++) {
          MarkerLabelT labelValue = imLabels.getPixel(i, j);
          if (imLabels.getPixel(i - 1, j - 1) != labelValue) {
            MarkerLabelT newLabel = imLabels.getPixel(i - 1, j - 1);
            lookupMapContours[labelValue].at(newLabel) =
                lookupMapContours[labelValue].at(newLabel) + 1;
          } else if (imLabels.getPixel(i - 1, j) != labelValue) {
            MarkerLabelT newLabel = imLabels.getPixel(i - 1, j);
            lookupMapContours[labelValue].at(newLabel) =
                lookupMapContours[labelValue].at(newLabel) + 1;
          } else if (imLabels.getPixel(i - 1, j + 1) != labelValue) {
            MarkerLabelT newLabel = imLabels.getPixel(i - 1, j + 1);
            lookupMapContours[labelValue].at(newLabel) =
                lookupMapContours[labelValue].at(newLabel) + 1;
          } else if (imLabels.getPixel(i, j - 1) != labelValue) {
            MarkerLabelT newLabel = imLabels.getPixel(i, j - 1);
            lookupMapContours[labelValue].at(newLabel) =
                lookupMapContours[labelValue].at(newLabel) + 1;
          } else if (imLabels.getPixel(i, j) != labelValue) {
            MarkerLabelT newLabel = imLabels.getPixel(i, j);
            lookupMapContours[labelValue].at(newLabel) =
                lookupMapContours[labelValue].at(newLabel) + 1;
          } else if (imLabels.getPixel(i, j + 1) != labelValue) {
            MarkerLabelT newLabel = imLabels.getPixel(i, j + 1);
            lookupMapContours[labelValue].at(newLabel) =
                lookupMapContours[labelValue].at(newLabel) + 1;
          } else if (imLabels.getPixel(i + 1, j - 1) != labelValue) {
            MarkerLabelT newLabel = imLabels.getPixel(i + 1, j - 1);
            lookupMapContours[labelValue].at(newLabel) =
                lookupMapContours[labelValue].at(newLabel) + 1;
          } else if (imLabels.getPixel(i + 1, j) != labelValue) {
            MarkerLabelT newLabel = imLabels.getPixel(i + 1, j);
            lookupMapContours[labelValue].at(newLabel) =
                lookupMapContours[labelValue].at(newLabel) + 1;
          } else if (imLabels.getPixel(i + 1, j + 1) != labelValue) {
            MarkerLabelT newLabel = imLabels.getPixel(i + 1, j + 1);
            lookupMapContours[labelValue].at(newLabel) =
                lookupMapContours[labelValue].at(newLabel) + 1;
          }
        }
      } // end imLabels traversal

      // lookupMapMoments traversal
      for (typename std::map<MarkerLabelT, std::vector<double>>::iterator iter =
               lookupMapMoments.begin();
           iter != lookupMapMoments.end(); ++iter) {
        MarkerLabelT        researchedLabel = iter->first;
        std::vector<double> momentsValues   = iter->second;
        //        cout << "label = " << researchedLabel << endl;
        //        cout << "moments " << endl;
        //        cout << "0 : " << momentsValues[0] << endl;
        //        cout << "1 : " << momentsValues[1] << endl;
        //        cout << "2 : " << momentsValues[2] << endl;
        //        cout << "3 : " << momentsValues[3] << endl;
        //        cout << "4 : " << momentsValues[4] << endl;
        researchLabel(researchedLabel)->setMoments(momentsValues);
      }

      // lookupMapContours traversal
      for (typename std::map<MarkerLabelT, std::vector<MarkerLabelT>>::iterator
               iter = lookupMapContours.begin();
           iter != lookupMapContours.end(); ++iter) {
        MarkerLabelT              researchedLabel     = iter->first;
        std::vector<MarkerLabelT> contoursCountValues = iter->second;
        //       cout << "label = " << researchedLabel << endl;
        //       cout << "contoursCountValues " << endl;
        //       cout << "0 : " << contoursCountValues[0] << endl;
        //       cout << "1 : " << contoursCountValues[1] << endl;
        //       cout << "2 : " << contoursCountValues[2] << endl;
        //       cout << "3 : " << contoursCountValues[3] << endl;
        //       cout << "4 : " << contoursCountValues[4] << endl;
        researchLabel(researchedLabel)->setContoursCount(contoursCountValues);
      }
      this->reorganize();
    };

    void reorganize()
    {
      size_t leavesNbr        = nbrNodes / 2 + 1;
      size_t internalNodesNbr = nbrNodes / 2;
      // Clean the kinship between nodes
      for (size_t i = 0; i < nbrNodes; i++) {
        DendroNodeType &curNode = *dendroNodes[i];
        curNode.setChildLeft(0);
        curNode.setChildRight(0);
        curNode.setFather(0);
        std::vector<MarkerLabelT> nLookupProgeny(leavesNbr, 0);
        curNode.setLookupProgeny(nLookupProgeny);

        if (curNode.getIsInternalNode()) {
          std::vector<MarkerLabelT> nMarkersCount(nbrMarkers, 0);
          curNode.setMarkersCount(nMarkersCount);
        }
      }
      // Filling the leaves
      for (size_t i = 0; i < nbrNodes; i++) {
        DendroNodeType &curNode = *dendroNodes[i];
        // initialize the lookupProgeny
        if (!curNode.getIsInternalNode()) {
          std::vector<MarkerLabelT> nLookupProgeny(leavesNbr, 0);
          nLookupProgeny.at(curNode.getLabel()) = 1;
          // new lookupProgeny for each leaf, with 1 for the label
          // of the leaf, and 0 elsewhere
          curNode.setLookupProgeny(nLookupProgeny);
        }
      }
      // we sort nodes by growing internalNodeValuationInitial
      this->sortNodes();
      // Filling the internal nodes
      for (size_t i = 0; i < nbrNodes; i++) {
        DendroNodeType &curNode = *dendroNodes[i];
        if (curNode.getIsInternalNode()) {
          curNode.setChildLeft(curNode.getNeighborLeft()->getAncestor());
          curNode.setChildRight(curNode.getNeighborRight()->getAncestor());

          curNode.getChildLeft()->setFather(&curNode);
          curNode.getChildRight()->setFather(&curNode);

          // Computing the new lookupProgeny
          std::vector<MarkerLabelT> lookupChildLeft =
              curNode.getChildLeft()->getLookupProgeny();
          std::vector<MarkerLabelT> lookupChildRight =
              curNode.getChildRight()->getLookupProgeny();
          std::vector<MarkerLabelT> nLookupProgeny(leavesNbr, 0);
          for (size_t i = 0; i < lookupChildLeft.size(); i++) {
            nLookupProgeny.at(i) =
                max(lookupChildRight.at(i), lookupChildLeft.at(i));
          }
          // new lookupProgeny for each leaf, with 1 for the label
          // of the leaf, and 0 elsewhere
          curNode.setLookupProgeny(nLookupProgeny);

          // Computing the new nbMarkersUnder
          MarkerLabelT nbMarkersUnderLeft =
              curNode.getChildLeft()->getNbMarkersUnder();
          MarkerLabelT nbMarkersUnderRight =
              curNode.getChildRight()->getNbMarkersUnder();
          curNode.setNbMarkersUnder(nbMarkersUnderLeft + nbMarkersUnderRight);

          // Computing the new markersCount
          std::vector<MarkerLabelT> markersCountLeft =
              curNode.getChildLeft()->getMarkersCount();
          std::vector<MarkerLabelT> markersCountRight =
              curNode.getChildRight()->getMarkersCount();
          std::vector<MarkerLabelT> nMarkersCount(nbrMarkers, 0);
          for (size_t i = 0; i < markersCountLeft.size(); i++) {
            nMarkersCount.at(i) =
                markersCountLeft.at(i) + markersCountRight.at(i);
          }
          curNode.setMarkersCount(nMarkersCount);

          // Computing the new moments vector
          std::vector<double> momentsChildLeft =
              curNode.getChildLeft()->getMoments();
          std::vector<double> momentsChildRight =
              curNode.getChildRight()->getMoments();
          std::vector<double> nMoments(momentsChildLeft.size(), 0);
          for (size_t i = 0; i < momentsChildLeft.size(); i++) {
            nMoments.at(i) = momentsChildLeft.at(i) + momentsChildRight.at(i);
          }
          // new moments vector for each node
          curNode.setMoments(nMoments);

          // Computing the new fidelity term

          // Computing the new contoursCount
          std::vector<MarkerLabelT> contoursCountLeft =
              curNode.getChildLeft()->getContoursCount();
          std::vector<MarkerLabelT> contoursCountRight =
              curNode.getChildRight()->getContoursCount();

          std::vector<MarkerLabelT> nContoursCount(leavesNbr, 0);
          for (size_t i = 0; i < contoursCountLeft.size(); i++) {
            nContoursCount.at(i) =
                contoursCountLeft.at(i) + contoursCountRight.at(i);
          }
          // new lookupProgeny for each leaf, with 1 for the label
          // of the leaf, and 0 elsewhere
          curNode.setContoursCount(nContoursCount);
          // Computing the new contoursSize
          double nContoursSize = 0;
          for (size_t i = 0; i < contoursCountLeft.size(); i++) {
            nContoursSize = nContoursSize +
                            nContoursCount.at(i) * (1 - nLookupProgeny.at(i));
          }
          curNode.setContoursSize(nContoursSize);
        }
        // set the contours size for leaves too
        else if (!curNode.getIsInternalNode()) {
          std::vector<MarkerLabelT> lookupProgeny = curNode.getLookupProgeny();
          std::vector<MarkerLabelT> contoursCount = curNode.getContoursCount();

          double nContoursSize = 0;
          for (size_t i = 0; i < contoursCount.size(); i++) {
            nContoursSize =
                nContoursSize + contoursCount.at(i) * (1 - lookupProgeny.at(i));
          }
          curNode.setContoursSize(nContoursSize);
        }
      }
      // last node parameters
      DendroNodeType &lastNode = *dendroNodes[leavesNbr + internalNodesNbr - 1];
      lastNode.setFather(&lastNode); // the last internal node is its own father
    };

    /** Dendrogram ultrametric values normalization
     *
     *
     * @param[in] typeOfNormalization Desired type of normalization:
     *              "reg" (according to the number of regions),
     *              "maxnorm" (according to the maximum value)
     */
    void normalize(const std::string typeOfNormalization = "reg")
    {
      //       size_t leavesNbr = nbrNodes/2+1;
      size_t internalNodesNbr = nbrNodes / 2;

      if (typeOfNormalization == "reg") {
        // we sort nodes by growing internalNodeValuationInitial
        this->sortNodes();
        double counter = 1;
        // Filling the internal nodes
        for (size_t i = 0; i < nbrNodes; i++) {
          DendroNodeType &curNode = *dendroNodes.at(i);
          if (curNode.getIsInternalNode()) {
            double newVal = counter / internalNodesNbr;
            curNode.setInternalNodeValuationFinal(newVal);
            counter = counter + 1;
          }
        }
        this->putValuationsFinalInInitial();
        this->reorganize();
      } else if (typeOfNormalization == "maxnorm") {
        // we sort nodes by decreasing internalNodeValuationInitial
        this->sortNodes(true);
        double maxVal;

        // get the biggest finite value
        for (size_t i = 0; i < nbrNodes; i++) {
          DendroNodeType &curNodeInit = *dendroNodes.at(i);
          maxVal = curNodeInit.getInternalNodeValuationInitial();
          // we avoid dividing by infinite value
          if (!isinf(maxVal)) {
            break;
          }
        }
        // Filling the internal nodes
        for (size_t i = 0; i < nbrNodes; i++) {
          DendroNodeType &curNode = *dendroNodes.at(i);
          if (curNode.getIsInternalNode()) {
            double newVal =
                double(curNode.getInternalNodeValuationInitial() / maxVal);
            curNode.setInternalNodeValuationFinal(newVal);
          }
        }
        this->putValuationsFinalInInitial();
        this->reorganize();
      } else {
        std::cerr << "normalize(const std::string typeOfNormalization) -> "
                     "typeOfNormalization must be chosen within: "
                  << "reg, maxnorm" << endl;
      }
    }
#ifndef SWIG
    //! Put the internalNodeValuationFinal in internalNodeValuationInitial and
    //! then set internalNodeValuationFinal = 0
    void putValuationsFinalInInitial()
    {
      for (size_t i = 0; i < nbrNodes; i++) {
        DendroNodeType &curNode = *dendroNodes[i];
        if (curNode.getIsInternalNode() == 1) {
          double temp = curNode.getInternalNodeValuationFinal();
          curNode.setInternalNodeValuationInitial(temp);
          curNode.setInternalNodeValuationFinal(0);
        }
      }
    };
    void setValuationsToZero()
    {
      for (size_t i = 0; i < nbrNodes; i++) {
        DendroNodeType &curNode = *dendroNodes[i];
        if (curNode.getIsInternalNode() == 1) {
          curNode.setValuation(0);
        }
        if (curNode.getIsInternalNode() == 0) {
          curNode.setValuation(curNode.getLeafValuation());
        }
      }
    }
#endif
    //! Given an internal node index lambda of the dendrogram, remove
    //! corresponding edge in the associated MST
    void removeMSTEdgesDendrogram(Graph<NodeT, ValGraphT> &associated_mst,
                                  double                   lbd)
    { // Dendrogram<MarkerLabelT,NodeT,ValGraphT>&
      // dendrogram
      std::vector<DendroNodeType *> &dendroNodes = this->getDendroNodes();
      size_t                         nodeNbr     = this->getNbrNodes();
      //       size_t internalNodesNbr = associated_mst.getEdgeNbr();
      //       size_t leavesNbr = associated_mst.getNodeNbr();
      for (size_t i = 0; i < nodeNbr; i++) {
        DendroNodeType &curNode = *dendroNodes.at(i);
        if (curNode.getInternalNodeValuationInitial() >= lbd &&
            curNode.getIsInternalNode() == true) {
          MarkerLabelT srcToRemove    = curNode.getNeighborLeft()->getLabel();
          MarkerLabelT targetToRemove = curNode.getNeighborRight()->getLabel();
          associated_mst.removeEdge(srcToRemove, targetToRemove);
          associated_mst.removeEdge(targetToRemove, srcToRemove);
        }
      }
    };

    /** Computes a new hierarchy from a given dendrogram hierarchy
     *
     *
     * @param[in] typeOfHierarchy Desired type of hierarchy:
     *              "none", "surfacic", "volumic", "stochasticSurfacic",
     * "stochasticVolumic" etc.
     * @param[in] nParam :
     * @param[in] imMosa (optional) Mosaic image - necessary for hierarchies
     * implying labels image deformations
     * @param[in] typeOfTransform (optional) Type of transform (deformation)
     * applied to each mosaic image region
     * @param[in] se (optional) Structuring element
     */
    void HierarchicalConstruction(
        const std::string typeOfHierarchy, const int nParam = 50,
        const smil::Image<MarkerLabelT> &imMosa = smil::Image<MarkerLabelT>(),
        const std::string                typeOfTransform = "erode",
        const StrElt &                   se              = DEFAULT_SE)
    {
      // sort by increasing values of internalNodeValuationInitial
      this->sortNodes();
      std::vector<DendroNodeType *> &dendroNodes = this->getDendroNodes();
      if (typeOfHierarchy == "none") {
      } else if (typeOfHierarchy == "surfacic") {
        // only one dendroNodes traversal, and only on internal nodes
        // because leaves already have surface as valuations
        for (size_t i = 0; i < dendroNodes.size(); i++) {
          DendroNodeType &curNode = *dendroNodes[i];
          // we verify that it's an internal node
          if (curNode.getIsInternalNode() == 1) {
            if (curNode.getValuation() == 0) {
              double childLeftValuation =
                  curNode.getChildLeft()->getValuation();
              double childRightValuation =
                  curNode.getChildRight()->getValuation();
              curNode.setValuation(childLeftValuation + childRightValuation);
              curNode.setInternalNodeValuationFinal(
                  fmin(childLeftValuation, childRightValuation));
            } else {
              curNode.setInternalNodeValuationFinal(
                  curNode.getInternalNodeValuationInitial());
            }
          }
        }
        this->putValuationsFinalInInitial();
        this->reorganize();
        this->setValuationsToZero();
      } else if (typeOfHierarchy == "surfacicImageReturn") {
        // only one dendroNodes traversal, and only on internal nodes,
        // because leaves already have surface as valuations
        for (size_t i = 0; i < dendroNodes.size(); i++) {
          DendroNodeType &curNode = *dendroNodes[i];
          // we verify that it's an internal node
          if (curNode.getIsInternalNode() == 1) {
            std::vector<MarkerLabelT> lookupChildLeft =
                curNode.getChildLeft()->getLookupProgeny();
            std::vector<MarkerLabelT> lookupChildRight =
                curNode.getChildRight()->getLookupProgeny();

            std::map<MarkerLabelT, MarkerLabelT> lookupMapChildLeft;
            for (MarkerLabelT j = 0; j < lookupChildLeft.size(); j++) {
              lookupMapChildLeft[j] = lookupChildLeft.at(j);
            }

            std::map<MarkerLabelT, MarkerLabelT> lookupMapChildRight;
            for (MarkerLabelT j = 0; j < lookupChildRight.size(); j++) {
              lookupMapChildRight[j] = lookupChildRight.at(j);
            }

            smil::Image<MarkerLabelT> imTmpLeft(imMosa);
            smil::Image<MarkerLabelT> imTmpRight(imMosa);

            applyLookup(imMosa, lookupMapChildLeft, imTmpLeft);
            applyLookup(imMosa, lookupMapChildRight, imTmpRight);

            if (typeOfTransform == "erode") {
              erode(imTmpLeft, imTmpLeft, se);
              erode(imTmpRight, imTmpRight, se);
            } else if (typeOfTransform == "dilate") {
              dilate(imTmpLeft, imTmpLeft, se);
              dilate(imTmpRight, imTmpRight, se);
            } else if (typeOfTransform == "open") {
              open(imTmpLeft, imTmpLeft, se);
              open(imTmpRight, imTmpRight, se);
            } else if (typeOfTransform == "close") {
              close(imTmpLeft, imTmpLeft, se);
              close(imTmpRight, imTmpRight, se);
            } else {
              cout << "Please choose typeOfTransform in the following: erode, "
                      "dilate, open, close"
                   << endl;
            }
            double childLeftSurf  = area(imTmpLeft);
            double childRightSurf = area(imTmpRight);

            double childLeftValuation  = childLeftSurf;
            double childRightValuation = childRightSurf;
            curNode.setValuation(childLeftValuation + childRightValuation);
            curNode.setInternalNodeValuationFinal(
                fmin(childLeftValuation, childRightValuation));
          }
        }
        this->putValuationsFinalInInitial();
        this->reorganize();
        this->setValuationsToZero();
      } else if (typeOfHierarchy == "stochasticSurfacic") {
        // First dendroNodes traversal, to get the "surfacic" dendrogram
        for (size_t i = 0; i < dendroNodes.size(); i++) {
          DendroNodeType &curNode = *dendroNodes[i];
          // we verify that it's an internal node
          if (curNode.getIsInternalNode() == 1) {
            if (curNode.getValuation() == 0) {
              double childLeftValuation =
                  curNode.getChildLeft()->getValuation();
              double childRightValuation =
                  curNode.getChildRight()->getValuation();
              curNode.setValuation(childLeftValuation + childRightValuation);
            } else {
              curNode.setInternalNodeValuationFinal(
                  curNode.getInternalNodeValuationInitial());
            }
          }
        }

        DendroNodeType &curNodeTmp = *dendroNodes[1];
        // get the total surface of the domain
        double totalSurface = curNodeTmp.getAncestor()->getValuation();
        // Second dendroNodes traversal, to get the stochasticSurfacic
        // dendrogram
        for (size_t i = 0; i < dendroNodes.size(); i++) {
          DendroNodeType &curNode = *dendroNodes[i];
          // we verify that it's an internal node
          if (curNode.getIsInternalNode() == 1) {
            double childLeftSurf  = curNode.getChildLeft()->getValuation();
            double childRightSurf = curNode.getChildRight()->getValuation();
            double newVal =
                1 - pow(1 - (childLeftSurf / totalSurface), nParam) -
                pow(1 - (childRightSurf / totalSurface), nParam) +
                pow(1 - ((childLeftSurf + childRightSurf) / totalSurface),
                    nParam);
            if (newVal <= 0) {
              newVal = 0.000000001;
            }
            curNode.setInternalNodeValuationFinal(newVal);
          }
        }
        this->putValuationsFinalInInitial();
        this->reorganize();
        this->setValuationsToZero();
      } else if (typeOfHierarchy == "stochasticExtinctionSurfacic") {
        // First dendroNodes traversal, to get the "surfacic" dendrogram
        for (size_t i = 0; i < dendroNodes.size(); i++) {
          DendroNodeType &curNode = *dendroNodes[i];
          // we verify that it's an internal node
          if (curNode.getIsInternalNode() == 1) {
            if (curNode.getValuation() == 0) {
              double childLeftValuation =
                  curNode.getChildLeft()->getValuation();
              double childRightValuation =
                  curNode.getChildRight()->getValuation();
              curNode.setValuation(childLeftValuation + childRightValuation);
            } else {
              curNode.setInternalNodeValuationFinal(
                  curNode.getInternalNodeValuationInitial());
            }
          }
        }

        DendroNodeType &curNodeTmp = *dendroNodes[1];
        // get the total surface of the domain
        double totalSurface = curNodeTmp.getAncestor()->getValuation();
        // Second dendroNodes traversal, to get the stochasticSurfacic
        // dendrogram
        for (size_t i = 0; i < dendroNodes.size(); i++) {
          DendroNodeType &curNode = *dendroNodes[i];
          // we verify that it's an internal node
          if (curNode.getIsInternalNode() == 1) {
            double childLeftSurf  = curNode.getChildLeft()->getValuation();
            double childRightSurf = curNode.getChildRight()->getValuation();
            double newVal =
                1 -
                pow(1 - std::min(childLeftSurf, childRightSurf) / totalSurface,
                    nParam);
            curNode.setInternalNodeValuationFinal(newVal);
          }
        }
        this->putValuationsFinalInInitial();
        this->reorganize();
        this->setValuationsToZero();
      } else if (typeOfHierarchy == "stochasticSurfacicImageReturn") {
        // First dendroNodes traversal, to get the "surfacic" dendrogram
        for (size_t i = 0; i < dendroNodes.size(); i++) {
          DendroNodeType &curNode = *dendroNodes[i];
          // we verify that it's an internal node
          if (curNode.getIsInternalNode() == 1) {
            if (curNode.getValuation() == 0) {
              double childLeftValuation =
                  curNode.getChildLeft()->getValuation();
              double childRightValuation =
                  curNode.getChildRight()->getValuation();
              curNode.setValuation(childLeftValuation + childRightValuation);
            }
          }
        }

        DendroNodeType &curNodeTmp = *dendroNodes[1];
        // get the total surface of the domain
        double totalSurface = curNodeTmp.getAncestor()->getValuation();
        // Second dendroNodes traversal, to get the stochasticSurfacic
        // dendrogram
        for (size_t i = 0; i < dendroNodes.size(); i++) {
          DendroNodeType &curNode = *dendroNodes[i];
          // we verify that it's an internal node
          if (curNode.getIsInternalNode() == 1) {
            std::vector<MarkerLabelT> lookupChildLeft =
                curNode.getChildLeft()->getLookupProgeny();
            std::vector<MarkerLabelT> lookupChildRight =
                curNode.getChildRight()->getLookupProgeny();

            std::map<MarkerLabelT, MarkerLabelT> lookupMapChildLeft;
            for (MarkerLabelT j = 0; j < lookupChildLeft.size(); j++) {
              lookupMapChildLeft[j] = lookupChildLeft.at(j);
            }

            std::map<MarkerLabelT, MarkerLabelT> lookupMapChildRight;
            for (MarkerLabelT j = 0; j < lookupChildRight.size(); j++) {
              lookupMapChildRight[j] = lookupChildRight.at(j);
            }

            smil::Image<MarkerLabelT> imTmpLeft(imMosa);
            smil::Image<MarkerLabelT> imTmpRight(imMosa);

            applyLookup(imMosa, lookupMapChildLeft, imTmpLeft);
            applyLookup(imMosa, lookupMapChildRight, imTmpRight);

            if (typeOfTransform == "erode") {
              erode(imTmpLeft, imTmpLeft, se);
              erode(imTmpRight, imTmpRight, se);
            } else if (typeOfTransform == "dilate") {
              dilate(imTmpLeft, imTmpLeft, se);
              dilate(imTmpRight, imTmpRight, se);
            } else if (typeOfTransform == "open") {
              open(imTmpLeft, imTmpLeft, se);
              open(imTmpRight, imTmpRight, se);
            } else if (typeOfTransform == "close") {
              close(imTmpLeft, imTmpLeft, se);
              close(imTmpRight, imTmpRight, se);
            } else {
              cout << "Please choose typeOfTransform in the following: erode, "
                      "dilate, open, close"
                   << endl;
            }
            double childLeftSurf  = area(imTmpLeft);
            double childRightSurf = area(imTmpRight);

            double newVal =
                1 - pow(1 - (childLeftSurf / totalSurface), nParam) -
                pow(1 - (childRightSurf / totalSurface), nParam) +
                pow(1 - ((childLeftSurf + childRightSurf) / totalSurface),
                    nParam);
            if (newVal <= 0) {
              newVal = 0.000000001;
            }
            curNode.setInternalNodeValuationFinal(newVal);
          }
        }
        this->putValuationsFinalInInitial();
        this->reorganize();
        this->setValuationsToZero();
      } else if (typeOfHierarchy == "stochasticExtinctionSurfacicImageReturn") {
        for (size_t i = 0; i < dendroNodes.size(); i++) {
          DendroNodeType &curNode = *dendroNodes[i];
          // we verify that it's an internal node
          if (curNode.getIsInternalNode() == 1) {
            if (curNode.getValuation() == 0) {
              double childLeftValuation =
                  curNode.getChildLeft()->getValuation();
              double childRightValuation =
                  curNode.getChildRight()->getValuation();
              curNode.setValuation(childLeftValuation + childRightValuation);
            } else {
              curNode.setInternalNodeValuationFinal(
                  curNode.getInternalNodeValuationInitial());
            }
          }
        }

        DendroNodeType &curNodeTmp = *dendroNodes[1];
        // get the total surface of the domain
        double totalSurface = curNodeTmp.getAncestor()->getValuation();
        // Second dendroNodes traversal, to get the stochasticSurfacic
        // dendrogram
        for (size_t i = 0; i < dendroNodes.size(); i++) {
          DendroNodeType &curNode = *dendroNodes[i];
          // we verify that it's an internal node
          if (curNode.getIsInternalNode() == 1) {
            std::vector<MarkerLabelT> lookupChildLeft =
                curNode.getChildLeft()->getLookupProgeny();
            std::vector<MarkerLabelT> lookupChildRight =
                curNode.getChildRight()->getLookupProgeny();

            std::map<MarkerLabelT, MarkerLabelT> lookupMapChildLeft;
            for (MarkerLabelT j = 0; j < lookupChildLeft.size(); j++) {
              lookupMapChildLeft[j] = lookupChildLeft.at(j);
            }

            std::map<MarkerLabelT, MarkerLabelT> lookupMapChildRight;
            for (MarkerLabelT j = 0; j < lookupChildRight.size(); j++) {
              lookupMapChildRight[j] = lookupChildRight.at(j);
            }

            smil::Image<MarkerLabelT> imTmpLeft(imMosa);
            smil::Image<MarkerLabelT> imTmpRight(imMosa);

            applyLookup(imMosa, lookupMapChildLeft, imTmpLeft);
            applyLookup(imMosa, lookupMapChildRight, imTmpRight);

            if (typeOfTransform == "erode") {
              erode(imTmpLeft, imTmpLeft, se);
              erode(imTmpRight, imTmpRight, se);
            } else if (typeOfTransform == "dilate") {
              dilate(imTmpLeft, imTmpLeft, se);
              dilate(imTmpRight, imTmpRight, se);
            } else if (typeOfTransform == "open") {
              open(imTmpLeft, imTmpLeft, se);
              open(imTmpRight, imTmpRight, se);
            } else if (typeOfTransform == "close") {
              close(imTmpLeft, imTmpLeft, se);
              close(imTmpRight, imTmpRight, se);
            } else {
              cout << "Please choose typeOfTransform in the following: erode, "
                      "dilate, open, close"
                   << endl;
            }
            double childLeftSurf  = area(imTmpLeft);
            double childRightSurf = area(imTmpRight);

            double newVal =
                1 -
                pow(1 - std::min(childLeftSurf, childRightSurf) / totalSurface,
                    nParam);

            if (newVal <= 0) {
              newVal = 0.000000001;
            }
            curNode.setInternalNodeValuationFinal(newVal);
          }
        }
        this->putValuationsFinalInInitial();
        this->reorganize();
        this->setValuationsToZero();
      } else if (typeOfHierarchy == "stochasticSurfacicImageReturnNonPoint") {
        // First dendroNodes traversal, to get the "surfacic" dendrogram
        for (size_t i = 0; i < dendroNodes.size(); i++) {
          DendroNodeType &curNode = *dendroNodes[i];
          // we verify that it's an internal node
          if (curNode.getIsInternalNode() == 1) {
            if (curNode.getValuation() == 0) {
              double childLeftValuation =
                  curNode.getChildLeft()->getValuation();
              double childRightValuation =
                  curNode.getChildRight()->getValuation();
              curNode.setValuation(childLeftValuation + childRightValuation);
            }
          }
        }

        DendroNodeType &curNodeTmp = *dendroNodes[1];
        // get the total surface of the domain
        double totalSurface = curNodeTmp.getAncestor()->getValuation();
        smil::Image<MarkerLabelT> imTmpLeft(imMosa);
        smil::Image<MarkerLabelT> imTmpRight(imMosa);
        smil::Image<MarkerLabelT> imTmpInter(imMosa);
        smil::Image<MarkerLabelT> imTmpInterbis(imMosa);

        // Second dendroNodes traversal, to get the stochasticSurfacic
        // dendrogram
        for (size_t i = 0; i < dendroNodes.size(); i++) {
          DendroNodeType &curNode = *dendroNodes[i];
          // we verify that it's an internal node
          if (curNode.getIsInternalNode() == 1) {
            std::vector<MarkerLabelT> lookupChildLeft =
                curNode.getChildLeft()->getLookupProgeny();
            std::vector<MarkerLabelT> lookupChildRight =
                curNode.getChildRight()->getLookupProgeny();

            std::map<MarkerLabelT, MarkerLabelT> lookupMapChildLeft;
            for (MarkerLabelT j = 0; j < lookupChildLeft.size(); j++) {
              lookupMapChildLeft[j] = lookupChildLeft.at(j);
            }

            std::map<MarkerLabelT, MarkerLabelT> lookupMapChildRight;
            for (MarkerLabelT j = 0; j < lookupChildRight.size(); j++) {
              lookupMapChildRight[j] = lookupChildRight.at(j);
            }

            applyLookup(imMosa, lookupMapChildLeft, imTmpLeft);
            applyLookup(imMosa, lookupMapChildRight, imTmpRight);
            dilate(imTmpLeft, imTmpLeft, se);
            dilate(imTmpRight, imTmpRight, se);

            mul(imTmpLeft, imTmpRight, imTmpInter);
            double LRSurf = area(imTmpInter);

            MarkerLabelT one  = 1;
            MarkerLabelT zero = 0;

            sub(imTmpLeft, imTmpRight, imTmpInter);

            compare(imTmpInter, "==", one, one, zero, imTmpInterbis);
            double L_RSurf = area(imTmpInterbis);

            sub(imTmpRight, imTmpLeft, imTmpInter);
            compare(imTmpInter, "==", one, one, zero, imTmpInterbis);
            double R_LSurf = area(imTmpInterbis);

            double newVal =
                pow(1 - (LRSurf / totalSurface), nParam) *
                (1 - pow(1 - (L_RSurf / (totalSurface - LRSurf)), nParam) -
                 pow(1 - (R_LSurf / (totalSurface - LRSurf)), nParam) +
                 pow(1 - (L_RSurf + R_LSurf) / (totalSurface - LRSurf),
                     nParam));

            if (newVal <= 0) {
              newVal = 0.000000001;
            }
            curNode.setInternalNodeValuationFinal(newVal);
          }
        }
        this->putValuationsFinalInInitial();
        this->reorganize();
        this->setValuationsToZero();
      } else if (typeOfHierarchy == "volumic") {
        // First dendroNodes traversal, to get the "surfacic" dendrogram
        for (size_t i = 0; i < dendroNodes.size(); i++) {
          DendroNodeType &curNode = *dendroNodes[i];
          // we verify that it's an internal node
          if (curNode.getIsInternalNode() == 1) {
            if (curNode.getValuation() == 0) {
              double childLeftValuation =
                  curNode.getChildLeft()->getValuation();
              double childRightValuation =
                  curNode.getChildRight()->getValuation();
              curNode.setValuation(childLeftValuation + childRightValuation);
            }
          }
        }
        // Second dendroNodes traversal, to get the "volumic" dendrogram
        for (size_t i = 0; i < dendroNodes.size(); i++) {
          DendroNodeType &curNode = *dendroNodes[i];
          // we verify that the node is not the ancestor
          if (curNode.getLabel() != curNode.getFather()->getLabel()) {
            double height =
                curNode.getFather()->getInternalNodeValuationInitial();
            double surface = curNode.getValuation();
            curNode.setValuation(height * surface);
          }
        }
        // Third dendroNodes traversal, to get the final
        // valuation of internal nodes
        for (size_t i = 0; i < dendroNodes.size(); i++) {
          DendroNodeType &curNode = *dendroNodes[i];
          // we verify that it's an internal node
          if (curNode.getIsInternalNode() == 1) {
            double childLeftValuation = curNode.getChildLeft()->getValuation();
            double childRightValuation =
                curNode.getChildRight()->getValuation();
            double height = curNode.getInternalNodeValuationInitial();
            curNode.setInternalNodeValuationFinal(
                height * fmin(childLeftValuation, childRightValuation));
          }
        }
        this->putValuationsFinalInInitial();
        this->reorganize();
        this->setValuationsToZero();
      } else if (typeOfHierarchy == "volumicImageReturn") {
        // First dendroNodes traversal, to get the "surfacic" dendrogram
        for (size_t i = 0; i < dendroNodes.size(); i++) {
          DendroNodeType &curNode = *dendroNodes[i];
          // we verify that it's an internal node
          if (curNode.getIsInternalNode() == 1) {
            std::vector<MarkerLabelT> lookupChildLeft =
                curNode.getChildLeft()->getLookupProgeny();
            std::vector<MarkerLabelT> lookupChildRight =
                curNode.getChildRight()->getLookupProgeny();

            std::map<MarkerLabelT, MarkerLabelT> lookupMapChildLeft;
            for (MarkerLabelT j = 0; j < lookupChildLeft.size(); j++) {
              lookupMapChildLeft[j] = lookupChildLeft.at(j);
            }

            std::map<MarkerLabelT, MarkerLabelT> lookupMapChildRight;
            for (MarkerLabelT j = 0; j < lookupChildRight.size(); j++) {
              lookupMapChildRight[j] = lookupChildRight.at(j);
            }

            smil::Image<MarkerLabelT> imTmpLeft(imMosa);
            smil::Image<MarkerLabelT> imTmpRight(imMosa);

            applyLookup(imMosa, lookupMapChildLeft, imTmpLeft);
            applyLookup(imMosa, lookupMapChildRight, imTmpRight);

            if (typeOfTransform == "erode") {
              erode(imTmpLeft, imTmpLeft, se);
              erode(imTmpRight, imTmpRight, se);
            } else if (typeOfTransform == "dilate") {
              dilate(imTmpLeft, imTmpLeft, se);
              dilate(imTmpRight, imTmpRight, se);
            } else if (typeOfTransform == "open") {
              open(imTmpLeft, imTmpLeft, se);
              open(imTmpRight, imTmpRight, se);
            } else if (typeOfTransform == "close") {
              close(imTmpLeft, imTmpLeft, se);
              close(imTmpRight, imTmpRight, se);
            } else {
              cout << "Please choose typeOfTransform in the following: erode, "
                      "dilate, open, close"
                   << endl;
            }
            double childLeftSurf  = area(imTmpLeft);
            double childRightSurf = area(imTmpRight);

            curNode.setValuation(childLeftSurf + childRightSurf);
          }
        }
        // Second dendroNodes traversal, to get the "volumic" dendrogram
        for (size_t i = 0; i < dendroNodes.size(); i++) {
          DendroNodeType &curNode = *dendroNodes[i];
          // we verify that the node is not the ancestor
          if (curNode.getLabel() != curNode.getFather()->getLabel()) {
            double height =
                curNode.getFather()->getInternalNodeValuationInitial();
            double surface = curNode.getValuation();
            curNode.setValuation(height * surface);
          }
        }
        // Third dendroNodes traversal, to get the final
        // valuation of internal nodes
        for (size_t i = 0; i < dendroNodes.size(); i++) {
          DendroNodeType &curNode = *dendroNodes[i];
          // we verify that it's an internal node
          if (curNode.getIsInternalNode() == 1) {
            double childLeftValuation = curNode.getChildLeft()->getValuation();
            double childRightValuation =
                curNode.getChildRight()->getValuation();
            double height = curNode.getInternalNodeValuationInitial();
            double newVal =
                height * fmin(childLeftValuation, childRightValuation);

            if (newVal <= 0) {
              newVal = 0.000000001;
            };
            curNode.setInternalNodeValuationFinal(newVal);
          }
        }
        this->putValuationsFinalInInitial();
        this->reorganize();
        this->setValuationsToZero();
      } else if (typeOfHierarchy == "stochasticVolumic") {
        double totalSurface = 0;
        double totalDepth   = 0;

        // First dendroNodes traversal, to get the "surfacic" dendrogram
        for (size_t i = 0; i < dendroNodes.size(); i++) {
          DendroNodeType &curNode = *dendroNodes[i];
          // we verify that it's an internal node
          if (curNode.getIsInternalNode() == 1) {
            if (curNode.getValuation() == 0) {
              double childLeftValuation =
                  curNode.getChildLeft()->getValuation();
              double childRightValuation =
                  curNode.getChildRight()->getValuation();
              curNode.setValuation(childLeftValuation + childRightValuation);
            }
          }
        }

        DendroNodeType &curNodeTmp = *dendroNodes[1];
        // get the total surface of the domain
        totalSurface = curNodeTmp.getAncestor()->getValuation();
        // get the total depth of the domain
        totalDepth =
            curNodeTmp.getAncestor()->getInternalNodeValuationInitial();
        double totalVolume = totalSurface * totalDepth;
        // Second dendroNodes traversal, to get the "volumic" dendrogram
        for (size_t i = 0; i < dendroNodes.size(); i++) {
          DendroNodeType &curNode = *dendroNodes[i];
          // we verify that the node is not the ancestor
          if (curNode.getLabel() != curNode.getFather()->getLabel()) {
            double height =
                curNode.getFather()->getInternalNodeValuationInitial();
            double surface = curNode.getValuation();
            curNode.setValuation(height * surface);
          }
        }

        // Third dendroNodes traversal, to get the
        // stochasticVolumic dendrogram
        for (size_t i = 0; i < dendroNodes.size(); i++) {
          DendroNodeType &curNode = *dendroNodes[i];
          // we verify that it's an internal node
          if (curNode.getIsInternalNode() == 1) {
            double childLeftVol  = curNode.getChildLeft()->getValuation();
            double childRightVol = curNode.getChildRight()->getValuation();

            double newVal =
                1 - pow(1 - (childLeftVol / totalVolume), nParam) -
                pow(1 - (childRightVol / totalVolume), nParam) +
                pow(1 - (childLeftVol + childRightVol) / totalVolume, nParam);
            curNode.setInternalNodeValuationFinal(newVal);
          }
        }
        this->putValuationsFinalInInitial();
        this->reorganize();
        this->setValuationsToZero();
      } else if (typeOfHierarchy == "stochasticExtinctionVolumic") {
        double totalSurface = 0;
        double totalDepth   = 0;

        // First dendroNodes traversal, to get the "surfacic" dendrogram
        for (size_t i = 0; i < dendroNodes.size(); i++) {
          DendroNodeType &curNode = *dendroNodes[i];
          // we verify that it's an internal node
          if (curNode.getIsInternalNode() == 1) {
            if (curNode.getValuation() == 0) {
              double childLeftValuation =
                  curNode.getChildLeft()->getValuation();
              double childRightValuation =
                  curNode.getChildRight()->getValuation();
              curNode.setValuation(childLeftValuation + childRightValuation);
            }
          }
        }

        DendroNodeType &curNodeTmp = *dendroNodes[1];
        // get the total surface of the domain
        totalSurface = curNodeTmp.getAncestor()->getValuation();
        // get the total depth of the domain
        totalDepth =
            curNodeTmp.getAncestor()->getInternalNodeValuationInitial();

        double totalVolume = totalSurface * totalDepth;
        // Second dendroNodes traversal, to get the "volumic" dendrogram
        for (size_t i = 0; i < dendroNodes.size(); i++) {
          DendroNodeType &curNode = *dendroNodes[i];
          // we verify that the node is not the ancestor
          if (curNode.getLabel() != curNode.getFather()->getLabel()) {
            //      double height =
            double height =
                curNode.getFather()->getInternalNodeValuationInitial();
            double surface = curNode.getValuation();
            curNode.setValuation(height * surface);
          }
        }
        // Third dendroNodes traversal, to get the stochasticVolumic dendrogram
        for (size_t i = 0; i < dendroNodes.size(); i++) {
          DendroNodeType &curNode = *dendroNodes[i];
          // we verify that it's an internal node
          if (curNode.getIsInternalNode() == 1) {
            double childLeftVol  = curNode.getChildLeft()->getValuation();
            double childRightVol = curNode.getChildRight()->getValuation();

            double newVal =
                1 - pow(1 - std::min(childLeftVol, childRightVol) / totalVolume,
                        nParam);

            if (newVal <= 0) {
              newVal = 0.000000001;
            }
            curNode.setInternalNodeValuationFinal(newVal);
          }
        }
        this->putValuationsFinalInInitial();
        this->reorganize();
        this->setValuationsToZero();
      } else if (typeOfHierarchy == "stochasticVolumicImageReturn") {
        double totalSurface = 0;
        double totalDepth   = 0;

        // First dendroNodes traversal, to get the "surfacic" dendrogram
        for (size_t i = 0; i < dendroNodes.size(); i++) {
          DendroNodeType &curNode = *dendroNodes[i];
          // we verify that it's an internal node
          if (curNode.getIsInternalNode() == 1) {
            std::vector<MarkerLabelT> lookupChildLeft =
                curNode.getChildLeft()->getLookupProgeny();
            std::vector<MarkerLabelT> lookupChildRight =
                curNode.getChildRight()->getLookupProgeny();

            std::map<MarkerLabelT, MarkerLabelT> lookupMapChildLeft;
            for (MarkerLabelT j = 0; j < lookupChildLeft.size(); j++) {
              lookupMapChildLeft[j] = lookupChildLeft.at(j);
            }

            std::map<MarkerLabelT, MarkerLabelT> lookupMapChildRight;
            for (MarkerLabelT j = 0; j < lookupChildRight.size(); j++) {
              lookupMapChildRight[j] = lookupChildRight.at(j);
            }

            smil::Image<UINT8> imTmpLeft(imMosa);
            smil::Image<UINT8> imTmpRight(imMosa);

            applyLookup(imMosa, lookupMapChildLeft, imTmpLeft);
            applyLookup(imMosa, lookupMapChildRight, imTmpRight);

            if (typeOfTransform == "erode") {
              erode(imTmpLeft, imTmpLeft, se);
              erode(imTmpRight, imTmpRight, se);
            } else if (typeOfTransform == "dilate") {
              dilate(imTmpLeft, imTmpLeft, se);
              dilate(imTmpRight, imTmpRight, se);
            } else if (typeOfTransform == "open") {
              open(imTmpLeft, imTmpLeft, se);
              open(imTmpRight, imTmpRight, se);
            } else if (typeOfTransform == "close") {
              close(imTmpLeft, imTmpLeft, se);
              close(imTmpRight, imTmpRight, se);
            } else {
              cout << "Please choose typeOfTransform in the following: erode, "
                      "dilate, open, close"
                   << endl;
            }
            double childLeftSurf  = area(imTmpLeft);
            double childRightSurf = area(imTmpRight);
            curNode.setValuation(childLeftSurf + childRightSurf);
          }
        }

        DendroNodeType &curNodeTmp = *dendroNodes[1];
        // get the total surface of the domain
        totalSurface = curNodeTmp.getAncestor()->getValuation();
        // get the total depth of the domain
        totalDepth =
            curNodeTmp.getAncestor()->getInternalNodeValuationInitial();
        double totalVolume = totalSurface * totalDepth;
        // Second dendroNodes traversal, to get the "volumic"
        for (size_t i = 0; i < dendroNodes.size(); i++) {
          DendroNodeType &curNode = *dendroNodes[i];
          // we verify that the node is not the ancestor
          if (curNode.getLabel() != curNode.getFather()->getLabel()) {
            double height =
                curNode.getFather()->getInternalNodeValuationInitial();
            double surface = curNode.getValuation();
            curNode.setValuation(height * surface);
          }
        }
        // Third dendroNodes traversal, to get the stochasticVolumic dendrogram
        for (size_t i = 0; i < dendroNodes.size(); i++) {
          DendroNodeType &curNode = *dendroNodes[i];
          // we verify that it's an internal node
          if (curNode.getIsInternalNode() == 1) {
            double childLeftVol  = curNode.getChildLeft()->getValuation();
            double childRightVol = curNode.getChildRight()->getValuation();

            double newVal =
                1 - pow(1 - (childLeftVol / totalVolume), nParam) -
                pow(1 - (childRightVol / totalVolume), nParam) +
                pow(1 - (childLeftVol + childRightVol) / totalVolume, nParam);
            if (newVal <= 0) {
              newVal = 0.000000001;
            };
            curNode.setInternalNodeValuationFinal(newVal);
          }
        }
        this->putValuationsFinalInInitial();
        this->reorganize();
        this->setValuationsToZero();
      } else if (typeOfHierarchy == "marker") {
        // First dendroNodes traversal, to set the markers values
        for (size_t i = 0; i < dendroNodes.size(); i++) {
          DendroNodeType &curNode = *dendroNodes[i];
          // we verify that it's a leaf
          if (curNode.getIsInternalNode() == 0) {
            if (curNode.getMarker() == 0) {
              curNode.setMarker(curNode.getValuation());
            }
          }
        }
        // Second dendroNodes traversal, to get the "marker" dendrogram
        for (size_t i = 0; i < dendroNodes.size(); i++) {
          DendroNodeType &curNode = *dendroNodes[i];
          // we verify that it's an internal node
          if (curNode.getIsInternalNode() == 1) {
            double markerLeft  = curNode.getChildLeft()->getMarker();
            double markerRight = curNode.getChildRight()->getMarker();
            curNode.setMarker(fmax(markerLeft, markerRight));
          }
        }
        // Third dendroNodes traversal, to get the final
        // valuation of internal nodes
        for (size_t i = 0; i < dendroNodes.size(); i++) {
          DendroNodeType &curNode = *dendroNodes[i];
          if (curNode.getIsInternalNode() == 1) {
            double markerLeft  = curNode.getChildLeft()->getMarker();
            double markerRight = curNode.getChildRight()->getMarker();
            curNode.setInternalNodeValuationFinal(
                fmin(markerLeft, markerRight));
          }
        }
        this->putValuationsFinalInInitial();
        this->reorganize();
        this->setValuationsToZero();
      } else if (typeOfHierarchy == "stochasticSurfacicCountMarkers") {
        // First dendroNodes traversal, to get the "surfacic" dendrogram
        for (size_t i = 0; i < dendroNodes.size(); i++) {
          DendroNodeType &curNode = *dendroNodes[i];
          curNode.setValuation(curNode.getNbMarkersUnder());
        }

        DendroNodeType &curNodeTmp = *dendroNodes[1];
        // get the total surface of the domain
        double totalSurface = curNodeTmp.getAncestor()->getValuation();
        //  cout << "totalSurface = " << totalSurface << endl;
        // Second dendroNodes traversal, to get the stochasticSurfacic
        // dendrogram
        for (size_t i = 0; i < dendroNodes.size(); i++) {
          DendroNodeType &curNode = *dendroNodes[i];
          // we verify that it's an internal node
          if (curNode.getIsInternalNode() == 1) {
            double childLeftSurf  = curNode.getChildLeft()->getValuation();
            double childRightSurf = curNode.getChildRight()->getValuation();
            double newVal =
                1 - pow(1 - (childLeftSurf / totalSurface), nParam) -
                pow(1 - (childRightSurf / totalSurface), nParam) +
                pow(1 - ((childLeftSurf + childRightSurf) / totalSurface),
                    nParam);
            curNode.setInternalNodeValuationFinal(newVal);
          }
        }
        this->putValuationsFinalInInitial();
        this->reorganize();
        this->setValuationsToZero();
      } else if (typeOfHierarchy == "stochasticSurfacicSumMarkers") {
        // First dendroNodes traversal, to get the "surfacic" dendrogram
        for (size_t i = 0; i < dendroNodes.size(); i++) {
          DendroNodeType &curNode = *dendroNodes[i];
          curNode.setValuation(curNode.getNbMarkersUnder());
        }

        DendroNodeType &curNodeTmp = *dendroNodes[1];
        // get the total surface of the domain
        double totalSurface = curNodeTmp.getAncestor()->getValuation();
        // Second dendroNodes traversal, to get the stochasticSurfacic
        // dendrogram
        for (size_t i = 0; i < dendroNodes.size(); i++) {
          DendroNodeType &curNode = *dendroNodes[i];
          // we verify that it's an internal node
          if (curNode.getIsInternalNode() == 1) {
            double childLeftSurf  = curNode.getChildLeft()->getValuation();
            double childRightSurf = curNode.getChildRight()->getValuation();
            double newVal =
                1 - pow(1 - (childLeftSurf / totalSurface), nParam) -
                pow(1 - (childRightSurf / totalSurface), nParam) +
                pow(1 - ((childLeftSurf + childRightSurf) / totalSurface),
                    nParam);
            curNode.setInternalNodeValuationFinal(newVal);
          }
        }
        this->putValuationsFinalInInitial();
        this->reorganize();
        this->setValuationsToZero();
      } else if (typeOfHierarchy == "stochasticVolumicCountMarkers") {
        // First dendroNodes traversal, to get the "surfacic" dendrogram
        for (size_t i = 0; i < dendroNodes.size(); i++) {
          DendroNodeType &curNode = *dendroNodes[i];
          curNode.setValuation(curNode.getNbMarkersUnder());
        }
        double          totalSurface = 0;
        DendroNodeType &curNodeTmp   = *dendroNodes[1];
        // get the total surface of the domain
        totalSurface = curNodeTmp.getAncestor()->getValuation();
        // Second dendroNodes traversal, to get the "volumic" dendrogram
        for (size_t i = 0; i < dendroNodes.size(); i++) {
          DendroNodeType &curNode = *dendroNodes[i];
          // we verify that the node is not the ancestor
          if (curNode.getLabel() != curNode.getFather()->getLabel()) {
            //      double height =
            // curNode.getFather()->getInternalNodeValuationInitial();
            double height =
                curNode.getFather()->getInternalNodeValuationInitial();
            double surface = curNode.getValuation();
            //      curNode.setValuation(surface);
            curNode.setValuation(height * surface);
          }
        }

        // Second dendroNodes traversal, to get the stochasticSurfacic
        // dendrogram
        for (size_t i = 0; i < dendroNodes.size(); i++) {
          DendroNodeType &curNode = *dendroNodes[i];
          // we verify that it's an internal node
          if (curNode.getIsInternalNode() == 1) {
            double childLeftSurf  = curNode.getChildLeft()->getValuation();
            double childRightSurf = curNode.getChildRight()->getValuation();
            double newVal =
                1 - pow(1 - (childLeftSurf / totalSurface), nParam) -
                pow(1 - (childRightSurf / totalSurface), nParam) +
                pow(1 - ((childLeftSurf + childRightSurf) / totalSurface),
                    nParam);
            curNode.setInternalNodeValuationFinal(newVal);
          }
        }
        this->putValuationsFinalInInitial();
        this->reorganize();
        this->setValuationsToZero();
      } else if (typeOfHierarchy == "waterfall") {
        // dendroNodes traversal, to set the waterfall and diameters values
        for (size_t i = 0; i < dendroNodes.size(); i++) {
          DendroNodeType &curNode = *dendroNodes[i];
          // we verify that it's an internal node
          if (curNode.getIsInternalNode() == 1) {
            if (curNode.getValuation() == 0) {
              double childLeftValuation =
                  curNode.getChildLeft()->getValuation();
              double childRightValuation =
                  curNode.getChildRight()->getValuation();
              double childLeftDiameter =
                  curNode.getChildLeft()->getInternalNodeValuationFinal();
              double childRightDiameter =
                  curNode.getChildRight()->getInternalNodeValuationFinal();

              double newVal = 1 + min(childLeftValuation, childRightValuation);
              curNode.setValuation(newVal);
              curNode.setInternalNodeValuationFinal(
                  max(newVal, max(childLeftDiameter, childRightDiameter)));
            }
          }
        }
        this->putValuationsFinalInInitial();
        this->reorganize();
        this->setValuationsToZero();
      } else if (typeOfHierarchy == "persistence") {
        for (size_t i = 0; i < dendroNodes.size(); i++) {
          DendroNodeType &curNode = *dendroNodes[i];
          // we verify that it's an internal node
          if (curNode.getIsInternalNode() == 1) {
            double childLeftINValuation =
                curNode.getChildLeft()->getInternalNodeValuationInitial();
            double childRightINValuation =
                curNode.getChildRight()->getInternalNodeValuationInitial();
            double INValuation = curNode.getInternalNodeValuationInitial();

            double newVal = INValuation - std::max(childLeftINValuation,
                                                   childRightINValuation);
            curNode.setInternalNodeValuationFinal(newVal);
          }
        }
        this->putValuationsFinalInInitial();
        this->reorganize();
        this->setValuationsToZero();
      } else {
        cout
            << "void "
               "Dendrogram::HierarchicalConstruction(Dendrogram& "
               "dendrogram,std::string typeOfHierarchy) \n"
            << "Please choose one of the following hierarchies: surfacic, "
               "volumic, stochasticSurfacic, stochasticVolumic, "
            << "surfacicImageReturn, volumicImageReturn, "
               "stochasticSurfacicImageReturn, stochasticVolumicImageReturn "
            << ",stochasticSurfacicCountMarkers,stochasticVolumicCountMarkers, "
            << "stochasticExtinctionSurfacic, stochasticExtinctionVolumic, "
            << "stochasticExtinctionSurfacicImageReturn, "
            << "waterfall,marker,persistence" << endl;
      }
    };

    //! Computes a new hierarchy from a given dendrogram hierarchy
    void EnergyConstruction(const std::string typeOfHierarchy)
    {
      // ,double lamb
      // sort by increasing values of internalNodeValuationInitial
      this->sortNodes();

      std::vector<DendroNodeType *> &dendroNodes = this->getDendroNodes();

      if (typeOfHierarchy == "none") {
      } else if (typeOfHierarchy == "simplifiedMumfordShah") {
        for (size_t i = 0; i < dendroNodes.size();
             i++) { // dendroNodes traversal
          DendroNodeType &curNode = *dendroNodes[i];
          if (curNode.getIsInternalNode() == 1) {
            std::vector<double> childLeftMoments =
                curNode.getChildLeft()->getMoments();
            std::vector<double> childRightMoments =
                curNode.getChildRight()->getMoments();
            std::vector<double> moments = curNode.getMoments();

            double contoursValue = curNode.getContoursSize();
            double contoursValueLeft =
                curNode.getChildLeft()->getContoursSize();
            double contoursValueRight =
                curNode.getChildRight()->getContoursSize();

            double fidelityTerm =
                moments.at(2) - (moments.at(1) * moments.at(1)) / moments.at(0);
            double fidelityTermLeft =
                childLeftMoments.at(2) -
                (childLeftMoments.at(1) * childLeftMoments.at(1)) /
                    childLeftMoments.at(0);
            double fidelityTermRight =
                childRightMoments.at(2) -
                (childRightMoments.at(1) * childRightMoments.at(1)) /
                    childRightMoments.at(0);

            double newLamb =
                (fidelityTerm - fidelityTermLeft - fidelityTermRight) /
                (contoursValueLeft + contoursValueRight - contoursValue);

            curNode.setInternalNodeValuationFinal(newLamb);
            curNode.setValuation(newLamb);
          }
        }
        this->putValuationsFinalInInitial();
        this->reorganize();

        // We compute the energies with the new organization of nodes obtained
        // with the reorganize function

        // sort nodes by decreasing values of
        // internalNodeValuationInitial
        this->sortNodes(true);
        // initialization of contoursValue and fidelityValue
        double              contoursValue      = 0.0;
        DendroNodeType &    highestNode        = *dendroNodes[0];
        std::vector<double> momentsHighestNode = highestNode.getMoments();
        double              fidelityValue =
            momentsHighestNode.at(2) -
            (momentsHighestNode.at(1) * momentsHighestNode.at(1)) /
                momentsHighestNode.at(0);

        // dendroNodes traversal
        for (size_t i = 0; i < dendroNodes.size(); i++) {
          DendroNodeType &curNode = *dendroNodes[i];
          // we verify that it's an internal node
          if (curNode.getIsInternalNode() == 1) {
            std::vector<double> moments = curNode.getMoments();
            std::vector<double> momentsLeft =
                curNode.getChildLeft()->getMoments();
            std::vector<double> momentsRight =
                curNode.getChildRight()->getMoments();

            double fidelityTerm =
                moments.at(2) - (moments.at(1) * moments.at(1)) / moments.at(0);
            double fidelityTermLeft =
                momentsLeft.at(2) -
                (momentsLeft.at(1) * momentsLeft.at(1)) / momentsLeft.at(0);
            double fidelityTermRight =
                momentsRight.at(2) -
                (momentsRight.at(1) * momentsRight.at(1)) / momentsRight.at(0);

            double Lamb = curNode.getInternalNodeValuationInitial();

            contoursValue = contoursValue + curNode.getContoursSize();
            fidelityValue = fidelityValue - fidelityTerm + fidelityTermLeft +
                            fidelityTermRight;

            double newEnergy = fidelityValue + Lamb * contoursValue;
            curNode.setEnergy(newEnergy);
          }
        }

        this->setValuationsToZero();
      }

      else if (typeOfHierarchy == "varianceEnergy") {
        for (size_t i = 0; i < dendroNodes.size();
             i++) { // dendroNodes traversal
          DendroNodeType &curNode = *dendroNodes[i];
          if (curNode.getIsInternalNode() == 1) {
            std::vector<double> childLeftMoments =
                curNode.getChildLeft()->getMoments();
            std::vector<double> childRightMoments =
                curNode.getChildRight()->getMoments();
            std::vector<double> moments = curNode.getMoments();

            double fidelityTerm =
                moments.at(2) - (moments.at(1) * moments.at(1)) / moments.at(0);
            double fidelityTermLeft =
                childLeftMoments.at(2) -
                (childLeftMoments.at(1) * childLeftMoments.at(1)) /
                    childLeftMoments.at(0);
            double fidelityTermRight =
                childRightMoments.at(2) -
                (childRightMoments.at(1) * childRightMoments.at(1)) /
                    childRightMoments.at(0);

            double newLamb =
                (fidelityTerm - fidelityTermLeft - fidelityTermRight);

            curNode.setInternalNodeValuationFinal(newLamb);
            curNode.setValuation(newLamb);
          }
        }
        this->putValuationsFinalInInitial();
        this->reorganize();

        // We compute the energies with the new organization of nodes obtained
        // with the reorganize function

        // sort nodes by decreasing values of
        // internalNodeValuationInitial
        this->sortNodes(true);
        // initialization of contoursValue and fidelityValue
        double              contoursValue      = 0.0;
        DendroNodeType &    highestNode        = *dendroNodes[0];
        std::vector<double> momentsHighestNode = highestNode.getMoments();
        double              fidelityValue =
            momentsHighestNode.at(2) -
            (momentsHighestNode.at(1) * momentsHighestNode.at(1)) /
                momentsHighestNode.at(0);

        // dendroNodes traversal
        for (size_t i = 0; i < dendroNodes.size(); i++) {
          DendroNodeType &curNode = *dendroNodes[i];
          // we verify that it's an internal node
          if (curNode.getIsInternalNode() == 1) {
            std::vector<double> moments = curNode.getMoments();
            std::vector<double> momentsLeft =
                curNode.getChildLeft()->getMoments();
            std::vector<double> momentsRight =
                curNode.getChildRight()->getMoments();

            double fidelityTerm =
                moments.at(2) - (moments.at(1) * moments.at(1)) / moments.at(0);
            double fidelityTermLeft =
                momentsLeft.at(2) -
                (momentsLeft.at(1) * momentsLeft.at(1)) / momentsLeft.at(0);
            double fidelityTermRight =
                momentsRight.at(2) -
                (momentsRight.at(1) * momentsRight.at(1)) / momentsRight.at(0);

            double Lamb = curNode.getInternalNodeValuationInitial();

            contoursValue = contoursValue + curNode.getContoursSize();
            fidelityValue = fidelityValue - fidelityTerm + fidelityTermLeft +
                            fidelityTermRight;

            double newEnergy = fidelityValue + Lamb * contoursValue;
            curNode.setEnergy(newEnergy);
          }
        }

        this->setValuationsToZero();
      }

      else {
        cout << "void Dendrogram::EnergyConstruction(Dendrogram& "
                "dendrogram,std::string typeOfHierarchy) \n"
             << "Please choose one of the following hierarchies: "
                "simplifiedMumfordShah, varianceEnergy "
             << endl;
      }
    }

    //! Setters and Getters
    std::vector<DendroNodeType *> &getDendroNodes()
    {
      return dendroNodes;
    };

    //     DendroNodeType*
    //     &getDendroNodeLabel(std::vector<DendroNodeType*>
    //     dendroNodes, MarkerLabelT label){
    //       DendroNodeType &curNode = *dendroNodes[0];
    //       for (size_t i = 0 ; i<dendroNodes.size() ; i++){
    //   &curNode = *dendroNodes[i];
    //   if (curNode.getLabel() == label){
    //     break;
    //  }
    //       }
    //       return &curNode;
    //     }
    void setNbrNodes(size_t nNbrNodes)
    {
      nbrNodes = nNbrNodes;
    }
    size_t getNbrNodes()
    {
      return nbrNodes;
    }
    void setNbrMarkers(size_t nNbrMarkers)
    {
      nbrMarkers = nNbrMarkers;
    }
    size_t getNbrMarkers()
    {
      return nbrMarkers;
    }

    /** Access a value of a node in the dendrogram
     *
     *
     * @param[in] nodeIndex Index of the node to access
     * @param[in] nameOfValueWanted Name of the value to access ("valuation",
     * "internalNodeValuationInitial", "label" etc.)
     */
    double getNodeValue(size_t nodeIndex, string nameOfValueWanted)
    {
      DendroNodeType &curNode = *dendroNodes.at(nodeIndex);
      if (nameOfValueWanted == "valuation") {
        return curNode.getValuation();
      } else if (nameOfValueWanted == "internalNodeValuationInitial") {
        return curNode.getInternalNodeValuationInitial();
      } else if (nameOfValueWanted == "internalNodeValuationFinal") {
        return curNode.getInternalNodeValuationFinal();
      } else if (nameOfValueWanted == "label") {
        return curNode.getLabel();
      } else if (nameOfValueWanted == "labelChildRight") {
        return curNode.getChildRight()->getLabel();
      } else if (nameOfValueWanted == "labelChildLeft") {
        return curNode.getChildLeft()->getLabel();
      } else if (nameOfValueWanted == "labelNeighborRight") {
        return curNode.getNeighborRight()->getLabel();
      } else if (nameOfValueWanted == "labelNeighborLeft") {
        return curNode.getNeighborLeft()->getLabel();
      } else if (nameOfValueWanted == "labelFather") {
        return curNode.getFather()->getLabel();
      } else if (nameOfValueWanted == "isInternalNode") {
        return curNode.getIsInternalNode();
      } else if (nameOfValueWanted == "marker") {
        return curNode.getMarker();
      } else if (nameOfValueWanted == "energy") {
        return curNode.getEnergy();
      } else if (nameOfValueWanted == "contours") {
        return curNode.getContoursSize();
      } else {
        cout
            << "int Dendrogram::getNodeValue(int nodeIndex,string "
               "nameOfValueWanted) -> nameOfValueWanted must be chosen in this "
               "list: valuation, internalNodeValuationInitial "
            << ", internalNodeValuationInitial, label, marker." << endl;
        return 0.0;
      }
    };

    /** Manually modify a value of a node in the dendrogram
     *
     *
     * @param[in] nodeIndex Index of the node to modify
     * @param[in] nameOfValueWanted Name of the value to modify ("valuation",
     * "internalNodeValuationInitial", "label" etc.)
     * @param[in] value New value
     */
    void setNodeValue(size_t nodeIndex, string nameOfValueWanted, double value)
    {
      DendroNodeType &curNode = *dendroNodes.at(nodeIndex);
      if (nameOfValueWanted == "valuation") {
        curNode.setValuation(value);
      } else if (nameOfValueWanted == "internalNodeValuationInitial") {
        curNode.setInternalNodeValuationInitial(value);
      } else if (nameOfValueWanted == "internalNodeValuationFinal") {
        curNode.setInternalNodeValuationFinal(value);
      } else if (nameOfValueWanted == "label") {
        curNode.setLabel(value);
      } else if (nameOfValueWanted == "labelChildRight") {
        curNode.getChildRight()->setLabel(value);
      } else if (nameOfValueWanted == "labelChildLeft") {
        curNode.getChildLeft()->setLabel(value);
      } else if (nameOfValueWanted == "labelNeighborRight") {
        curNode.getNeighborRight()->setLabel(value);
      } else if (nameOfValueWanted == "labelNeighborLeft") {
        curNode.getNeighborLeft()->setLabel(value);
      } else if (nameOfValueWanted == "labelFather") {
        curNode.getFather()->setLabel(value);
      } else if (nameOfValueWanted == "isInternalNode") {
        curNode.setIsInternalNode(value);
      } else if (nameOfValueWanted == "marker") {
        curNode.setMarker(value);
      } else {
        cout << "int Dendrogram::setNodeValue(int nodeIndex,string "
                "nameOfValueWanted, double value) -> nameOfValueWanted must be "
                "chosen in this list: valuation, internalNodeValuationInitial "
             << ", internalNodeValuationInitial, label, marker." << endl;
      }
    };

    /** Get the lookup of the progeny of a node
     *
     *
     * @param[in] nodeIndex Index of the node to access
     * @param[in] nameOfValueWanted Name of the exact node ("current",
     * "childLeft", "childRight")
     */
    std::vector<MarkerLabelT> getLookupProgeny(size_t nodeIndex,
                                               string nameOfValueWanted)
    {
      DendroNodeType &          curNode = *dendroNodes.at(nodeIndex);
      std::vector<MarkerLabelT> lookupToReturn;
      if (nameOfValueWanted == "current") {
        lookupToReturn = curNode.getLookupProgeny();
      } else if (nameOfValueWanted == "childLeft") {
        lookupToReturn = curNode.getChildLeft()->getLookupProgeny();
      } else if (nameOfValueWanted == "childRight") {
        lookupToReturn = curNode.getChildRight()->getLookupProgeny();
      } else {
        cout << "std::vector<MarkerLabelT> getLookupProgeny(size_t "
                "nodeIndex,string nameOfValueWanted) -> nameOfValueWanted must "
                "be chosen in this list: current, "
             << ", childLeft, childRight." << endl;
      }
      return lookupToReturn;
    }

    /** Get the lookup of the progeny below a certain ultrametric threshold
     *
     *
     * @param[in] thresh Ultrametric value threshold
     * @param[in] nameOfValueWanted Name of the exact node ("current",
     * "childLeft", "childRight")
     */
    std::vector<MarkerLabelT> getThreshLookupProgeny(double thresh,
                                                     string nameOfValueWanted)
    {
      std::vector<MarkerLabelT> lookupToReturn;
      this->sortNodes(
          "true"); // sort nodes by decreasing internalNodeValuationInitial
      if (thresh >= 0) {
        if (nameOfValueWanted == "current") {
          for (MarkerLabelT i = 0; i < nbrNodes; i++) {
            DendroNodeType &curNode = *dendroNodes.at(i);
            if (curNode.getInternalNodeValuationInitial() <= thresh) {
              lookupToReturn = curNode.getLookupProgeny();
              break;
            }
          }
        }

        else if (nameOfValueWanted == "childLeft") {
          for (MarkerLabelT i = 0; i < nbrNodes; i++) {
            DendroNodeType &curNode = *dendroNodes.at(i);
            if (curNode.getInternalNodeValuationInitial() <= thresh) {
              lookupToReturn = curNode.getChildLeft()->getLookupProgeny();
              break;
            }
          }
        }

        else if (nameOfValueWanted == "childRight") {
          for (MarkerLabelT i = 0; i < nbrNodes; i++) {
            DendroNodeType &curNode = *dendroNodes.at(i);
            if (curNode.getInternalNodeValuationInitial() <= thresh) {
              lookupToReturn = curNode.getChildRight()->getLookupProgeny();
              break;
            }
          }
        } else {
          cout << "std::vector<MarkerLabelT> "
                  "getThreshLookupProgeny(thresh,nameOfValueWanted) -> "
                  "nameOfValueWanted must be chosen in this list: current, "
               << ", childLeft, childRight." << endl;
        }
      } else {
        cout << "getThreshLookupProgeny(thresh,nameOfValueWanted) "
             << "-> the thresh specified must be higher than 0" << endl;
      }

      return lookupToReturn;
    }

    /** Compute persistences matrix (persistence measures only the order of
     * fusion)
     *
     *
     * @param[in] typeOfMatrix "sparse" (only between adjacent regions) or
     * "full"
     * @param[in] momentOfLife "death" (when regions fuse) or "life" (when
     * regions appear)
     */
    std::vector<std::vector<double>>
    getPersistences(string typeOfMatrix = "sparse",
                    string momentOfLife = "death")
    {
      size_t nbrRegions = nbrNodes / 2 + 1;
      // persistences matrix
      std::vector<std::vector<double>> persistences(
          nbrRegions, vector<double>(nbrRegions, -1.0));

      if (typeOfMatrix == "sparse" && momentOfLife == "death") {
        // First dendrogram traversal to initialize the links between nodes in
        // the persistences matrix (we put the value to -0.5 when there is a
        // link)
        for (MarkerLabelT i = 0; i < nbrNodes; i++) {
          DendroNodeType &curNode = *dendroNodes.at(i);
          // we check if it is a leaf
          if (curNode.getIsInternalNode() == 0) {
            MarkerLabelT         label1        = curNode.getLabel();
            vector<MarkerLabelT> contoursCount = curNode.getContoursCount();
            for (MarkerLabelT t = 0; t < contoursCount.size(); t++) {
              MarkerLabelT label2 = t;
              if (contoursCount.at(t) != 0) {
                if (persistences[label1][label2] == -1) {
                  persistences[label1][label2] = -0.5;
                  persistences[label2][label1] = -0.5;
                }
              }
            }
          }
        }
        // sort nodes by decreasing internalNodeValuationInitial
        this->sortNodes(true);
        // Second dendrogram traversal by decreasing
        // internalNodeValuationInitial to complete the persistences matrix
        MarkerLabelT count = 0;
        for (MarkerLabelT i = 0; i < nbrNodes; i++) {
          DendroNodeType &curNode = *dendroNodes.at(i);
          if (curNode.getIsInternalNode() == 1) {
            std::vector<MarkerLabelT> lookupProgenyChildLeft =
                curNode.getChildLeft()->getLookupProgeny();
            std::vector<MarkerLabelT> lookupProgenyChildRight =
                curNode.getChildRight()->getLookupProgeny();
            for (MarkerLabelT p = 0; p < lookupProgenyChildLeft.size(); p++) {
              for (MarkerLabelT q = 0; q < lookupProgenyChildRight.size();
                   q++) {
                if (lookupProgenyChildLeft.at(p) == 1 &&
                    lookupProgenyChildRight.at(q) == 1 &&
                    persistences[p][q] == -0.5) {
                  persistences[p][q] = nbrRegions - count;
                  persistences[q][p] = nbrRegions - count;
                }
              }
            }
            count++;
          }
        }
        // we set negative values of the matrix to zero
        for (size_t i = 0; i < persistences.size(); i++) {
          for (size_t j = 0; j < persistences[i].size(); j++)
            if (persistences[i][j] <= 0.0) {
              persistences[i][j] = 0.0;
              persistences[j][i] = 0.0;
            }
        }
      } // end if "sparse" and "death"

      else if (typeOfMatrix == "full" && momentOfLife == "death") {
        persistences = std::vector<std::vector<double>>(
            nbrRegions, vector<double>(nbrRegions, -0.5));
        // sort nodes by decreasing internalNodeValuationInitial
        this->sortNodes(true);
        // Second dendrogram traversal by decreasing
        // internalNodeValuationInitial to complete the persistences matrix
        MarkerLabelT count = 0;
        for (MarkerLabelT i = 0; i < nbrNodes; i++) {
          DendroNodeType &curNode = *dendroNodes.at(i);
          if (curNode.getIsInternalNode() == 1) {
            std::vector<MarkerLabelT> lookupProgenyChildLeft =
                curNode.getChildLeft()->getLookupProgeny();
            std::vector<MarkerLabelT> lookupProgenyChildRight =
                curNode.getChildRight()->getLookupProgeny();
            for (MarkerLabelT p = 0; p < lookupProgenyChildLeft.size(); p++) {
              for (MarkerLabelT q = 0; q < lookupProgenyChildRight.size();
                   q++) {
                if (lookupProgenyChildLeft.at(p) == 1 &&
                    lookupProgenyChildRight.at(q) == 1 &&
                    persistences[p][q] == -0.5) {
                  persistences[p][q] = nbrRegions - count;
                  persistences[q][p] = nbrRegions - count;
                }
              }
            }
            count++;
          }
        }
        // we set negative values of the matrix to zero
        for (size_t i = 0; i < persistences.size(); i++) {
          for (size_t j = 0; j < persistences[i].size(); j++)
            if (persistences[i][j] <= 0.0) {
              persistences[i][j] = 0.0;
              persistences[j][i] = 0.0;
            }
        }
      } // end if "full" and "death"
      else if (typeOfMatrix == "sparse" && momentOfLife == "life") {
        // First dendrogram traversal initialize the links between nodes in the
        // persistences matrix
        // (we put the value to -0.5 when there is a link)
        for (MarkerLabelT i = 0; i < nbrNodes; i++) {
          DendroNodeType &curNode = *dendroNodes.at(i);
          if (curNode.getIsInternalNode() == 0) { // we check if it is a leaf
            MarkerLabelT         label1        = curNode.getLabel();
            vector<MarkerLabelT> contoursCount = curNode.getContoursCount();
            for (MarkerLabelT t = 0; t < contoursCount.size(); t++) {
              MarkerLabelT label2 = t;
              if (contoursCount.at(t) != 0) {
                if (persistences[label1][label2] == -1) {
                  persistences[label1][label2] = 1;
                  persistences[label2][label1] = 1;
                }
              }
            }
          }
        }
        // we set negative values of the matrix to zero
        for (size_t i = 0; i < persistences.size(); i++) {
          for (size_t j = 0; j < persistences[i].size(); j++)
            if (persistences[i][j] <= 0.0) {
              persistences[i][j] = 0.0;
              persistences[j][i] = 0.0;
            }
        }
      } // end if "sparse" and "life"
      else if (typeOfMatrix == "full" && momentOfLife == "life") {
        // we begin by generating the same matrix as in the "sparse"/"death"
        // version
        // First dendrogram traversal to initialize the links between nodes in
        // the persistences matrix (we put the value to -0.5 when there is a
        // link)
        for (MarkerLabelT i = 0; i < nbrNodes; i++) {
          DendroNodeType &curNode = *dendroNodes.at(i);
          if (curNode.getIsInternalNode() == 0) { // we check if it is a leaf
            MarkerLabelT         label1        = curNode.getLabel();
            vector<MarkerLabelT> contoursCount = curNode.getContoursCount();
            for (MarkerLabelT t = 0; t < contoursCount.size(); t++) {
              MarkerLabelT label2 = t;
              if (contoursCount.at(t) != 0) {
                if (persistences[label1][label2] == -1) {
                  persistences[label1][label2] = -0.5;
                  persistences[label2][label1] = -0.5;
                }
              }
            }
          }
        }
        // sort nodes by decreasing internalNodeValuationInitial
        this->sortNodes(true);
        // Second dendrogram traversal by decreasing
        // internalNodeValuationInitial to complete the persistences matrix
        MarkerLabelT count = 0;
        for (MarkerLabelT i = 0; i < nbrNodes; i++) {
          DendroNodeType &curNode = *dendroNodes.at(i);
          if (curNode.getIsInternalNode() == 1) {
            std::vector<MarkerLabelT> lookupProgenyChildLeft =
                curNode.getChildLeft()->getLookupProgeny();
            std::vector<MarkerLabelT> lookupProgenyChildRight =
                curNode.getChildRight()->getLookupProgeny();
            for (MarkerLabelT p = 0; p < lookupProgenyChildLeft.size(); p++) {
              for (MarkerLabelT q = 0; q < lookupProgenyChildRight.size();
                   q++) {
                if (lookupProgenyChildLeft.at(p) == 1 &&
                    lookupProgenyChildRight.at(q) == 1 &&
                    persistences[p][q] == -0.5) {
                  persistences[p][q] = nbrRegions - count;
                  persistences[q][p] = nbrRegions - count;
                }
              }
            }
            count++;
          }
        }
        // we set negative values of the matrix to zero
        for (size_t i = 0; i < persistences.size(); i++) {
          for (size_t j = 0; j < persistences[i].size(); j++)
            if (persistences[i][j] <= 0.0) {
              persistences[i][j] = 0.0;
              persistences[j][i] = 0.0;
            }
        }

        // Now that we have the "sparse"/"death" matrix, we apply a recursive
        // lexicographic-distance-based
        // algorithm to infer the desired matrix
        std::vector<std::vector<double>> persistencesInit(persistences);
        std::vector<std::vector<double>> persistencesTmp(
            nbrRegions, vector<double>(nbrRegions, 0.0));

        // recursive max-min term-by-term product of the persistences matrix
        // until equilibrium state
        while (persistences != persistencesTmp) {
          persistencesTmp = persistences;
          for (size_t i = 0; i < nbrRegions; i++) {
            for (size_t j = i; j < nbrRegions; j++) {
              std::vector<double> minVec(nbrRegions, 0.0);
              for (size_t k = 0; k < nbrRegions; k++) {
                minVec[k] = min(persistences[i][k], persistencesInit[k][j]);
              }
              persistences[i][j] = *max_element(minVec.begin(), minVec.end());
              persistences[j][i] = persistences[i][j];
            }
          }
        }
      } // end if "full" and "life"
      else {
        cout << "getPersistences(string typeOfMatrix = ''sparse'',string "
                "momentOfLife = ''death'') -> please choose "
             << "typeOfMatrix in the following : ''sparse'', ''full''; and "
                "momentOfLife in the following : ''death'',''life'' "
             << endl;
      }

      return persistences;
    } // end getPersistences

    /** Compute saliences matrix (salience expresses ultrametric values)
     *
     *
     * @param[in] typeOfMatrix "sparse" (only between adjacent regions) or
     * "full"
     * @param[in] momentOfLife "death" (when regions fuse) or "life" (when
     * regions appear)
     */
    std::vector<std::vector<double>>
    getSaliences(string typeOfMatrix = "sparse", string momentOfLife = "death")
    {
      size_t nbrRegions = nbrNodes / 2 + 1;
      // saliences matrix
      std::vector<std::vector<double>> saliences(
          nbrRegions, vector<double>(nbrRegions, -1.0));

      if (typeOfMatrix == "sparse" && momentOfLife == "death") {
        // First dendrogram traversal initialize the links between nodes in the
        // saliences matrix (we put the value to -0.5 when there is a link)
        for (MarkerLabelT i = 0; i < nbrNodes; i++) {
          DendroNodeType &curNode = *dendroNodes.at(i);
          if (curNode.getIsInternalNode() == 0) { // we check if it is a leaf
            MarkerLabelT         label1        = curNode.getLabel();
            vector<MarkerLabelT> contoursCount = curNode.getContoursCount();
            for (MarkerLabelT t = 0; t < contoursCount.size(); t++) {
              MarkerLabelT label2 = t;
              if (contoursCount.at(t) != 0) {
                if (saliences[label1][label2] == -1) {
                  saliences[label1][label2] = -0.5;
                  saliences[label2][label1] = -0.5;
                }
              }
            }
          }
        }
        // sort nodes by decreasing internalNodeValuationInitial
        this->sortNodes(true);
        // Second dendrogram traversal by decreasing
        // internalNodeValuationInitial to complete the saliences matrix
        for (MarkerLabelT i = 0; i < nbrNodes; i++) {
          DendroNodeType &curNode = *dendroNodes.at(i);
          if (curNode.getIsInternalNode() == 1) {
            std::vector<MarkerLabelT> lookupProgenyChildLeft =
                curNode.getChildLeft()->getLookupProgeny();
            std::vector<MarkerLabelT> lookupProgenyChildRight =
                curNode.getChildRight()->getLookupProgeny();
            for (MarkerLabelT p = 0; p < lookupProgenyChildLeft.size(); p++) {
              for (MarkerLabelT q = 0; q < lookupProgenyChildRight.size();
                   q++) {
                if (lookupProgenyChildLeft.at(p) == 1 &&
                    lookupProgenyChildRight.at(q) == 1 &&
                    saliences[p][q] == -0.5) {
                  saliences[p][q] = curNode.getInternalNodeValuationInitial();
                  saliences[q][p] = curNode.getInternalNodeValuationInitial();
                }
              }
            }
          }
        }

        // we set the non positive values of the matrix to zero
        for (size_t i = 0; i < saliences.size(); i++) {
          for (size_t j = 0; j < saliences[i].size(); j++)
            if (saliences[i][j] <= 0.0) {
              saliences[i][j] = 0.0;
              saliences[j][i] = 0.0;
            }
        }
      } // end if "sparse" && "death"
      else if (typeOfMatrix == "full" && momentOfLife == "death") {
        saliences = std::vector<std::vector<double>>(
            nbrRegions, vector<double>(nbrRegions, -0.5));

        // sort nodes by decreasing internalNodeValuationInitial
        this->sortNodes(true);
        // Second dendrogram traversal by decreasing
        // internalNodeValuationInitial to complete the saliences matrix
        for (MarkerLabelT i = 0; i < nbrNodes; i++) {
          DendroNodeType &curNode = *dendroNodes.at(i);
          if (curNode.getIsInternalNode() == 1) {
            std::vector<MarkerLabelT> lookupProgenyChildLeft =
                curNode.getChildLeft()->getLookupProgeny();
            std::vector<MarkerLabelT> lookupProgenyChildRight =
                curNode.getChildRight()->getLookupProgeny();
            for (MarkerLabelT p = 0; p < lookupProgenyChildLeft.size(); p++) {
              for (MarkerLabelT q = 0; q < lookupProgenyChildRight.size();
                   q++) {
                if (lookupProgenyChildLeft.at(p) == 1 &&
                    lookupProgenyChildRight.at(q) == 1 &&
                    saliences[p][q] == -0.5) {
                  saliences[p][q] = curNode.getInternalNodeValuationInitial();
                  saliences[q][p] = curNode.getInternalNodeValuationInitial();
                }
              }
            }
          }
        }

        // we set the non positive values of the matrix to zero
        for (size_t i = 0; i < saliences.size(); i++) {
          for (size_t j = 0; j < saliences[i].size(); j++)
            if (saliences[i][j] <= 0.0) {
              saliences[i][j] = 0.0;
              saliences[j][i] = 0.0;
            }
        }
      } // end if "full" && "death"
      else if (typeOfMatrix == "sparse" && momentOfLife == "life") {
        // First dendrogram traversal initialize the links between nodes in the
        // saliences matrix (we put the value to -0.5 when there is a link)
        for (MarkerLabelT i = 0; i < nbrNodes; i++) {
          DendroNodeType &curNode = *dendroNodes.at(i);
          if (curNode.getIsInternalNode() == 0) { // we check if it is a leaf
            MarkerLabelT         label1        = curNode.getLabel();
            vector<MarkerLabelT> contoursCount = curNode.getContoursCount();
            for (MarkerLabelT t = 0; t < contoursCount.size(); t++) {
              MarkerLabelT label2 = t;
              if (contoursCount.at(t) != 0) {
                if (saliences[label1][label2] == -1) {
                  saliences[label1][label2] = 0.01;
                  saliences[label2][label1] = 0.01;
                }
              }
            }
          }
        }
        // we set the non positive values of the matrix to zero
        for (size_t i = 0; i < saliences.size(); i++) {
          for (size_t j = 0; j < saliences[i].size(); j++)
            if (saliences[i][j] <= 0.0) {
              saliences[i][j] = 0.0;
              saliences[j][i] = 0.0;
            }
        }

      } // end if "sparse" && "life"
      else if (typeOfMatrix == "full" && momentOfLife == "life") {
        saliences = std::vector<std::vector<double>>(
            nbrRegions, vector<double>(nbrRegions, -0.5));
        // we begin by generating the same matrix as in the "sparse"/"death"
        // version
        // First dendrogram traversal initialize the links between nodes in the
        // saliences matrix
        //(we put the value to -0.5 when there is a link)
        for (MarkerLabelT i = 0; i < nbrNodes; i++) {
          DendroNodeType &curNode = *dendroNodes.at(i);
          if (curNode.getIsInternalNode() == 0) { // we check if it is a leaf
            MarkerLabelT         label1        = curNode.getLabel();
            vector<MarkerLabelT> contoursCount = curNode.getContoursCount();
            for (MarkerLabelT t = 0; t < contoursCount.size(); t++) {
              MarkerLabelT label2 = t;
              if (contoursCount.at(t) != 0) {
                if (saliences[label1][label2] == -1) {
                  saliences[label1][label2] = -0.5;
                  saliences[label2][label1] = -0.5;
                }
              }
            }
          }
        }

        // sort nodes by decreasing internalNodeValuationInitial
        this->sortNodes(true);
        // Second dendrogram traversal by decreasing
        // internalNodeValuationInitial to complete the saliences matrix
        for (MarkerLabelT i = 0; i < nbrNodes; i++) {
          DendroNodeType &curNode = *dendroNodes.at(i);
          if (curNode.getIsInternalNode() == 1) {
            std::vector<MarkerLabelT> lookupProgenyChildLeft =
                curNode.getChildLeft()->getLookupProgeny();
            std::vector<MarkerLabelT> lookupProgenyChildRight =
                curNode.getChildRight()->getLookupProgeny();
            for (MarkerLabelT p = 0; p < lookupProgenyChildLeft.size(); p++) {
              for (MarkerLabelT q = 0; q < lookupProgenyChildRight.size();
                   q++) {
                if (lookupProgenyChildLeft.at(p) == 1 &&
                    lookupProgenyChildRight.at(q) == 1 &&
                    saliences[p][q] == -0.5) {
                  saliences[p][q] = curNode.getInternalNodeValuationInitial();
                  saliences[q][p] = curNode.getInternalNodeValuationInitial();
                }
              }
            }
          }
        }
        // we set the non positive values of the matrix to zero
        for (size_t i = 0; i < saliences.size(); i++) {
          for (size_t j = 0; j < saliences[i].size(); j++)
            if (saliences[i][j] <= 0.0) {
              saliences[i][j] = 0.0;
              saliences[j][i] = 0.0;
            }
        }
        // Now that we have the "sparse"/"death" matrix, we apply a recursive
        // lexicographic-distance-based
        // algorithm to infer the desired matrix
        std::vector<std::vector<double>> saliencesInit(saliences);
        std::vector<std::vector<double>> saliencesTmp(
            nbrRegions, vector<double>(nbrRegions, 0.0));

        // recursive max-min term-by-term product of the persistences matrix
        // until equilibrium state
        while (saliences != saliencesTmp) {
          saliencesTmp = saliences;
          for (size_t i = 0; i < nbrRegions; i++) {
            for (size_t j = i; j < nbrRegions; j++) {
              std::vector<double> minVec(nbrRegions, 0.0);
              for (size_t k = 0; k < nbrRegions; k++) {
                minVec[k] = min(saliences[i][k], saliencesInit[k][j]);
              }
              saliences[i][j] = *max_element(minVec.begin(), minVec.end());
              saliences[j][i] = saliences[i][j];
            }
          }
        }
      } // end if "full" && "life"
      else {
        cerr << "getSaliences(string typeOfMatrix = ''sparse'',momentOfLife = "
                "''death'') -> please choose typeOfMatrix"
             << " in the following : sparse, full; and please choose "
                "momentOfLife in the following : ''life'', ''death'' "
             << endl;
      }

      return saliences;
    } // end getSaliences

    /** Compute dasgupta score (paper "A cost function for similarity-based
     * hierarchical clustering")
     *
     *
     * @param[in] completeGraph complete graph (with all adjacent regions
     * connected)
     */
    double computeDasguptaCF(Graph<NodeT, ValGraphT> completeGraph)
    {
      double dasguptaCF = 0;

      // sort nodes by decreasing internalNodeValuationInitial
      this->sortNodes(true);

      // We extract pairs of labels in the form of labelsin and labelsout
      // vectors
      std::vector<Edge<NodeT, ValGraphT>> edges = completeGraph.getEdges();

      std::vector<NodeT>     labelsin  = std::vector<NodeT>(edges.size(), 0);
      std::vector<NodeT>     labelsout = std::vector<NodeT>(edges.size(), 0);
      std::vector<ValGraphT> weights = std::vector<ValGraphT>(edges.size(), 0);

      for (MarkerLabelT i = 0; i < edges.size(); i++) {
        labelsin.at(i)  = edges.at(i).source;
        labelsout.at(i) = edges.at(i).source;
        weights.at(i)   = edges.at(i).source;
      }

      // dendrogram traversal to compute Dasgupta cost function
      for (MarkerLabelT i = 0; i < nbrNodes; i++) {
        DendroNodeType &curNode = *dendroNodes.at(i);
        // we check if it is an internal node
        if (curNode.getIsInternalNode() == 1) {
          std::vector<MarkerLabelT> lookupProgeny = curNode.getLookupProgeny();
          std::vector<MarkerLabelT> lookupProgenyChildLeft =
              curNode.getChildLeft()->getLookupProgeny();
          std::vector<MarkerLabelT> lookupProgenyChildRight =
              curNode.getChildRight()->getLookupProgeny();

          double nbLeavesUnder =
              std::accumulate(lookupProgeny.begin(), lookupProgeny.end(), 0.);

          for (MarkerLabelT s = 0; s < edges.size(); s++) {
            MarkerLabelT labelIn       = labelsin.at(s);
            MarkerLabelT labelOut      = labelsout.at(s);
            double       currentWeight = weights.at(s);

            if (lookupProgeny.at(labelIn) == lookupProgeny.at(labelOut) &&
                lookupProgenyChildLeft.at(labelIn) !=
                    lookupProgenyChildRight.at(labelOut)) {
              dasguptaCF += nbLeavesUnder * currentWeight;
            }
          }

          //    for (MarkerLabelT s = 0; s < lookupProgeny.size(); s++) {
          //      for (MarkerLabelT t = s+1; t < lookupProgeny.size(); t++) {
          //        if (lookupProgeny.at(s)==lookupProgeny.at(t) &&
          //      lookupProgenyChildLeft.at(s)!=lookupProgenyChildRight.at(t)){
          //    dasguptaCF += nbLeavesUnder*curVal;
          //        }
          //      }
          //    }
        }
      }
      return dasguptaCF;
    }

    double computeScoreInteractiveSeg()
    {
      double score = 0;

      std::map<MarkerLabelT, std::vector<MarkerLabelT>> lookupMapMarkersLabels;
      std::map<std::pair<MarkerLabelT, MarkerLabelT>, std::vector<double>>
          lookupMapMarkersPairsValuations;
      std::map<std::pair<MarkerLabelT, MarkerLabelT>, std::vector<double>>
          lookupMapSameMarkersPairsValuations;

      // first dendrogram traversal to get marked nodes by labels in
      // lookupMapMarkers
      for (MarkerLabelT i = 0; i < nbrNodes; i++) {
        DendroNodeType &curNode = *dendroNodes.at(i);
        if (curNode.getMarker() != 0) {
          MarkerLabelT markerValue = curNode.getMarker();
          MarkerLabelT labelValue  = curNode.getLabel();
          lookupMapMarkersLabels[markerValue].push_back(labelValue);
        }
      }
      // lookupMapMarkersLabels traversal to initialize
      // lookupMapMarkersPairsValuations
      for (typename std::map<MarkerLabelT, std::vector<MarkerLabelT>>::iterator
               iter = lookupMapMarkersLabels.begin();
           iter != lookupMapMarkersLabels.end(); ++iter) {
        typename std::map<MarkerLabelT, std::vector<MarkerLabelT>>::iterator
            iter2 = iter;
        iter2++;
        for (; iter2 != lookupMapMarkersLabels.end(); ++iter2) {
          if (iter->first != iter2->first) {
            std::pair<MarkerLabelT, MarkerLabelT> pairMarkers =
                make_pair(iter->first, iter2->first);
            lookupMapMarkersPairsValuations[pairMarkers] =
                std::vector<double>();
          }
        }
        std::pair<MarkerLabelT, MarkerLabelT> pairSameMarkers =
            make_pair(iter->first, iter->first);
        lookupMapSameMarkersPairsValuations[pairSameMarkers] =
            std::vector<double>();
      }

      // lookupMapMarkersPairsValuations traversal to complete the valuations
      // field
      for (typename std::map<std::pair<MarkerLabelT, MarkerLabelT>,
                             std::vector<double>>::iterator iter =
               lookupMapMarkersPairsValuations.begin();
           iter != lookupMapMarkersPairsValuations.end(); iter++) {
        MarkerLabelT              marker1 = iter->first.first;
        MarkerLabelT              marker2 = iter->first.second;
        std::vector<MarkerLabelT> labelsMarker1 =
            lookupMapMarkersLabels[marker1];
        std::vector<MarkerLabelT> labelsMarker2 =
            lookupMapMarkersLabels[marker2];

        for (size_t i = 0; i < labelsMarker1.size(); i++) {
          for (size_t j = 0; j < labelsMarker2.size(); j++) {
            double       scoreTmp = 0;
            MarkerLabelT label1   = labelsMarker1[i];
            MarkerLabelT label2   = labelsMarker2[j];

            for (size_t d = 0; d < nbrNodes; d++) {
              DendroNodeType &          curNode = *dendroNodes.at(d);
              std::vector<MarkerLabelT> lookupProgenyd =
                  curNode.getLookupProgeny();
              if (lookupProgenyd[label1] != 0 && lookupProgenyd[label2] != 0) {
                if (scoreTmp == 0) {
                  scoreTmp = curNode.getInternalNodeValuationInitial();
                } else {
                  if (curNode.getInternalNodeValuationInitial() < scoreTmp) {
                    scoreTmp = curNode.getInternalNodeValuationInitial();
                  }
                }
              }
            }
            if (scoreTmp != 0) {
              (iter->second).push_back(scoreTmp);
            }
          } // end for (int j =0; j<labelsMarker2.size();j++){
        }   // end for (int i = 0;i<labelsMarker1.size();i++){
      }

      // lookupMapSameMarkersPairsValuations traversal to complete the
      // valuations field
      for (typename std::map<std::pair<MarkerLabelT, MarkerLabelT>,
                             std::vector<double>>::iterator iter =
               lookupMapSameMarkersPairsValuations.begin();
           iter != lookupMapSameMarkersPairsValuations.end(); iter++) {
        MarkerLabelT              marker       = iter->first.first;
        std::vector<MarkerLabelT> labelsMarker = lookupMapMarkersLabels[marker];
        for (size_t i = 0; i < labelsMarker.size(); i++) {
          for (size_t j = 0; j < labelsMarker.size(); j++) {
            double       scoreTmp = 0;
            MarkerLabelT label1   = labelsMarker[i];
            MarkerLabelT label2   = labelsMarker[j];
            for (size_t d = 0; d < nbrNodes; d++) {
              DendroNodeType &          curNode = *dendroNodes.at(d);
              std::vector<MarkerLabelT> lookupProgenyd =
                  curNode.getLookupProgeny();
              if (lookupProgenyd[label1] != 0 && lookupProgenyd[label2] != 0) {
                if (scoreTmp == 0) {
                  scoreTmp = curNode.getInternalNodeValuationInitial();
                } else {
                  if (curNode.getInternalNodeValuationInitial() < scoreTmp) {
                    scoreTmp = curNode.getInternalNodeValuationInitial();
                  }
                }
              }
            }
            (iter->second).push_back(scoreTmp);
          } // end for (int j =0; j<labelsMarker2.size();j++){
        }   // end for (int i = 0;i<labelsMarker1.size();i++){
      }

      // check if lookupMapMarkersPairsValuations is correctly filled
      //    for (typename
      //    std::map<std::pair<MarkerLabelT,MarkerLabelT>,std::vector<double>
      //    >::iterator iter = lookupMapMarkersPairsValuations.begin();
      //     iter!=lookupMapMarkersPairsValuations.end();iter++){
      //      cout << "m1 = " << iter->first.first << "  m2 = " <<
      //      iter->first.second << endl;
      //       for (int i = 0; i<(iter->second).size();i++){
      //  cout << (iter->second)[i]<<endl;
      //       }
      //    }

      // computation of the score
      // Mean and variances for pairs of different markers
      double meanDiffMarkers = 0;
      double varDiffMarkers  = 0;
      double sizeDiffMarkers = 0;
      for (typename std::map<std::pair<MarkerLabelT, MarkerLabelT>,
                             std::vector<double>>::iterator iter =
               lookupMapMarkersPairsValuations.begin();
           iter != lookupMapMarkersPairsValuations.end(); iter++) {
        sizeDiffMarkers = sizeDiffMarkers + 1;
        double mean_ind = 0;
        double var_ind  = 0;
        for (MarkerLabelT i = 0; i < (iter->second).size(); i++) {
          mean_ind = mean_ind + (iter->second)[i];
        }
        mean_ind = mean_ind / (iter->second).size();
        for (MarkerLabelT i = 0; i < (iter->second).size(); i++) {
          var_ind = var_ind + (mean_ind - (iter->second)[i]) *
                                  (mean_ind - (iter->second)[i]);
        }
        var_ind         = var_ind / (iter->second).size();
        meanDiffMarkers = meanDiffMarkers + mean_ind;
        varDiffMarkers  = varDiffMarkers + var_ind;
      }
      meanDiffMarkers = meanDiffMarkers / sizeDiffMarkers;
      varDiffMarkers  = varDiffMarkers / sizeDiffMarkers;

      // mean and variances for pairs of same markers
      double meanSameMarkers = 0;
      double varSameMarkers  = 0;
      double sizeSameMarkers = 0;
      for (typename std::map<std::pair<MarkerLabelT, MarkerLabelT>,
                             std::vector<double>>::iterator iter =
               lookupMapSameMarkersPairsValuations.begin();
           iter != lookupMapSameMarkersPairsValuations.end(); iter++) {
        sizeSameMarkers = sizeSameMarkers + 1;
        double mean_ind = 0;
        double var_ind  = 0;
        for (MarkerLabelT i = 0; i < (iter->second).size(); i++) {
          mean_ind = mean_ind + (iter->second)[i];
        }
        mean_ind = mean_ind / (iter->second).size();
        for (MarkerLabelT i = 0; i < (iter->second).size(); i++) {
          varSameMarkers = varSameMarkers + (mean_ind - (iter->second)[i]) *
                                                (mean_ind - (iter->second)[i]);
        }
        var_ind         = var_ind / (iter->second).size();
        meanSameMarkers = meanSameMarkers + mean_ind;
        varSameMarkers  = varSameMarkers + var_ind;
      }
      meanSameMarkers = meanSameMarkers / sizeSameMarkers;
      varSameMarkers  = varSameMarkers / sizeSameMarkers;

      score = (varSameMarkers * varSameMarkers) /
              (std::abs(meanSameMarkers - meanDiffMarkers));
      return score;
    }; // end computeScoreInteractiveSeg

  }; // end Dendrogram

  // TODO: debug distDendro (problem with wrapping in Python using SWIG -
  // segfault)
  //   template <class MarkerLabelT, class NodeT,class ValGraphT>
  //   double distDendro(Dendrogram<MarkerLabelT,NodeT,ValGraphT> &
  //   d1,Dendrogram<MarkerLabelT,NodeT,ValGraphT> & d2,int order){
  //     double distToReturn = 0;
  //     if (d1.getNbrNodes()==d2.getNbrNodes()){
  //       int nbrNodes = d1.getNbrNodes();
  //       std::vector<DendroNode<MarkerLabelT>*>&dendroNodes1 =
  //       d1.getDendroNodes();
  //       for (int i = 0; i<nbrNodes; i++){
  //  DendroNode<MarkerLabelT> &node1 = *dendroNodes1[i];
  //  MarkerLabelT researchedLabel = node1.getLabel();
  //  DendroNode<MarkerLabelT> &node2 =
  // *d2.researchLabel(researchedLabel);
  //  distToReturn = distToReturn
  //        +
  // std::abs(std::pow(node1.getInternalNodeValuationInitial()-node2.getInternalNodeValuationInitial(),order));
  //       }
  //     }
  //     else{
  //       cout << "The two dendrograms do not have the same number of nodes" <<
  //       endl;
  //     }
  //     return distToReturn;
  //   };

  //   template <class MarkerLabelT, class NodeT,class ValGraphT>
  //   Dendrogram<MarkerLabelT,NodeT,ValGraphT>
  //   infDendro(Dendrogram<MarkerLabelT,NodeT,ValGraphT> &
  //   d1,Dendrogram<MarkerLabelT,NodeT,ValGraphT> & d2){
  //     Dendrogram<MarkerLabelT,NodeT,ValGraphT> dendroToReturn(d1);
  //     if (d1.getNbrNodes()==d2.getNbrNodes()){
  //       int nbrNodes = d1.getNbrNodes();
  //       std::vector<DendroNode<MarkerLabelT>*>&dendroNodes1 =
  //       d1.getDendroNodes();
  //       for (int i = 0; i<nbrNodes; i++){
  //  DendroNode<MarkerLabelT> &node1 = *dendroNodes1[i];
  //  MarkerLabelT researchedLabel = node1.getLabel();
  //  DendroNode<MarkerLabelT> &node2 =
  // *d2.researchLabel(researchedLabel);
  //  DendroNode<MarkerLabelT> &nodeToModify =
  // *dendroToReturn.researchLabel(researchedLabel);
  //  nodeToModify.setInternalNodeValuationInitial(std::min(node1.getInternalNodeValuationInitial(),
  //                    node2.getInternalNodeValuationInitial()));
  //       }
  //     }
  //     return dendroToReturn;
  //   }

  //     template <class MarkerLabelT, class NodeT,class ValGraphT>
  //   void supDendro(Dendrogram<MarkerLabelT,NodeT,ValGraphT> &
  //   d1,Dendrogram<MarkerLabelT,NodeT,ValGraphT> & d2,
  //     Dendrogram<MarkerLabelT,NodeT,ValGraphT> & dOut){
  //     dOut = d1.clone();
  //     if (dOut.getNbrNodes()==d2.getNbrNodes()){
  //       int nbrNodes = dOut.getNbrNodes();
  //       std::vector<DendroNode<MarkerLabelT>*>&dendroNodesOut =
  //       dOut.getDendroNodes();
  //       for (int i = 0; i<nbrNodes; i++){
  //  DendroNode<MarkerLabelT> &node1 = *dendroNodesOut[i];
  //  MarkerLabelT researchedLabel = node1.getLabel();
  //  DendroNode<MarkerLabelT> &node2 =
  // *d2.researchLabel(researchedLabel);
  //  DendroNode<MarkerLabelT> &nodeToModify =
  // *dOut.researchLabel(researchedLabel);
  //  nodeToModify.setInternalNodeValuationInitial(std::max(node1.getInternalNodeValuationInitial(),
  //                    node2.getInternalNodeValuationInitial()));
  //       };
  //     }
  // //     dOut.reorganize();
  //   }
  //
  //     template <class MarkerLabelT, class NodeT,class ValGraphT>
  //   void infDendro(Dendrogram<MarkerLabelT,NodeT,ValGraphT> &
  //   d1,Dendrogram<MarkerLabelT,NodeT,ValGraphT> & d2,
  //     Dendrogram<MarkerLabelT,NodeT,ValGraphT> & dOut){
  //     dOut = d1.clone();
  //     if (dOut.getNbrNodes()==d2.getNbrNodes()){
  //       int nbrNodes = dOut.getNbrNodes();
  //       std::vector<DendroNode<MarkerLabelT>*>&dendroNodesOut =
  //       dOut.getDendroNodes();
  //       for (int i = 0; i<nbrNodes; i++){
  //  DendroNode<MarkerLabelT> &node1 = *dendroNodesOut[i];
  //  MarkerLabelT researchedLabel = node1.getLabel();
  //  DendroNode<MarkerLabelT> &node2 =
  // *d2.researchLabel(researchedLabel);
  //  DendroNode<MarkerLabelT> &nodeToModify =
  // *dOut.researchLabel(researchedLabel);
  //  nodeToModify.setInternalNodeValuationInitial(std::min(node1.getInternalNodeValuationInitial(),
  //                    node2.getInternalNodeValuationInitial()));
  //       };
  //     }
  // //     dOut.reorganize();
  //   }

  /** Compute cut of a hierarchy according to the number of regions
   *
   *
   * @param[in] imLabels labels image
   * @param[in] dendrogram dendrogram of the hierarchy
   * @param[in] associated_mst associated mst
   * @param[in] desiredNbrRegions desired number of regions in the output cut
   * @param[out] imOut output cut (labels image)
   */
  template <class MarkerLabelT, class NodeT, class ValGraphT>
  void
  getLevelHierarchySeg(const smil::Image<MarkerLabelT> &           imLabels,
                       Dendrogram<MarkerLabelT, NodeT, ValGraphT> &dendrogram,
                       Graph<NodeT, ValGraphT> &  associated_mst,
                       double                     desiredNbrRegions,
                       smil::Image<MarkerLabelT> &imOut)
  {
    Dendrogram<MarkerLabelT, NodeT, ValGraphT> dendrobis = dendrogram.clone();
    Graph<NodeT, ValGraphT>                    mstbis = associated_mst.clone();
    dendrobis.sortNodes(true);

    MarkerLabelT desiredNbrRegionsbis = desiredNbrRegions;
    if (desiredNbrRegions <= 1) {
      desiredNbrRegionsbis =
          (dendrobis.getNbrNodes() / 2 + 1) * desiredNbrRegions;
    }
    if ((desiredNbrRegionsbis > 1) &&
        (desiredNbrRegionsbis <= (dendrobis.getNbrNodes() / 2 + 1))) {
      double lamb = dendrobis.getNodeValue(desiredNbrRegionsbis - 2,
                                           "internalNodeValuationInitial");
      dendrobis.removeMSTEdgesDendrogram(mstbis, lamb);
      graphToMosaic(imLabels, mstbis, imOut);
    } else {
      cerr << "getLevelHierarchySeg(imFinePartition,dendro,associated_mst,"
              "desiredNbrRegions,imOut) "
           << "-> the desiredNbrRegions specified is higher than the number of "
              "regions"
           << endl;
    }
  }
  /** Compute cut of a hierarchy according to an ultrametric threshold
   *
   *
   * @param[in] imLabels labels image
   * @param[in] dendrogram dendrogram of the hierarchy
   * @param[in] associated_mst associated mst
   * @param[in] thresh threshold of ultrametric values
   * @param[out] imOut output cut (labels image)
   */
  template <class MarkerLabelT, class NodeT, class ValGraphT>
  void
  getThreshHierarchySeg(const smil::Image<MarkerLabelT> &           imLabels,
                        Dendrogram<MarkerLabelT, NodeT, ValGraphT> &dendrogram,
                        Graph<NodeT, ValGraphT> &associated_mst, double thresh,
                        smil::Image<MarkerLabelT> &imOut)
  {
    if (thresh >= 0) {
      Dendrogram<MarkerLabelT, NodeT, ValGraphT> dendrobis = dendrogram.clone();
      Graph<NodeT, ValGraphT> mstbis = associated_mst.clone();
      dendrobis.sortNodes(true);
      dendrobis.removeMSTEdgesDendrogram(mstbis, thresh);
      graphToMosaic(imLabels, mstbis, imOut);
    } else {
      cout << "getLevelHierarchySeg(imFinePartition,dendro,associated_mst,"
              "thresh,"
              "imOut) "
           << "-> the thresh specified must be higher than 0" << endl;
    }
  }

} // namespace smil

#endif // _DENDROGRAM_HPP
