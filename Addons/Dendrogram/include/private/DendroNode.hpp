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

#ifndef _DENDRO_NODE_HPP
#define _DENDRO_NODE_HPP

#include "Core/include/DCore.h"
#include "Morpho/include/DMorpho.h"
#include "Morpho/include/DStructuringElement.h"

#include <unistd.h> // For usleep

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>
#include <vector>

namespace smil {
/**
 * DendroNode : node of a Dendrogram
 */
template <class MarkerLabelT = size_t> class DendroNode {
protected:
  double valuation;
  double internalNodeValuationInitial;
  double internalNodeValuationFinal;
  double leafValuation;
  double energy;
  MarkerLabelT marker;
  MarkerLabelT label;
  MarkerLabelT nbMarkersUnder;
  vector<MarkerLabelT> markersCount;
  vector<MarkerLabelT> lookupProgeny;
  bool isInternalNode;
  DendroNode *father;
  DendroNode *childLeft;
  DendroNode *childRight;
  DendroNode *neighborLeft;
  DendroNode *neighborRight;
  std::vector<double> moments;
  vector<MarkerLabelT> contoursCount;
  double contoursSize;
  double fidelityTerm;

public:
  typedef DendroNode<MarkerLabelT> DendroNodeType;
  //! Default constructor
  DendroNode()
      : valuation(0), internalNodeValuationInitial(0),
        internalNodeValuationFinal(0), leafValuation(0), energy(0), marker(0),
        label(0), nbMarkersUnder(0), markersCount(0), lookupProgeny(0),
        isInternalNode(0), father(0), childLeft(0), childRight(0),
        neighborLeft(0), neighborRight(0), moments(0), contoursCount(0),
        contoursSize(0), fidelityTerm(0) {}
  //! Copy constructor
  DendroNode(
      const DendroNodeType &dendroNodeToCopy) { // shallow copy
    valuation = dendroNodeToCopy.valuation;
    internalNodeValuationInitial =
        dendroNodeToCopy.internalNodeValuationInitial;
    internalNodeValuationFinal = dendroNodeToCopy.internalNodeValuationFinal;
    leafValuation = dendroNodeToCopy.leafValuation;
    energy = dendroNodeToCopy.energy;
    marker = dendroNodeToCopy.marker;
    label = dendroNodeToCopy.label;
    nbMarkersUnder = dendroNodeToCopy.nbMarkersUnder;
    markersCount = dendroNodeToCopy.markersCount;
    lookupProgeny = dendroNodeToCopy.lookupProgeny;
    isInternalNode = dendroNodeToCopy.isInternalNode;
    moments = dendroNodeToCopy.moments;
    contoursCount = dendroNodeToCopy.contoursCount;
    contoursSize = dendroNodeToCopy.contoursSize;
    fidelityTerm = dendroNodeToCopy.fidelityTerm;

    if (dendroNodeToCopy.father != NULL) {
      father = new DendroNodeType(*(dendroNodeToCopy.father));
    } else {
      father = NULL;
    }
    if (dendroNodeToCopy.childLeft != NULL) {
      childLeft = new DendroNodeType(*(dendroNodeToCopy.childLeft));
    } else {
      childLeft = NULL;
    }
    if (dendroNodeToCopy.childRight != NULL) {
      childRight = new DendroNodeType(*(dendroNodeToCopy.childRight));
    } else {
      childRight = NULL;
    }
    if (dendroNodeToCopy.neighborLeft != NULL) {
      neighborLeft = new DendroNodeType(*(dendroNodeToCopy.neighborLeft));
    } else {
      neighborLeft = NULL;
    }
    if (dendroNodeToCopy.neighborRight != NULL) {
      neighborRight =
          new DendroNodeType(*(dendroNodeToCopy.neighborRight));
    } else {
      neighborRight = NULL;
    }
  }
  //! Assignment operator
  DendroNode &operator=(DendroNode const &dendroNodeToCopy) {
    if (this != &dendroNodeToCopy) {
      valuation = dendroNodeToCopy.valuation;
      internalNodeValuationInitial =
          dendroNodeToCopy.internalNodeValuationInitial;
      internalNodeValuationFinal = dendroNodeToCopy.internalNodeValuationFinal;
      leafValuation = dendroNodeToCopy.leafValuation;
      marker = dendroNodeToCopy.marker;
      markersCount = dendroNodeToCopy.markersCount;
      lookupProgeny = dendroNodeToCopy.lookupProgeny;
      label = dendroNodeToCopy.label;
      nbMarkersUnder = dendroNodeToCopy.nbMarkersUnder;
      isInternalNode = dendroNodeToCopy.isInternalNode;
      moments = dendroNodeToCopy.moments;
      contoursCount = dendroNodeToCopy.contoursCount;
      contoursSize = dendroNodeToCopy.contoursSize;
      fidelityTerm = dendroNodeToCopy.fidelityTerm;
      if (dendroNodeToCopy.father != NULL) {
        delete father;
        father = new DendroNodeType(*(dendroNodeToCopy.father));
      } else {
        father = NULL;
      }
      if (dendroNodeToCopy.childLeft != NULL) {
        delete childLeft;
        childLeft = new DendroNodeType(*(dendroNodeToCopy.childLeft));
      } else {
        childLeft = NULL;
      }
      if (dendroNodeToCopy.childRight != NULL) {
        delete childRight;
        childRight = new DendroNodeType(*(dendroNodeToCopy.childRight));
      } else {
        childRight = NULL;
      }
      if (dendroNodeToCopy.neighborLeft != NULL) {
        delete neighborLeft;
        neighborLeft =
            new DendroNodeType(*(dendroNodeToCopy.neighborLeft));
      } else {
        neighborLeft = NULL;
      }
      if (dendroNodeToCopy.neighborRight != NULL) {
        delete neighborRight;
        neighborRight =
            new DendroNodeType(*(dendroNodeToCopy.neighborRight));
      } else {
        neighborRight = NULL;
      }
    }
    return *this;
  }
  //! Destructor
  virtual ~DendroNode() {}
  //! Setters and getters
  double getInternalNodeValuationInitial() {
    return internalNodeValuationInitial;
  };
  void setInternalNodeValuationInitial(double nValuation) {
    internalNodeValuationInitial = nValuation;
  };
  double getInternalNodeValuationFinal() { return internalNodeValuationFinal; };
  void setInternalNodeValuationFinal(double nValuation) {
    internalNodeValuationFinal = nValuation;
  };
  double getValuation() { return valuation; };
  void setValuation(double nValuation) { valuation = nValuation; };
  double getLeafValuation() { return leafValuation; };
  void setLeafValuation(double nValuation) { leafValuation = nValuation; };
  void setEnergy(double nEnergy) { energy = nEnergy; }
  double getEnergy() { return energy; };
  MarkerLabelT getMarker() { return marker; };
  void setMarker(double nMarker) { marker = nMarker; };
  MarkerLabelT getLabel() { return label; };
  void setLabel(double nLabel) { label = nLabel; };
  MarkerLabelT getNbMarkersUnder() { return nbMarkersUnder; };
  void setNbMarkersUnder(MarkerLabelT nNbMarkersUnder) {
    nbMarkersUnder = nNbMarkersUnder;
  };
  std::vector<MarkerLabelT> &getMarkersCount() { return markersCount; };
  void setMarkersCount(std::vector<MarkerLabelT> nMarkersCount) {
    markersCount = nMarkersCount;
  };
  std::vector<MarkerLabelT> &getLookupProgeny() { return lookupProgeny; }
  void setLookupProgeny(std::vector<MarkerLabelT> nLookupProgeny) {
    lookupProgeny = nLookupProgeny;
  }
  bool getIsInternalNode() { return isInternalNode; };
  void setIsInternalNode(bool nIsInternalNode) {
    isInternalNode = nIsInternalNode;
  };
  DendroNode *getFather() { return father; };
  void setFather(DendroNode *nFather) { father = nFather; };
  DendroNode *getChildLeft() { return childLeft; };
  void setChildLeft(DendroNode *nChildLeft) { childLeft = nChildLeft; };
  DendroNode *getChildRight() { return childRight; };
  void setChildRight(DendroNode *nChildRight) {
    childRight = nChildRight;
  };
  DendroNode *getNeighborLeft() { return neighborLeft; };
  void setNeighborLeft(DendroNode *nNeighborLeft) {
    neighborLeft = nNeighborLeft;
  };
  DendroNode *getNeighborRight() { return neighborRight; };
  void setNeighborRight(DendroNode *nNeighborRight) {
    neighborRight = nNeighborRight;
  };
  std::vector<double> getMoments() { return moments; }
  void setMoments(std::vector<double> nMoments) { moments = nMoments; }
  vector<MarkerLabelT> getContoursCount() { return contoursCount; }
  void setContoursCount(vector<MarkerLabelT> nContoursCount) {
    contoursCount = nContoursCount;
  };
  double getContoursSize() { return contoursSize; }
  void setContoursSize(double nContoursSize) { contoursSize = nContoursSize; }
  double getFidelityTerm() { return fidelityTerm; };
  void setFidelityTerm(double nFidelityTerm) { fidelityTerm = nFidelityTerm; }
  DendroNode *getAncestor() {
    DendroNode *refToReturn = this;
    if (refToReturn->getFather() != NULL) {
      while (refToReturn != refToReturn->getFather() &&
             refToReturn->getFather() != NULL) {
        refToReturn = refToReturn->getFather();
      }
    }
    return refToReturn;
  };
  DendroNode *getSelf() { return this; };
  //! Comparison functions
  static bool isInferior(DendroNode *dendroNodeL,
                         DendroNode *dendroNodeR) {
    return dendroNodeL->getInternalNodeValuationInitial() <
           dendroNodeR->getInternalNodeValuationInitial();
  };
  static bool isSuperior(DendroNode *dendroNodeL,
                         DendroNode *dendroNodeR) {
    return dendroNodeL->getInternalNodeValuationInitial() >
           dendroNodeR->getInternalNodeValuationInitial();
  };
  //! Operators overriding
  bool operator<(const DendroNode &nDendroNode) {
    return internalNodeValuationInitial <
           nDendroNode.internalNodeValuationInitial;
  };
  bool operator>(const DendroNode &nDendroNode) {
    return internalNodeValuationInitial >
           nDendroNode.internalNodeValuationInitial;
  };
  bool operator==(const DendroNode &nDendroNode) {
    return internalNodeValuationInitial ==
           nDendroNode.internalNodeValuationInitial;
  };
}; // end DendroNode

} // namespace smil

#endif // _DENDRO_NODE_HPP
