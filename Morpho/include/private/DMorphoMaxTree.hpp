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
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS AND CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _D_MAX_TREE
#define _D_MAX_TREE

#include "DMorphImageOperations.hpp" //BMI
#include "DMorphoHierarQ.hpp"        //BMI
#include "Core/include/DImage.h"
#include "Base/include/private/DImageHistogram.hpp"
#include "Morpho/include/private/DMorphoMaxTreeCriteria.hpp"

#include <complex>
#include <math.h>

namespace smil
{
  /**
   * @ingroup Morpho
   * @defgroup MaxTree MaxTree based algorithms
   *
   * MaxTree based Attribute Opening and Closing as proposed by Salembier
   * @cite Salembier1988
   *
   * @{
   */

#ifndef SWIG

  typedef size_t Offset_T;
  typedef UINT32 Label_T;

  template <class T, class CriterionT, class Offset_T = size_t,
            class Label_T = UINT32>
  class MaxTree2
  {
  private:
    size_t          GRAY_LEVEL_NBR;
    Image<T> const *img;

    HierarchicalQueue<T, Offset_T> hq;

    std::vector<T>          levels;
    std::vector<Label_T>    children;
    std::vector<Label_T>    brothers;
    std::vector<CriterionT> criteria;
    std::vector<Label_T>    labels;

    Label_T curLabel;

    bool initialized;

    size_t imWidth;
    size_t imHeight;

  public:
    MaxTree2() : hq(true) // TOTO BMI
    {
      GRAY_LEVEL_NBR = ImDtTypes<T>::max() - ImDtTypes<T>::min() + 1;
      //  hq.reverse();// change priority order (max first)
      //  hq =   HierarchicalQueue<T,Offset_T> (true);

      labels.resize(GRAY_LEVEL_NBR);

      initialized = false;
    }

    ~MaxTree2()
    {
      reset();
    }

    void reset()
    {
      if (!initialized)
        return;

      levels.clear();
      children.clear();
      brothers.clear();
      criteria.clear();
      labels.clear();
    }

    void allocatePage()
    {
      levels.resize(levels.size() + 1024 * 1024);
      children.resize(children.size() + 1024 * 1024);
      brothers.resize(brothers.size() + 1024 * 1024);
      criteria.resize(criteria.size() + 1024 * 1024);
    }

    T initialize(const Image<T> &imIn, Label_T *img_eti, const StrElt &se)
    {
      imIn.getSize(imSize);

      // BMI BEGIN
      sePtsNbr = sePts.size();
      dOffsets.clear();
      sePts.clear();
      oddSE = se.odd;

      // set an offset distance for each se point (!=0,0,0)
      for (std::vector<IntPoint>::const_iterator it = se.points.begin();
           it != se.points.end(); it++) {
        if (it->x != 0 || it->y != 0 || it->z != 0) {
          sePts.push_back(*it);
          dOffsets.push_back(it->x + it->y * imSize[0] +
                             it->z * imSize[0] * imSize[1]);
        }
      }

      sePtsNbr = sePts.size();
      // BMI END

      if (initialized)
        reset();

      this->img                           = &imIn;
      typename ImDtTypes<T>::lineType pix = img->getPixels();

      T        minValue = ImDtTypes<T>::max();
      T        tMinV    = ImDtTypes<T>::min();
      Offset_T minOff   = 0;
      for (size_t i = 0; i < img->getPixelCount(); i++) {
        if (pix[i] < minValue) {
          minValue = pix[i];
          minOff   = i;
          if (minValue == tMinV)
            break;
        }
      }

      allocatePage();

      curLabel         = 1;
      levels[curLabel] = minValue;

      // BMI labels? quel rapport avec img_eti?
      // memset(old_labels, 0, GRAY_LEVEL_NBR * sizeof(Label_T));

      img_eti[minOff]  = curLabel;
      labels[minValue] = curLabel;
      labels[minValue] = curLabel;

      hq.initialize(*img);
      hq.push(minValue, minOff);

      size_t x, y, z;
      img->getCoordsFromOffset(minOff, x, y, z);
      //  std::cout<<"PUSH:offset="<<minOff<<"(x,y)="<<x<<", "<<y<<",
      //  val="<<minValue<<"\n";

      //  getCriterion(curLabel).xmin = getCriterion(curLabel).xmax = x;
      //  // A voir comment on peut melanger des criteres... BMI
      //  getCriterion(curLabel).ymin = getCriterion(curLabel).ymax = y;
      //  getCriterion(curLabel).zmin = getCriterion(curLabel).zmax = z;

      getCriterion(curLabel).initialize();
      curLabel++;

      initialized = true;

      imWidth  = img->getWidth();
      imHeight = img->getHeight();

      return minValue;
    }

    int nextLowerLabel(T valeur)
    {
      if (curLabel + 256 > levels.size())
        allocatePage();

      getLevel(curLabel) = valeur;
      int i;
      for (i = valeur - 1; labels[i] == 0; i--)
        ;

      getChild(curLabel)             = getChild(labels[i]);
      getChild(labels[i])            = curLabel;
      getBrother(curLabel)           = getBrother(getChild(curLabel));
      getBrother(getChild(curLabel)) = 0;
      getCriterion(curLabel).reset();
      return curLabel++;
    }

    int nextHigherLabel(T parent_valeur, T valeur)
    {
      if (curLabel + 256 > levels.size())
        allocatePage();

      getLevel(curLabel)              = valeur;
      getBrother(curLabel)            = getChild(labels[parent_valeur]);
      getChild(labels[parent_valeur]) = curLabel;
      getCriterion(curLabel).reset();
      return curLabel++;
    }

    bool subFlood(typename ImDtTypes<T>::lineType imgPix, Label_T *img_eti,
                  Offset_T p, Offset_T p_suiv)
    {
      Label_T indice;

      if (imgPix[p_suiv] > imgPix[p]) {
        Label_T j;
        for (j = imgPix[p] + 1; j < imgPix[p_suiv]; j++)
          labels[j] = 0;
        indice = img_eti[p_suiv] = labels[j] =
            nextHigherLabel(imgPix[p], imgPix[p_suiv]);

      } else if (labels[imgPix[p_suiv]] == 0) {
        indice = img_eti[p_suiv] = labels[imgPix[p_suiv]] =
            nextLowerLabel(imgPix[p_suiv]);
      } else {
        indice = img_eti[p_suiv] = labels[imgPix[p_suiv]];
      }

      size_t x, y, z;
      img->getCoordsFromOffset(p_suiv, x, y, z);

      getCriterion(indice).update(x, y, z);
      hq.push(imgPix[p_suiv], p_suiv);

      if (imgPix[p_suiv] > imgPix[p]) {
        hq.push(imgPix[p], p);
        return true;
      }
      return false;
    }

    void flood(const Image<T> &img, UINT *img_eti, unsigned int level)
    {
      Offset_T p;

      typename ImDtTypes<T>::lineType imgPix = img.getPixels();
      size_t                          x0, y0, z0;

      while ((hq.getHigherLevel() >= level) && !hq.isEmpty()) {
        p = hq.pop();

        img.getCoordsFromOffset(p, x0, y0, z0);

        bool oddLine = oddSE && ((y0) % 2);

        // not size_t in order to (possibly be negative!)
        off_t    x, y, z;
        Offset_T p_suiv;

        for (UINT i = 0; i < sePtsNbr; i++) {
          IntPoint &pt = sePts[i];
          x            = x0 + pt.x;
          y            = y0 + pt.y;
          z            = z0 + pt.z;

          if (oddLine)
            x += (((y + 1) % 2) != 0);

          if (x >= 0 && x < (int) imSize[0] && y >= 0 && y < (int) imSize[1] &&
              z >= 0 && z < (int) imSize[2]) {
            p_suiv = p + dOffsets[i];
            if (oddLine)
              p_suiv += (((y + 1) % 2) != 0);

            if (img_eti[p_suiv] == 0) {
              if (subFlood(imgPix, img_eti, p, p_suiv))
                break;
            }
          }
        } // for each ngb

      } // while hq.notEmpty
    } // void flood

  protected:
    size_t imSize[3];

    std::vector<IntPoint> sePts;
    UINT                  sePtsNbr;
    bool                  oddSE;
    std::vector<int>      dOffsets;

  public:
    inline CriterionT &getCriterion(const Label_T node)
    {
      return criteria[node];
    }

    inline T &getLevel(const Label_T node)
    {
      return levels[node];
    }

    inline Label_T &getChild(const Label_T node)
    {
      return children[node];
    }

    inline Label_T &getBrother(const Label_T node)
    {
      return brothers[node];
    }

    inline Label_T getLabelMax()
    {
      return curLabel;
    }

    inline int getImWidth()
    {
      return imWidth;
    }

    inline int getImHeight()
    {
      return imHeight;
    }

    int build(const Image<T> &img, Label_T *img_eti, const StrElt &se)
    {
      T minValue = initialize(img, img_eti, se);

      // BMI: dOffset already contains se information
      flood(img, img_eti, minValue);
      updateCriteria(labels[minValue]);

      return labels[minValue];
    }

    // BMI (from Andres)
    // Update criteria of a given max-tree node
    // void updateCriteria(const int node);

    // Update criteria on a given max-tree node.// From Andres
    // template <class T, class CriterionT, class Offset_T, class Label_T>
    void updateCriteria(const int node)
    {
      Label_T child = getChild(node);
      while (child != 0) {
        updateCriteria(child);
        getCriterion(node).merge(&getCriterion(child));
        child = getBrother(child);
      }
    }
  };

  // END BMI

  // NEW BMI    # ##################################################
  //(tree, transformee_node, indicatrice_node, child, stopSize, (T)0, 0,
  // hauteur, tree.getLevel(root), tree.getLevel(root));
  template <class T, class CriterionT, class Offset_T, class Label_T,
            class Attr_T>
  void ComputeDeltaUO(MaxTree2<T, CriterionT, Offset_T, Label_T> &tree,
                      T *transformee_node, Attr_T *indicatrice_node, int node,
                      int nParent, T prev_residue, Attr_T stop, UINT delta,
                      int isPrevMaxT)
  {
    // self,node = 1, nParent =0, stop=0, delta = 0, isPrevMaxT = 0):
    int  child; // index node
    T    current_residue;
    UINT cNode, cParent; // attributes
    T    lNode, lParent; // node levels, the same type than input image

    // ymax-tree.getCriterion(node).ymin+1;
    // #current criterion
    cNode = tree.getCriterion(node).getAttributeValue();
    // #current level
    lNode = tree.getLevel(node);

    // ymax-tree.getCriterion(nParent).ymin+1;//
    // #current criterion
    cParent = tree.getCriterion(nParent).getAttributeValue();
    // #current level
    lParent = tree.getLevel(nParent);

    bool flag = false;

    if ((cParent - cNode) <= delta) {
      flag = true;
    }

    if (flag) {
      current_residue = prev_residue + lNode - lParent;
    } else {
      current_residue = lNode - lParent;
    }

    transformee_node[node] = transformee_node[nParent];
    indicatrice_node[node] = indicatrice_node[nParent];

    int isMaxT = 0;

    if (cNode < stop) {
      if (current_residue > transformee_node[node]) {
        //          std::cout<<"UPDATE RES\n";
        isMaxT                 = 1;
        transformee_node[node] = current_residue;
        if (!(isPrevMaxT && flag)) {
          indicatrice_node[node] = cNode + 1;
        }
      }
    } else {
      indicatrice_node[node] = 0;
    }
    child = tree.getChild(node);
    while (child != 0) {
      ComputeDeltaUO(tree, transformee_node, indicatrice_node, child, node,
                     current_residue, stop, delta, isMaxT);
      child = tree.getBrother(child);
    }
  }

  template <class T1, class T2>
  void compute_max(MaxTree2<T1, HeightCriterion, size_t, UINT32> &tree,
                   T1 *transformee_node, T2 *indicatrice_node, UINT32 node,
                   T2 stop, T1 max_tr, T2 max_in, T2 hauteur_parent,
                   T1 valeur_parent, T1 previous_value)
  {
    T1     m;
    T1     max_node;
    T2     max_criterion;
    UINT32 child;
    T2     hauteur = tree.getCriterion(node).getAttributeValue();
    // ymax-tree.getCriterion(node).ymin+1;
    //  std::cout<<"IN COMPUTE_MAX:"<<"; node="<<node<<"\n";
    m = (hauteur == hauteur_parent) ? tree.getLevel(node) - previous_value
                                    : tree.getLevel(node) - valeur_parent;
    if (hauteur >= stop) {
      max_node               = max_tr;
      max_criterion          = 0; // max_in;
      transformee_node[node] = max_node;

      indicatrice_node[node] = 0;
      child                  = tree.getChild(node);
    } else {
      if (m > max_tr) {
        max_node      = m;
        max_criterion = hauteur;
      } else {
        max_node      = max_tr;
        max_criterion = max_in;
      }
      transformee_node[node] = max_node;
      indicatrice_node[node] = max_criterion + 1;
      child                  = tree.getChild(node);
    }
    if (hauteur == hauteur_parent) {
      while (child != 0) {
        //      if(child > tree.getLabelMax()){
        //      std::cout<<"ERROR call child:"<<child<<"\n";
        //      }
        if (hauteur_parent > stop)
          compute_max(tree, transformee_node, indicatrice_node, child, stop,
                      max_node, max_criterion, hauteur, tree.getLevel(node),
                      previous_value);
        else
          compute_max(tree, transformee_node, indicatrice_node, child, stop,
                      max_node, max_criterion, hauteur,
                      tree.getLevel(node) /*valeur_parent*/, previous_value);
        child = tree.getBrother(child);
      }
    } else {
      while (child != 0) {
        //      if(child > tree.getLabelMax()){
        //      std::cout<<"ERROR call child:"<<child<<"\n";
        //}

        compute_max(tree, transformee_node, indicatrice_node, child, stop,
                    max_node, max_criterion, hauteur, tree.getLevel(node),
                    valeur_parent);
        child = tree.getBrother(child);
      }
    }
  }

  template <class T1, class T2>
  void compute_contrast(MaxTree2<T1, HeightCriterion, size_t, UINT32> &tree,
                        T1 *transformee_node, T2 *indicatrice_node, UINT32 root,
                        T2 stopSize, UINT delta = 0)
  {
    UINT32 child;
    T2     hauteur;
    //  std::cout<<"IN compute_contrast\n";

    transformee_node[root] = 0;
    indicatrice_node[root] = 0;

    hauteur = tree.getCriterion(root).getAttributeValue();
    child   = tree.getChild(root);
    // ymax - tree.getCriterion(root).ymin+1;
    if (delta == 0) {
      while (child != 0) {
        compute_max(tree, transformee_node, indicatrice_node, child, stopSize,
                    (T1) 0, (T2) 0, hauteur, tree.getLevel(root),
                    tree.getLevel(root));
        child = tree.getBrother(child);
      }
    } // END delta == 0
    else {
      while (child != 0) {
        ComputeDeltaUO(tree, transformee_node, indicatrice_node, child,
                       root /*parent*/, (T1) 0 /* prev_residue*/,
                       stopSize /*stop*/, delta, 0 /*isPrevMaxT*/);

        child = tree.getBrother(child);
      }
    } // END dela != 0
  }

#endif // SWIG

  /**
   * Ultimate Opening using the max-trees
   *
   * Max-tree based algorithm as described by Fabrizio and Marcotegui (2009)
   * @cite fabrizio_fast_2009
   * @param[in] imIn Input image
   * @param[out] imTrans The transformation image
   * @param[out] imIndic The indicator image
   * @param[in] stopSize (optional)
   * @param[in] delta (optional)
   * @param[in] se structuring element
   */

  template <class T1, class T2>
  RES_T ultimateOpen(const Image<T1> &imIn, Image<T1> &imTrans,
                     Image<T2> &imIndic, const StrElt &se = DEFAULT_SE,
                     T2 stopSize = 0, UINT delta = 0)
  {
    ASSERT_ALLOCATED(&imIn, &imTrans, &imIndic);
    ASSERT_SAME_SIZE(&imIn, &imTrans, &imIndic);

    if (stopSize == 0)
      stopSize = imIn.getHeight() - 1;

    int   imSize  = imIn.getPixelCount();
    UINT *img_eti = new UINT[imSize]();

    MaxTree2<T1, HeightCriterion, size_t, UINT32> tree;
    UINT32 root = tree.build(imIn, img_eti, se);

    // std::cout<<"ULTIMATE OPEN, after tree.build"<<"NB
    // VERTEX="<<tree.getLabelMax()<<"\n";
    T1 *transformee_node = new T1[tree.getLabelMax()]();
    T2 *indicatrice_node = new T2[tree.getLabelMax()]();
    //  std::cout<<"ULTIMATE OPEN, after memory allocation\n";
    compute_contrast(tree, transformee_node, indicatrice_node, root, stopSize,
                     delta);

    typename ImDtTypes<T1>::lineType transformeePix = imTrans.getPixels();
    typename ImDtTypes<T2>::lineType indicatricePix = imIndic.getPixels();

    for (int i = 0; i < imSize; i++) {
      transformeePix[i] = transformee_node[img_eti[i]];
      indicatricePix[i] = indicatrice_node[img_eti[i]];
    }

    delete[] img_eti;
    delete[] transformee_node;
    delete[] indicatrice_node;

    imTrans.modified();
    imIndic.modified();

    return RES_OK;
  }

#ifndef SWIG
  // NEW BMI    # ##################################################
  //(tree, transformee_node, indicatrice_node, child, stopSize, (T)0, 0,
  // hauteur, tree.getLevel(root), tree.getLevel(root));
  template <class T, class CriterionT, class Offset_T, class Label_T,
            class Attr_T>
  void ComputeDeltaUOMSER(MaxTree2<T, CriterionT, Offset_T, Label_T> &tree,
                          T *transformee_node, Attr_T *indicatrice_node,
                          int node, int nParent, int first_ancestor,
                          Attr_T stop, UINT delta, UINT method, int isPrevMaxT,
                          UINT minArea = 0, T threshold = 0, T mymax = 0)
  {
    // method:  1 (MSER), 2 (RGR)

    // "node": the current node; "nParent": its direct parent (allows
    // attribute comparison for Delta versions); "first_ancestor": the
    // first ancestor with a different attribute (used for residue
    // computation level(first_ancestor) - level(node) and for area stability
    // computation

    // self,node = 1, nParent =0, stop=0, delta = 0, isPrevMaxT = 0):
    int    child; // index node
    T      current_residue, stab_residue = 0;
    UINT   hNode, hParent; // attributes
    size_t aNode, aParent, aAncestor;
    T      lNode, lParent, lAncestor;
    // node levels, the same type than input image
    float stability;

    hNode = tree.getCriterion(node).getAttributeValue().H; // #current criterion
    aNode = tree.getCriterion(node).getAttributeValue().A;
    lNode = tree.getLevel(node); // #current level

    // #current criterion
    hParent   = tree.getCriterion(nParent).getAttributeValue().H;
    aParent   = tree.getCriterion(nParent).getAttributeValue().A;
    aAncestor = tree.getCriterion(first_ancestor).getAttributeValue().A;

    lParent   = tree.getLevel(nParent);        // #current level
    lAncestor = tree.getLevel(first_ancestor); // #current level

    bool flag = false;
    if ((hParent - hNode) <= delta) {
      flag = true;
    }

    float  relativeGrowthRate; // BMI RGR
    double lgARoot = std::log((double) tree.getImWidth() * tree.getImHeight());
    double factor;

    if (method == 2) {
      // RGR
      // BMI RGR La valeur max sera max/10 pour un
      // changement d'aire de 1 a taille_im en pixels
      factor = ImDtTypes<T>::max() / ((double) 100 * lgARoot);
    } else if (method == 3) {                          // MSER_sub
      factor = ImDtTypes<T>::max() / ((double) 100.0); // BMI
      factor = mymax / ((double) 25.0);                // BMI
    }
    if (flag) {
      current_residue = lNode - lAncestor;
      if (method == 1) {
        // mser stability
        stability    = 1 - ((aAncestor - aNode) * 1.0 / (aAncestor));
        stab_residue = round(current_residue * stability);
      } else if (method == 2) { // relative growth rate
        relativeGrowthRate =
            factor * (std::log((double) aAncestor) - std::log((double) aNode));
        if (current_residue > relativeGrowthRate) {
          stab_residue = round(current_residue - relativeGrowthRate);
        } else {
          stab_residue = 0;
        }
      } // end RGR
      else if (method == 3) { // MSER sustraire
        stability = factor * (((aAncestor - aNode) * 1.0 / (aAncestor)));
        if (current_residue > stability) {
          stab_residue = round(current_residue - stability);
        } else {
          stab_residue = 0;
        }
      } // end MSER_sub

    } // if flag
    else {
      current_residue = (lNode - lParent);
      if (method == 1) { // mser stability
        // relative growth rate
        stability    = 1 - ((aParent - aNode) * 1.0 / (aParent));
        stab_residue = round(current_residue * stability);
      } else if (method == 2) { // relative growth rate
        relativeGrowthRate =
            factor * (std::log((double) aParent) - std::log((double) aNode));
        if (current_residue > relativeGrowthRate) {
          stab_residue = round(current_residue - relativeGrowthRate);
        } else {
          stab_residue = 0;
        }
      } // end RGR
      else if (method == 3) {
        stability = factor * (((aParent - aNode) * 1.0 / (aParent))); // msersub
        if (current_residue > stability) {
          stab_residue = round(current_residue - stability);
        } else {
          stab_residue = 0;
        }

      } else {
        std::cout << "ERROR UNKOWN METHOD\n";
      }
    } // if not flag

    transformee_node[node] = transformee_node[nParent];
    indicatrice_node[node] = indicatrice_node[nParent];
    int isMaxT             = 0;
    if (hNode < stop) {
      if (stab_residue > transformee_node[node] && stab_residue > threshold &&
          aNode > minArea) {
        isMaxT                 = 1;
        transformee_node[node] = stab_residue;

        if (!(isPrevMaxT && flag)) {
          indicatrice_node[node] = hNode + 1;
        }
      }
    } else {
      indicatrice_node[node] = 0;
    }
    child = tree.getChild(node);
    while (child != 0) {
      if (flag && (hNode < stop)) {
        ComputeDeltaUOMSER(tree, transformee_node, indicatrice_node, child,
                           node, first_ancestor, stop, delta, method, isMaxT,
                           minArea, threshold, mymax);
      } else {
        ComputeDeltaUOMSER(tree, transformee_node, indicatrice_node, child,
                           node, nParent, stop, delta, method, isMaxT, minArea,
                           threshold, mymax);
      }
      child = tree.getBrother(child);
    }
  }

  inline void computeFillAspectRatioFactor(UINT wNode, UINT cNode, UINT area,
                                           UINT width, UINT height,
                                           float &fillRatio, float &AspectRatio)
  {
    // compute fillRatio and AspectRatio
    UINT minHW, maxHW;
    minHW       = MIN(wNode, cNode);
    maxHW       = MAX(wNode, cNode);
    fillRatio   = area * 1.0 / (wNode * cNode * 1.0);
    AspectRatio = minHW * 1.0 / maxHW;

    if (AspectRatio <= 0.4) // && fillRatio < 0.4)
      AspectRatio = 0.0;
    else
      AspectRatio = std::pow(AspectRatio, 1.0 / 3);

    if (fillRatio <= 0.2 || fillRatio >= 0.9)
      fillRatio = 0.0;
    else {
      fillRatio = abs(fillRatio - 0.55);
      fillRatio = std::pow((1.0 - (fillRatio * 1.0 / 0.35)), 1.0 / 3);
    }

    if (AspectRatio <= 0.4 && fillRatio >= 0.4) { // a verifier
      AspectRatio = 1;                            // 0.9
      fillRatio   = 1;
    } // 0.9

    if (area < 20 || area > (0.9 * width * height) || cNode < 5) {
      //|| cNode > (tree.getImHeight()/3)){
      AspectRatio = 0.0;
      fillRatio   = 0.0;
    }
  }

  // NEW BMI    # ##################################################
  //(tree, transformee_node, indicatrice_node, child, stopSize, (T)0, 0,
  // hauteur, tree.getLevel(root), tree.getLevel(root));
  template <class T, class CriterionT, class Offset_T, class Label_T,
            class Attr_T>
  void ComputeDeltaUOMSERSC(MaxTree2<T, CriterionT, Offset_T, Label_T> &tree,
                            T *transformee_node, Attr_T *indicatrice_node,
                            int node, int nParent, int first_ancestor,
                            Attr_T stop, UINT delta, int isPrevMaxT)
  {
    // "node": the current node; "nParent": its direct parent (allows
    // attribute comparison for Delta versions); "first_ancestor": the
    // first ancestor with a different attribute (used for residue
    // computation level(first_ancestor) - level(node) and for area stability
    // computation

    // self,node = 1, nParent =0, stop=0, delta = 0, isPrevMaxT = 0):
    int    child; // index node
    T      current_residue, stab_residue;
    UINT   hNode, hParent, wNode; // attributes
    size_t aNode, aParent, aAncestor;
    T      lNode, lParent, lAncestor;
    /// wParent set but not used*/
    // node levels, the same type than input image
    float stability, fillRatio, AspectRatio, fac;

    hNode = tree.getCriterion(node).getAttributeValue().H; // #current criterion
    wNode = tree.getCriterion(node).getAttributeValue().W; // #width
    aNode = tree.getCriterion(node).getAttributeValue().A;
    lNode = tree.getLevel(node); // #current level

    hParent = tree.getCriterion(nParent).getAttributeValue().H;
    // #current criterion
    // wParent =
    // tree.getCriterion(nParent).xmax-tree.getCriterion(nParent).xmin+1;//
    //  #width
    aParent   = tree.getCriterion(nParent).getAttributeValue().A;
    aAncestor = tree.getCriterion(first_ancestor).getAttributeValue().A;
    lParent   = tree.getLevel(nParent);        // #current level
    lAncestor = tree.getLevel(first_ancestor); // #current level

    bool flag = false;
    if ((hParent - hNode) <= delta) {
      flag = true;
    }

    computeFillAspectRatioFactor(wNode, hNode, aNode, tree.getImWidth(),
                                 tree.getImHeight(), fillRatio, AspectRatio);

    if (flag) { // no significant attribute change
      stability =
          std::pow((1.0 - ((aAncestor - aNode) * 1.0 / aAncestor)), 1.0 / 3);
      fac             = (stability * AspectRatio * fillRatio);
      current_residue = lNode - lAncestor;
      stab_residue    = round(current_residue * fac);
    } else {
      stability =
          std::pow((1.0 - ((aParent - aNode) * 1.0 / aParent)), 1.0 / 3);
      fac             = (stability * AspectRatio * fillRatio);
      current_residue = (lNode - lParent);
      stab_residue    = round(current_residue * fac);
    }

    transformee_node[node] = transformee_node[nParent];
    indicatrice_node[node] = indicatrice_node[nParent];

    int isMaxT = 0;
    if (hNode < stop) {
      if (stab_residue > transformee_node[node]) {
        isMaxT                 = 1;
        transformee_node[node] = stab_residue;

        if (!(isPrevMaxT && flag)) {
          indicatrice_node[node] = hNode + 1;
        }
      }
    } else
      indicatrice_node[node] = 0;

    child = tree.getChild(node);
    while (child != 0) {
      if (flag && (hNode < stop)) {
        ComputeDeltaUOMSERSC(tree, transformee_node, indicatrice_node, child,
                             node, first_ancestor, stop, delta, isMaxT);
      } else {
        ComputeDeltaUOMSERSC(tree, transformee_node, indicatrice_node, child,
                             node, nParent, stop, delta, isMaxT);
      }
      child = tree.getBrother(child);
    }
  }

  template <class T1, class T2>
  void compute_contrast_MSER(MaxTree2<T1, HWACriterion, size_t, UINT32> &tree,
                             T1 *transformee_node, T2 *indicatrice_node,
                             UINT32 root, T2 stopSize, UINT delta = 0,
                             UINT method = 2, UINT minArea = 0,
                             T1 threshold = 0, bool use_textShape = 0)
  {
    int child;
    //   UINT hauteur;

    transformee_node[root] = 0;
    indicatrice_node[root] = 0;

    //  tree.updateCriteria(root);
    // already in tree building function
    //  hauteur = tree.getCriterion(root).ymax - tree.getCriterion(root).ymin+1;
    child = tree.getChild(root);

    // BEGIN COMPUTE DYNAMIC, BMI

    UINT32 mynode;
    T1     mylevel, mymax;
    mymax = 0;
    for (mynode = 0; mynode < tree.getLabelMax(); mynode++) {
      mylevel = tree.getLevel(mynode);
      if (mylevel > mymax) {
        mymax = mylevel;
      }
    }
    // std::cout << "mymax=" << mymax << "\n";
    // END COMPUTE DYNAMIC, BMI
    while (child != 0) {
      if (!use_textShape)
        ComputeDeltaUOMSER(tree, transformee_node, indicatrice_node, child,
                           root /*parent*/, root /*first_ancestor*/,
                           stopSize /*stop*/, delta, method, 0 /*isPrevMaxT*/,
                           minArea, threshold, mymax);

      else {
        ComputeDeltaUOMSERSC(tree, transformee_node, indicatrice_node, child,
                             root /*parent*/, root /*first_ancestor*/,
                             stopSize /*stop*/, delta, 0 /*isPrevMaxT*/);
      }
      child = tree.getBrother(child);
    }
  }

#endif // SWIG

  template <class T1, class T2>
  RES_T ultimateOpenMSER(const Image<T1> &imIn, Image<T1> &imTrans,
                         Image<T2> &imIndic, const StrElt &se = DEFAULT_SE,
                         T2 stopSize = 0, UINT delta = 0, UINT method = 2,
                         UINT minArea = 0, T1 threshold = 0,
                         bool use_textShape = 0)
  {
    ASSERT_ALLOCATED(&imIn, &imTrans, &imIndic);
    ASSERT_SAME_SIZE(&imIn, &imTrans, &imIndic);

    int   imSize  = imIn.getPixelCount();
    UINT *img_eti = new UINT[imSize]();

    MaxTree2<T1, HWACriterion> tree;
    int                        root = tree.build(imIn, img_eti, se);

    if (stopSize == 0) {
      stopSize = imIn.getHeight() - 1;
    }

    T1 *transformee_node = new T1[tree.getLabelMax()]();
    T2 *indicatrice_node = new T2[tree.getLabelMax()]();

    compute_contrast_MSER(tree, transformee_node, indicatrice_node, root,
                          stopSize, delta, method, minArea, threshold,
                          use_textShape);

    typename ImDtTypes<T1>::lineType transformeePix = imTrans.getPixels();
    typename ImDtTypes<T2>::lineType indicatricePix = imIndic.getPixels();

    for (int i = 0; i < imSize; i++) {
      transformeePix[i] = transformee_node[img_eti[i]];
      indicatricePix[i] = indicatrice_node[img_eti[i]];
    }

    delete[] img_eti;
    delete[] transformee_node;
    delete[] indicatrice_node;

    imTrans.modified();
    imIndic.modified();

    return RES_OK;
  }

#ifndef SWIG

  template <class T, class CriterionT, class Offset_T, class Label_T,
            class Attr_T>
  void processMaxTree(MaxTree2<T, CriterionT, Offset_T, Label_T> &tree,
                      Label_T node, T *lut_out, T previousLevel, Attr_T stop)
  {
    // MORPHEE_ENTER_FUNCTION("s_OpeningTree::computeMaxTree");
    Label_T child;
    T       nodeLevel = tree.getLevel(node);

    T currentLevel = (tree.getCriterion(node).getAttributeValue() < stop)
                         ? previousLevel
                         : nodeLevel;

    lut_out[node] = currentLevel;

    child = tree.getChild(node);
    while (child != 0) {
      processMaxTree(tree, child, lut_out, currentLevel, stop);
      child = tree.getBrother(child);
    }
  }

  template <class T, class CriterionT, class Offset_T, class Label_T,
            class Attr_T>
  void
  compute_AttributeOpening(MaxTree2<T, CriterionT, Offset_T, Label_T> &tree,
                           T *lut_node, Label_T root, Attr_T stopSize)
  {
    Label_T child;

    lut_node[root] = tree.getLevel(root);

    //    tree.updateCriteria(root);

    // DEBUG
    // Label_T lab;

    // for(lab = 1; lab < tree.getLabelMax(); lab ++){
    //   std::cout<<"lab="<<lab<<",
    //   att="<<tree.getCriterion(lab).getAttributeValue()<<"\n";
    // }

    // END DEBUG
    child = tree.getChild(root);

    while (child != 0) {
      processMaxTree(tree, child, lut_node, tree.getLevel(root), stopSize);
      child = tree.getBrother(child);
    }

  } // compute_AttributeOpening

  template <class T, class CriterionT, class Offset_T = size_t,
            class Label_T = UINT32>
  RES_T attributeOpen(const Image<T> &imIn, Image<T> &imOut, size_t stopSize,
                      const StrElt &se)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    size_t   imSize  = imIn.getPixelCount();
    Label_T *img_eti = new Label_T[imSize]();

    MaxTree2<T, CriterionT> tree;
    Label_T                 root = tree.build(imIn, img_eti, se);

    // BEGIN BMI, DEBUG
    // for(size_t i=0;i<imSize;i++) {
    //   if(i%5 == 0){
    //     std::cout<<"\n";
    //   }
    //   std::cout<<img_eti[i]<<",";
    // }
    // END BEGIN BMI, DEBUG
    T *out_node = new T[tree.getLabelMax()]();

    compute_AttributeOpening(tree, out_node, root, stopSize);

    typename ImDtTypes<T>::lineType outPix = imOut.getPixels();

    for (size_t i = 0; i < imSize; i++)
      outPix[i] = out_node[img_eti[i]];

    delete[] img_eti;
    delete[] out_node;

    imOut.modified();

    return RES_OK;
  }

#endif // SWIG

  /**
   * Image opening based on height attribute
   *
   * Max-tree based algorithm
   * @warning 4-connex only (6-connex in 3D)
   * @param[in] imIn Input image
   * @param[in] stopSize The size of the opening
   * @param[out] imOut Output image
   * @param[in] se Structuring element
   */
  template <class T>
  RES_T heightOpen(const Image<T> &imIn, size_t stopSize, Image<T> &imOut,
                   const StrElt &se = DEFAULT_SE)
  {
    return attributeOpen<T, HeightCriterion>(imIn, imOut, stopSize, se);
  } // END heightOpen

  /**
   * Image closing based on height attribute
   *
   * Max-tree based algorithm
   * @warning 4-connex only (6-connex in 3D)
   * @param[in] imIn Input image
   * @param[in] stopSize The size of the opening
   * @param[out] imOut Output image
   * @param[in] se Structuring element
   */
  template <class T>
  RES_T heightClose(const Image<T> &imIn, size_t stopSize, Image<T> &imOut,
                    const StrElt &se = DEFAULT_SE)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    ImageFreezer freeze(imOut);

    Image<T> tmpIm(imIn);
    inv(imIn, tmpIm);
    RES_T res = attributeOpen<T, HeightCriterion>(tmpIm, imOut, stopSize, se);
    inv(imOut, imOut);

    return res;
  } // END heightClose

  /**
   * Image opening based on width attribute
   *
   * Max-tree based algorithm
   * @warning 4-connex only (6-connex in 3D)
   * @param[in] imIn Input image
   * @param[in] stopSize The size of the opening
   * @param[out] imOut Output image
   * @param[in] se Structuring element
   */
  template <class T>
  RES_T widthOpen(const Image<T> &imIn, size_t stopSize, Image<T> &imOut,
                  const StrElt &se = DEFAULT_SE)
  {
    return attributeOpen<T, WidthCriterion>(imIn, imOut, stopSize, se);
  } // END widthOpen

  /**
   * Image opening based on width attribute
   *
   * Max-tree based algorithm
   * @warning 4-connex only (6-connex in 3D)
   * @param[in] imIn Input image
   * @param[in] stopSize The size of the opening
   * @param[out] imOut Output image
   * @param[in] se Structuring element
   */
  template <class T>
  RES_T widthClose(const Image<T> &imIn, size_t stopSize, Image<T> &imOut,
                   const StrElt &se = DEFAULT_SE)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    ImageFreezer freeze(imOut);

    Image<T> tmpIm(imIn);
    inv(imIn, tmpIm);
    RES_T res = attributeOpen<T, WidthCriterion>(tmpIm, imOut, stopSize, se);
    inv(imOut, imOut);

    return res;
  } // END widthClose

  /**
   * Image opening based on area attribute
   *
   * Max-tree based algorithm
   *
   * @param[in] imIn Input image
   * @param[in] stopSize The size of the opening
   * @param[out] imOut Output image
   * @param[in] se Structuring element
   */
  template <class T>
  RES_T areaOpen(const Image<T> &imIn, size_t stopSize, Image<T> &imOut,
                 const StrElt &se = DEFAULT_SE)
  {
    return attributeOpen<T, AreaCriterion>(imIn, imOut, stopSize, se);
  }

  /**
   * Image closing based on area attribute
   *
   * Max-tree based algorithm
   *
   * @param[in] imIn Input image
   * @param[in] stopSize The size of the closing
   * @param[out] imOut Output image
   * @param[in] se Structuring element
   */
  template <class T>
  RES_T areaClose(const Image<T> &imIn, size_t stopSize, Image<T> &imOut,
                  const StrElt &se = DEFAULT_SE)
  {
    ASSERT_ALLOCATED(&imIn, &imOut);
    ASSERT_SAME_SIZE(&imIn, &imOut);

    ImageFreezer freeze(imOut);

    Image<T> tmpIm(imIn);
    inv(imIn, tmpIm);
    RES_T res = attributeOpen<T, AreaCriterion>(tmpIm, imOut, stopSize, se);
    inv(imOut, imOut);

    return res;
  }

  /** @} */

} // namespace smil

#endif // _D_MAX_TREE_HPP
