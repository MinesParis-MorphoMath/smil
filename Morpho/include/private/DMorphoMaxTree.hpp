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


#ifndef _D_MAX_TREE
#define _D_MAX_TREE

#include "Core/include/DImage.h"
#include "Base/include/private/DImageHistogram.hpp"
#include <complex>
namespace smil
{
    /**
    * \ingroup Morpho
    * \defgroup MaxTree
    * \{
    */

    
#ifndef SWIG

template <class T, class TokenType=UINT>
class PriorityQueue
{
private:
    typedef TokenType* StackType;
    
    size_t GRAY_LEVEL_NBR;
    size_t TYPE_FLOOR;
    StackType *stacks;
    size_t *tokenNbr;
    size_t size;
    size_t higherLevel;
    
    bool initialized;
    
public:
    PriorityQueue()
    {
        GRAY_LEVEL_NBR = ImDtTypes<T>::max()-ImDtTypes<T>::min()+1;
        TYPE_FLOOR = -ImDtTypes<T>::min();
        
        stacks = new StackType[GRAY_LEVEL_NBR];
        for (size_t i=0;i<GRAY_LEVEL_NBR;i++)
          stacks[i] = NULL;
        tokenNbr = new size_t[GRAY_LEVEL_NBR];
        initialized = false;
    }
    ~PriorityQueue()
    {
        reset();
        delete[] stacks;
        delete[] tokenNbr;
    }
    
    void reset()
    {
        if (!initialized)
          return;
        
        for(size_t i=0;i<GRAY_LEVEL_NBR;i++)
        {
            if (stacks[i])
                delete[] stacks[i];
            stacks[i] = NULL;
        }
        
        initialized = false;
    }
    
    void initialize(const Image<T> &img)
    {
        if (initialized)
          reset();
        
        size_t *h = new size_t[GRAY_LEVEL_NBR];
        histogram(img, h);

        for(size_t i=0;i<GRAY_LEVEL_NBR;i++)
            if (h[i]!=0)
              stacks[i] = new TokenType[h[i]+1];
            
        delete[] h;
        memset(tokenNbr, 0, GRAY_LEVEL_NBR*sizeof(size_t));
        size = 0;
        higherLevel = 0;
        
        initialized = true;
    }
    
    inline size_t getSize()
    {
        return size;
    }
    
    inline bool isEmpty()
    {
        return size==0;
    }
    
    inline size_t getHigherLevel()
    {
        return higherLevel;
    }
    
    inline void push(T value, TokenType dOffset)
    {
        size_t level = TYPE_FLOOR + value;
        if (level>higherLevel)
          higherLevel = level;
        stacks[level][tokenNbr[level]++] = dOffset;
        size++;
    }
    
    inline TokenType pop()
    {
        size_t hlSize = tokenNbr[higherLevel];
        TokenType dOffset = stacks[higherLevel][hlSize-1];
        if (hlSize>1)
          tokenNbr[higherLevel]--;
        else if (size>1) // Find new higher level (non empty stack)
        {
            tokenNbr[higherLevel] = 0;
            for (size_t i=higherLevel-1;i<numeric_limits<size_t>::max();i--)
              if (tokenNbr[i]>0)
              {
                  higherLevel = i;
                  break;
              }
        }
        size--;
        
        return dOffset;
    }
  
};

#define COLUMN_NBR 16
#define COLUMN_SIZE 131072

#define COLUMN_SHIFT (COLUMN_NBR+1)
#define COLUMN_MOD (COLUMN_SIZE-1)


#define GET_TREE_OBJ(type, node) type[node >> COLUMN_SHIFT][node & COLUMN_MOD]




struct EmptyCriterion
{
  inline void init() {}
  inline void reset() {}
  inline void merge(EmptyCriterion &/*other*/) { }
  inline void update() {  }
};

template <class T, class BaseCriterionT=EmptyCriterion, class OffsetT=UINT>
class MaxTree
{
public:
    struct CriterionT
    {
        OffsetT ymin, ymax, xmin, xmax, zmin, zmax;
        BaseCriterionT crit;
        inline void initialize() { crit.init(); }
        inline void reset() { crit.reset(); }
        inline void merge(CriterionT &other) { crit.merge(other.crit); }
        inline void update() { crit.update(); }
    };
  
private:
    size_t GRAY_LEVEL_NBR;
    Image<T> const *img;
    
    PriorityQueue<T, OffsetT> pq;
    
    T **levels;
    
    OffsetT **children;
    OffsetT **brothers;
    CriterionT **criteria;
    
    size_t *labels;
    size_t curLabel;
    
    bool initialized;

    UINT imWidth ;
    UINT imHeight ;

    void reset()
    {
        if (!initialized)
          return;
        
        for (OffsetT i=0;i<COLUMN_NBR;i++)
        {
            if (!levels[i])
              break;
            delete[] levels[i]; levels[i] = NULL;
            delete[] children[i]; children[i] = NULL;
            delete[] brothers[i]; brothers[i] = NULL;
            delete[] criteria[i]; criteria[i] = NULL;
        }
    }

    void allocatePage(UINT page)
    {
        levels[page] =  new T[COLUMN_SIZE]();
        children[page] = new OffsetT[COLUMN_SIZE]();
        brothers[page] =  new OffsetT[COLUMN_SIZE]();
        criteria[page] = new CriterionT[COLUMN_SIZE]();
    }

    T initialize(const Image<T> &imIn, OffsetT *img_eti)
    {
        if (initialized)
          reset();
        
        this->img = &imIn;
        typename ImDtTypes<T>::lineType pix = img->getPixels();
        
        T minValue = ImDtTypes<T>::max();
        T tMinV = ImDtTypes<T>::min();
        OffsetT minOff = 0;
        for (size_t i=0;i<img->getPixelCount();i++)
          if (pix[i]<minValue)
          {
              minValue = pix[i];
              minOff = i;
              if (minValue==tMinV)
                break;
          }
          
        allocatePage(0);
        
        curLabel = 1;
        levels[0][curLabel] = minValue;
        
        memset(labels, 0, GRAY_LEVEL_NBR*sizeof(size_t));
        
        img_eti[minOff] = curLabel;
        labels[minValue] = curLabel;

        pq.initialize(*img);
        pq.push(minValue, minOff);
        
        size_t x, y, z;
        img->getCoordsFromOffset(minOff, x, y, z);
        
        getCriterion(curLabel).xmin = getCriterion(curLabel).xmax = x;
        getCriterion(curLabel).ymin = getCriterion(curLabel).ymax = y;
        getCriterion(curLabel).zmin = getCriterion(curLabel).zmax = z;

        getCriterion(curLabel).initialize();
        curLabel++;
        
        initialized = true;

        imWidth = img->getWidth();
        imHeight = img->getHeight();

        return minValue;
    }
    
    int nextLowerLabel(T valeur)
    {
        if ((curLabel & COLUMN_MOD) == 0)
            allocatePage(curLabel >> COLUMN_SHIFT);

        getLevel(curLabel) = valeur;
        int i;
        for(i=valeur-1;labels[i]==0;i--);

        getChild(curLabel) = getChild(labels[i]);
        getChild(labels[i]) = curLabel;
        getBrother(curLabel) = getBrother(getChild(curLabel));
        getBrother(getChild(curLabel)) = 0;
        getCriterion(curLabel).ymin = numeric_limits<unsigned short>::max();
        getCriterion(curLabel).xmin = numeric_limits<unsigned short>::max();
        getCriterion(curLabel).reset();
        return curLabel++;
    }

    int nextHigherLabel(T parent_valeur, T valeur)
    {
            if ((curLabel & COLUMN_MOD) == 0)
              allocatePage(curLabel >> COLUMN_SHIFT);

        getLevel(curLabel) = valeur;
        getBrother(curLabel) = getChild(labels[parent_valeur]);
        getChild(labels[parent_valeur]) = curLabel;
        getCriterion(curLabel).ymin = numeric_limits<unsigned short>::max();
        getCriterion(curLabel).xmin = numeric_limits<unsigned short>::max();
        getCriterion(curLabel).reset();
        return curLabel++;
    }
    
    bool subFlood(typename ImDtTypes<T>::lineType imgPix, UINT *img_eti, OffsetT p, OffsetT p_suiv)
    {
        int indice;
        
        if (imgPix[p_suiv]>imgPix[p]) 
        {
              int j;
              for(j=imgPix[p]+1;j<imgPix[p_suiv];j++) 
                labels[j]=0;
              indice = img_eti[p_suiv] = labels[j] = nextHigherLabel(imgPix[p], imgPix[p_suiv]);
          
        } 
        else if (labels[imgPix[p_suiv]]==0) 
            indice = img_eti[p_suiv] = labels[imgPix[p_suiv]] = nextLowerLabel(imgPix[p_suiv]);
        else 
            indice = img_eti[p_suiv] = labels[imgPix[p_suiv]];
        
        size_t x, y, z;
        img->getCoordsFromOffset(p_suiv, x, y, z);
        
        getCriterion(indice).xmin = MIN(getCriterion(indice).xmin, x);
        getCriterion(indice).xmax = MAX(getCriterion(indice).xmax, x);
        getCriterion(indice).ymin = MIN(getCriterion(indice).ymin, y);
        getCriterion(indice).ymax = MAX(getCriterion(indice).ymax, y);
        getCriterion(indice).zmin = MIN(getCriterion(indice).zmin, z);
        getCriterion(indice).zmax = MAX(getCriterion(indice).zmax, z);
        getCriterion(indice).update();
        pq.push(imgPix[p_suiv], p_suiv);
        
        if (imgPix[p_suiv]>imgPix[p])
        {
                pq.push(imgPix[p], p);
                return true;
        }
        return false;
    }

    void flood(const Image<T> &img, UINT *img_eti, unsigned int level)
    {
            OffsetT p;
            size_t imWidth = img.getWidth();
            size_t imHeight = img.getHeight();
            size_t imDepth = img.getDepth();
            size_t pixPerSlice = imWidth*imHeight;
            typename ImDtTypes<T>::lineType imgPix = img.getPixels();
            size_t x, y, z;
            
            while( (pq.getHigherLevel()>=level) && !pq.isEmpty())
            {
                p = pq.pop();
                img.getCoordsFromOffset(p, x, y, z);
                
                size_t p_suiv;

                if (imDepth>1) // for 3D
                {
                    if ( z<imDepth-1 && img_eti[p_suiv=p+pixPerSlice]==0) //z+1
                      if (subFlood(imgPix, img_eti, p, p_suiv))
                        continue;
                      
                    if ( z>0 && img_eti[p_suiv=p-pixPerSlice]==0) //z-1
                      if (subFlood(imgPix, img_eti, p, p_suiv))
                        continue;
                }
                
                if ( y<imHeight-1 && img_eti[p_suiv=p+imWidth]==0) //y+1
                  if (subFlood(imgPix, img_eti, p, p_suiv))
                    continue;

                if ( y>0 && img_eti[p_suiv=p-imWidth]==0) //y-1
                  if (subFlood(imgPix, img_eti, p, p_suiv))
                    continue;
                  
                if ( x<imWidth-1 && img_eti[p_suiv=p+1]==0) //x+1
                  if (subFlood(imgPix, img_eti, p, p_suiv))
                    continue;
                  
                if ( x>0 && img_eti[p_suiv=p-1]==0) //x-1
                  if (subFlood(imgPix, img_eti, p, p_suiv))
                    continue;
                  
            }
    }
    
public:
    MaxTree()
    {
        GRAY_LEVEL_NBR = ImDtTypes<T>::max()-ImDtTypes<T>::min()+1;
        
        children = new OffsetT*[COLUMN_NBR]();
        brothers =  new OffsetT*[COLUMN_NBR]();
        levels = new T*[COLUMN_NBR]();
        criteria = new CriterionT*[COLUMN_NBR]();
        labels = new size_t[GRAY_LEVEL_NBR];
        
        initialized = false;
    }
    ~MaxTree()
    {
        reset();
        
        delete[] children;
        delete[] brothers;
        delete[] levels;
        delete[] criteria;
        delete[] labels;
    }
    
    inline CriterionT &getCriterion(const OffsetT node)
    {
        return GET_TREE_OBJ(criteria, node);
    }
  
    inline T &getLevel(const OffsetT node)
    {
        return GET_TREE_OBJ(levels, node);
    }
  
    inline OffsetT &getChild(const OffsetT node)
    {
        return GET_TREE_OBJ(children, node);
    }
  
    inline OffsetT &getBrother(const OffsetT node)
    {
        return GET_TREE_OBJ(brothers, node);
    }
    
    inline int getLabelMax()
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

    int build(const Image<T> &img, OffsetT *img_eti)
    {
            T minValue = initialize(img, img_eti);

            flood(img, img_eti, minValue);
            return labels[minValue];
    }
    
    CriterionT updateCriteria(int node) 
    {
        int child = getChild(node);
        while (child!=0)
        {
            CriterionT c = updateCriteria(child);
            getCriterion(node).ymin = MIN(getCriterion(node).ymin, c.ymin);
            getCriterion(node).ymax = MAX(getCriterion(node).ymax, c.ymax);
            getCriterion(node).xmin = MIN(getCriterion(node).xmin, c.xmin);
            getCriterion(node).xmax = MAX(getCriterion(node).xmax, c.xmax);
            getCriterion(node).merge(getCriterion(child));
            child = getBrother(child);
        }
        return getCriterion(node);
    }
    
};



// NEW BMI    # ##################################################
//(tree, transformee_node, indicatrice_node, child, stopSize, (T)0, 0, hauteur, tree.getLevel(root), tree.getLevel(root));
template <class T, class CiterionT, class OffsetT>
void  ComputeDeltaUO(MaxTree<T,CiterionT,OffsetT> &tree, T* transformee_node, UINT* indicatrice_node, int node, int nParent, T prev_residue, UINT stop, UINT delta, int isPrevMaxT){

                    //self,node = 1, nParent =0, stop=0, delta = 0, isPrevMaxT = 0):
  int child; // index node
      T current_residue;
      UINT cNode, cParent; // attributes
      T lNode, lParent; // node levels, the same type than input image

      cNode =  tree.getCriterion(node).ymax-tree.getCriterion(node).ymin+1;// #current criterion
      lNode =  tree.getLevel(node);// #current level


      cParent =  tree.getCriterion(nParent).ymax-tree.getCriterion(nParent).ymin+1;// #current criterion
      lParent =  tree.getLevel(nParent);// #current level

      int flag;

      if ((cParent - cNode) <= delta){
        flag = 1;
      }
      else{
        flag = 0;
      }
      if (flag){
        current_residue  = prev_residue + lNode - lParent ;
      }
      else{
        current_residue  = lNode - lParent;
      }

      transformee_node[node] = transformee_node[nParent];
      indicatrice_node[node] = indicatrice_node[nParent];

      int isMaxT = 0;

      if(cNode < stop){
        if (current_residue > transformee_node[node]){
          //          std::cout<<"UPDATE RES\n";
          isMaxT = 1;
          transformee_node[node] = current_residue;
          if(! (isPrevMaxT && flag)){
            indicatrice_node[node]  = cNode + 1;
          }
          
        }
      }
      else{
        indicatrice_node[node]  = 0;
      }
      child=tree.getChild(node);
      while (child!=0){
        ComputeDeltaUO(tree, transformee_node, indicatrice_node, child, node, current_residue, stop, delta,isMaxT);
        child = tree.getBrother(child);
      }

}
template <class T, class CiterionT, class OffsetT>
void compute_max(MaxTree<T,CiterionT,OffsetT> &tree, T* transformee_node, UINT* indicatrice_node, int node, UINT stop, T max_tr, unsigned int max_in, unsigned int hauteur_parent, T valeur_parent, T previous_value)
{
        T m;
        T max_node;
        unsigned int max_criterion;
        UINT child;
        UINT hauteur = tree.getCriterion(node).ymax-tree.getCriterion(node).ymin+1;
        m = (hauteur==hauteur_parent) ? tree.getLevel(node)-previous_value : tree.getLevel(node)-valeur_parent;
        if (hauteur>=stop) 
        {
                max_node = max_tr;
                max_criterion = 0;//max_in;
                transformee_node[node] = max_node;

                indicatrice_node[node]=0;
                child=tree.getChild(node);
        } 
        else 
        {
                if (m>max_tr) 
                {
                        max_node=m;
                        max_criterion=hauteur;
                } else 
                {
                        max_node=max_tr;
                        max_criterion=max_in;
                }
                transformee_node[node]=max_node;
                indicatrice_node[node]=max_criterion+1;
                child=tree.getChild(node);
        }
        if (hauteur==hauteur_parent) 
        {
                while (child!=0) 
                {
                    if (hauteur_parent>stop) 
                      compute_max(tree, transformee_node, indicatrice_node, child, stop, max_node, max_criterion, hauteur, tree.getLevel(node), previous_value);
                    else 
                      compute_max(tree, transformee_node, indicatrice_node, child, stop, max_node, max_criterion, hauteur, tree.getLevel(node)/*valeur_parent*/, previous_value);
                    child = tree.getBrother(child);
                }
        } 
        else 
        {
                while (child!=0) 
                {
                    compute_max(tree, transformee_node, indicatrice_node, child, stop, max_node, max_criterion, hauteur, tree.getLevel(node), valeur_parent);
                    child = tree.getBrother(child);
                }
        }
}

template <class T, class CiterionT, class OffsetT>
void compute_contrast(MaxTree<T,CiterionT,OffsetT> &tree, T* transformee_node, UINT* indicatrice_node, int root, UINT stopSize,UINT delta = 0)
{

  int child;
  UINT hauteur;

  transformee_node[root]=0;
  indicatrice_node[root]=0;

  tree.updateCriteria(root);

  hauteur = tree.getCriterion(root).ymax - tree.getCriterion(root).ymin+1;
  child = tree.getChild(root);
  if(delta == 0){
    while (child!=0) 
      {
        compute_max(tree, transformee_node, indicatrice_node, child, stopSize, (T)0, 0, hauteur, tree.getLevel(root), tree.getLevel(root));
        child = tree.getBrother(child);
      }
  }// END delta == 0
  else{
    while (child!=0) 
      {
        ComputeDeltaUO(tree, transformee_node, indicatrice_node, child, root/*parent*/, (T)0/* prev_residue*/, stopSize /*stop*/, delta, 0 /*isPrevMaxT*/);

        child = tree.getBrother(child);
      }
  }//END dela != 0
}
template <class T, class CiterionT, class OffsetT>
void compute_contrast_matthieuNoDelta(MaxTree<T,CiterionT,OffsetT> &tree, T* transformee_node, UINT* indicatrice_node, int root, UINT stopSize)
{
        int child;
        UINT hauteur = tree.getCriterion(root).ymax - tree.getCriterion(root).ymin+1;
        transformee_node[root]=0;
        indicatrice_node[root]=0;
        tree.updateCriteria(root);
        child = tree.getChild(root);
        while (child!=0) 
        {
            compute_max(tree, transformee_node, indicatrice_node, child, stopSize, (T)0, 0, hauteur, tree.getLevel(root), tree.getLevel(root));
            child = tree.getBrother(child);
        }
}


#endif // SWIG

    /**
     * Ultimate Opening using the max-trees
     * 
     * Max-tree based algorithm as described by Fabrizio and Marcotegui (2009) \cite fabrizio_fast_2009
     * \warning 4-connex only (6-connex in 3D)
     * \param[in] imIn Input image
     * \param[out] imOut The transformation image
     * \param[out] imIndic The indicator image
     * \param[in] stopSize (optional)
     */
    template <class T1, class T2>
    RES_T ultimateOpen(const Image<T1> &imIn, Image<T1> &imTrans, Image<T2> &imIndic, int stopSize=-1, UINT delta = 0)
    {
        ASSERT_ALLOCATED(&imIn, &imTrans, &imIndic);
        ASSERT_SAME_SIZE(&imIn, &imTrans, &imIndic);
        
        if (stopSize==-1)
          stopSize = imIn.getHeight()-1;
          
        int imSize = imIn.getPixelCount();
        UINT *img_eti = new UINT[imSize]();
        
        MaxTree<T1> tree;
        int root = tree.build(imIn, img_eti);
        
        T1 *transformee_node = new T1[tree.getLabelMax()]();
        UINT *indicatrice_node = new UINT[tree.getLabelMax()]();

        compute_contrast(tree, transformee_node, indicatrice_node, root, stopSize,delta);

        
        typename ImDtTypes<T1>::lineType transformeePix = imTrans.getPixels();
        typename ImDtTypes<T2>::lineType indicatricePix = imIndic.getPixels();

        for(int i=0;i<imSize;i++) 
        {
            transformeePix[i]=transformee_node[img_eti[i]];
            indicatricePix[i]=indicatrice_node[img_eti[i]];
        }
        
        delete[] img_eti;
        delete[] transformee_node;
        delete[] indicatrice_node;
            
        imTrans.modified();
        imIndic.modified();
        
        return RES_OK;
    }  



#ifndef SWIG

struct AreaCriterion
{
  size_t value;
  inline void init() { value = 1; }
  inline void reset() { value = 0; }
  inline void merge(AreaCriterion &other) { value += other.value; }
  inline void update() { value += 1; }
};


// NEW BMI    # ##################################################
//(tree, transformee_node, indicatrice_node, child, stopSize, (T)0, 0, hauteur, tree.getLevel(root), tree.getLevel(root));
template <class T, class CriterionT, class OffsetT>
void  ComputeDeltaUOMSER(MaxTree<T,CriterionT,OffsetT> &tree, T* transformee_node, UINT* indicatrice_node, int node, int nParent,  int first_ancestor, UINT stop, UINT delta, int isPrevMaxT,UINT minArea=0,T threshold=0){

  // "node": the current node; "nParent": its direct parent (allows
  // attribute comparison for Delta versions); "first_ancestor": the
  // first ancestor with a different attribute (used for residue
  // computation level(first_ancestor) - level(node) and for area stability computation

                    //self,node = 1, nParent =0, stop=0, delta = 0, isPrevMaxT = 0):
  int child; // index node
  T current_residue, stab_residue;
      UINT cNode, cParent; // attributes
      CriterionT aNode, aParent,aAncestor;
      T lNode, lParent, lAncestor; // node levels, the same type than input image
      float stability;


      cNode =  tree.getCriterion(node).ymax-tree.getCriterion(node).ymin+1;// #current criterion
      aNode =  tree.getCriterion(node).crit;
      lNode =  tree.getLevel(node);// #current level


      cParent =  tree.getCriterion(nParent).ymax-tree.getCriterion(nParent).ymin+1;// #current criterion
      aParent  = tree.getCriterion(nParent).crit;
      aAncestor  = tree.getCriterion(first_ancestor).crit;
      lParent =  tree.getLevel(nParent);// #current level
      lAncestor =  tree.getLevel(first_ancestor);// #current level
      int flag;


      if ((cParent - cNode) <= delta){
        flag = 1;
      }
      else{
        flag = 0;
      }
      if (flag){
        stability  = 1 - ((aAncestor.value - aNode.value)*1.0/aAncestor.value);
        current_residue  =  lNode - lAncestor ;
        stab_residue = round(current_residue * stability);
      }
      else{
        stability = 1 - ((aParent.value - aNode.value)*1.0/aParent.value);
        current_residue  = (lNode - lParent);
        stab_residue = round(current_residue * stability);

      }


      transformee_node[node] = transformee_node[nParent];
      indicatrice_node[node] = indicatrice_node[nParent];
      int isMaxT = 0;
      if(cNode < stop){
        if (stab_residue > transformee_node[node] && stab_residue > threshold && aNode.value > minArea){
          isMaxT = 1;
          transformee_node[node] = stab_residue;

          if(! (isPrevMaxT && flag)){
            indicatrice_node[node]  = cNode + 1;
          }
          
        }
      }
      else{
        indicatrice_node[node]  = 0;
      }
      child=tree.getChild(node);
      while (child!=0){
        if(flag && (cNode < stop)){
          ComputeDeltaUOMSER(tree, transformee_node, indicatrice_node, child, node,first_ancestor, stop, delta,isMaxT,minArea,threshold);
        }
        else{

          ComputeDeltaUOMSER(tree, transformee_node, indicatrice_node, child, node, nParent,stop, delta,isMaxT,minArea,threshold);
        }
        child = tree.getBrother(child);
      }

}

inline void computeFillAspectRatioFactor(UINT wNode,UINT cNode,UINT area,UINT width,UINT height,float &fillRatio,float &AspectRatio)
{
    //compute fillRatio and AspectRatio
    UINT minHW, maxHW;
    minHW = MIN(wNode,cNode);
    maxHW = MAX(wNode,cNode);
    fillRatio = area*1.0/(wNode * cNode * 1.0);
    AspectRatio = minHW*1.0/maxHW ;

    if(AspectRatio <= 0.4)// && fillRatio < 0.4)
        AspectRatio = 0.0;
    else
        AspectRatio = pow(AspectRatio,1.0/3);

    if(fillRatio <= 0.2  || fillRatio >= 0.9 )
        fillRatio = 0.0;
    else{
        fillRatio = abs(fillRatio - 0.55);
        fillRatio = pow((1.0- (fillRatio*1.0/0.35)),1.0/3);
    }

    if(AspectRatio <= 0.4 && fillRatio >= 0.4){//a verifier
        AspectRatio = 1;//0.9
        fillRatio = 1;
    }//0.9

    if(area < 20 ||  area > (0.9* width* height) ||  cNode < 5 ){//|| cNode > (tree.getImHeight()/3)){
        AspectRatio = 0.0;
        fillRatio   = 0.0;
    }
}

// NEW BMI    # ##################################################
//(tree, transformee_node, indicatrice_node, child, stopSize, (T)0, 0, hauteur, tree.getLevel(root), tree.getLevel(root));
template <class T, class CriterionT, class OffsetT>
void  ComputeDeltaUOMSERSC(MaxTree<T,CriterionT,OffsetT> &tree, T* transformee_node, UINT* indicatrice_node,int node, int nParent,  int first_ancestor, UINT stop, UINT delta, int isPrevMaxT){

    // "node": the current node; "nParent": its direct parent (allows
    // attribute comparison for Delta versions); "first_ancestor": the
    // first ancestor with a different attribute (used for residue
    // computation level(first_ancestor) - level(node) and for area stability computation

    //self,node = 1, nParent =0, stop=0, delta = 0, isPrevMaxT = 0):
    int child; // index node
    T current_residue, stab_residue;
    UINT cNode, cParent, wNode; // attributes
    CriterionT aNode, aParent,aAncestor;
    T lNode, lParent,lAncestor; /*wParent set but not used*/ // node levels, the same type than input image
    float stability, fillRatio, AspectRatio, fac;


    cNode =  tree.getCriterion(node).ymax-tree.getCriterion(node).ymin+1;// #current criterion
    wNode =  tree.getCriterion(node).xmax-tree.getCriterion(node).xmin+1;// #width
    aNode =  tree.getCriterion(node).crit;
    lNode =  tree.getLevel(node);// #current level


    cParent =  tree.getCriterion(nParent).ymax-tree.getCriterion(nParent).ymin+1;// #current criterion
//     wParent =  tree.getCriterion(nParent).xmax-tree.getCriterion(nParent).xmin+1;// #width   
    aParent  = tree.getCriterion(nParent).crit;
    aAncestor  = tree.getCriterion(first_ancestor).crit;
    lParent =  tree.getLevel(nParent);// #current level
    lAncestor =  tree.getLevel(first_ancestor);// #current level
    int flag;


    if ((cParent - cNode) <= delta){
        flag = 1;
    }
    else{
        flag = 0;
    }
    
    computeFillAspectRatioFactor(wNode,cNode,aNode.value,tree.getImWidth(),tree.getImHeight(),fillRatio,AspectRatio);

    if (flag){//no significant attribute change
        stability = pow((1.0 - ((aAncestor.value - aNode.value)*1.0/aAncestor.value)),1.0/3);
        fac =  (stability * AspectRatio * fillRatio);
        current_residue  =  lNode - lAncestor ;
        stab_residue = round(current_residue * fac);
    }
    else{
        stability = pow((1.0 - ((aParent.value - aNode.value)*1.0/aParent.value)),1.0/3);
        fac =  (stability * AspectRatio * fillRatio);
        current_residue  = (lNode - lParent);
        stab_residue = round(current_residue * fac);
    }


    transformee_node[node] = transformee_node[nParent];
    indicatrice_node[node] = indicatrice_node[nParent];


    int isMaxT = 0;
    if(cNode < stop){

        if (stab_residue > transformee_node[node]){
            isMaxT = 1;
            transformee_node[node] = stab_residue;

            if(! (isPrevMaxT and flag)){
                indicatrice_node[node]  = cNode + 1;                
            }

        }
    }
    else
        indicatrice_node[node]  = 0;

    child=tree.getChild(node);
    while (child!=0){
        if(flag && (cNode < stop)){
            ComputeDeltaUOMSERSC(tree, transformee_node, indicatrice_node, child, node,first_ancestor, stop, delta,isMaxT);
        }
        else{
            ComputeDeltaUOMSERSC(tree, transformee_node, indicatrice_node, child, node, nParent,stop, delta,isMaxT);
        }
        child = tree.getBrother(child);
    }

}




template <class T, class CriterionT, class OffsetT>
void compute_contrast_MSER(MaxTree<T,CriterionT,OffsetT> &tree, T* transformee_node, UINT* indicatrice_node, int root, UINT stopSize,UINT delta = 0, UINT minArea=0,T threshold=0, bool use_textShape =0)
{

  int child;
//   UINT hauteur;

  transformee_node[root]=0;
  indicatrice_node[root]=0;

  tree.updateCriteria(root);
  //  hauteur = tree.getCriterion(root).ymax - tree.getCriterion(root).ymin+1;
  child = tree.getChild(root);

    while (child!=0) 
      {
       if(!use_textShape)
        ComputeDeltaUOMSER(tree, transformee_node, indicatrice_node, child, root/*parent*/,root /*first_ancestor*/, stopSize /*stop*/, delta, 0 /*isPrevMaxT*/,minArea,threshold);
       else
        ComputeDeltaUOMSERSC(tree, transformee_node, indicatrice_node, child, root/*parent*/,root /*first_ancestor*/, stopSize /*stop*/, delta, 0 /*isPrevMaxT*/);

        child = tree.getBrother(child);
      }

}

#endif // SWIG

    template <class T1, class T2>
    RES_T ultimateOpenMSER(const Image<T1> &imIn, Image<T1> &imTrans, Image<T2> &imIndic, int stopSize=-1, UINT delta = 0, UINT minArea=0,T1 threshold=0, bool use_textShape =0)
    {
        ASSERT_ALLOCATED(&imIn, &imTrans, &imIndic);
        ASSERT_SAME_SIZE(&imIn, &imTrans, &imIndic);
        
        int imSize = imIn.getPixelCount();
        UINT *img_eti = new UINT[imSize]();
        
        MaxTree<T1, AreaCriterion> tree;
        int root = tree.build(imIn, img_eti);

        if(stopSize == -1){
          stopSize= imIn.getHeight()-1;
        }

        
        T1 *transformee_node = new T1[tree.getLabelMax()]();
        UINT *indicatrice_node = new UINT[tree.getLabelMax()]();

        compute_contrast_MSER(tree, transformee_node, indicatrice_node, root, stopSize,delta,minArea,threshold,use_textShape);

        
        typename ImDtTypes<T1>::lineType transformeePix = imTrans.getPixels();
        typename ImDtTypes<T2>::lineType indicatricePix = imIndic.getPixels();

        for(int i=0;i<imSize;i++) 
        {
            transformeePix[i]=transformee_node[img_eti[i]];
            indicatricePix[i]=indicatrice_node[img_eti[i]];
        }
        
        delete[] img_eti;
        delete[] transformee_node;
        delete[] indicatrice_node;
            
        imTrans.modified();
        imIndic.modified();
        
        return RES_OK;
    }  

    
    
#ifndef SWIG

template <class T, class CriterionT, class OffsetT>
void processMaxTree(MaxTree<T,CriterionT,OffsetT> &tree, UINT node, T* lut_out, T previousLevel, UINT stop)
{
                                //MORPHEE_ENTER_FUNCTION("s_OpeningTree::computeMaxTree");
        UINT child;
        T nodeLevel = tree.getLevel(node);
        T currentLevel = (tree.getCriterion(node).crit.value < stop)? previousLevel:nodeLevel;

        lut_out[node] = currentLevel;
        
        child = tree.getChild(node);
        while (child!=0) 
          {
            processMaxTree(tree, child, lut_out, currentLevel, stop);
            child = tree.getBrother(child);
          }
        
}

template <class T, class CriterionT, class OffsetT>
void compute_AttributeOpening(MaxTree<T,CriterionT,OffsetT> &tree, T* lut_node, int root, UINT stopSize)
{

    int child;

    lut_node[root] = tree.getLevel(root);

    tree.updateCriteria(root);

    child = tree.getChild(root);

    while (child!=0) 
      {
        processMaxTree(tree, child, lut_node, tree.getLevel(root), stopSize);
        child = tree.getBrother(child);
      }
  
  
}

    
    template <class T, class CriterionT>
    RES_T attributeOpen(const Image<T> &imIn, Image<T> &imOut, int stopSize)
    {
        ASSERT_ALLOCATED(&imIn, &imOut);
        ASSERT_SAME_SIZE(&imIn, &imOut);
        
        int imSize = imIn.getPixelCount();
        UINT *img_eti = new UINT[imSize]();
        
        MaxTree<T,CriterionT> tree;
        int root = tree.build(imIn, img_eti);
        
        T *out_node = new T[tree.getLabelMax()]();

        compute_AttributeOpening(tree, out_node, root, stopSize);

        
        typename ImDtTypes<T>::lineType outPix = imOut.getPixels();

        for(int i=0;i<imSize;i++) 
            outPix[i] = out_node[img_eti[i]];
        
        delete[] img_eti;
        delete[] out_node;
            
        imOut.modified();
        
        return RES_OK;
    }  

#endif // SWIG

    /**
    * Area opening
     * 
     * Max-tree based algorithm
     * \warning 4-connex only (6-connex in 3D)
     * \param[in] imIn Input image
     * \param[in] size The size of the opening
     * \param[out] imOut Output image
    */
    template <class T>
    RES_T areaOpen(const Image<T> &imIn, int stopSize, Image<T> &imOut)
    {
        return attributeOpen<T, AreaCriterion>(imIn, imOut, stopSize);
    }
    
    /**
    * Area closing
     * 
     * Max-tree based algorithm
     * \warning 4-connex only (6-connex in 3D)
     * \param[in] imIn Input image
     * \param[in] size The size of the closing
     * \param[out] imOut Output image
    */
    template <class T>
    RES_T areaClose(const Image<T> &imIn, int stopSize, Image<T> &imOut)
    {
        ASSERT_ALLOCATED(&imIn, &imOut);
        ASSERT_SAME_SIZE(&imIn, &imOut);
        
        ImageFreezer freeze(imOut);
        
        Image<T> tmpIm(imIn);
        inv(imIn, tmpIm);        
        RES_T res = attributeOpen<T, AreaCriterion>(tmpIm, imOut, stopSize);
        inv(imOut, imOut);
        
        return res;
    }
    
    /** \} */

} // namespace smil


#endif // _D_SKELETON_HPP

