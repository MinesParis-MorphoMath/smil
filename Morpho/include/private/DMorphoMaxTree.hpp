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

#include "DMorphImageOperations.hpp"//BMI
#include "DMorphoHierarQ.hpp"//BMI
#include "Core/include/DImage.h"
#include "Base/include/private/DImageHistogram.hpp"
#include "Morpho/include/private/DMorphoMaxTreeCriteria.hpp"

#include <complex>
#include <math.h>
namespace smil
{
    /**
    * \ingroup Morpho
    * \defgroup MaxTree
    * \{
    */

    
#ifndef SWIG


#define COLUMN_NBR 20// 16
#define COLUMN_SIZE 2097152//131072

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

  // BEGIN BMI


template <class T, class CriterionT, class OffsetT=size_t, class LabelT = UINT32>
class MaxTree2
{
private:
    size_t GRAY_LEVEL_NBR;
    Image<T> const *img;
    
  //    PriorityQueue<T, OffsetT> pq;
  //  HierarchicalQueue<T,size_t, STD_Queue<size_t> > &hq;
  HierarchicalQueue<T,OffsetT> hq;

    T **levels;
    
    LabelT **children;
    LabelT **brothers;
    CriterionT **criteria;
    
  LabelT *labels, curLabel;
    
    bool initialized;

    size_t imWidth ;
    size_t imHeight ;


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
        children[page] = new LabelT[COLUMN_SIZE]();
        brothers[page] =  new LabelT[COLUMN_SIZE]();
        criteria[page] = new CriterionT[COLUMN_SIZE]();
    }

    T initialize(const Image<T> &imIn, LabelT *img_eti, const StrElt &se)
    {
      imIn.getSize(imSize);
      pixPerSlice = imSize[0]*imSize[1];

	// BMI BEGIN
	sePtsNbr = sePts.size();
	dOffsets.clear();
	sePts.clear();
	oddSE = se.odd;
            
	// set an offset distance for each se point (!=0,0,0)
	for(vector<IntPoint>::const_iterator it = se.points.begin() ; it!=se.points.end() ; it++)
	  if (it->x!=0 || it->y!=0 || it->z!=0)
            {
	      sePts.push_back(*it);
	      dOffsets.push_back(it->x + it->y*imSize[0] + it->z*imSize[0]*imSize[1]);
            }
            
	sePtsNbr = sePts.size();
	// BMI END

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
        
        memset(labels, 0, GRAY_LEVEL_NBR*sizeof(LabelT));// BMI labels? quel rapport avec img_eti?
        
        img_eti[minOff] = curLabel;
        labels[minValue] = curLabel;

        hq.initialize(*img);
        hq.push(minValue, minOff);


        size_t x, y, z;
        img->getCoordsFromOffset(minOff, x, y, z);
	//	std::cout<<"PUSH:offset="<<minOff<<"(x,y)="<<x<<", "<<y<<", val="<<minValue<<"\n";

        
	//        getCriterion(curLabel).xmin = getCriterion(curLabel).xmax = x; // A voir comment on peut melanger des criteres... BMI
	//        getCriterion(curLabel).ymin = getCriterion(curLabel).ymax = y;
	//        getCriterion(curLabel).zmin = getCriterion(curLabel).zmax = z;

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
        getCriterion(curLabel).reset();
        return curLabel++;
    }
    
    bool subFlood(typename ImDtTypes<T>::lineType imgPix, LabelT *img_eti, OffsetT p, OffsetT p_suiv)
    {
        LabelT indice;
        
        if (imgPix[p_suiv]>imgPix[p]) 
        {
              LabelT j;
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
        
        getCriterion(indice).update(x,y,z);
        hq.push(imgPix[p_suiv], p_suiv);

	//	std::cout<<"PUSH:offset="<<p_suiv<<"("<<x<<","<<y<<"), val="<<int(imgPix[p_suiv])<<"\n";
        
        if (imgPix[p_suiv]>imgPix[p])
        {
                hq.push(imgPix[p], p);
		//		std::cout<<"PUSH_P:offset="<<p<<", val="<<int(imgPix[p])<<"\n";
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
            size_t x0, y0, z0;
            

            while( (hq.getHigherLevel()>=level) && !hq.isEmpty())
            {
                p = hq.pop();

                img.getCoordsFromOffset(p, x0, y0, z0);
		//		std::cout<<"POP:offset="<<p<<"(x,y)="<<x0<<", "<<y0<<"\n";
		//                std::cout<<"-------------------------\n";

                
                bool oddLine = oddSE && ((y0)%2);
                
                int x, y, z; // not size_t in order to (possibly be negative!)
                OffsetT p_suiv;
                
                for(UINT i=0;i<sePtsNbr;i++)
                {
                    IntPoint &pt = sePts[i];
                    x = x0 + pt.x;
                    y = y0 + pt.y;
                    z = z0 + pt.z;
                    
                    if (oddLine)
                      x += (((y+1)%2)!=0);
                  
                    if (x>=0 && x<(int)imSize[0] && y>=0 && y<(int)imSize[1] && z>=0 && z<(int)imSize[2])
                    {
                        p_suiv = p + dOffsets[i];
			if (oddLine)
			  p_suiv += (((y+1)%2)!=0);

			if(img_eti[p_suiv]==0){                   
			  //			  std::cout<<"("<<x0<<","<<y0<<"),"<<"("<<x<<","<<y<<"),"<<p<<","<<p_suiv<<"\n";     
			  if(subFlood(imgPix, img_eti, p, p_suiv))
			    break;
			}
                    }
                }// for each ngb
                  
            }// while hq.notEmpty
    }//void flood
    
public:
  MaxTree2():hq(true)// TOTO BMI
    {
        GRAY_LEVEL_NBR = ImDtTypes<T>::max()-ImDtTypes<T>::min()+1;
	//	hq.reverse();// change priority order (max first)
	//	hq =   HierarchicalQueue<T,OffsetT> (true);

        children = new LabelT*[COLUMN_NBR]();
        brothers =  new LabelT*[COLUMN_NBR]();
        levels = new T*[COLUMN_NBR]();
        criteria = new CriterionT*[COLUMN_NBR]();
        labels = new LabelT[GRAY_LEVEL_NBR];
        
        initialized = false;

    }
    ~MaxTree2()
    {
        reset();
        
        delete[] children;
        delete[] brothers;
        delete[] levels;
        delete[] criteria;
        delete[] labels;
    }
protected:
        size_t imSize[3], pixPerSlice;

        vector<IntPoint> sePts;
        UINT sePtsNbr;
        bool oddSE;
        vector<int> dOffsets;

public:

    inline CriterionT &getCriterion(const LabelT node)
    {
        return GET_TREE_OBJ(criteria, node);
    }
  
    inline T &getLevel(const LabelT node)
    {
        return GET_TREE_OBJ(levels, node);
    }
  
    inline LabelT &getChild(const LabelT node)
    {
        return GET_TREE_OBJ(children, node);
    }
  
    inline LabelT &getBrother(const LabelT node)
    {
        return GET_TREE_OBJ(brothers, node);
    }
    
    inline LabelT getLabelMax()
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

    int build(const Image<T> &img, LabelT *img_eti, const StrElt &se)
    {
      T minValue = initialize(img, img_eti,se);

      flood(img, img_eti, minValue);// BMI: dOffset already contains se information
      return labels[minValue];
    }

  // BMI (from Andres)
  /// Update criteria of a given max-tree node
  CriterionT updateCriteria(const int node);
    

    CriterionT PREVIOUS_OLD_FROMJONATHAN_updateCriteria(LabelT node) 
    {
        LabelT child = getChild(node);
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

  // END BMI

// Update criteria on a given max-tree node.// From Andres
template <class T, class CriterionT, class OffsetT, class LabelT>
inline CriterionT MaxTree2<T, CriterionT,OffsetT,LabelT>::updateCriteria( const int node )
{
  LabelT child = getChild(node);
  while ( child != 0 )
  {
    //    std::cout<<"call update with child:"<<child<<"\n";
    CriterionT c = updateCriteria( child );
    //    std::cout<<"BEFORE MERGE("<<node<<","<<child<<")="<<getCriterion ( node ).getAttributeValue()<<","<<getCriterion ( child ).getAttributeValue()<<"\n";
    getCriterion( node ).merge(&getCriterion( child ) );
    //    std::cout<<"AFTER MERGE:"<<getCriterion ( node ).getAttributeValue()<<"\n";

    //DEBUG    LabelT child_toto = child;
    child = getBrother( child );
    //    std::cout<<"get brother:"<<child_toto<<","<<child<<"\n";
    //    std::cout<<"------------------\n";

  }
  return getCriterion( node );
}







// NEW BMI    # ##################################################
//(tree, transformee_node, indicatrice_node, child, stopSize, (T)0, 0, hauteur, tree.getLevel(root), tree.getLevel(root));
template <class T, class CriterionT, class OffsetT, class LabelT, class tAttType>
void  ComputeDeltaUO(MaxTree2<T,CriterionT,OffsetT,LabelT> &tree, T* transformee_node, tAttType* indicatrice_node, int node, int nParent, T prev_residue, tAttType stop, UINT delta, int isPrevMaxT){

                    //self,node = 1, nParent =0, stop=0, delta = 0, isPrevMaxT = 0):
  int child; // index node
      T current_residue;
      UINT cNode, cParent; // attributes
      T lNode, lParent; // node levels, the same type than input image

      cNode =  tree.getCriterion(node).getAttributeValue();//ymax-tree.getCriterion(node).ymin+1;// #current criterion
      lNode =  tree.getLevel(node);// #current level


      cParent =  tree.getCriterion(nParent).getAttributeValue();//ymax-tree.getCriterion(nParent).ymin+1;// #current criterion
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
  template <class T1, class T2>
  void compute_max(MaxTree2<T1,HeightCriterion,size_t,UINT32> &tree, T1* transformee_node, T2* indicatrice_node, UINT32 node, T2 stop, T1 max_tr, T2 max_in, T2 hauteur_parent, T1 valeur_parent, T1 previous_value)
{
        T1 m;
        T1 max_node;
        T2 max_criterion;
        UINT32 child;
        T2 hauteur = tree.getCriterion(node).getAttributeValue();//ymax-tree.getCriterion(node).ymin+1;
	//	std::cout<<"IN COMPUTE_MAX:"<<"; node="<<node<<"\n";
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
		  //		  if(child > tree.getLabelMax()){
		  //		  std::cout<<"ERROR call child:"<<child<<"\n";
		  //		  }
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
		  //		  if(child > tree.getLabelMax()){
		  //		  std::cout<<"ERROR call child:"<<child<<"\n";
		  //		  }

                    compute_max(tree, transformee_node, indicatrice_node, child, stop, max_node, max_criterion, hauteur, tree.getLevel(node), valeur_parent);
                    child = tree.getBrother(child);
                }
        }
}

  template <class T1, class T2>
void compute_contrast(MaxTree2<T1,HeightCriterion,size_t,UINT32> &tree, T1* transformee_node, T2* indicatrice_node, UINT32 root, T2 stopSize,UINT delta = 0)
{

  UINT32 child;
  T2 hauteur;
  //  std::cout<<"IN compute_contrast\n";

  transformee_node[root]=0;
  indicatrice_node[root]=0;
  //  std::cout<<"IN compute_contrast 2\n";
  tree.updateCriteria(root);

  hauteur = tree.getCriterion(root).getAttributeValue();//ymax - tree.getCriterion(root).ymin+1;
  child = tree.getChild(root);
  if(delta == 0){
    while (child!=0) 
      {
	//	std::cout<<"child"<<child<<"; level"<<tree.getLevel(child);

      compute_max(tree, transformee_node, indicatrice_node, child, stopSize, (T1)0, (T2)0, hauteur, tree.getLevel(root), tree.getLevel(root));
        child = tree.getBrother(child);
      }
  }// END delta == 0
  else{
    while (child!=0) 
      {
	//	std::cout<<"child"<<child<<"; level"<<tree.getLevel(child);
        ComputeDeltaUO(tree, transformee_node, indicatrice_node, child, root/*parent*/, (T1)0/* prev_residue*/, stopSize /*stop*/, delta, 0 /*isPrevMaxT*/);

        child = tree.getBrother(child);
      }
  }//END dela != 0
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
    RES_T ultimateOpen(const Image<T1> &imIn, Image<T1> &imTrans, Image<T2> &imIndic, const StrElt &se, T2 stopSize=-1, UINT delta = 0)
    {



        ASSERT_ALLOCATED(&imIn, &imTrans, &imIndic);
        ASSERT_SAME_SIZE(&imIn, &imTrans, &imIndic);
        
        if (stopSize==-1)
          stopSize = imIn.getHeight()-1;
          
        int imSize = imIn.getPixelCount();
        UINT *img_eti = new UINT[imSize]();
     

        MaxTree2<T1,HeightCriterion,size_t,UINT32> tree;
        UINT32 root = tree.build(imIn, img_eti,se);

	std::cout<<"ULTIMATE OPEN, after tree.build"<<"NB VERTEX="<<tree.getLabelMax()<<"\n";
        T1 *transformee_node = new T1[tree.getLabelMax()]();
        T2 *indicatrice_node = new T2[tree.getLabelMax()]();
	//	std::cout<<"ULTIMATE OPEN, after memory allocation\n";
        compute_contrast(tree, transformee_node, indicatrice_node, root, stopSize,delta);
	//	std::cout<<"ULTIMATE OPEN, compute_contrast\n";
        
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

struct BMIAreaCriterion
{
  size_t value;
  inline void init() { value = 1; }
  inline void reset() { value = 0; }
  inline void merge(BMIAreaCriterion &other) { value += other.value; }
  inline void update() { value += 1; }
};
struct BMIHeightCriterion
{
  size_t ymax;
  size_t ymin;
  size_t value;
  inline void init() { ymax = 0; ymin=6500000; }
  inline void reset() { ymax = 0; ymin = 6500000;}
  inline void merge(BMIHeightCriterion &other) { ymax = max(ymax,other.ymax);ymin = min(ymin,other.ymin); }
  inline void update() { value = ymax-ymin+1; }
};



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


#endif // SWIG

    
    
#ifndef SWIG

  template <class T, class CriterionT, class OffsetT, class LabelT, class tAttType>
  void processMaxTree(MaxTree2<T,CriterionT,OffsetT,LabelT> &tree, LabelT node, T* lut_out, T previousLevel, tAttType stop)
{
                                //MORPHEE_ENTER_FUNCTION("s_OpeningTree::computeMaxTree");
        LabelT child;
        T nodeLevel = tree.getLevel(node);

	T currentLevel = ( tree.getCriterion( node ) < stop ) ? previousLevel : nodeLevel;

        lut_out[node] = currentLevel;
	//	std::cout<<"node:"<<node<<"->"<<int(currentLevel)<<"\n";
        
        child = tree.getChild(node);
        while (child!=0) 
          {
            processMaxTree(tree, child, lut_out, currentLevel, stop);
            child = tree.getBrother(child);
          }
        
}

  template <class T, class CriterionT, class OffsetT, class LabelT, class tAttType>
  void compute_AttributeOpening(MaxTree2<T,CriterionT,OffsetT,LabelT> &tree, T* lut_node, LabelT root, tAttType stopSize)
{
    LabelT child;

    lut_node[root] = tree.getLevel(root);

    tree.updateCriteria(root);

    child = tree.getChild(root);

    while (child!=0) 
      {
        processMaxTree(tree, child, lut_node, tree.getLevel(root), stopSize);
        child = tree.getBrother(child);
      }
  
  
}// compute_AttributeOpening

    
  template <class T, class CriterionT, class OffsetT, class LabelT>
    RES_T attributeOpen(const Image<T> &imIn, Image<T> &imOut, size_t stopSize, const StrElt &se)
    {
        ASSERT_ALLOCATED(&imIn, &imOut);
        ASSERT_SAME_SIZE(&imIn, &imOut);

        size_t imSize = imIn.getPixelCount();
        LabelT *img_eti = new LabelT[imSize]();
        
        MaxTree2<T,CriterionT,OffsetT,LabelT> tree;
        LabelT root = tree.build(imIn, img_eti,se);


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

        for(size_t i=0;i<imSize;i++) 
            outPix[i] = out_node[img_eti[i]];
        
        delete[] img_eti;
        delete[] out_node;
            
        imOut.modified();
        
        return RES_OK;
    }  

#endif // SWIG

    /**
    * Height opening
     * 
     * Max-tree based algorithm
     * \warning 4-connex only (6-connex in 3D)
     * \param[in] imIn Input image
     * \param[in] size The size of the opening
     * \param[out] imOut Output image
    */
    template <class T>
    RES_T heightOpen(const Image<T> &imIn, size_t stopSize, Image<T> &imOut, const StrElt &se=DEFAULT_SE)
    {
      return attributeOpen<T, HeightCriterion,size_t,UINT32>(imIn, imOut, stopSize,se);
    }// END heightOpen

    template <class T>
    RES_T heightClose(const Image<T> &imIn, size_t stopSize, Image<T> &imOut, const StrElt &se=DEFAULT_SE)
    {
        ASSERT_ALLOCATED(&imIn, &imOut);
        ASSERT_SAME_SIZE(&imIn, &imOut);
        
        ImageFreezer freeze(imOut);
        
        Image<T> tmpIm(imIn);
        inv(imIn, tmpIm);        
        RES_T res = attributeOpen<T, HeightCriterion,size_t,UINT32>(tmpIm, imOut, stopSize, se);
        inv(imOut, imOut);
        
        return res;
    }// END heightClose

    /**
    * Width opening
     * 
     * Max-tree based algorithm
     * \warning 4-connex only (6-connex in 3D)
     * \param[in] imIn Input image
     * \param[in] size The size of the opening
     * \param[out] imOut Output image
    */
    template <class T>
    RES_T widthOpen(const Image<T> &imIn, size_t stopSize, Image<T> &imOut, const StrElt &se=DEFAULT_SE)
    {
      return attributeOpen<T, WidthCriterion,size_t,UINT32>(imIn, imOut, stopSize,se);
    }// END widthOpen

    template <class T>
    RES_T widthClose(const Image<T> &imIn, size_t stopSize, Image<T> &imOut, const StrElt &se=DEFAULT_SE)
    {
        ASSERT_ALLOCATED(&imIn, &imOut);
        ASSERT_SAME_SIZE(&imIn, &imOut);
        
        ImageFreezer freeze(imOut);
        
        Image<T> tmpIm(imIn);
        inv(imIn, tmpIm);        
        RES_T res = attributeOpen<T, WidthCriterion,size_t,UINT32>(tmpIm, imOut, stopSize, se);
        inv(imOut, imOut);
        
        return res;
    }// END widthClose

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
    RES_T areaOpen(const Image<T> &imIn, size_t stopSize, Image<T> &imOut, const StrElt &se=DEFAULT_SE)
    {
      return attributeOpen<T, AreaCriterion,size_t,UINT32>(imIn, imOut, stopSize,se);
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
    RES_T areaClose(const Image<T> &imIn, size_t stopSize, Image<T> &imOut, const StrElt &se=DEFAULT_SE)
    {
        ASSERT_ALLOCATED(&imIn, &imOut);
        ASSERT_SAME_SIZE(&imIn, &imOut);
        
        ImageFreezer freeze(imOut);
        
        Image<T> tmpIm(imIn);
        inv(imIn, tmpIm);        
        RES_T res = attributeOpen<T, AreaCriterion,size_t,UINT32>(tmpIm, imOut, stopSize, se);
        inv(imOut, imOut);
        
        return res;
    }
    
    /** \} */

} // namespace smil


#endif // _D_MAX_TREE_HPP

