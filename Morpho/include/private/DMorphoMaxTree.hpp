/*
 * Copyright (c) 2011, Matthieu FAESSEL and ARMINES
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

#include "DImage.h"
#include "DImageHistogram.hpp"

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
	size_t level = TYPE_FLOOR + size_t(value);
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
	    for (size_t i=higherLevel-1;i>=0;i--)
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
#define ORDONNEE(offset,largeur) ((offset)/(largeur))

template <class T, class OffsetT=UINT>
class MaxTree
{
public:
    struct Criterion
    {
	OffsetT ymin, ymax;
    };
  
private:
    size_t GRAY_LEVEL_NBR;
    
    PriorityQueue<T, OffsetT> pq;
    
    T **levels;
    
    OffsetT **children;
    OffsetT **brothers;
    Criterion **criteria;
    
    size_t *labels;
    size_t curLabel;
    
    bool initialized;


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
	criteria[page] = new Criterion[COLUMN_SIZE]();
    }

    T initialize(const Image<T> &img, OffsetT *img_eti)
    {
	if (initialized)
	  reset();
	
	typename ImDtTypes<T>::lineType pix = img.getPixels();
	
	T minValue = ImDtTypes<T>::max();
	T tMinV = ImDtTypes<T>::min();
	OffsetT minOff;
	for (size_t i=0;i<img.getPixelCount();i++)
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
	labels[UINT(minValue)] = curLabel;

	pq.initialize(img);
	pq.push(minValue, minOff);
	getCriterion(curLabel).ymin = ORDONNEE(minOff, img.getWidth());
	getCriterion(curLabel).ymax = ORDONNEE(minOff, img.getWidth());
	    
	curLabel++;
	
	initialized = true;

	return minValue;
    }
    
    int nextLowerLabel(const T &value)
    {
	    if ((curLabel & COLUMN_MOD) == 0)
	      allocatePage(curLabel >> COLUMN_SHIFT);
	    
	    getLevel(curLabel) = value;
	    int i;
	    for(i=int(value)-1;labels[i]==0;i--);

	    getChild(curLabel) = getChild(labels[i]);
	    getChild(labels[i]) = curLabel;
	    getBrother(curLabel) = getBrother(getChild(curLabel));
	    getBrother(getChild(curLabel)) = 0;
	    getCriterion(curLabel).ymin = numeric_limits<unsigned short>::max();
	    return curLabel++;
    }

    int nextHigherLabel(T parent_valeur, T valeur)
    {
	    if ((curLabel & COLUMN_MOD) == 0)
	      allocatePage(curLabel >> COLUMN_SHIFT);

	    getLevel(curLabel) = valeur;
	    getBrother(curLabel) = getChild(labels[UINT(parent_valeur)]);
	    getChild(labels[UINT(parent_valeur)]) = curLabel;
	    getCriterion(curLabel).ymin = numeric_limits<unsigned short>::max();
	    return curLabel++;
    }
    
    bool subFlood(typename ImDtTypes<T>::lineType imgPix, int imWidth, UINT *img_eti, OffsetT p, OffsetT p_suiv)
    {
	int indice;
	
	if (imgPix[p_suiv]>imgPix[p]) 
	{
	      int j;
	      for(j=imgPix[p]+1;j<int(imgPix[p_suiv]);j++) 
		labels[j]=0;
	      indice = img_eti[p_suiv] = labels[j] = nextHigherLabel(imgPix[p], imgPix[p_suiv]);
	  
	} 
	else if (labels[UINT(imgPix[p_suiv])]==0) 
	    indice = img_eti[p_suiv] = labels[UINT(imgPix[p_suiv])] = nextLowerLabel(imgPix[p_suiv]);
	else 
	    indice = img_eti[p_suiv] = labels[UINT(imgPix[p_suiv])];
	
	getCriterion(indice).ymax = MAX(getCriterion(indice).ymax, ORDONNEE(p_suiv,imWidth));
	getCriterion(indice).ymin = MIN(getCriterion(indice).ymin, ORDONNEE(p_suiv,imWidth));
	pq.push(imgPix[p_suiv], p_suiv);
	
	if (imgPix[p_suiv]>imgPix[p])
	{
		pq.push(imgPix[p], p);
		return true;
	}
	return false;
    }

    void flood(const Image<T> &img, UINT *img_eti, int level)
    {
	    int indice;
	    int p;
	    int imWidth = img.getWidth();
	    int imHeight = img.getHeight();
	    int imDepth = img.getDepth();
	    int pixelCount = img.getPixelCount();
	    int pixPerSlice = imWidth*imHeight;
	    typename ImDtTypes<T>::lineType imgPix = img.getPixels();
	    
	    while( (pq.getHigherLevel()>=level) && !pq.isEmpty())
	    {
		p = pq.pop();
		int p_suiv;

		if ( p%pixPerSlice<pixPerSlice-imWidth && img_eti[p_suiv=p+imWidth]==0) //y+1
		  if (subFlood(imgPix, imWidth, img_eti, p, p_suiv))
		    continue;

		if ( p%pixPerSlice>imWidth-1 && img_eti[p_suiv=p-imWidth]==0) //y-1
		  if (subFlood(imgPix, imWidth, img_eti, p, p_suiv))
		    continue;
		  
		if ( ((p_suiv=p+1) % imWidth !=0) && img_eti[p_suiv]==0) //x+1
		  if (subFlood(imgPix, imWidth, img_eti, p, p_suiv))
		    continue;
		  
		if ( (((p_suiv=p-1) % imWidth )!=imWidth-1) && p_suiv>=0 && img_eti[p_suiv]==0) //x-1
		  if (subFlood(imgPix, imWidth, img_eti, p, p_suiv))
		    continue;
		  
		if (imDepth>1) // for 3D
		{
		    if ( p/pixPerSlice<imDepth-1 && img_eti[p_suiv=p+pixPerSlice]==0) //z+1
		      if (subFlood(imgPix, imWidth, img_eti, p, p_suiv))
			continue;
		      
		    if ( p/pixPerSlice>1 && img_eti[p_suiv=p-pixPerSlice]==0) //z-1
		      if (subFlood(imgPix, imWidth, img_eti, p, p_suiv))
			continue;
		}
	    }
    }
    
public:
    MaxTree()
    {
	GRAY_LEVEL_NBR = ImDtTypes<T>::max()-ImDtTypes<T>::min()+1;
	
	children = new OffsetT*[COLUMN_NBR]();
	brothers =  new OffsetT*[COLUMN_NBR]();
	levels = new T*[COLUMN_NBR]();
	criteria = new Criterion*[COLUMN_NBR]();
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
    
    inline Criterion &getCriterion(const OffsetT node)
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
  

    int build(const Image<T> &img, OffsetT *img_eti) 
    {
	    T minValue = initialize(img, img_eti);

	    flood(img, img_eti, minValue);
	    return labels[UINT(minValue)];
    }
    
    Criterion updateCriteria(int node) 
    {
	    int child = getChild(node);
	    while (child!=0) 
	    {
		    Criterion c = updateCriteria(child);
		    getCriterion(node).ymin = MIN(getCriterion(node).ymin, c.ymin);
		    getCriterion(node).ymax = MAX(getCriterion(node).ymax, c.ymax);
		    child = getBrother(child);
	    }
	    return getCriterion(node);
    }
    
};


template <class T, class OffsetT>
void compute_max(MaxTree<T,OffsetT> &tree, T* transformee_node, UINT* indicatrice_node, int node, UINT stop, T max_tr, unsigned int max_in, unsigned int hauteur_parent, T valeur_parent, T previous_value)
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

template <class T, class OffsetT>
void compute_contrast(MaxTree<T,OffsetT> &tree, T* transformee_node, UINT* indicatrice_node, int root, UINT stopSize)
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


#endif
    /**
     * Ultimate Opening using the max-trees
     * 
     * Max-tree based algorithm as described by Fabrizio and Marcotegui (2009) \cite hutchison_fast_2009
     * \warning 4-connex only (6-connex in 3D)
     * \param[in] imIn Input image
     * \param[out] imOut The transformation image
     * \param[out] imIndic The indicator image
     * \param[in] stopSize (optional)
     */
    template <class T1, class T2>
    RES_T ultimateOpen(const Image<T1> &imIn, Image<T1> &imTrans, Image<T2> &imIndic, UINT stopSize=-1)
    {
	ASSERT_ALLOCATED(&imIn, &imTrans, &imIndic);
	ASSERT_SAME_SIZE(&imIn, &imTrans, &imIndic);
	
	int imSize = imIn.getPixelCount();
	UINT *img_eti = new UINT[imSize]();
	
	MaxTree<T1> tree;
	int root = tree.build(imIn, img_eti);
	
	T1 *transformee_node = new T1[tree.getLabelMax()]();
	UINT *indicatrice_node = new UINT[tree.getLabelMax()]();
	compute_contrast(tree, transformee_node, indicatrice_node, root, stopSize);
	
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


    
    /** \} */

} // namespace smil


#endif // _D_SKELETON_HPP

