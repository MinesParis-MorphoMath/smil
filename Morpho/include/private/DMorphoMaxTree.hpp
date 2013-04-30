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

namespace smil
{
    /**
    * \ingroup Morpho
    * \defgroup MaxTree
    * \{
    */

#ifndef SWIG


#define NB_COLONE 16
#define SIZE_COLONE 131072
#define DECALAGE 17
#define MODULO 131071

#define ORDONNEE(offset,largeur) ((offset)/(largeur))

//#define OU_ITERATIF

typedef struct criteres {
	unsigned short ymin, ymax;
} critere_type, *CRITERE_TYPE;


template <class T>
struct UO_Struct
{
    int higher_level;//=0;
    UINT NB_NIV_GRIS;
    typedef int* int_ptr;
    int_ptr *pile;
    int *p_pile;
    
    T **niveaux;
    
    unsigned int **fils;
    unsigned int **frere;
    critere_type **critere;

    int next_eti;//=1;
    T* transformee_node;
    unsigned int* indicatrice_node;
    //JFPLIFO_PLIFO jfplifo;
    
    UO_Struct()
    {
	NB_NIV_GRIS = ImDtTypes<T>::max()-ImDtTypes<T>::min()+1;
	pile = new int_ptr[NB_NIV_GRIS]();
	p_pile = new int[NB_NIV_GRIS]();
    }
    ~UO_Struct()
    {
	delete[] pile;
	delete[] p_pile;
    }
    

    void init_pile(const Image<T> &img)
    {
	    typename ImDtTypes<T>::lineType p=img.getPixels(), end = p + img.getPixelCount() - 1;
	    unsigned int histogramme[NB_NIV_GRIS];
	    unsigned int *p_h;
	    int i;
	    histogram(img, histogramme);
	    p_h=histogramme;

	    for(i=0;i<NB_NIV_GRIS;i++) {
			    pile[i]=(*(p_h)>0)?(int *)malloc((*(p_h))*sizeof(int)):NULL;
			    p_h++;
	    }
	    higher_level=0;
    }


    void push(int e, int niveau)
    {
	    if (higher_level<niveau) higher_level=niveau;
	    pile[niveau][p_pile[niveau]++]=e;
    }

    int pop()
    {
	    if (p_pile[higher_level] == 0) {
		    int i;
		    for(i=higher_level;p_pile[i]==0 && i>0;i--);
		    higher_level=i;
		    if (p_pile[higher_level]==0)  {
			    return -1;
		    }
	    }
	    return pile[higher_level][--p_pile[higher_level]];
    }

    int get_higher_level()
    {
	    if (p_pile[higher_level] == 0) {
		    int i;
		    for(i=higher_level;p_pile[i]==0 && i>0;i--);
		    higher_level=i;
	    }
	    return higher_level;
    }

    int first_etiquette(unsigned char valeur)
    {
	    fils=(unsigned int **)malloc(NB_COLONE*sizeof(unsigned int *));
	    frere=(unsigned int **)malloc(NB_COLONE*sizeof(unsigned int *));
	    niveaux=(T **)malloc(NB_COLONE*sizeof(T *));

	    critere=(critere_type **)malloc(NB_COLONE*sizeof(critere_type *));

	    fils[0]=(unsigned int *)calloc(SIZE_COLONE, sizeof(unsigned int));
	    frere[0]=(unsigned int *)calloc(SIZE_COLONE, sizeof(unsigned int));
	    niveaux[0]=(T *)calloc(SIZE_COLONE, sizeof(T));

	    critere[0]=(CRITERE_TYPE)calloc(SIZE_COLONE, sizeof(critere_type));

	    next_eti=1;
	    niveaux[0][next_eti]=valeur;

	    return next_eti++;
    }

    int next_etiquette_lower(T valeur, int *eti)
    {
	    int i;
	    int offset_next_eti = next_eti & MODULO;
	    int page_next_eti = next_eti >> DECALAGE;
	    int offset_eti, page_eti;

	    if (offset_next_eti == 0) {
		    //printf("ALLOCATION ________________________________________ %d \n", page_next_eti);
		    fils[page_next_eti]=(unsigned int *)calloc(SIZE_COLONE, sizeof(unsigned int));
		    frere[page_next_eti]=(unsigned int *)calloc(SIZE_COLONE, sizeof(unsigned int));
		    niveaux[page_next_eti]=(T *)calloc(SIZE_COLONE, sizeof(T));

		    critere[page_next_eti]=(CRITERE_TYPE)calloc(SIZE_COLONE, sizeof(critere_type));
	    }
	    niveaux[page_next_eti][offset_next_eti]=valeur;
	    for(i=valeur-1;eti[i]==0;i--);

	    page_eti=eti[i] >> DECALAGE;
	    offset_eti = eti[i] & MODULO;

	    fils[page_next_eti][offset_next_eti]=fils[page_eti][offset_eti];
	    fils[page_eti][offset_eti]=next_eti;
	    frere[page_next_eti][offset_next_eti]=frere[fils[page_next_eti][offset_next_eti]>> DECALAGE][fils[page_next_eti][offset_next_eti]& MODULO];
	    frere[fils[page_next_eti][offset_next_eti]>> DECALAGE][fils[page_next_eti][offset_next_eti]& MODULO]=0;
	    critere[page_next_eti][offset_next_eti].ymin=16384;
	    return next_eti++;
    }

    int next_etiquette_higher(T parent_valeur, T valeur, int *eti)
    {
	    int offset_next_eti = next_eti & MODULO;
	    int page_next_eti = next_eti >> DECALAGE;
	    int offset_eti, page_eti;

	    if (offset_next_eti == 0) {

		    fils[page_next_eti]=(unsigned int *)calloc(SIZE_COLONE, sizeof(unsigned int));
		    frere[page_next_eti]=(unsigned int *)calloc(SIZE_COLONE, sizeof(unsigned int));
		    niveaux[page_next_eti]=(T *)calloc(SIZE_COLONE, sizeof(T));
		    critere[page_next_eti]=(CRITERE_TYPE)calloc(SIZE_COLONE, sizeof(critere_type));
	    }

	    niveaux[page_next_eti][offset_next_eti]=valeur;
	    page_eti=eti[parent_valeur] >> DECALAGE;
	    offset_eti = eti[parent_valeur] & MODULO;
	    frere[page_next_eti][offset_next_eti]=fils[page_eti][offset_eti];
	    fils[page_eti][offset_eti]=next_eti;
	    critere[page_next_eti][offset_next_eti].ymin=16384;
	    return next_eti++;
    }

    int build_maxtree(const Image<T> &img, int *img_eti) {
	    int *eti = new int[NB_NIV_GRIS];
	    int img_size = img.getPixelCount();
	    typename ImDtTypes<T>::lineType imgPix = img.getPixels();
	    int i,p;
	    int min=imgPix[0], min_index=0;
	    int indice;
	    if (min) 
	    {
		    for(i=0;i<img_size;i++)
			    if (imgPix[i]<min) {min=imgPix[i];min_index=i;if (!min) break;}
	    }
	    memset(eti,0,NB_NIV_GRIS*sizeof(int));
	    indice=img_eti[min_index]=eti[min/*imgPix[min_index]*/]=first_etiquette(min/*imgPix[min_index]*/);

	    push(min_index, min);
	    critere[indice >> DECALAGE][indice & MODULO].ymin=ORDONNEE(min_index,img.getWidth());
	    critere[indice >> DECALAGE][indice & MODULO].ymax=ORDONNEE(min_index,img.getWidth());

	    flood(img, img_eti, eti, min);
	    int retVal = eti[min];
	    delete[] eti;
	    return retVal;
    }
    #define TRACE_IT 15404
    #define TRACE_PSUIV 41083

    void flood(const Image<T> &img, int *img_eti, int *eti, int level)
    {
	    int indice;
	    int i,p;
	    int img_size = img.getPixelCount();
	    int imWidth = img.getWidth();
	    typename ImDtTypes<T>::lineType imgPix = img.getPixels();
	    while( (get_higher_level()>=level) && (p=pop())!=-1)
	    {
		    int p_suiv;

		    if ( (p_suiv=p+imWidth)<img_size && img_eti[p_suiv]==0) {  //y+1
			    if (imgPix[p_suiv]>imgPix[p]) {
				    int j;

				    for(j=imgPix[p]+1;j<imgPix[p_suiv];j++) eti[j]=0;

				    indice=img_eti[p_suiv]=eti[j]=next_etiquette_higher(imgPix[p], imgPix[p_suiv], eti);

			      
			    } else if (eti[imgPix[p_suiv]]==0) {
				    indice=img_eti[p_suiv]=eti[imgPix[p_suiv]]=next_etiquette_lower(imgPix[p_suiv], eti);
			    } else indice=img_eti[p_suiv]=eti[imgPix[p_suiv]];
				    critere[indice >> DECALAGE][indice & MODULO].ymax=MAX(critere[indice >> DECALAGE][indice & MODULO].ymax, ORDONNEE(p_suiv,imWidth));
				    critere[indice >> DECALAGE][indice & MODULO].ymin=MIN(critere[indice >> DECALAGE][indice & MODULO].ymin, ORDONNEE(p_suiv,imWidth));
			    push(p_suiv, imgPix[p_suiv]);
			    if (imgPix[p_suiv]>imgPix[p]) {
				    push(p, imgPix[p]);
				    continue;
			    }
		    }
		    if ( (p_suiv=p-imWidth)>=0 && img_eti[p_suiv]==0) {  //y-1
			    if (imgPix[p_suiv]>imgPix[p]) {
				    int j;
				    for(j=imgPix[p]+1;j<imgPix[p_suiv];j++) eti[j]=0;
				    indice=img_eti[p_suiv]=eti[j]=next_etiquette_higher(imgPix[p], imgPix[p_suiv], eti);
			    } else if (eti[imgPix[p_suiv]]==0) {
				    indice=img_eti[p_suiv]=eti[imgPix[p_suiv]]=next_etiquette_lower(imgPix[p_suiv], eti);
			    } else indice=img_eti[p_suiv]=eti[imgPix[p_suiv]];
				    critere[indice >> DECALAGE][indice & MODULO].ymax=MAX(critere[indice >> DECALAGE][indice & MODULO].ymax, ORDONNEE(p_suiv,imWidth));
				    critere[indice >> DECALAGE][indice & MODULO].ymin=MIN(critere[indice >> DECALAGE][indice & MODULO].ymin, ORDONNEE(p_suiv,imWidth));
			    push(p_suiv, imgPix[p_suiv]);
			    if (imgPix[p_suiv]>imgPix[p]) {
				    push(p, imgPix[p]);
				    continue;
			    }
		    }
		    if ( ((p_suiv=p+1) % imWidth !=0) && img_eti[p_suiv]==0) {  //x+1
			    if (imgPix[p_suiv]>imgPix[p]) {
				    int j;
				    for(j=imgPix[p]+1;j<imgPix[p_suiv];j++) eti[j]=0;
				    indice=img_eti[p_suiv]=eti[j]=next_etiquette_higher(imgPix[p], imgPix[p_suiv], eti);
			    } else if (eti[imgPix[p_suiv]]==0) {
				    indice=img_eti[p_suiv]=eti[imgPix[p_suiv]]=next_etiquette_lower(imgPix[p_suiv], eti);
			    } else indice=img_eti[p_suiv]=eti[imgPix[p_suiv]];
				    critere[indice >> DECALAGE][indice & MODULO].ymax=MAX(critere[indice >> DECALAGE][indice & MODULO].ymax, ORDONNEE(p_suiv,imWidth));
				    critere[indice >> DECALAGE][indice & MODULO].ymin=MIN(critere[indice >> DECALAGE][indice & MODULO].ymin, ORDONNEE(p_suiv,imWidth));
			    push(p_suiv, imgPix[p_suiv]);
			    if (imgPix[p_suiv]>imgPix[p]) {
				    push(p, imgPix[p]);
				    continue;
			    }
		    }
		    if ( (((p_suiv=p-1) % imWidth )!=imWidth-1) && p_suiv>=0 && img_eti[p_suiv]==0) {  //x-1
			    if (imgPix[p_suiv]>imgPix[p]) {
				    int j;
				    for(j=imgPix[p]+1;j<imgPix[p_suiv];j++) eti[j]=0;
				    indice=img_eti[p_suiv]=eti[j]=next_etiquette_higher(imgPix[p], imgPix[p_suiv], eti);
			    } else if (eti[imgPix[p_suiv]]==0) {
				    indice=img_eti[p_suiv]=eti[imgPix[p_suiv]]=next_etiquette_lower(imgPix[p_suiv], eti);
			    } else indice=img_eti[p_suiv]=eti[imgPix[p_suiv]];
				    critere[indice >> DECALAGE][indice & MODULO].ymax=MAX(critere[indice >> DECALAGE][indice & MODULO].ymax, ORDONNEE(p_suiv,imWidth));
				    critere[indice >> DECALAGE][indice & MODULO].ymin=MIN(critere[indice >> DECALAGE][indice & MODULO].ymin, ORDONNEE(p_suiv,imWidth));
			    push(p_suiv, imgPix[p_suiv]);
			    if (imgPix[p_suiv]>imgPix[p]) {
				    push(p, imgPix[p]);
				    continue;
			    }
		    }
	    }
    }
    
    void compute_max(int node, int stop, T max_tr, unsigned int max_in, unsigned int hauteur_parent, T valeur_parent, T previous_value)
    {
	    T m;
	    T max_node;
	    unsigned int max_critere;
	    int child;
	    int hauteur = critere[node >> DECALAGE][node & MODULO].ymax-critere[node >> DECALAGE][node & MODULO].ymin+1;

	    m = (hauteur==hauteur_parent)?niveaux[node >> DECALAGE][node & MODULO]-previous_value:niveaux[node >> DECALAGE][node & MODULO]-valeur_parent;
	    if (hauteur>=stop) {
			    max_node=max_tr;
			    max_critere=0;//max_in;
			    transformee_node[node]=max_node;
			    if (transformee_node[node]!=0) printf("BANG !!! %d\n", transformee_node[node]);

			    indicatrice_node[node]=0;
			    child=fils[node >> DECALAGE][node & MODULO];
	    } else {
		    if (m>max_tr) 
		    {
			    max_node=m;
			    max_critere=hauteur;//critere[node >> DECALAGE][node & MODULO];
		    } else 
		    {
			    max_node=max_tr;
			    max_critere=max_in;
		    }
		    transformee_node[node]=max_node;
		    indicatrice_node[node]=max_critere+1;
		    child=fils[node >> DECALAGE][node & MODULO];
	    }
	    if (indicatrice_node[node]==0 && transformee_node[node]!=0) printf("BANG!!! %d\n", transformee_node[node]);
    #ifdef OU_ITERATIF
    { unsigned char max_tr_propage;
      unsigned int max_indi_propage;
	    if (max_node>10 && hauteur>100) {max_tr_propage=(max_tr>0)?0:max_tr;max_indi_propage=max_critere;}
	    else {max_tr_propage=max_node;max_indi_propage=max_critere;}
	    if (hauteur==hauteur_parent) {
		    while (child!=0) {
		      if (hauteur_parent>stop) compute_max(child, stop, max_tr_propage, max_indi_propage, hauteur, niveaux[node >> DECALAGE][node & MODULO], previous_value);
		      else compute_max(child, stop, max_tr_propage, max_indi_propage, hauteur, niveaux[node >> DECALAGE][node & MODULO]/*valeur_parent*/, previous_value);
			    child=frere[child >> DECALAGE][child & MODULO];
		    }
	    } else {
		    while (child!=0) {
		      compute_max(child, stop, max_tr_propage, max_indi_propage, hauteur, niveaux[node >> DECALAGE][node & MODULO], valeur_parent);
			    child=frere[child >> DECALAGE][child & MODULO];
		    }
	    }
    }
    #else
	    if (hauteur==hauteur_parent) {
		    while (child!=0) {
		      if (hauteur_parent>stop) compute_max(child, stop, max_node, max_critere, hauteur, niveaux[node >> DECALAGE][node & MODULO], previous_value);
		      else compute_max(child, stop, max_node, max_critere, hauteur, niveaux[node >> DECALAGE][node & MODULO]/*valeur_parent*/, previous_value);
			    child=frere[child >> DECALAGE][child & MODULO];
		    }
	    } else {
		    while (child!=0) {
		      compute_max(child, stop, max_node, max_critere, hauteur, niveaux[node >> DECALAGE][node & MODULO], valeur_parent);
			    child=frere[child >> DECALAGE][child & MODULO];
		    }
	    }
    #endif
    }


    template <class T2>
    void fill_in_image(Image<T2> &indicatrice, Image<T> &transformee, int *img_eti)
    {
	    int i;
	    int img_size = transformee.getPixelCount();
	    typename ImDtTypes<T2>::lineType indicatricePix = indicatrice.getPixels();
	    typename ImDtTypes<T>::lineType transformeePix = transformee.getPixels();

	    for(i=0;i<img_size;i++) 
	    {
		transformeePix[i]=transformee_node[img_eti[i]];
		indicatricePix[i]=indicatrice_node[img_eti[i]];
	    }
    }

    critere_type update_critere(int node) {
	    int child=fils[node >> DECALAGE][node & MODULO];
	    while (child!=0) 
	    {
		    critere_type c=update_critere(child);
		    critere[node >> DECALAGE][node & MODULO].ymin=MIN(critere[node >> DECALAGE][node & MODULO].ymin, c.ymin);
		    critere[node >> DECALAGE][node & MODULO].ymax=MAX(critere[node >> DECALAGE][node & MODULO].ymax, c.ymax);
		    child=frere[child >> DECALAGE][child & MODULO];
	    }
	    return critere[node >> DECALAGE][node & MODULO];
    }


    void seuil_maxtree(int node, T seuil)
    {
	    int m=niveaux[node >> DECALAGE][node & MODULO];
	    int child=fils[node >> DECALAGE][node & MODULO];
	    while (child!=0) 
	    {
		    seuil_maxtree(child, seuil);
		    child=frere[child >> DECALAGE][child & MODULO];
	    }
    }

    void compute_contrast(int root, UINT stopValue)
    {
	    int child;
	    int hauteur=critere[root >> DECALAGE][root & MODULO].ymax-critere[root >> DECALAGE][root & MODULO].ymin+1;
	    transformee_node=(T*)malloc(next_eti*sizeof(T));
	    indicatrice_node=(unsigned int*)malloc(next_eti*sizeof(int));
	    transformee_node[root]=0;
	    indicatrice_node[root]=0;
	    update_critere(root);
	    child=fils[root >> DECALAGE][root & MODULO];
	    while (child!=0) 
	    {
		compute_max(child, stopValue, 0, 0, hauteur, niveaux[root >> DECALAGE][root & MODULO], niveaux[root >> DECALAGE][root & MODULO]);
		child=frere[child >> DECALAGE][child & MODULO];
	    }
    }

    void _delete() {
	    int i;
	    int last_page = (next_eti-1) >> DECALAGE;
	    free(transformee_node);
	    free(indicatrice_node);
	    for(i=0;i<=last_page;i++) {
		    free(fils[i]);
		    free(frere[i]);
		    free(niveaux[i]);
		    free(critere[i]);
	    }
	    free(fils);
	    free(frere);
	    free(niveaux);
	    free(critere);
	    for(i=0;i<NB_NIV_GRIS;i++) {
		    if (pile[i]!=NULL) { free(pile[i]);}
	    }
    }

    template <class T2>
    void ouvert_ultime(const Image<T> &img_in, int stopValue, Image<T> &transformee, Image<T2> &indicatrice)
    {
	    int arbre;
	    int *img_eti=(int *)calloc(img_in.getPixelCount(),sizeof(int));
	    init_pile(img_in);
	    arbre=build_maxtree(img_in, img_eti);
	    compute_contrast(arbre, stopValue);
	    fill_in_image(indicatrice, transformee, img_eti);
	    free(img_eti);
	    _delete();
    }

};

#endif
    /**
     * 2D Ultimate Opening using the max-trees
     * 
     * Max-tree based algorithm as described by Fabrizio and Marcotegui (2009) \cite hutchison_fast_2009
     * \warning 4-connex only
     * \param[in] imIn Input image
     * \param[in] stopSize (optional)
     * \param[out] imOut The transformation image
     * \param[out] imIndic The indicator image
     */
    template <class T1, class T2>
    RES_T ultimateOpen(const Image<T1> &imIn, int stopSize, Image<T1> &imOut, Image<T2> &imIndic)
    {
	ASSERT_ALLOCATED(&imIn, &imOut, &imIndic);
	ASSERT_SAME_SIZE(&imIn, &imOut, &imIndic);
	
	UO_Struct<T1> uo;
	uo.ouvert_ultime(imIn, stopSize, imOut, imIndic);
	imOut.modified();
	imIndic.modified();
	
	return RES_OK;
    }  

    template <class T1, class T2>
    RES_T ultimateOpen(const Image<T1> &imIn, Image<T1> &imOut, Image<T2> &imIndic)
    {
	ASSERT_ALLOCATED(&imIn, &imOut, &imIndic);
	ASSERT_SAME_SIZE(&imIn, &imOut, &imIndic);
	
	UO_Struct<T1> uo;
	uo.ouvert_ultime(imIn, numeric_limits<int>::max(), imOut, imIndic);
	
	return RES_OK;
    }  

    
    /** \} */

} // namespace smil


#endif // _D_SKELETON_HPP

