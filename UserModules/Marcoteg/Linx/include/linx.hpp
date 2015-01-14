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


#ifndef _D_LINX
#define _D_LINX

#include "Core/include/private/DImage.hxx"
#include "Morpho/include/DMorpho.h"
#include "Morpho/include/private/DMorphoMaxTree.hpp"
#include "Base/include/private/DBlobMeasures.hpp"
#include "UserModules/Morard/include/DMorard.hpp"
#include <queue>          // std::queue

#define CANDIDATE 0
#define QUEUED 1
#define DONE 2

#define lut_OK 250
#define lut_KO 50

#define Min_Nb_CC  6 //# minimal number of CC for accepting a bar code
#define dist_factor  8 //# max_dist between grouped CC: 
// # dist_centers < dist_factor * min(B1,B2)

#define max_reg_ratio  0.1 //# Region should be smaller than max_reg_ration*imsize
//#-> 10%, for max_reg_ratio  0.1
#define mim_Area_region  20 //# pixels
#define MinAspectRatio  10 //# if ( (max(A,B)/min(A,B))> MinAspectRatio) -> valid CCs
#define MinFillRatio  50 //#  FillRatio = int(100.0 *Area/float(A*B)); A, B inertia axes


#define AspectRatioMerge 1.5// AspectRatio = 1.0*max(A1,A2)/min(A1,A2); A1 and A2 two major axes of 2 CCs

namespace smil
{
    /**
     * @{
     */

      struct inertiaParams
      {
	UINT XC;
	UINT YC;
	double A;
	double B;
	double THETA;
      };


      struct maStruct
      {
	Image_UINT16 *imLbl;
	map<UINT16, vector<double> > mats;
	map<UINT16, double > areaList;
	map<UINT16, vector<UINT> > bbl;
	map<UINT16, inertiaParams > inertiaList;
	map<UINT16, UINT16 > lutList;
      };

    UINT16 TakeDecision(size_t imxsize,size_t imysize,size_t x0,size_t y0,size_t x1,size_t y1,size_t xsize,size_t ysize, double Area)
    {
      size_t AreaBB = xsize * ysize;
      UINT16 FillRatio = int(100.0 * Area/ float(AreaBB));
      UINT16 AspectRatio = int(max(xsize,ysize)/min(xsize,ysize));

      if(Area < 20)
      {
	  return 0;
      }
      else if ((x0 == 0) || (y0==0) || (x1 ==imxsize-1) || (y1 == imysize-1))
      {
        return 0;
      }
      else if((FillRatio>12)&&(AspectRatio>=4)){
        return 255;
      }
      else if((FillRatio<15)&&(AspectRatio<=2))
      {
        return 255;
      }
      else{
        return 0;
      }     

    }


  //##################################################
  //# TakeDecisionInertia
  //##################################################

  UINT16 TakeDecisionInertia(size_t imxsize, size_t imysize, maStruct features, UINT16 lab)
    {
      size_t xsize = 1+features.bbl[lab][2]-features.bbl[lab][0];
      size_t ysize = 1+features.bbl[lab][3]-features.bbl[lab][1];
      size_t x0= features.bbl[lab][0];
      size_t y0= features.bbl[lab][1];
      size_t x1= features.bbl[lab][2];
      size_t y1= features.bbl[lab][3];

      size_t Area = features.areaList[lab];

      double L1 = features.inertiaList[lab].A;
      double L2 = features.inertiaList[lab].B;

      size_t AreaBB = xsize * ysize;
      UINT16 FillRatio = int(100.0 * Area/ float(AreaBB));
      UINT16 AspectRatio = int(max(xsize,ysize)/min(xsize,ysize));
      
      if(L2 == 0){
        L2 = 1;}
      size_t imsize = imxsize * imysize;


      if(L1 > 0){
	AreaBB = L1 * L2; }
      else if (L1==0){
	AreaBB = 1;}

      FillRatio = int(100.0 * Area/ (1.0* (AreaBB)));

      AspectRatio = int(1.0*L1/L2);
      if(0){// DEBUG
	if((x0>1130)and(y0>380)){
	  std::cout<<"L1="<<int(L1)<<";L2="<<int(L2)<<";Aspect="<<AspectRatio<<";Fill="<<FillRatio<<"\n";
	}
      }
                      
      //    # Compare "Area" with L1*L2 ??
	if((Area > imsize*max_reg_ratio) || (Area < mim_Area_region)){ //# remove small CCs
	  return lut_KO;}
	  //    # remove CCs touching borders
	else if ((x0 == 0) || (y0==0) || (x1 ==imxsize-1) || (y1 == imysize-1)){
	    return lut_KO;}
	//    # remove if too small Aspect Ratio || FillRatio not convinient
	else if((AspectRatio>=MinAspectRatio)and(FillRatio > MinFillRatio)){
	  return lut_OK;}
	else{
	  return lut_KO;}
    }	  //# END TakeDecisionInertia ..................................................


    void SelectCCs(const Image_UINT8 &imCC, Image_UINT8 &imOut)
    {
	Image_UINT16 imlabel(imCC);
	label(imCC, imlabel);
	map<UINT16, vector<UINT> > bbl = measBoundBoxes(imlabel);//BMI

	size_t imxsize = imCC.getWidth();
	size_t imysize = imCC.getHeight();
	
	map<UINT16,UINT16> lut;
	Image_UINT16 imtmp16(imCC);
	
	map<UINT16, double> AreaList = measAreas(imlabel);//BMI
	
	map<UINT16, vector<UINT> >::iterator it;//BMI
	for (it=bbl.begin();it!=bbl.end();it++)
	  {
	    vector<UINT> box = it->second;
	    UINT16 lab = it->first;
	    size_t Area = AreaList[lab];
	    
	    size_t xsize = 1+bbl[lab][2]-bbl[lab][0];
	    size_t ysize = 1+bbl[lab][3]-bbl[lab][1];
	    size_t x0= bbl[lab][0];
	    size_t y0= bbl[lab][1];
	    size_t x1= bbl[lab][2];
	    size_t y1= bbl[lab][3];
	    lut[lab] = TakeDecision(imxsize,imysize,x0,y0,x1,y1,xsize,ysize,Area);
	  }
	    
	applyLookup(imlabel,lut,imtmp16);
	copy(imtmp16,imOut);

	write(imOut,"selectCC_decision.png");

    }// END Select CC


  //##################################################
  //    # fitRectangle
  //##################################################
  void fitRectangle(vector<double>  &mat, double &xc, double &yc, double &A, double &B, double &theta)
  {
    double m00= mat[0];
    double m10=mat[1]; 
    double m01=mat[2];
    double m11=mat[3];
    double m20=mat[4];
    double m02=mat[5];
    
    if (m00==0){
      return;
    }

    // # COM
    xc = int (1.0*m10/m00);
    yc = int (1.0*m01/m00);
    //    # centered matrix (central moments)
    double u00 = m00;
    double u20 = m20 - (m10*m10/m00);
    double u02 = m02 - (m01*m01/m00);
    double u11 = m11 - (m10*m01/m00);

      //    # eigen values
    double delta = 4*u11*u11 + (u20-u02)*(u20-u02);
    double I1 = (u20+u02+sqrt(delta))/2;
    double I2 = (u20+u02-sqrt(delta))/2;

    theta = 0.5 * atan2(-2*u11, (u20-u02));

    //    # Equivalent rectangle
    //    # I1 = a**2*S/12, I2 = b**2*S/12
    A = int (sqrt(12*I1/u00));
    B = int (sqrt(12*I2/u00));

    return;

  }// fitRectangle
      //..................................................
      //# END fitRectangle ..................................................
      //..................................................

    void SelectCCsInertia(const Image_UINT8 &imCC,  maStruct &features,Image_UINT8 &imOut)
    {
      UINT sum_L1, sum_L2, nb_valid;
      UINT16 lab;


      //      features.mats[lab][0];
      
      sum_L1=0; sum_L2=0; nb_valid = 0;

      size_t imxsize = imCC.getWidth();
      size_t imysize = imCC.getHeight();
	
      
      map<UINT16, vector<UINT> >::iterator it;
      for (it=features.bbl.begin();it!=features.bbl.end();it++)
	{
	  vector<UINT> box = it->second;
	  lab = it->first;
	  
	  UINT lut_elem = TakeDecisionInertia(imxsize,imysize,features,lab);
	    features.lutList[lab]=lut_elem;
	    if(lut_elem == lut_OK){// valid CC
	      sum_L1 = sum_L1 + features.inertiaList[lab].A;
	      sum_L2 = sum_L2 + features.inertiaList[lab].B;
	      nb_valid = nb_valid + 1;
	    }
	}

	double L1_avg = 1.0*sum_L1/nb_valid;
	double L2_avg = 1.0 * sum_L2/nb_valid;
	UINT Area_thresh = (L1_avg * L2_avg)/2;

	for (it=features.bbl.begin();it!=features.bbl.end();it++)
	{
	  lab = it->first;
	  if(features.lutList[lab] == lut_OK){//lut_OK
	    if(features.areaList[lab]<Area_thresh){
	      features.lutList[lab] = lut_KO;//lut_OKKO
	    }
	  }
	}
	Image_UINT16 &imlabel = *(features.imLbl);

	Image<UINT16> imtmp16(imCC);
	applyLookup(imlabel,features.lutList,imtmp16);
	copy(imtmp16,imOut);

	write(imOut,"selectCC_decision.png");

    }// end SelectCCsInertia
    


    //##################################################
    //# Verify if two labels are compatible for grouping, using inertia moments
    //#    if((dist < dist_factor * max(B1,B2))and (AspectRatio < AspectRatioMerge)):
    //#        merge_flag = 1
    //##################################################

    UINT AreLabelsCompatibleInertia(maStruct &features, UINT16 lab, UINT16 lab2){
      UINT x_center_1, y_center_1, x_center_2, y_center_2, dist;
      int dist_x, dist_y;
      double A1, A2, B1, B2;

      x_center_1 = features.inertiaList[lab].XC;
      y_center_1 = features.inertiaList[lab].YC;

      x_center_2 = features.inertiaList[lab2].XC;
      y_center_2 = features.inertiaList[lab2].YC;

      dist_x = x_center_2 - x_center_1;
      dist_y = y_center_2 - y_center_1;

      dist = sqrt(dist_x*dist_x+dist_y*dist_y);
      A1 = features.inertiaList[lab].A;
      A2 = features.inertiaList[lab2].A;
      double AspectRatio = 1.0*max(A1,A2)/min(A1,A2);
    
      B1 = features.inertiaList[lab].B;
      B2 = features.inertiaList[lab2].B;
      UINT merge_flag = 0;// # by default, do not merge boxes

      if((dist < dist_factor * max(B1,B2))and (AspectRatio < AspectRatioMerge)){
        merge_flag = 1;
      }
      return merge_flag;
      }// END AreLabelsCompatible


  //##################################################
  //# Recursively merge all compatible CCs, starting from CC with label = "lab"
  //##################################################
  UINT MergeLab(maStruct &features, map<UINT16, UINT8 > &status,  map<UINT16, UINT16 > &mergeList,UINT16 lab,UINT &group_box_label,UINT &nb_CC, std::queue<UINT16> &myqueue)
  {
    UINT16 lab0, lab2;
    UINT merge_flag;
    status[lab]=DONE;

    nb_CC = nb_CC + 1;
    mergeList[lab]=group_box_label;
    //      map<UINT, vector<UINT> >::iterator it;
    //      for (it=bbl.begin();it!=bbl.end();it++){
    map<UINT16, vector<UINT> >::iterator it;

    for (it=features.bbl.begin();it!=features.bbl.end();it++){
      lab2 = it->first;

	if(status[lab2]==CANDIDATE){
	  //#            merge_flag=AreLabelsCompatible(bbl,lab,lab2)
	  merge_flag=AreLabelsCompatibleInertia(features,lab,lab2);

	  if(merge_flag){
	    //#                print "MergeLabs",lab,lab2
	    mergeList[lab2] = group_box_label;
	    myqueue.push (lab2);//	  queue.append(lab2);

	    status[lab2]=QUEUED;
	  }// if (merge_flag)
	}// if status[labb2] == CANDIDATE
    }// for lab2
    while (myqueue.size()>0)//	while(len(queue) > 0){
      {
	lab0=myqueue.front();//	  lab0=queue.pop();

	myqueue.pop();

	if(status[lab0]<DONE){
	  nb_CC=MergeLab(features,status,mergeList,lab0,group_box_label,nb_CC,myqueue);
	}
      }// END while
    return nb_CC;
 }// END MergeLab

  //##################################################
  //# Call MergeLab for each label
  //##################################################
  void MergeCCs(Image_UINT8 &imCC, maStruct &features,Image_UINT8 &imOut)
  {
    map<UINT16,UINT8> status;
    map<UINT16,UINT16> mergeList;
    UINT16 x;

    map<UINT16, vector<UINT> >::iterator it;
    map<UINT16, vector<UINT> >::iterator it2;

    for (it=features.bbl.begin();it!=features.bbl.end();it++){
      x = it->first;
      status[x] = CANDIDATE;
      mergeList[x]=0;
    }
    UINT group_box_label = 1;
    UINT16 lab;
    
    std::queue<UINT16> myqueue;

    for (it=features.bbl.begin();it!=features.bbl.end();it++){
      x = it->first;
      if(features.lutList[x] != lut_OK){
	status[x] = DONE;
      }
    }

    // voir mail de Matthieu (mardi 20 janvier, 14h13)
    UINT8 min_status = DONE;
    for (it=features.bbl.begin();it!=features.bbl.end();it++){
      x = it->first;
      if(status[x] < min_status){
	min_status = status[x];
      }
    }

    if(min_status == DONE){//# This loop is required if we deal with all
      //# CC, not only those that are validated
      return ;
    }

    for (it=features.bbl.begin();it!=features.bbl.end();it++){
      lab = it->first;
      if(status[lab]==CANDIDATE){
	myqueue.push (lab);//      queue.append(lab);
	status[lab] = QUEUED;
	  UINT nb_CC = 0;
	  //            # print "lab,group_box_label",lab, group_box_label
	  nb_CC=MergeLab(features, status,mergeList,lab,group_box_label,nb_CC,myqueue);
	  
	  if(nb_CC < Min_Nb_CC){//# If less than 4 CC, it is not a barcode
	    for (it2=features.bbl.begin();it2!=features.bbl.end();it2++){
	      x = it2->first;
	      if (mergeList[x] == group_box_label){
		mergeList[x] = 0;}
	      else{
		group_box_label = group_box_label + 1;
	      }
	    }
	  }// END if (nb_CC < Min_Nb_CC // NO BAR CODE
      }// if status CANDIDATE
    }// for lab = 1


    Image_UINT16 &imlabel = *(features.imLbl);
    Image<UINT16> imtmp16(imlabel);

    applyLookup(imlabel,mergeList,imtmp16);
    copy<UINT16,UINT8>(imtmp16,imOut);

    return;

  }// END MergeCCs
      // ..................................................
      // END MergeCCs
      // ..................................................


    void FindCB(const Image_UINT8 &imIn, Image_UINT8 &imOut)
    {
	Image_UINT8 imCC(imIn, true);

	maStruct features;
// 

	inv(imIn, imCC);
	Image_UINT8 imT(imIn);
	Image<UINT16> imInd(imIn);
	//std::cout <<"################BEGIN UO 0\n";
	std::cout <<"IMAGE SIZE ="<<imIn.getHeight()<<"\n";
	//ultimateOpen(imIn, imT, imInd, (tmpIm.getHeight()-1),0);//######### imIn or tmpIm

	//write(imT,"imT_smil0.png");
	std::cout <<"########BEGIN UO 1\n";
	size_t stop;
	stop = imCC.getHeight()-10;
	ultimateOpen(imCC, imT, imInd, stop,1);
	write(imT,"imT_smil1.png");

	threshold(imT, imCC);


	// ..........Compute Features -> maStruct

	Image_UINT16 imLabel(imIn);
	label(imCC, imLabel);

	map<UINT16, Blob> blobs = computeBlobs(imLabel);//BMI
	map<UINT16, double> AreaList = measAreas(blobs);
	map<UINT16, vector<UINT> > bbl = measBoundBoxes(imLabel,blobs);
	map<UINT16, vector<double> > mats = measInertiaMatrices(imCC,blobs);//# imin?
	map<UINT16, inertiaParams > inertia;
	map<UINT16, UINT16>lutList;

	inertiaParams inertiaPar;
	vector<double> mat;
	map<UINT16, vector<double> >::iterator it;
	for (it=mats.begin();it!=mats.end();it++){
	  double XC, YC, A, B, theta;
	  mat = (*it).second;
	  UINT16 lab = it->first;

	  fitRectangle(mat, XC, YC, A, B, theta);
	  inertiaPar.XC =  XC;
	  inertiaPar.YC =  YC;
	  inertiaPar.A =  A;
	  inertiaPar.B =  B;
	  inertiaPar.THETA =  theta;
	  inertia[lab] = inertiaPar;
	  lutList[lab] = lut_OK;
	}
	features.imLbl = &imLabel;
	features.mats = mats;
	features.areaList = AreaList;
	features.bbl = bbl;
	features.inertiaList = inertia;
	features.lutList = lutList;
	// .......... END Compute Features -> maStruct

	if(0){ // previous version
	  SelectCCs(imCC, imOut);
	}
	else{
	  // features.lutList  = lut_OK (250, if CC is validated), lut_KO (50, otherwise)
	  // lookuptable(features.imlabel, features.lutList) -> imOut
	  SelectCCsInertia(imCC,features,imOut);

	  // Merge CCs, result in imOut. label ==1 for first detected barcode, =2 for second...
	  MergeCCs(imCC,features,imOut);

	}// END Inertia selection
    }// void FindCB

  void ToggleMapping(const Image_UINT8 im, UINT size, Image_UINT8 imdil, Image_UINT8 imero, Image_UINT8 imtmp1, Image_UINT8 imtmp2, Image_UINT8 imres, const StrElt &se=DEFAULT_SE){
    dilate(im,imdil,se(size));
    absDiff(im,imdil,imtmp1);
    
    erode(im,imero,se(size));
    absDiff(im,imero,imtmp2);
      
    compare(imtmp1,">",imtmp2,imero,imdil,imres);
    //    copy(imres,imero);// compare does not allow twice the same image?
    //    compare(imtmp2,">",imtmp1,imdil,imero,imres);
    }
  
  void ToggleMappingRes(const Image_UINT8 &im, UINT size, Image_UINT8 &imdil, Image_UINT8 &imero, Image_UINT8 &imres, const StrElt &se=DEFAULT_SE){// compute directly the residues
    dilate(im,imdil,se(size));
    absDiff(im,imdil,imdil);

    erode(im,imero,se(size));
    absDiff(im,imero,imero);

    inf(imdil,imero,imres);
      //    compare(imtmp1,">",imtmp2,imero,imdil,imres);
    //    copy(imres,imero);// compare does not allow twice the same image?
    //    compare(imtmp2,">",imtmp1,imdil,imero,imres);
    }

    void Local_QA(const Image_UINT8 &imdiff,UINT min_thresh,UINT min_area, Image_UINT8 &imOut)
    {

	Image_UINT8 imCC(imOut);

	compare(imdiff,">",(UINT8)min_thresh,(UINT8)255,(UINT8)0,imCC);

	Image_UINT16 imlabel(imCC);

	//----------------------------------------
	//Compute Blobs
	//----------------------------------------
	label(imCC, imlabel);
	  
	map<UINT16, Blob> blobs = computeBlobs(imlabel);//BMI

	// ..........Compute Features -> maStruct
	map<UINT16, double> AreaList = measAreas(blobs);
	map<UINT16, Vector_double > measAvg = measMeanVals(imdiff,blobs);



	// .......... END Compute Features -> maStruct

	size_t imxsize = imCC.getWidth();
	size_t imysize = imCC.getHeight();
	
	map<UINT16,UINT16> lutList;
	Image_UINT16 imtmp16(imCC);
	
	//	map<UINT, vector<UINT> >::iterator it;

	UINT8 avg;
	double area;

	map<UINT16, Blob>::iterator it;
	for (it=blobs.begin();it!=blobs.end();it++){
	  UINT16 lab = it->first;
	  area= AreaList[lab];
	  avg = measAvg[lab][0];
	  if(area<min_area){
	    lutList[lab] = 0;
	  }
	  else{
	    lutList[lab] = avg;
	    //	    std::cout<<"avg("<<lab<<"="<<(int)avg<<"\n";
	  }

	}// for each lab
	    
	applyLookup(imlabel,lutList,imtmp16);
	copy(imtmp16,imOut);
	//END BMI

    }// END Select CC




    void BlurEstimation(const Image_UINT8 &imIn, Image_UINT8 &imOut)
    {

	maStruct features;

	Image_UINT8 imres1(imIn);
	Image_UINT8 imres5(imIn);
	Image_UINT8 imgra(imIn);
	Image_UINT8 imFil(imIn);
	Image_UINT8 imtmp(imIn);
	Image_UINT8 imtmp2(imIn);

	// Parameters SETUP
	UINT8 min_thresh=3;
	double min_area=5;
	int size1=1,size2=2;

	if(1){//filter
	  fastBilateralFilter(imIn,2,5,5,20,imFil);
	}
	else{
	  copy(imIn,imFil);
	}

	ToggleMappingRes(imFil,size1,imtmp,imtmp2,imres1,CrossSE());//4 temporary images (
	ToggleMappingRes(imFil,size2,imtmp,imtmp2,imres5,CrossSE());
	gradient(imFil,imgra,CrossSE());

	sub(imgra,imres5,imtmp);

	Local_QA(imtmp,min_thresh,min_area, imOut);

    }// void BlurEstimation

} // namespace smil

#endif // _D_LINX

 
