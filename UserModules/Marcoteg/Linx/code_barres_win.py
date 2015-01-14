from smilPython import *
from marcoteg_utilities_smil import *
import math
import pdb

def GetValue(val,imsize):
  if (val < 0):
    new_val = 0
  elif(val>imsize):
    new_val = imsize
  else:
    new_val = val
  return new_val

def ToggleMapping(im,nl,size):
  imdil,imero = Image(im),Image(im)
  imtmp1,imtmp2 = Image(im),Image(im)
  imres = Image(im)
  dilate(im,imdil,nl(size))
  absDiff(im,imdil,imtmp1)

  erode(im,imero,nl(size))
  absDiff(im,imero,imtmp2)

  compare(imtmp1,">",imtmp2,imero,im,imres)
  copy(imres,imero)# compare does not allow twice the same image?
  compare(imtmp2,">",imtmp1,imdil,imero,imres)

  return imres


def DrawInertiaAxes(imDraw,xc,yc,A,B,theta,lab,imxsize,imysize):
  dx = A/2*math.cos(math.pi-theta)
  dy = A/2*math.sin(math.pi-theta)
  x_start = GetValue(int(xc-dx),imxsize)
  y_start = GetValue(int(yc-dy),imysize)
  x_end = GetValue(int(xc+dx),imxsize)
  y_end = GetValue(int(yc+dy),imysize)

  drawLine(imDraw, x_start,y_start,x_end,y_end, lab)
  dx = B/2*math.sin(theta)
  dy = B/2*math.cos(theta)
  x_start = GetValue(int(xc-dx),imxsize)
  y_start = GetValue(int(yc-dy),imysize)
  x_end = GetValue(int(xc+dx),imxsize)
  y_end = GetValue(int(yc+dy),imysize)
  drawLine(imDraw, x_start,y_start,x_end,y_end, lab)
#  drawLine(imDraw, int(xc-dx), int(yc-dy), int(xc+dx), int(yc+dy), lab)


def WriteResults(im,imT,imthresh,imtmp,name):
    write(im,"./results/"+name+"step00_ori.png")
    write(imT,"./results/"+name+"step01_T.png")
    write(imthresh,"./results/"+name+"step02_thresh.png")
    write(imtmp,"./results/"+name+"step03_out.png")

def fitRectangle(mat):
    m00, m10, m01, m11, m20, m02 = mat

    if m00==0:
      return 0, 0, 0, 0, 0

    # COM
    xc = int (m10/m00)
    yc = int (m01/m00)

    # centered matrix (central moments)
    u00 = m00
    u20 = m20 - m10**2/m00
    u02 = m02 - m01**2/m00
    u11 = m11 - m10*m01/m00

    # eigen values
    delta = 4*u11**2 + (u20-u02)**2
    I1 = (u20+u02+math.sqrt(delta))/2
    I2 = (u20+u02-math.sqrt(delta))/2

    theta = 0.5 * math.atan2(-2*u11, (u20-u02))

    # Equivalent rectangle
    # I1 = a**2*S/12, I2 = b**2*S/12
    a = int (math.sqrt(12*I1/u00))
    b = int (math.sqrt(12*I2/u00))

    return xc, yc, a, b, theta

    
def TakeDecision(imxsize,imysize,x0,y0,x1,y1,xsize,ysize,Area):
    AreaBB = xsize * ysize
    FillRatio = int(100.0 * Area/ float(AreaBB))
    AspectRatio = int(max(xsize,ysize)/min(xsize,ysize))

    if(Area < 20):
        return 0,FillRatio,AspectRatio
    elif ((x0 == 0) or (y0==0) or (x1 ==imxsize-1) or (y1 == imysize-1)):
        return 0,FillRatio,AspectRatio
    elif((FillRatio>12)and(AspectRatio>=4)):
        return 255,FillRatio,AspectRatio
    elif((FillRatio<15)and(AspectRatio<=2)):
        return 255,FillRatio,AspectRatio
    else:
        return 0,FillRatio,AspectRatio
lut_OK = 250
lut_KO = 50           
lut_OKKO = 75
def TakeDecisionInertia(imxsize,imysize,x0,y0,x1,y1,xsize,ysize,Area,L1,L2,theta):
    if(L2 == 0):
        L2 = 1
    imsize = imxsize * imysize

#        print "WARNING: L2 is ZERO!!!!!!!!!!!"
#        pdb.set_trace()
    if(L1 > 0):
      AreaBB = L1 * L2
    elif (L1==0):
      AreaBB = 1

    FillRatio = int(100.0 * Area/ float(AreaBB))
#    AspectRatio = int(max(xsize,ysize)/min(xsize,ysize))
    AspectRatio = int(1.0*L1/L2)
                      
    # Compare "Area" with L1*L2 ??
    
    if((Area > imsize/10) or (Area < 20)): # remove small CCs
        return lut_KO,FillRatio,AspectRatio
    # remove CCs touching borders
    elif ((x0 == 0) or (y0==0) or (x1 ==imxsize-1) or (y1 == imysize-1)): 
        return lut_KO,FillRatio,AspectRatio
    # remove if too small Aspect Ratio or FillRatio not convinient
    elif((AspectRatio>=10)and(FillRatio > 50)):
        return lut_OK,FillRatio,AspectRatio
    else:
        return lut_KO,FillRatio,AspectRatio


def ThreshUO(imT,low_thresh):
  imCC = Image(imT)
  compare(imT,">",low_thresh, 255,0,imCC)
  ImDisplayX(imCC,"CCC")
  pdb.set_trace()
  imlabel = Image(imCC,"UINT16")
  label(imCC,imlabel)
  blobs = computeBlobs(imlabel)
  means = measMeanVals(imT, blobs)
  maxs = measMaxVals(imT, blobs)
  ThreshLUT = Map_UINT16_UINT16()
  for lbl in blobs.keys():
    if(means[lbl][0] > maxs[lbl]/2):
      ThreshLUT[lbl] = int(means[lbl][0]/2)
    else:
#      raw_input()
      ThreshLUT[lbl] = int(means[lbl][0]+1)

  imtmp16=Image(imCC,"UINT16")

  applyLookup(imlabel,ThreshLUT,imtmp16)
  copy(imtmp16,imCC)
  imthresh = Image(imT)
  compare(imT,">",imCC,255,0,imthresh)
  pdb.set_trace()
  write(imT,"trans1.png")
  write(imCC,"compare.png")
  write(imthresh,"thresh.png")
  ImDisplayX(imthresh,"thresh")
  ImDisplayX(imCC,"compare")
  ImDisplayX(imT,"Trans")
  pdb.set_trace()
  return imthresh
#    print str(lbl) + "\t" + str(areas[lbl]) + "\t" + str(vols[lbl]) + "\t" + str(barys[lbl])
  


def SelectCCs(imCC,name):
    imlabel = Image(imCC,"UINT16")
    label(imCC,imlabel)
    bbl=measBoundBoxes(imlabel)
    AreaList = measAreas(imlabel)

    imxsize,imysize = imCC.getWidth(),imCC.getHeight()
    lut = Map_UINT16_UINT16()
    FillRatioLut=Map_UINT16()
    AspectRatioLut = Map_UINT16()
    imtmp16=Image(imCC,"UINT16")
    for lab in bbl.keys():
        Area = AreaList[lab]

        x0= bbl[lab][0]
        y0= bbl[lab][1]
        x1= bbl[lab][2]
        y1= bbl[lab][3]
        xsize = 1+x1-x0
        ysize = 1+y1-y0
        decision,FillRatio,AspectRatio = TakeDecision(imxsize,imysize,x0,y0,x1,y1,xsize,ysize,Area)
        lut[lab] = decision
        FillRatioLut[lab]=FillRatio
        AspectRatioLut[lab]=AspectRatio
        
        if(Area>500):
            print lab,xsize,ysize,Area,xsize*ysize,FillRatio,AspectRatio,decision
        
    applyLookup(imlabel,lut,imtmp16)
    imout8 = Image(imCC)
    copy(imtmp16,imout8)

    write(imout8,"./results/"+name+"_decision.png")

    applyLookup(imlabel,FillRatioLut,imtmp16)
    copy(imtmp16,imout8)
    write(imout8,"./results/"+name+"_FillRatio.png")

    applyLookup(imlabel,AspectRatioLut,imtmp16)
    copy(imtmp16,imout8)
    write(imout8,name+"./results/"+name+"_AspectRatio.png")

    return imout8

def SelectCCsInertia(imCC,name):
    imlabel = Image(imCC,"UINT16")
    label(imCC,imlabel)

    # Compute Blobs
    blobs = computeBlobs(imlabel)
    # Compute Inertia Matrices

    mats  = measInertiaMatrices(imCC, blobs)# imin?
    bbl=measBoundBoxes(imlabel)
    AreaList = measAreas(imlabel)

    imxsize,imysize = imCC.getWidth(),imCC.getHeight()
    lut = Map_UINT16_UINT16()
    FillRatioLut=Map_UINT16_UINT16()
    AspectRatioLut = Map_UINT16_UINT16()
    imtmp16=Image(imCC,"UINT16")
#    imDraw=Image(imCC)
#    copy(imCC,imDraw)
    orientation_list=[]
    L1_list,L2_list = [],[]
    for lab in bbl.keys():

        xc, yc, A, B, theta = fitRectangle(mats[lab])

        Area = AreaList[lab]
        x0= bbl[lab][0]
        y0= bbl[lab][1]
        x1= bbl[lab][2]
        y1= bbl[lab][3]
        xsize = 1+x1-x0
        ysize = 1+y1-y0
#        print x0,y0,x1,y1,lab
        decision,FillRatio,AspectRatio = TakeDecisionInertia(imxsize,imysize,x0,y0,x1,y1,xsize,ysize,Area,A,B,theta)
        if(decision == 250):
          angle = theta*180.0 /math.pi 
          orientation_list.append(angle)
          L1_list.append(A)
          L2_list.append(B)
          
#        print "lab,decision=",lab,decision
#        DrawInertiaAxes(imDraw,xc,yc,A,B,theta,decision+1,imxsize,imysize)
        lut[lab] = decision
        FillRatioLut[lab]=FillRatio
        AspectRatioLut[lab]=AspectRatio
        
        if(Area>500):
            print lab,xsize,ysize,Area,xsize*ysize,FillRatio,AspectRatio,decision
    if(len(L1_list) > 0):
      L1_avg = sum(L1_list)/len(L1_list)
      L2_avg = sum(L2_list)/len(L1_list)
      Area_thresh = (L1_avg * L2_avg)/2
      

      for lab in bbl.keys():
        if(lut[lab] == lut_OK):
          if(AreaList[lab]<Area_thresh):
            lut[lab] = lut_OKKO

    else:
      print "NO VALID BAR CODE"

    applyLookup(imlabel,lut,imtmp16)
    imout8,imtmp = Image(imCC),Image(imCC)
    copy(imtmp16,imout8)
#    ImDisplayX(imout8,"decisionsionsion")
    write(imout8,"./results/"+name+"_decision.png")
    if(len(L1_list) > 0):
      compare(imout8,"==",lut_OK, lut_OK,0,imtmp)
      close(imtmp,imout8,SquSE(L2_avg * 2))
      write(imout8,"./results/"+name+"_decision_close.png")

    applyLookup(imlabel,FillRatioLut,imtmp16)
    copy(imtmp16,imout8)
#    pdb.set_trace()
    write(imout8,"./results/"+name+"_FillRatio.png")

    applyLookup(imlabel,AspectRatioLut,imtmp16)
    copy(imtmp16,imout8)
    write(imout8,"./results/"+name+"_AspectRatio.png")
#    ImDisplayX(imDraw,"draw")
    
#    pdb.set_trace()
    print "look to draw"
    return imout8


    
def FindCB(im,name):
#    ImDisplayX(im,"oririri")

    inv(im,im)
    imT,imInd = Image(im),Image(im,"UINT16")
    ultimateOpen (im, imT,imInd, (im.getHeight()-10),1)
    write(imT,"mytrans.png")
    write(im,"myori.png")
    write(imInd,"myind.png")
    pdb.set_trace()

    low_thresh = 8
    ThreshUO(imT,low_thresh)
    imtmp,imthresh= Image(imT),Image(imT)

    threshold(imT,imthresh)

    imout8 =  SelectCCsInertia(imthresh,name)
    open(imthresh,imtmp,VertSE(10))
    close(imtmp,imtmp,HorizSE(10))

#    ImDisplayX(imthresh,"thresh")

#    ImDisplayX(imtmp,"resres")
#    pdb.set_trace()
    if(1):
        WriteResults(im,imT,imthresh,imtmp,name)
    return imtmp



def ImDrawInertiaAxes(imCC):
    imlabel = Image(imCC,"UINT16")
    label(imCC,imlabel)
    imDraw = Image(imCC)
    copy(imCC,imDraw)

    imxsize,imysize = imCC.getWidth(),imCC.getHeight()

    # Compute Blobs
    blobs = computeBlobs(imlabel)
    # Compute Inertia Matrices

    mats  = measInertiaMatrices(imCC, blobs)# imin?
    bbl=measBoundBoxes(imlabel)
    AreaList = measAreas(imlabel)
    draw_label = 0
    for lab in bbl.keys():

        xc, yc, A, B, theta = fitRectangle(mats[lab])

        Area = AreaList[lab]
        x0= bbl[lab][0]
        y0= bbl[lab][1]
        x1= bbl[lab][2]
        y1= bbl[lab][3]
        xsize = 1+x1-x0
        ysize = 1+y1-y0

        print lab, "Area=",Area,A*B,A,B,"diff=",Area-A*B
        if (draw_label==256):
          draw_label = 0
        else:
          draw_label = draw_label + 1
        DrawInertiaAxes(imDraw,xc,yc,A,B,theta,draw_label,imxsize,imysize)
    return imDraw

if(0):# Test inertia on single image
# MAIN
  imCC = Image("C:\\Morph-M 2.6\\share\\Images\\Bin\\\metal1.png")
  imDraw=ImDrawInertiaAxes(imCC)
  ImDisplayX(imDraw,"draw")
  pdb.set_trace()



if(1): #bench #main
#    mydir = "/home/marcoteg/src/images/code_barres/Gray/"
    mydir = "C:\Users\Marcotegui\Desktop\projects\LINX\images\code_barres\from_bea\Gray\\"
    files_to_transform = getFiles(mydir, 'png')
    nfiles = len(files_to_transform)
    n = 1
#    for cur_file in files_to_transform:
#        file_i = os.path.splitext(os.path.split(cur_file)[1])[0]
#        imout = FindCB(im,file_i)
#    if(1):
    im_list = ["IMG_20130413_153024Gray","IMG_20130413_154959Gray","IMG_20130413_155019Gray","IMG_20130413_153322Gray","IMG_20130413_153347Gray"]
#    for file_i in im_list:
#        file_i = "IMG_20130413_154959Gray"# vertical
#        file_i = "IMG_20130413_155019Gray"# incline
#        file_i = "IMG_20130413_153322Gray"# nutella vertical
#        file_i = "IMG_20130413_153347Gray"# nutella horizontal
    for cur_file in files_to_transform:
        file_i = os.path.splitext(os.path.split(cur_file)[1])[0]
        print n,"/",nfiles, file_i
        totoim=Image(mydir+file_i+".png")
        im = Image(totoim)
        scale(totoim,0.5,0.5,im)

        if(1):
          imtoggle = ToggleMapping(im,SquSE,1)
          copy(imtoggle,im)
        
        FindCB(im,file_i)
#        pdb.set_trace()
        n = n + 1


        # imtmp,imthresh= Image(im),Image(im)
        # threshold(im,imthresh)
        # inv(imthresh,imthresh)
        # imlabel = Image(im,"UINT16")
        # label(imthresh,imlabel)
        # bbl=measBoundBoxes(imlabel)
        # AreaList = measAreas(imlabel)
        # lut = Map_UINT16()
        # imtmp16=Image(imtmp,"UINT16")
        # for lab in bbl.keys():
        #     Area = AreaList[lab]
            
        #     AreaBB = (1+bbl[lab].getXSize())*(1+bbl[lab].getYSize())
        #     FillRatio = int(100.0 * Area/ float(AreaBB))
        #     lut[lab] = FillRatio

        #     print lab,FillRatio
        
        # applyLookup(imlabel,lut,imtmp16)

        # imFillRatio = Image(im)
        # copy(imtmp16,imFillRatio)
        # compare(imFillRatio,"==",100,255,0,imtmp)
        # imtmp.show()
        # n = n + 1

if(0):

    import pdb
#totoim = Image("../../images/code_barres/Gray/HPIM6404Gray.png")
    totoim = Image("../../images/code_barres/Gray/HPIM6413Gray.png")
    im=Image(totoim.getWidth(),totoim.getHeight())
    ImDisplayX(im)
    scale(totoim,0.5,0.5,im)
    inv(im,im)
    imT,imInd = Image(im),Image(im,"UINT16")
    ultimateOpen (im, imT,imInd, (im.getHeight()/3))
    imtmp,imthresh= Image(imT),Image(imT)
    threshold(imT,imthresh)
    ImDisplayX(imthresh,"thresh")
    open(imthresh,imtmp,VertSE(10))
    close(imtmp,imtmp,HorizSE(10))
    imtmp.show()
#im2.show()
#im3.show()
#pdb.set_trace()
