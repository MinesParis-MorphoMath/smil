from smilPython import *

def tgdskiz(im1, im2, imMask):
    tmp1 = Image(im1)
    tmp2 = Image(im1)
    sumIm = Image(im1)
    vol0 = -1
    vol1 = 0
    while vol0<vol1:
      dilate(im1, tmp1, se)
      dilate(im2, tmp2, se)
      addNoSat(tmp1, tmp2, sumIm)




def skizBin(label1, label2, maskIm, se=hSE()):
    tmp1 = Image(label1)
    tmp2 = Image(label1)
    lpe = Image(label1)	
    oldVol = 0
    newVol = -1
    while oldVol != newVol:
        dilate(label1, tmp1, se)
        dilate(label2, tmp2, se)
        addNoSat(label1, label2, lpe)
        threshold(lpe, 0, 254, lpe)
        inf(maskIm, lpe, lpe)
        mask(tmp1, lpe, tmp1)
        mask(tmp2, lpe, tmp2)
        sup(label1, tmp1, label1)
        sup(label2, tmp2, label2)

        oldVol = newVol
        newVol = vol(label1)
        print(newVol)
        #raw_input()

def trueWatershed(imIn, imMark, imOut, se=hSE()):
    #global label1, label2, maskIm
    label1 = Image(imIn)
    label2 = Image(imIn)
    maskIm = Image(imIn)
    label1.showLabel()
    label2.showLabel()
    maskIm.show()

    label(imMark, label1)
    mask(~label1, label1, label2)

    for i in range(rangeVal(imIn)[1]+1):
      threshold(imIn, 0, i, maskIm)
      sup(maskIm, imMark, maskIm)
      skizBin(label1, label2, maskIm, se)

    copy(label1, imOut)
    

def skizIsotrop(imIn, imOut, se=hSE()):
    label1 = Image(imIn)
    label2 = Image(imIn)
    maskIm = Image(imIn)
    fill(maskIm, 255)

    label(imIn, label1)
    mask(~label1, label1, label2)

    label1.showLabel()
    label2.showLabel()
    lpe.show()
    tmp1.showLabel()
    tmp2.showLabel()
    
    skizBin(label1, label2, maskIm, se)
    
    
