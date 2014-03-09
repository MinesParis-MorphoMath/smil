package org.smil.demo;

import android.content.Context;
import android.util.AttributeSet;

import smilJava.*;
import static smilJava.smilBaseJava.*;
import static smilJava.smilMorphoJava.*;

class Preview extends PreviewBase 
{
    public Preview(Context context, AttributeSet attrs) 
    {
        super(context, attrs);
    }
    
    private Image_UINT8 im1 = new Image_UINT8();
    private Image_UINT8 im2 = new Image_UINT8();
    private Image_UINT8 im3 = new Image_UINT8();

    private int fWidth = 0;
    private int fHeight = 0;

    @Override
    protected void processImage(Image_UINT8 imIn, Image_UINT8 imOut) 
    {
    	if (fWidth!=getFrameWidth())
    	{
        	fWidth = getFrameWidth();
        	fHeight = getFrameHeight();

        	im1.setSize(fWidth, fHeight);
        	im2.setSize(fWidth, fHeight);
        	im3.setSize(fWidth, fHeight);
    	}
        
    	if (SmilDemo.algoType==SmilDemo.ALGO_GRADIENT)
    	{
    		gradient(imIn, im1);
    		enhanceContrast(im1, imOut);
    	}
    	else if (SmilDemo.algoType==SmilDemo.ALGO_UO)
    	{
    		ultimateOpen(imIn, imOut, im2, fHeight/3);
   			enhanceContrast(imOut, imOut);
    	}
    	else if (SmilDemo.algoType==SmilDemo.ALGO_TOPHAT)
    	{
    		dualTopHat(imIn, im1, smilMorphoJava.hSE(10));
    		threshold(im1, imOut);
    	}
        
    }
}