package org.smil.demo;

import java.io.ByteArrayInputStream;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Typeface;
import android.util.AttributeSet;
import android.widget.TextView;

import smilJava.*;

class Preview extends PreviewBase {
    public Preview(Context context, AttributeSet attrs) {
        super(context, attrs);
    }
    
    private int[] rgba = null;
    private Image_UINT8 im1 = new Image_UINT8();
    private Image_UINT8 im2 = new Image_UINT8();
    private Image_UINT8 im3 = new Image_UINT8();
    private Image_UINT16 imLbl = new Image_UINT16();
    private int fWidth = 0;
    private int fHeight = 0;

    //YUV Space to Grayscale
    static public void YUVtoGrayScale(int[] rgb, byte[] yuv420sp, int width, int height){
        final int frameSize = width * height;
        for (int pix = 0; pix < frameSize; pix++){
            int pixVal = (0xff & ((int) yuv420sp[pix])) - 16;
            if (pixVal < 0) pixVal = 0;
            if (pixVal > 255) pixVal = 255;
            rgb[pix] = 0xff000000 | (pixVal << 16) | (pixVal << 8) | pixVal;
        }
    }
    
    @Override
    protected Bitmap processFrame(byte[] data) {
    	if (fWidth!=getFrameWidth())
    	{
        	fWidth = getFrameWidth();
        	fHeight = getFrameHeight();
        	rgba = new int[fWidth*fHeight];
        	im1.setSize(fWidth, fHeight);
        	im2.setSize(fWidth, fHeight);
        	im3.setSize(fWidth, fHeight);
        	imLbl.setSize(fWidth, fHeight);
    	}
        int frameSize = fWidth * fHeight;

    	
    	im1.fromCharArray(data);
    	if (SmilDemo.algoType==SmilDemo.ALGO_GRADIENT)
    	{
    		smilMorphoJava.gradient(im1, im2);
    		smilBaseJava.enhanceContrast(im2, im3, 0.1);
            im3.toCharArray(data);
    	}
    	else if (SmilDemo.algoType==SmilDemo.ALGO_UO)
    	{
    		smilMorphoJava.ultimateOpen(im1, fHeight/3, im2, imLbl);
    		smilBaseJava.enhanceContrast(im2, im3, 0.1);
            im3.toCharArray(data);
    	}
    	else if (SmilDemo.algoType==SmilDemo.ALGO_TOPHAT)
    	{
    		smilMorphoJava.topHat(im1, im2, smilMorphoJava.hSE(10));
    		smilBaseJava.threshold(im2, im3);
            im3.toCharArray(data);
    	}
        
        for (int i = 0; i < frameSize; i++) {
            int y = data[i];
            rgba[i] = 0xff000000 + (y << 16) + (y << 8) + y;
        }

        Bitmap bmp = Bitmap.createBitmap(fWidth, fHeight, Bitmap.Config.ARGB_8888);
        bmp.setPixels(rgba, 0/* offset */, fWidth /* stride */, 0, 0, fWidth, fHeight);
        return bmp;
    }
}