package org.smil.demo;

import java.io.IOException;
import java.util.List;

import smilJava.Image_UINT8;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.hardware.Camera;
import android.hardware.Camera.PreviewCallback;
import android.util.AttributeSet;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.widget.TextView;

public abstract class PreviewBase extends SurfaceView implements SurfaceHolder.Callback, Runnable {
    private static final String TAG = "Sample::SurfaceView";

    private Camera              mCamera;
    private List<Camera.Size> 			cameraSizes;
    private SurfaceHolder       mHolder;
    private static int                 mFrameWidth = 0;
    private static int                 mFrameHeight = 0;
    private byte[]              mFrame;
    private boolean             mThreadRun;
    public TextView			textView = null;
    
    private Image_UINT8 imIn = new Image_UINT8();
    private Image_UINT8 imOut = new Image_UINT8();
    private int rgba[] = null;
    
    public PreviewBase(Context context, AttributeSet attrs) {
        super(context, attrs);
//        Log.i(TAG, "Preview Created");
        mHolder = getHolder();
        mHolder.addCallback(this);
    }

    public int getFrameWidth() {
        return mFrameWidth;
    }

    public int getFrameHeight() {
        return mFrameHeight;
    }
    
    public List<Camera.Size> getCameraSizes()
    {
    	return cameraSizes;
    }

    public void setFrameSize(int w, int h) 
    {
    	mThreadRun = false;
    	
        if (mCamera != null) 
        {
            mHolder.removeCallback(this);
	        
	    	mFrameWidth = w;
	    	mFrameHeight = h;
	    	
	       	rgba = new int[mFrameWidth*mFrameHeight];
	    	imIn.setSize(mFrameWidth, mFrameHeight);
	    	imOut.setSize(mFrameWidth, mFrameHeight);
	    	
			if (textView!=null)
				textView.setText("Im Size: " + mFrameWidth + "x" + mFrameHeight);
			
	    	Camera.Parameters params = mCamera.getParameters();
	        params.setPreviewSize(mFrameWidth, mFrameHeight);
	        mCamera.setParameters(params);
	        try 
	        {
	        	SurfaceView fakeview = (SurfaceView) ((View)getParent()).findViewById(R.id.fakeCameraView); 
				mCamera.setPreviewDisplay(fakeview.getHolder());
			} catch (IOException e) 
			{
	//			Log.e(TAG, "mCamera.setPreviewDisplay fails: " + e);
			}
	        
	        mHolder.addCallback(this);
	        mCamera.startPreview();
        }
        mThreadRun = true;
    }

    public void surfaceChanged(SurfaceHolder _holder, int format, int width, int height) {
        Log.i(TAG, "surfaceChanged");
        if (mCamera != null) {
            
        	int newWidth = 0;
        	int newHeight = 0;
        	
            // selecting optimal camera preview size
            if (mFrameWidth==0)
            {
                double minDiff = Double.MAX_VALUE;
                for (Camera.Size size : cameraSizes) {
                    if (Math.abs(size.height - height) < minDiff) {
                        newWidth = size.width;
                        newHeight = size.height;
                        minDiff = Math.abs(size.height - height);
                    }
                }
            }
            
          mCamera.setDisplayOrientation(90);
            
          setFrameSize(newWidth, newHeight);
        }
    }

    public void surfaceCreated(SurfaceHolder holder) {
        Log.i(TAG, "surfaceCreated");
        mCamera = Camera.open();
        cameraSizes = mCamera.getParameters().getSupportedPreviewSizes();
        mCamera.setPreviewCallback(new PreviewCallback() {
            public void onPreviewFrame(byte[] data, Camera camera) {
                synchronized (PreviewBase.this) {
                    mFrame = data;
                    PreviewBase.this.notify();
                }
            }
        });
        (new Thread(this)).start();
    }

    public void surfaceDestroyed(SurfaceHolder holder) {
        Log.i(TAG, "surfaceDestroyed");
        mThreadRun = false;
        if (mCamera != null) {
            synchronized (this) {
                mCamera.stopPreview();
                mCamera.setPreviewCallback(null);
                mCamera.release();
                mCamera = null;
            }
        }
    }

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
    
    protected abstract void processImage(Image_UINT8 imIn, Image_UINT8 imOut);

    protected Bitmap processFrame(byte[] data)
    {
    	Bitmap bmp = Bitmap.createBitmap(mFrameWidth, mFrameHeight, Bitmap.Config.ARGB_8888);
    	
    	imIn.fromCharArray(data);
    	processImage(imIn, imOut);
    	imOut.toCharArray(data);
    	
        for (int i = 0; i < mFrameWidth*mFrameHeight; i++) 
        {
            int y = data[i];
            rgba[i] = 0xff000000 + (y << 16) + (y << 8) + y;
        }

        bmp.setPixels(rgba, 0/* offset */, mFrameWidth /* stride */, 0, 0, mFrameWidth, mFrameHeight);
        
        return bmp;
    }
    
    public void run() {
        mThreadRun = true;
//        Log.i(TAG, "Starting processing thread");
        while (mThreadRun) {
            Bitmap bmp = null;

            synchronized (this) {
                try {
                    this.wait();
                    bmp = processFrame(mFrame);
               } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }

            if (bmp != null) {
                Canvas canvas = mHolder.lockCanvas();
                if (canvas != null) {
                	canvas.drawBitmap(bmp, (canvas.getWidth() - getFrameWidth()) / 2, (canvas.getHeight() - getFrameHeight()) / 2, null);
                    mHolder.unlockCanvasAndPost(canvas);
                }
                bmp.recycle();
            }
        }
    }
}