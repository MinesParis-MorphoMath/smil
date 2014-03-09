package org.smil.demo;

import java.io.IOException;
import java.util.List;

import smilJava.Image_UINT8;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.hardware.Camera;
import android.hardware.Camera.PreviewCallback;
import android.util.AttributeSet;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;

public abstract class PreviewBase extends SurfaceView implements SurfaceHolder.Callback, Runnable {
    private static final String TAG = "Sample::SurfaceView";

    private static Camera              mCamera;
    private static List<Camera.Size> 			cameraSizes;
    private boolean cameraInitialized = false;
    private static SurfaceHolder       mHolder;
    private static int                 mFrameWidth = 0;
    private static int                 mFrameHeight = 0;
    private byte[]              mFrame;
    
    protected static String infoMsg;
    
    public Bitmap processBmp = null;
    private static Image_UINT8 imIn = new Image_UINT8();
    private static Image_UINT8 imOut = new Image_UINT8();
    private static int rgba[] = null;
    
    protected boolean mThreadRun = false;
    protected boolean processing = false;
    private static long processTime = 0;
    private static long lastTime = 0;
    private double fps = 0; 
    
    public PreviewBase(Context context, AttributeSet attrs) 
    {
        super(context, attrs);
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
    	configure(0, w, h);
    }

    public void surfaceChanged(SurfaceHolder _holder, int format, int width, int height) 
    {
        configure(format, width, height);
        
		infoMsg = null;
    	
    }

    protected void setPictureFormat(int format) 
    {
        Camera.Parameters params = mCamera.getParameters();
        List<Integer> supported = params.getSupportedPictureFormats();
        if (supported != null) 
        {
                for (int f : supported) 
                {
                        if (f == format) 
                        {
                                params.setPreviewFormat(format);
                                mCamera.setParameters(params);
                                break;
                        }
                }
        }
}

	protected void setPreviewSize(int width, int height) 
	{
        Log.i(TAG, "setPreviewSize: " + width + "x" + height);
        Camera.Parameters params = mCamera.getParameters();
        List<Camera.Size> supported = params.getSupportedPreviewSizes();
        if (!cameraInitialized)
        {
        	width = 800;
        	height = 600;
        	cameraInitialized = true;
        }
        if (supported != null) 
        {
            for (Camera.Size size : supported) 
            {
                if (size.width <= width && size.height <= height) 
                {
                    mFrameWidth = size.width;
                    mFrameHeight = size.height;
                    params.setPreviewSize(mFrameWidth, mFrameHeight);
                    mCamera.setParameters(params);
                    break;
                }
            }
        }
	}
	
	protected void setImagesSize(int width, int height) 
	{
		processBmp = Bitmap.createBitmap(mFrameWidth, mFrameHeight, Bitmap.Config.ARGB_8888);
		
		rgba = new int[mFrameWidth*mFrameHeight];
		imIn.setSize(mFrameWidth, mFrameHeight);
		imOut.setSize(mFrameWidth, mFrameHeight);
	}
	
    public void configure(int format, int width, int height) 
    {
        if (mCamera == null)
        	return;

        mThreadRun = false;
        mCamera.stopPreview();
        while (processing)
        {
        	synchronized (this)
        	{
		        try {
					this.wait();
				} catch (InterruptedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
        	}
        }
	    	
        if (format!=0)
        	setPictureFormat(format);
        setPreviewSize(width, height);
        setImagesSize(width, height);
        mCamera.startPreview();
        mThreadRun = true;
    }
    
    public void surfaceCreated(SurfaceHolder holder) 
    {
        Log.i(TAG, "surfaceCreated");
        mCamera = Camera.open();
        cameraSizes = mCamera.getParameters().getSupportedPreviewSizes();
//        mCamera.setPreviewCallbackWithBuffer(new PreviewCallback() 
        mCamera.setPreviewCallback(new PreviewCallback() 
        {
            public void onPreviewFrame(byte[] data, Camera camera) 
            {
                synchronized (PreviewBase.this) 
                {
                    mFrame = data;
                    PreviewBase.this.notify();
                }
            }
        });
        SurfaceView fakeview = (SurfaceView) ((View)getParent()).findViewById(R.id.fakeCameraView); 
		try 
		{
			mCamera.setPreviewDisplay(fakeview.getHolder());
		} 
		catch (IOException e) 
		{
		//			Log.e(TAG, "mCamera.setPreviewDisplay fails: " + e);
		}
		mThreadRun = true;
		(new Thread(this)).start();
    }

    public void surfaceDestroyed(SurfaceHolder holder) 
    {
        if (mCamera == null)
        	return;

        mCamera.stopPreview();
        mCamera.setPreviewCallback(null);
        mCamera.release();
        mCamera = null;

        Log.i(TAG, "surfaceDestroyed");
    }

    //YUV Space to Grayscale
    static public void YUVtoGrayScale(int[] rgb, byte[] yuv420sp, int width, int height)
    {
        final int frameSize = width * height;
        for (int pix = 0; pix < frameSize; pix++)
        {
            int pixVal = (0xff & ((int) yuv420sp[pix])) - 16;
            if (pixVal < 0) pixVal = 0;
            if (pixVal > 255) pixVal = 255;
            rgb[pix] = 0xff000000 | (pixVal << 16) | (pixVal << 8) | pixVal;
        }
    }
    
    protected abstract void processImage(Image_UINT8 imIn, Image_UINT8 imOut);

    protected void processFrame(byte[] data)
    {
    	imIn.fromCharArray(data);
		long t0 = System.currentTimeMillis();
    	processImage(imIn, imOut);
		processTime = System.currentTimeMillis() - t0;
    	imOut.toCharArray(data);
    	
        for (int i = 0; i < mFrameWidth*mFrameHeight; i++) 
        {
            int y = data[i];
            rgba[i] = 0xff000000 + (y << 16) + (y << 8) + y;
        }

        processBmp.setPixels(rgba, 0/* offset */, mFrameWidth /* stride */, 0, 0, mFrameWidth, mFrameHeight);
        
		long tickFrameTime = System.currentTimeMillis();
		long curFrameTime = tickFrameTime - lastTime;
		fps = (long)(1000.0 / curFrameTime * 100) / 100.0;
		lastTime = tickFrameTime;
    }
    
    public void run() 
    {
//        Log.i(TAG, "Starting processing thread");
        while (true)
        {
        	
        	if (!mThreadRun)
        	{
                try 
                {
                    Thread.sleep(500);
                } catch (InterruptedException e) 
                {
                    e.printStackTrace();
                }
        	}
        	else
        	{
	        	processing = true;
	            synchronized (this) 
	            {
	                try 
	                {
	                    this.wait();
	                    processFrame(mFrame);
	                } catch (InterruptedException e) 
	                {
	                    e.printStackTrace();
	                }
	            }
	
	            if (processBmp != null) 
	            {
	                Canvas canvas = mHolder.lockCanvas();
	                if (canvas != null) 
	                {
	                	Paint paint = new Paint();
	                	
	        			canvas.drawColor(Color.BLACK);
	        			Bitmap scaledBmp = Bitmap.createScaledBitmap(processBmp, canvas.getWidth(), canvas.getHeight(), false);
	        			canvas.drawBitmap(scaledBmp, 0, 0, null);
	        			scaledBmp.recycle();
//	        			canvas.drawBitmap(processBmp, 0, 0, null);
	        			
	                	paint.setStyle(Paint.Style.FILL);
	        			paint.setStrokeWidth(3);
	        			paint.setColor(Color.RED);
	        			paint.setTextSize(30);
	        			canvas.drawText("Im Size: " + mFrameWidth + "x" + mFrameHeight , 20, 40, paint);
	        			canvas.drawText("Fps: " + fps , 20, 80, paint);
	        			canvas.drawText("Process time: " + processTime + " msec", 20, 120, paint);
	        			if (infoMsg!=null)
	            			canvas.drawText(infoMsg, 20, 120, paint);
	        			
	                    mHolder.unlockCanvasAndPost(canvas);
	                }
	            }
	        	processing = false;
        	}
        }
    }
}