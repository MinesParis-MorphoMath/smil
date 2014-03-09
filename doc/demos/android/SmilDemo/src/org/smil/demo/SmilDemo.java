package org.smil.demo;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;

import android.app.Activity;
import android.graphics.Bitmap;
import android.hardware.Camera.Size;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.SubMenu;
import android.view.SurfaceView;
import android.view.Window;

public class SmilDemo extends Activity {
    private static final String TAG            = "Sample::Activity";

    public static final int     ALGO_GRADIENT = 0;
    public static final int     ALGO_UO = 1;
    public static final int     ALGO_TOPHAT = 2;

    private SubMenu            submAlgo;
    private SubMenu            submImSize;
    private SurfaceView 		fakeview;
    private Preview 		trueview;

    public static int           algoType       = ALGO_GRADIENT;

    static 
    {
		System.loadLibrary("smilBaseJava");
		System.loadLibrary("smilCoreJava");
		System.loadLibrary("smilIOJava");
		System.loadLibrary("smilGuiJava");
		System.loadLibrary("smilMorphoJava");
    }
    
    public SmilDemo() {
//        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
//        Log.i(TAG, "onCreate");
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        setContentView(R.layout.main);
        fakeview = (SurfaceView)this.findViewById(R.id.fakeCameraView);
        fakeview.setZOrderMediaOverlay(false);
        trueview = (Preview)this.findViewById(R.id.trueView);
        trueview.setZOrderMediaOverlay(true);
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
//        Log.i(TAG, "onCreateOptionsMenu");
    	
    	submAlgo = menu.addSubMenu(0, -1, 0, "Algorithm");
        submAlgo.add(0, 0, 0, "Gradient");
        submAlgo.add(0, 1, 0, "UO");
        submAlgo.add(0, 2, 0, "TopHat");
        
    	submImSize = menu.addSubMenu(1, -1, 0, "Img Size");
    	int i = 0;
    	for (Size size : trueview.getCameraSizes())
    	{
    		submImSize.add(1, i, 0, size.width + "x" + size.height);
    		i += 1;
    	}

    	menu.add(99, 0, 0, "Take snapshot");
        
        return true;
    }

	
    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
//        Log.i(TAG, "Menu Item selected " + item);
    	int groupId = item.getGroupId();
    	int itemId = item.getItemId();
    	
    	if (itemId<0)
    		return true;
    	
    	if (groupId==0)
    		algoType = itemId;
    	else if (groupId==1)
    	{
    		Size size = trueview.getCameraSizes().get(itemId);
    		trueview.setFrameSize(size.width, size.height);
    	}
    	else if (groupId==99)
    	{
    		String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
   			File imageFile = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DCIM), 
   				"SmilDemo_" + timeStamp + ".png");
    		try {
				FileOutputStream file = new FileOutputStream(imageFile);
				Bitmap bmp = trueview.processBmp;
				if (bmp!=null)
				{
					bmp.compress(Bitmap.CompressFormat.PNG, 100,  file);
				}
				file.flush();
				file.close();
				MediaStore.Images.Media.insertImage(getContentResolver(),imageFile.getAbsolutePath(),imageFile.getName(),imageFile.getName());
				
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} 
    		         
    	}
    	
        return true;
    }
}
