package org.smil.demo;

import android.app.Activity;
import android.content.pm.ActivityInfo;
import android.hardware.Camera.Size;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.SubMenu;
import android.view.SurfaceView;
import android.view.ViewGroup;
import android.view.Window;
import android.widget.TextView;

public class SmilDemo extends Activity {
    private static final String TAG            = "Sample::Activity";

    public static final int     ALGO_GRADIENT = 0;
    public static final int     ALGO_UO = 1;
    public static final int     ALGO_TOPHAT = 2;

    private SubMenu            submAlgo;
    private SubMenu            submImSize;
    private SurfaceView 		fakeview;
    private Preview 		trueview;
    public TextView			textView;

    public static int           algoType       = ALGO_UO;

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
        textView = (TextView)this.findViewById(R.id.textView1);
        trueview = (Preview)this.findViewById(R.id.trueView);
        trueview.setZOrderMediaOverlay(true);
        trueview.textView = textView;
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
//        Log.i(TAG, "onCreateOptionsMenu");
    	
    	submImSize = menu.addSubMenu(0, -1, 0, "Img Size");
    	int i = 0;
    	for (Size size : trueview.getCameraSizes())
    	{
    		submImSize.add(0, i, 0, size.width + "x" + size.height);
    		i += 1;
    	}
    	
    	submAlgo = menu.addSubMenu(1, -1, 0, "Algorithm");
        submAlgo.add(1, 0, 0, "Gradient");
        submAlgo.add(1, 1, 0, "UO");
        submAlgo.add(1, 2, 0, "TopHat");
        
        
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
    	{
    		Size size = trueview.getCameraSizes().get(itemId);
    		trueview.setFrameSize(size.width, size.height);
//    		trueview.destroyDrawingCache();
    	}
    	else if (groupId==1)
    		algoType = itemId;
    	
        return true;
    }
}
