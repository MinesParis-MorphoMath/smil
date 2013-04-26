package org.smil.demo;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.SurfaceView;
import android.view.Window;

public class SmilDemo extends Activity {
    private static final String TAG            = "Sample::Activity";

    public static final int     ALGO_GRADIENT = 0;
    public static final int     ALGO_TEXT = 1;

    private MenuItem            mItemAlgoGradient;
    private MenuItem            mItemAlgoText;
    private SurfaceView 		fakeview;
    private SurfaceView 		trueview;

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
        trueview = (SurfaceView)this.findViewById(R.id.cvsurface);
        trueview.setZOrderMediaOverlay(true);
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
//        Log.i(TAG, "onCreateOptionsMenu");
        mItemAlgoGradient = menu.add("Preview Gradient");
        mItemAlgoText = menu.add("Preview TopHat&Otsu");
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
//        Log.i(TAG, "Menu Item selected " + item);
        if (item == mItemAlgoGradient)
            algoType = ALGO_GRADIENT;
        else if (item == mItemAlgoText)
        	algoType = ALGO_TEXT;
        return true;
    }
}
