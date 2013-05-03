package org.smil.demo;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.SurfaceView;
import android.view.Window;
import android.widget.TextView;

public class SmilDemo extends Activity {
    private static final String TAG            = "Sample::Activity";

    public static final int     ALGO_GRADIENT = 0;
    public static final int     ALGO_UO = 1;
    public static final int     ALGO_TOPHAT = 2;

    private MenuItem            mItemAlgoGradient;
    private MenuItem            mItemAlgoUO;
    private MenuItem            mItemAlgoTopHat;
    private SurfaceView 		fakeview;
    private Preview 		trueview;
    public TextView			textView;

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
        trueview = (Preview)this.findViewById(R.id.prevSurface);
        trueview.setZOrderMediaOverlay(true);
        textView = (TextView)this.findViewById(R.id.textView1);
        trueview.textView = textView;
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
//        Log.i(TAG, "onCreateOptionsMenu");
        mItemAlgoGradient = menu.add("Preview Gradient");
        mItemAlgoUO = menu.add("Preview UO");
        mItemAlgoTopHat = menu.add("Preview TopHat");
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
//        Log.i(TAG, "Menu Item selected " + item);
        if (item == mItemAlgoGradient)
            algoType = ALGO_GRADIENT;
        else if (item == mItemAlgoUO)
        	algoType = ALGO_UO;
        else if (item == mItemAlgoTopHat)
        	algoType = ALGO_TOPHAT;
        return true;
    }
}
