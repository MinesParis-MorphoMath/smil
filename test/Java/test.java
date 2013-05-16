
// Compile test: 
// java -classpath smilJava/ test.java
// Execute:
// java -classpath .:smilJava -Djava.library.path=lib test
// Linux users:
// Don't forget to add the smil-libs path to LD_LIBRARY_PATH:
// LD_LIBRARY_PATH=./lib/ java -classpath .:smilJava -Djava.library.path=lib test

public class test 
{

  static 
  {
       System.loadLibrary("smilBaseJava");
       System.loadLibrary("smilCoreJava");
       System.loadLibrary("smilIOJava");
       System.loadLibrary("smilGuiJava");
       System.loadLibrary("smilMorphoJava");
  }
  
  public static void main(String argv[]) 
  {
      Image_UINT8 im = new Image_UINT8(512, 512);
      smilBaseJava.fill(im, (short)127);
      smilBaseJava.drawRectangle(im, 128, 128, 256, 256);
      smilMorphoJava.dilate(im, im);
      
      // Warning: gui (Qt) stuff seems to crash in debug mode...
      im.show();
      Gui.execLoop();
  }

}
