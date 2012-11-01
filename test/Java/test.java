
// Linux users:
// Don't forget to add the smil-libs path to LD_LIBRARY_PATH

public class test 
{

  static 
  {
       System.loadLibrary("smilCoreJava");
       System.loadLibrary("smilIOJava");
       System.loadLibrary("smilGuiJava");
       System.loadLibrary("smilMorphoJava");
  }
  
  public static void main(String argv[]) 
  {
      Image_UINT8 im = new Image_UINT8(512, 512);
      smilCoreJava.fill(im, (short)127);
      smilCoreJava.drawRectangle(im, 128, 128, 256, 256);
      smilMorphoJava.dilate(im, im);
      
      // Warning: gui (Qt) stuff seems to crash in debug mode...
      im.show();
      Core.execLoop();
  }

}
