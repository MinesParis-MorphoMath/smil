

#include "Core/include/DTest.h"
#include "linx.hpp"
#include <string>

using namespace smil;


int mainCB(int argc, char *argv[])
{
    Image_UINT8 im1;
    
    read(argv[1], im1);
    //    read("titi.png", im1);
    Image_UINT8 im2(im1);
    FindCB(im1, im2);
    im1.show();
    im2.show();
write(im2,argv[2]);
    
    Gui::execLoop();
  
}
int main(int argc, char *argv[])/*BlurEstimation*/
{
    Image_UINT8 im1;
    
    read(argv[1], im1);
    //    read("titi.png", im1);
    Image_UINT8 im2(im1);
    BlurEstimation(im1, im2);
    //    im1.show();
    //    im2.show();
    write(im2,argv[2]);
    
    Gui::execLoop();
  
}

