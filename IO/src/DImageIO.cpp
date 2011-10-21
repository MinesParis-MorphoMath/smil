#include "DImageIO.h"
#include "DImage.h"

#ifdef USE_PNG
#include "DImageIO_PNG.h"
#endif // USE_PNG

int read(const char* filename, Image<UINT8> *image)
{
    #ifdef USE_PNG
    readPNG(filename, image);
    #endif // USE_PNG
}

int write(Image<UINT8> *image, const char *filename)
{
    #ifdef USE_PNG
    writePNG(image, filename);
    #endif // USE_PNG
}

