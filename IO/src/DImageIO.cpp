#include "DImageIO.h"
#include "DImage.h"

#include <string>
#include <algorithm>

#ifdef USE_PNG
#include "DImageIO_PNG.h"
#endif // USE_PNG

const char *getFileExtension(const char *fileName)
{
    string fName(fileName);
    string::size_type idx = fName.rfind('.');
    string fExt = fName.substr(idx+1).c_str();
    transform(fExt.begin(), fExt.end(), fExt.begin(), ::toupper);
    return fExt.c_str();
}

int read(const char* filename, Image<UINT8> *image)
{
    string fileExt = getFileExtension(filename);
    
    if (fileExt=="BMP")
	readBMP(filename, image);
    
    #ifdef USE_PNG
    else if (fileExt=="PNG")
      readPNG(filename, image);
//     image->modified();
    #endif // USE_PNG
      
    else 
    {
      cout << "File type not supported" << endl;
    }
}

int write(Image<UINT8> *image, const char *filename)
{
    string fileExt = getFileExtension(filename);
    
    if (fileExt=="BMP")
	writeBMP(image, filename);
    
    #ifdef USE_PNG
    else if (fileExt=="PNG")
      writePNG(image, filename);
    #endif // USE_PNG
      
    else 
    {
      cout << "File type not supported" << endl;
    }
}

template <>
Image<UINT8>& Image<UINT8>::operator << (const char *filename)
{
//     cout << "here ok" << endl;
    read(filename, this);
    modified();
    return *this;
}

template <>
Image<UINT8>& Image<UINT8>::operator >> (const char *filename)
{
    write(this, filename);
    return *this;
}
