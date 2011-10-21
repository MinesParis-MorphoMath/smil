#ifndef _D_IMAGE_IO_H
#define _D_IMAGE_IO_H


// #include "D_Types.h"
#include "DImage.h"

using namespace std;


int read(const char* filename, Image<UINT8> *image);
int write(Image<UINT8> *image, const char *filename);

// template <class T> Image<T>& operator << (Image<T> &im, const char *filename)
// {
//   cout << "oki" << endl;
// }
// 
// Image<UINT8>& operator << (Image<UINT8> &im, const char *filename)
// {
//   cout << "oki" << endl;
// }

template <>
Image<UINT8>& Image<UINT8>::operator << (const char *filename)
{
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




#endif // _D_IMAGE_IO_H
