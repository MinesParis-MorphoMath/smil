#include "DImageIO_BMP.h"
#include "DImage.h"


RES_T readBMP(const char *filename, Image<UINT8> *image)
{
    FILE* fp = fopen( filename, "rb" );

    if ( fp == NULL )
    {
        cout << "Error: Cannot open file " << filename << " for input." << endl;
        return RES_ERR;
    }
    bmpFileHeader fHeader;
    bmpInfoHeader iHeader;

    //read the bitmap file header
    if (!fread(&fHeader, sizeof(bmpFileHeader), 1 ,fp))
        return RES_ERR;

    //verify that this is a bmp file by check bitmap id
    if (fHeader.bfType !=0x4D42)
    {
        fclose(fp);
        return RES_ERR;
    }
    //read the bitmap info header
    if (!fread(&iHeader, sizeof(bmpInfoHeader), 1, fp))
        return RES_ERR;

    //move file point to the begging of bitmap data (skip palette information)
    fseek(fp, fHeader.bfOffBits, SEEK_SET);

    int width = iHeader.biWidth;
    int height = iHeader.biHeight;

    image->setSize(width, height);
    Image<UINT8>::lineType *lines = image->getLines();

    for (int i=height-1;i>=0;i--)
        if (!fread(lines[i], width*sizeof(UINT8), 1, fp))
            break;

    fclose(fp);

    return RES_OK;
}




/* write a png file */
RES_T writeBMP(Image<UINT8> *image, const char *filename)
{
    FILE* fp = fopen( filename, "wb" );

    if ( fp == NULL )
    {
        cout << "Error: Cannot open file " << filename << " for output." << endl;
        return RES_ERR;
    }
    bmpFileHeader fHeader;
    bmpInfoHeader iHeader;

    int width = image->getWidth();
    int height = image->getHeight();

    int nColors = 256;

    fHeader.bfType = 0x4D42;
    fHeader.bfSize = width*height*sizeof(UINT8) + sizeof(bmpFileHeader) + sizeof(bmpInfoHeader);
    fHeader.bfReserved1 = 0;
    fHeader.bfReserved2 = 0;
    fHeader.bfOffBits = sizeof(bmpFileHeader) + sizeof(bmpInfoHeader) + nColors*4;

    iHeader.biSize = sizeof(bmpInfoHeader);  // number of bytes required by the struct
    iHeader.biWidth = width;  // width in pixels
    iHeader.biHeight = height;  // height in pixels
    iHeader.biPlanes = 1; // number of color planes, must be 1
    iHeader.biBitCount = 8; // number of bit per pixel
    iHeader.biCompression = 0;// type of compression
    iHeader.biSizeImage = 0;  //size of image in bytes
    iHeader.biXPelsPerMeter = 2835;  // number of pixels per meter in x axis
    iHeader.biYPelsPerMeter = 2835;  // number of pixels per meter in y axis
    iHeader.biClrUsed = nColors;  // number of colors used by the bitmap
    iHeader.biClrImportant = nColors;  // number of colors that are important


    //write the bitmap file header
    fwrite(&fHeader, sizeof(bmpFileHeader), 1 ,fp);

    //write the bitmap image header
    fwrite(&iHeader, sizeof(bmpInfoHeader), 1 ,fp);

    // write palette
    for (int i=0;i<256;i++)
    {
        fputc(i, fp);
        fputc(i, fp);
        fputc(i, fp);
        fputc(0, fp);
    }

    Image<UINT8>::lineType *lines = image->getLines();

    for (int i=height-1;i>=0;i--)
        fwrite(lines[i], width*sizeof(UINT8), 1, fp);

    fclose(fp);

    return RES_OK;
}

