#ifndef _D_TYPES_H
#define _D_TYPES_H


typedef unsigned int UINT;
typedef unsigned char UINT8;
typedef unsigned short UINT16;
typedef unsigned int UINT32;
typedef char INT8;
typedef short INT16;
typedef int INT32;

enum RES_T
{
    RES_OK = 0,
    RES_ERR = -1,
    RES_ERR_BAD_ALLOCATION
};

#endif
