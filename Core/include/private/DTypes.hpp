#ifndef _D_TYPES_H
#define _D_TYPES_H


typedef unsigned int UINT;
typedef unsigned char UINT8;
typedef unsigned short UINT16;
typedef unsigned int UINT32;
#ifndef _MSC_VER
typedef char INT8;
#endif // _MSC_VER
typedef short INT16;
typedef int INT32;

enum RES_T
{
    RES_OK = 0,
    RES_ERR = -1,
    RES_ERR_BAD_ALLOCATION,
	RES_NOT_IMPLEMENTED
};

template <class T>
inline const char *getDataTypeAsString(T &val)
{
    return "Unknown";
}

template <>
inline const char *getDataTypeAsString(UINT8 &val)
{
    return "UINT8 (unsigned char)";
}

template <>
inline const char *getDataTypeAsString(UINT16 &val)
{
    return "UINT16 (unsigned short)";
}


#endif
