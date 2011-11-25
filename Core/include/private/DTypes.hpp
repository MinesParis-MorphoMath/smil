/*
 * Smil
 * Copyright (c) 2010 Matthieu Faessel
 *
 * This file is part of Smil.
 *
 * Smil is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * Smil is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Smil.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 */


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
