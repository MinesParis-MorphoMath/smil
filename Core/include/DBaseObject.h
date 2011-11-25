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


#ifndef _D_BASEOBJECT_H
#define _D_BASEOBJECT_H

#include <iostream>
#include <sstream>

using namespace std;


#include <string>

class D_BaseObject
{
public:
    D_BaseObject() {};

    typedef void parentClass;

    virtual string getInfoString(string indent = "") {};

    virtual void printSelf() {};

    inline virtual const char * getClassName();


protected:
    const char * className;

};
inline const char * D_BaseObject::getClassName()
{
    return className;
}

#endif
