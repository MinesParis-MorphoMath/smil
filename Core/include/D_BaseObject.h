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
