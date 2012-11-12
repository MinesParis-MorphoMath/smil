/*
 * Copyright (c) 2011, Matthieu FAESSEL and ARMINES
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of Matthieu FAESSEL, or ARMINES nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS AND CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#ifndef _D_BASEOBJECT_H
#define _D_BASEOBJECT_H

#include <iostream>
#include <sstream>

#include <string>


#include "DCommon.h"
#include "DSlot.h"

using namespace std;

class Core;

/**
 * Base Smil Object
 */
class _DCORE BaseObject
{
public:
    BaseObject(bool _register=true);
    BaseObject(const string _className, bool _register=true);
    BaseObject(const BaseObject &rhs, bool _register=true);
    
    // Assignment operator
    BaseObject& operator=(const BaseObject &rhs)
    {
	this->_clone(rhs);
	return *this;
    }
    
private:    
    void _clone(const BaseObject &rhs);
    
public:

    virtual ~BaseObject() ;
    
    Core *getCoreInstance();
    typedef void parentClass;
    virtual const char *getInfoString(const char *indent = "") const { return NULL; }
    virtual void printSelf(ostream &os=std::cout, string indent="") const {}
    virtual const char *getClassName() const;

    virtual void setName(const char *_name)
    {
	name = _name;
    }
    virtual const char *getName() const
    {
	return name.c_str();
    }
    typedef void(BaseObject::*voidMemberFunc)();

    bool triggerEvents;
protected:
    string className;
    string name;
    bool registered;

    friend class Core;
};

#endif
