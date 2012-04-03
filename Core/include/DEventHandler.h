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
 *     * Neither the name of the University of California, Berkeley nor the
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


#ifndef _DEVENTHANDLER_H
#define _DEVENTHANDLER_H


#include "DEvent.h"
#include "DBaseObject.h"
#include <map>
using namespace std;

// class baseObject;
class slotOwner;

template <class R, class A>
class slotBase
{
  public:
    inline virtual ~slotBase() {};

    inline R exec(const A *arg) { call(arg); };


  private:
    virtual R call(const A * ) {};
};

class slot : public slotBase<void, event>
{
};

// class slot 
// {
//   public:
//     inline virtual ~slot() {};
// 
//     inline void exec(const event * event) { call(event); };
// 
// 
//   private:
//     virtual void call(const event * ) = 0;
// 
// };

template<class T, class eventT>
class memberFunctionSlot : public slotBase<T, eventT>
{
  public:
    typedef void(T::*memberFunc)(eventT*);

    inline memberFunctionSlot(T * instance, memberFunc memFn) : _instance(instance), _function(memFn) {};

    inline void call(const event * event) { (_instance->*_function)(static_cast<eventT*>(event)); };

    inline memberFunc getFunction() const { return _function; };


  private:
    T * _instance;

    memberFunc _function;

};

class eventHandler : public baseObject 
{
  public:
    eventHandler(baseObject * obj);

    ~eventHandler();

    typedef baseObject parentClass;

    baseObject * owner;

    inline virtual void trigger(const event * event);

    typedef map<slot*,slotOwner*> handlers;

    slotOwner * rec;

    handlers _handlers;

    //! Connect an object's event with a member function.  Multiple
    //! connections which are identical are treated as separate connections.
    template<class T, class EventT>
      inline slot * connect(T * obj, void (T::*memFn)(EventT*));

    void disconnect(slot * hf);

    inline void operator ()(const event * event) {trigger(event);};

};
inline void eventHandler::trigger(const event * event) 
{
    handlers::iterator it = _handlers.begin();
    while(it != _handlers.end())
    {
	slot* f = it->first;
	f->exec(event);
	++it;
    }
}

//! Connect an object's event with a member function.  Multiple
//! connections which are identical are treated as separate connections.
template<class T, class EventT>
inline slot * eventHandler::connect(T * obj, void (T::*memFn)(EventT*)) 
{
    memberFunctionSlot<T, EventT> *h = new memberFunctionSlot<T, EventT>(obj, memFn);
    _handlers[h] = obj;
    obj->registerFunctionHandler(this, h);
    return h;
}

#endif // _DEVENTHANDLER_H
