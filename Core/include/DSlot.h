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

#ifndef _DSLOT_H
#define _DSLOT_H

#include <vector>
#include <iostream>

using namespace std;

#include "DCommon.h"

class Event;
class Signal;

class _DCORE BaseSlot
{
  friend class Signal;
public:
  BaseSlot() {}
  virtual ~BaseSlot() 
  {
    unregisterAll();
  }
protected:
  virtual void _run(Event *e=NULL) = 0;
  virtual void registerSignal(Signal *signal);
  virtual void unregisterSignal(Signal *signal, bool _disconnect=true);
  virtual void unregisterAll();
  vector<Signal*> _signals;
};

template <class eventT>
class Slot : public BaseSlot
{
public:
  Slot() {}
  virtual ~Slot() {}
  virtual void run(eventT * = NULL)
  {
  }
  void operator() (eventT *e=NULL)
  {
  }
protected:
  virtual void _run(Event *e=NULL)
  {
    if (e)
      run(static_cast<eventT*>(e));
    else run();
  }
};

template <class T, class eventT=Event>
class MemberFunctionSlot : public Slot<eventT>
{
public:
  typedef void(T::*memberFunc)(eventT*);
  typedef void(T::*voidMemberFunc)();
  MemberFunctionSlot()
  {
      _instance = NULL;
      _function = NULL;
  }
  MemberFunctionSlot(T *inst, memberFunc func)
  {
      init(inst, func);
  }
  MemberFunctionSlot(T *inst, voidMemberFunc func)
  {
      init(inst, func);
  }
  void init(T *inst, memberFunc func)
  {
      _instance = inst;
      _function = func;
  }
  void init(T *inst, voidMemberFunc func)
  {
      _instance = inst;
      _void_function = func;
  }
protected:
  T *_instance;
  memberFunc _function;
  voidMemberFunc _void_function;
  virtual void run(eventT *e=NULL) 
  { 
    if (!_instance)
      return;
    
    if (_function)
      (_instance->*_function)(e);
    if (_void_function)
      (_instance->*_void_function)();
  }
};

template <class eventT=Event>
class FunctionSlot : public Slot<eventT>
{
public:
  typedef void(*funcPtr)(eventT*);
  typedef void(*voidFuncPtr)();
  FunctionSlot(funcPtr func)
  {
    _function = func;
  }
protected:
  funcPtr _function;
  virtual void run(eventT *e) 
  { 
    (*_function)(e);
  }
};



#endif // _DSLOT_H
