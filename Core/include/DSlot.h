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

#ifndef _DSLOT_H
#define _DSLOT_H

#include <vector>

using namespace std;

class Event;
class Signal;

class Slot
{
  friend class Signal;
public:
  Slot() {}
  virtual ~Slot() 
  {
    unregisterAll();
  }
  virtual void run(Event &e) = 0;
protected:
  virtual void registerSignal(Signal &signal);
  virtual void unregisterSignal(Signal &signal, bool _disconnect=true);
  virtual void unregisterAll();
  vector<Signal*> _signals;
};


template <class T, class eventT=Event>
class MemberFunctionSlot : public Slot
{
public:
  typedef void(T::*memberFunc)(eventT&);
  MemberFunctionSlot(T *inst, memberFunc func)
  {
    _instance = inst;
    _function = func;
  }
protected:
  T *_instance;
  memberFunc _function;
  virtual void run(Event &e) 
  { 
    eventT *ePtr = static_cast<eventT*>(&e);
    (_instance->*_function)(*ePtr);
  }
};



#endif // _DSLOT_H
