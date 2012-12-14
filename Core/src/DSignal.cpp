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

#include "DSlot.h"
#include "DSignal.h"

#include <algorithm>

namespace smil
{

    void Signal::connect(BaseSlot *slot, bool _register)
    {
      vector<BaseSlot*>::iterator it = std::find(_slots.begin(), _slots.end(), slot);
      
      if (it!=_slots.end())
	return;
      
      _slots.push_back(slot);
      if (_register)
	slot->registerSignal(this);
    }

    void Signal::disconnect(BaseSlot *slot, bool _unregister)
    {
      vector<BaseSlot*>::iterator it = std::find(_slots.begin(), _slots.end(), slot);
      
      if (it==_slots.end())
	return;
      
      _slots.erase(it);
      
      if (_unregister)
	slot->unregisterSignal(this, false);
    }

    void Signal::disconnectAll()
    {
      vector<BaseSlot*>::iterator it = _slots.begin();
      
      while(it!=_slots.end())
      {
	(*it)->unregisterSignal(this, false);
	it++;
      }
    }

    void Signal::trigger(Event *e)
    {
      if (e && sender)
	e->sender = sender;
      
      vector<BaseSlot*>::iterator it = _slots.begin();
      
      while(it!=_slots.end())
      {
	(*it)->_run(e);
	it++;
      }
    }
    
} // namespace smil

