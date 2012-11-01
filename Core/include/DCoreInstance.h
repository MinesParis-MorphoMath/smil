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


#ifndef _DCORE_INSTANCE_H
#define _DCORE_INSTANCE_H

#include <iostream>
#include <algorithm>

#include "DCommon.h"
#include "DTimer.h"
#include "DSignal.h"

class BaseObject;
class BaseImage;

struct stat;

#include "DInstance.hpp"


class _DCORE Core : public UniqueInstance<Core>
{
    friend class UniqueInstance<Core>;

protected:
  Core ();
  ~Core ();
  
public:
  // Public interface
  bool keepAlive;
  
  void registerObject(BaseObject *obj);

  void unregisterObject(BaseObject *obj);
  
  vector<BaseObject*> getRegisteredObjects() { return registeredObjects; }
  vector<BaseImage*> getImages()  { return registeredImages; }
  
  UINT getNumberOfThreads();
  long getAllocatedMemory();
  void showAllImages();
  void hideAllImages();
  
  Signal onBaseImageCreated;
  
  
protected:
    vector<BaseObject*> registeredObjects;
    vector<BaseImage*> registeredImages;
    void deleteRegisteredObjects();
    UINT threadNumber;
  
public:
  static void initialize();
  
};



#endif // _DCORE_INSTANCE_H

