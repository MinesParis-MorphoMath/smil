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


#include "DBaseObject.h"
#include "DBaseImage.h"
#include "DCoreInstance.h"
#include "DGui.h"

Core::Core ()
// : BaseObject("Core", false),
  : keepAlive(false)
{
#if DEBUG_LEVEL > 1
     cout << "Core created" << endl;
#endif // DEBUG_LEVEL > 1
}

Core::~Core ()
{
    deleteRegisteredObjects();
#if DEBUG_LEVEL > 1
      cout << "Core deleted" << endl;
#endif // DEBUG_LEVEL > 1
}


void Core::initialize()
{
  if (_instance == NULL)
  {
      _instance =  new Core;
      Gui::initialize();
  }
}

void Core::registerObject(BaseObject *obj)
{
    if (obj->registered)
      return;

    registeredObjects.push_back(obj);

    obj->registered = true;

    if (string(obj->getClassName())=="Image")
	registeredImages.push_back(static_cast<BaseImage*>(obj));

#if DEBUG_LEVEL > 1
    cout << "Core::registerObject: " << obj->getClassName() << " " << obj << " created." << endl;
#endif // DEBUG_LEVEL > 1
}


void Core::unregisterObject(BaseObject *obj)
{
    if (!obj->registered)
      return;

    registeredObjects.erase(std::remove(registeredObjects.begin(), registeredObjects.end(), obj));

    obj->registered = false;

    if (string(obj->getClassName())=="Image")
	registeredImages.erase(std::remove(registeredImages.begin(), registeredImages.end(), static_cast<BaseImage*>(obj)));

#if DEBUG_LEVEL > 1
    cout << "Core::unregisterObject: " << obj->getClassName() << " " << obj << " deleted." << endl;
#endif // DEBUG_LEVEL > 1

    if (!keepAlive && registeredObjects.size()==0)
	kill();
}


void Core::deleteRegisteredObjects()
{
    BaseObject *obj;
    vector<BaseObject*>::iterator it = registeredObjects.begin();

    while (it!=registeredObjects.end())
    {
	obj = *it++;
	delete obj;
    }
}


long Core::getAllocatedMemory()
{
    vector<BaseImage*>::iterator it = registeredImages.begin();
    long totAlloc = 0;

    while (it!=registeredImages.end())
	totAlloc += (*it++)->getAllocatedSize();
    return totAlloc;
}

void Core::showAllImages()
{
    vector<BaseImage*>::iterator it = registeredImages.begin();

    while (it!=registeredImages.end())
	(*it++)->show();
}

void Core::hideAllImages()
{
    vector<BaseImage*>::iterator it = registeredImages.begin();

    while (it!=registeredImages.end())
	(*it++)->hide();
}


