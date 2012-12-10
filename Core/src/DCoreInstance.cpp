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

#ifdef USE_OPEN_MP
#include <omp.h>
#endif // USE_OPEN_MP


Core::Core ()
// : BaseObject("Core", false),
  : keepAlive(false),
    autoResizeImages(true),
    threadNumber(0)
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
//       Gui::initialize();
  }
}

void Core::registerObject(BaseObject *obj)
{
    if (obj->registered)
      return;

    Core *inst = Core::getInstance();
    inst->registeredObjects.push_back(obj);

    obj->registered = true;

    if (string(obj->getClassName())=="Image")
	inst->registeredImages.push_back(static_cast<BaseImage*>(obj));

#if DEBUG_LEVEL > 1
    cout << "Core::registerObject: " << obj->getClassName() << " " << obj << " created." << endl;
#endif // DEBUG_LEVEL > 1
}


void Core::unregisterObject(BaseObject *obj)
{
    if (!obj->registered)
      return;

    Core *inst = Core::getInstance();
    inst->registeredObjects.erase(std::remove(inst->registeredObjects.begin(), inst->registeredObjects.end(), obj));

    obj->registered = false;

    if (string(obj->getClassName())=="Image")
	inst->registeredImages.erase(std::remove(inst->registeredImages.begin(), inst->registeredImages.end(), static_cast<BaseImage*>(obj)));

#if DEBUG_LEVEL > 1
    cout << "Core::unregisterObject: " << obj->getClassName() << " " << obj << " deleted." << endl;
#endif // DEBUG_LEVEL > 1

    if (!inst->keepAlive && inst->registeredObjects.size()==0)
	inst->kill();
}


void Core::deleteRegisteredObjects()
{
    BaseObject *obj;
    Core *inst = Core::getInstance();
    vector<BaseObject*>::iterator it = inst->registeredObjects.begin();

    while (it!=inst->registeredObjects.end())
    {
	obj = *it++;
	delete obj;
    }
}

size_t Core::_getNumberOfThreads()
{
#ifdef USE_OPEN_MP
    if (threadNumber!=0)
      return threadNumber;
    
    int nthreads;
    #pragma omp parallel shared(nthreads)
    { 
	nthreads = omp_get_num_threads();
    }
    threadNumber = nthreads;
    return threadNumber;
#else // USE_OPEN_MP
    return 1;
#endif // USE_OPEN_MP
}

size_t Core::getNumberOfThreads()
{
    return Core::getInstance()->_getNumberOfThreads();
}

size_t Core::_getAllocatedMemory()
{
    vector<BaseImage*>::iterator it = registeredImages.begin();
    size_t totAlloc = 0;

    while (it!=registeredImages.end())
	totAlloc += (*it++)->getAllocatedSize();
    return totAlloc;
}

size_t Core::getAllocatedMemory()
{
    return Core::getInstance()->_getAllocatedMemory();
}

vector<BaseObject*> Core::_getRegisteredObjects() 
{ 
    return registeredObjects; 
}

vector<BaseObject*> Core::getRegisteredObjects() 
{ 
    return Core::getInstance()->_getRegisteredObjects(); 
}

vector<BaseImage*> Core::_getImages()  
{ 
    return registeredImages; 
}

vector<BaseImage*> Core::getImages()  
{ 
    return Core::getInstance()->_getImages();
}

void Core::_showAllImages()
{
    vector<BaseImage*>::iterator it = registeredImages.begin();

    while (it!=registeredImages.end())
	(*it++)->show();
}

void Core::showAllImages()
{
    Core::getInstance()->_showAllImages();
}

void Core::_hideAllImages()
{
    vector<BaseImage*>::iterator it = registeredImages.begin();

    while (it!=registeredImages.end())
	(*it++)->hide();
}

void Core::hideAllImages()
{
    Core::getInstance()->_hideAllImages();
}

