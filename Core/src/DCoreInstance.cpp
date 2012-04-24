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


#include "DBaseObject.h"
#include "DBaseImage.h"
#include "DCoreInstance.h"

// Initialization of singleton to NULL
// Core Core::_singleton;
Core *Core::_singleton = NULL;


guiInstance::guiInstance()
  : _qapp(NULL)
{
      if (!qApp)
      {
	    cout << "Core qt created" << endl;
	  int ac = 1;
	  char **av = NULL;
	  _qapp = new QApplication(ac, av);
      }
      _timer = new timer();
      _timer->app = _qapp;
      _timer->start();
}

guiInstance::~guiInstance()
{
    delete _timer;
}

void guiInstance::execLoop() 
{ 
    if (_qapp)
      _qapp->exec(); 
    else if (qApp)
      qApp->exec();
}



Core::Core ()
: baseObject("Core", false), 
  _value (0), 
  keepAlive(false)
{ 
	cout << "Core created" << endl;
    guiInst = new guiInstance();
  
}

Core::~Core () 
{
  deleteRegisteredObjects();
  delete guiInst;
      cout << "Core deleted" << endl;
}


Core *Core::getInstance ()
{
  initialize();
  return _singleton;
}

void Core::kill ()
{
  if (_singleton==NULL)
    return;
  
//       std::cout << "Bye" << std::endl;
    
//       qApp->exit(0);
    
    delete _singleton;
    _singleton = NULL;
  
}


void Core::initialize()
{
  if (_singleton == NULL)
  {
      _singleton =  new Core;
  }
}

void Core::execLoop() 
{ 
    getInstance()->guiInst->execLoop();
}

void Core::registerObject(baseObject *obj)
{
    if (obj->registered)
      return;
    
    registeredObjects.push_back(obj);
    obj->registered = true;
//     cout << obj->getClassName() << " created." << endl;
}

void Core::unregisterObject(baseObject *obj)
{
    if (!obj->registered)
      return;
    
    std::vector<baseObject*>::iterator newEnd = std::remove(registeredObjects.begin(), registeredObjects.end(), obj);

    registeredObjects.erase(newEnd, registeredObjects.end());
    obj->registered = false;
//     cout << obj->getClassName() << " deleted." << endl;
    
    if (!keepAlive && registeredObjects.size()==0)
	kill();
}

void Core::deleteRegisteredObjects()
{
    baseObject *obj;
    vector<baseObject*>::iterator it = registeredObjects.begin();
    
    while (it!=registeredObjects.end())
    {
	obj = *it++;
	delete obj;
    }
}

