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



using namespace smil;

Core::Core ()
// : BaseObject("Core", false),
  : keepAlive(false),
    autoResizeImages(true),
    threadNumber(1),
    maxThreadNumber(1),
    systemName(SYSTEM_NAME),
    targetArchitecture(TARGET_ARCHITECTURE),
#ifdef USE_OPEN_MP
    supportOpenMP(true)
#else // USE_OPEN_MP
    supportOpenMP(false)
#endif // USE_OPEN_MP
{
    detectNumProcs();
    // Initialize threadNumber with the number of cores
    threadNumber = coreNumber;
}

Core::~Core ()
{
    deleteRegisteredObjects();
#if DEBUG_LEVEL > 1
      cout << "Core deleted" << endl;
#endif // DEBUG_LEVEL > 1
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

UINT Core::getNumberOfThreads()
{
    return this->threadNumber;
}

UINT Core::getNumberOfCores()
{
    return this->threadNumber;
}

UINT Core::getMaxNumberOfThreads()
{
    return this->maxThreadNumber;
}

RES_T Core::setNumberOfThreads(UINT nbr)
{
    ASSERT((nbr<=maxThreadNumber), "Nbr of thread exceeds system capacity !", RES_ERR);
    this->threadNumber = nbr;
    return RES_OK;
}

void Core::resetNumberOfThreads()
{
    this->threadNumber = this->maxThreadNumber;
}


size_t Core::getAllocatedMemory()
{
    vector<BaseImage*>::iterator it = this->registeredImages.begin();
    size_t totAlloc = 0;

    while (it!=this->registeredImages.end())
	totAlloc += (*it++)->getAllocatedSize();
    return totAlloc;
}

vector<BaseObject*> Core::getRegisteredObjects() 
{ 
    return this->registeredObjects; 
}

vector<BaseImage*> Core::getImages()  
{ 
    return this->registeredImages; 
}

void Core::showAllImages()
{
    vector<BaseImage*>::iterator it = this->registeredImages.begin();

    while (it!=this->registeredImages.end())
	(*it++)->show();
}

void Core::hideAllImages()
{
    vector<BaseImage*>::iterator it = this->registeredImages.begin();

    while (it!=this->registeredImages.end())
	(*it++)->hide();
}

void Core::getCompilationInfos(ostream &outStream)
{
    outStream << "System: " << this->systemName << endl;
    outStream << "Target Architecture: " << this->targetArchitecture << endl;
    outStream << "OpenMP support: " << (this->supportOpenMP ? "On" : "Off") << endl;
}


void cpuID(unsigned i, unsigned regs[4]) 
{
#ifdef _WIN32
  __cpuid((int *)regs, (int)i);

#else
  asm volatile
    ("cpuid" : "=a" (regs[0]), "=b" (regs[1]), "=c" (regs[2]), "=d" (regs[3])
     : "a" (i), "c" (0));
  // ECX is set to zero for CPUID function 4
#endif
}

void Core::detectNumProcs() 
{
    unsigned regs[4];

    // Get vendor
    char vendor[12];
    cpuID(0, regs);
    ((unsigned *)vendor)[0] = regs[1]; // EBX
    ((unsigned *)vendor)[1] = regs[3]; // EDX
    ((unsigned *)vendor)[2] = regs[2]; // ECX
    string cpuVendor = string(vendor, 12);

    // Get CPU features
    cpuID(1, regs);
    unsigned cpuFeatures = regs[3]; // EDX

    // Logical core count per CPU
    cpuID(1, regs);
    unsigned logical = (regs[1] >> 16) & 0xff; // EBX[23:16]
    unsigned cores = logical;

    if (cpuVendor == "GenuineIntel") 
    {
      // Get DCP cache info
      cpuID(4, regs);
      cores = ((regs[0] >> 26) & 0x3f) + 1; // EAX[31:26] + 1

    } 
    else if (cpuVendor == "AuthenticAMD") 
    {
      // Get NC: Number of CPU cores - 1
      cpuID(0x80000008, regs);
      cores = ((unsigned)(regs[2] & 0xff)) + 1; // ECX[7:0] + 1
    }

    int hyperThreadsCoef = (cpuFeatures & (1 << 28) && cores < logical) ? 2 : 1;
    
    maxThreadNumber = logical / hyperThreadsCoef;
    coreNumber = cores / hyperThreadsCoef;
}

