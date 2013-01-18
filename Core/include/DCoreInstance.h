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
#include "DErrors.h"

#include "private/DInstance.hpp"

namespace smil
{
    class BaseObject;
    class BaseImage;

   /**
    * \ingroup Core
    * @{
    */
    
    /**
     * Core module instance
     */
    class _DCORE Core : public UniqueInstance<Core>
    {
	friend class UniqueInstance<Core>;

    protected:
      Core ();
      ~Core ();
      
    public:
      
	// Public interface
	
	bool keepAlive;
	bool autoResizeImages;
	
	UINT getNumberOfThreads();
	RES_T setNumberOfThreads(UINT nbr);
	void resetNumberOfThreads();
	size_t getAllocatedMemory();
	void registerObject(BaseObject *obj);
	void unregisterObject(BaseObject *obj);
	void showAllImages();
	void hideAllImages();
	vector<BaseObject*> getRegisteredObjects();
	vector<BaseObject*> getObjectsByClassName(const char* cName);
	vector<BaseImage*> getImages();
	void getCompilationInfos(ostream &outStream = std::cout);
	
	Signal onBaseImageCreated;
	
      
    protected:
	UINT threadNumber;
	UINT maxThreadNumber;
	
	const char *systemName;
	const char *targetArchitecture;
	const bool supportOpenMP;
	
	vector<BaseObject*> registeredObjects;
	void deleteRegisteredObjects();
      
    };

    /*@}*/
    
} // namespace smil


#endif // _DCORE_INSTANCE_H

