/*
 * Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES
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


#include "Gui/include/DGuiInstance.h"

#ifdef USE_QT
#include "Gui/Qt/DQtGuiInstance.h"
#include "Gui/Qt/DQtImageViewer.hxx"
#endif // USE_QT

namespace smil
{


        template <>
        RES_T UniqueInstance<Gui>::initialize()
        {
                if (Gui::_instance == NULL)
                {
        #ifdef USE_QT
                // Check if there is already a qapplication running
                if (!qApp)
                  Gui::_instance =  new QtAppGui;
                else
                  Gui::_instance =  new QtGui;
        #else // USE_QT
                        Gui::_instance =  new Gui;
        #endif // USE_QT
                }
                return RES_OK;
        }

        void Gui::execLoop()
        {
                Gui::getInstance()->_execLoop();
        }

        void Gui::processEvents()
        {
                Gui::getInstance()->_processEvents();
        }

        void Gui::showHelp()
        {
                Gui::getInstance()->_showHelp();
        }

}
