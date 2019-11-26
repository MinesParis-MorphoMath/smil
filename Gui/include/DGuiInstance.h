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



#ifndef _D_GUI_INSTANCE_H
#define _D_GUI_INSTANCE_H

#include "Core/include/private/DInstance.hpp"
#include "Core/include/private/DImage.hpp"
#include "private/DImageViewer.hpp"

#ifdef USE_QT
#include "Gui/Qt/DQtImageViewer.hpp"
#include "Gui/Qt/DQtImageViewer.hxx"
#endif // USE_QT

#ifdef USE_AALIB
#include "Gui/AALib/DAAImageViewer.hpp"
#endif // USE_AALIB



namespace smil
{
   /**
    * @ingroup Gui
    */
    /**@{*/

    template <class T>
    class ImageViewer;
    
    class Gui;
    
    
    /**
     * Gui module instance
     */
    class Gui : public UniqueInstance<Gui>
    {
        friend class UniqueInstance<Gui>;

    protected:
        Gui () {}
        virtual ~Gui () {}

    public:
        // Public interface
    //     static void kill();

        static RES_T initialize();
        /**
         * Run the event loop
         */
        static void execLoop();
        static void processEvents();
        static void showHelp();
        
        /**
         * Create a default viewer for type T
         */
        template <class T>
        ImageViewer<T> *createDefaultViewer(Image<T> &im=NULL);
    protected:
        virtual void _execLoop() {}
        virtual void _processEvents() {}
        virtual void _showHelp() {}
    private:
    };

    
    template <class T>
    ImageViewer<T> *Gui::createDefaultViewer(Image<T> &im)
    {
      #ifdef USE_QT
        return new QtImageViewer<T>(im);
      #elif defined USE_AALIB
        return new AaImageViewer<T>(im);
      #else
        return new ImageViewer<T>(im);
      #endif
    }

/**@}*/

} // namespace smil



#endif // _D_GUI_INSTANCE_H
