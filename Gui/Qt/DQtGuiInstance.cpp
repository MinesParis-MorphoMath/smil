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


#ifdef USE_QT

#include <QTimer>
#include <unistd.h>

#if defined(Q_OS_WIN)
#include <conio.h>
#include <QTimer>
#else
#include <QSocketNotifier>
#endif
#include <QThread>

#include "DQtGuiInstance.h"

using namespace smil;


namespace smil 
{
    void QtGui::_execLoop() 
    { 
	qtLoop();
    }

    void QtGui::_processEvents() 
    { 
	if (qApp)
	  qApp->processEvents();
    }


    QtAppGui::QtAppGui()
      : QApplication(_argc, NULL)
    {
    }

    QtAppGui::~QtAppGui()
    {
    }

    void QtAppGui::_execLoop() 
    { 
	qtLoop();
    }

    void QtAppGui::_processEvents() 
    { 
	QApplication::processEvents();
    }

    int qtLoop()
    {
	QCoreApplication *app = QCoreApplication::instance();

	if (app && app->thread()==QThread::currentThread())
	{
    #if defined(Q_OS_WIN)
	    QTimer timer;
	    bool lastWindowClosed;

	    do
	    {
		timer.start(100);
		QCoreApplication::processEvents();
		timer.stop();
		
		lastWindowClosed = true;
		foreach (QWidget *widget, QApplication::topLevelWidgets()) 
		{
		    if (widget->isVisible())
			lastWindowClosed = false;
		}
	    }
	    while (!_kbhit() && !lastWindowClosed);

	    QObject::disconnect(&timer, SIGNAL(timeout()), app, SLOT(quit()));
    #else
	    QSocketNotifier notifier(0, QSocketNotifier::Read, 0);
	    QObject::connect(&notifier, SIGNAL(activated(int)), app, SLOT(quit()));
	    QCoreApplication::exec();
	    QObject::disconnect(&notifier, SIGNAL(activated(int)), app, SLOT(quit()));
    #endif
	}

	return 0;
    }
    
} // namespace smil


#endif // USE_QT
