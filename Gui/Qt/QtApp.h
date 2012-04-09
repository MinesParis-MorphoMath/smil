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


#ifndef QT_APP_H
#define QT_APP_H

#include <QApplication>
#include <QThread>
#include <QTimer>
#include <iostream>

using namespace std;

class Thread : public QThread
{
public:
Thread(QApplication *a): qa(a) {}
void run() 
{ 
      if (!qApp)
      {
	cout << "created" << endl;
	  int ac = 1;
	  char **av = NULL;
	  _qapp = new QApplication(ac, av, true);
      }
//     while(true)
//     {
//       usleep(100000); 
      qApp->processEvents();
//     }
}
private:
  QApplication *_qapp;
    QApplication *qa;
};

class QtApp
{
public:
  QtApp()
    : _qapp(NULL)
  {
      if (!qApp)
      {
	cout << "QtApp: qt app created" << endl;
	  int ac = 1;
	  char **av = NULL;
	  _qapp = new QApplication(ac, av);
      }
//       else _qapp = qApp;
//       th = new Thread(_qapp);
//       th->start();
  }
//   void start() { th->start(); }
  void exec() { if (_qapp) _qapp->exec(); }
  void processEvents() { if (_qapp) _qapp->processEvents(); }
  QApplication *_qapp;
//   Thread *th;
  QTimer *timer;
};


class appTimer : public QObject
{
  Q_OBJECT
public:
  appTimer()
  {
      timer = new QTimer(this);
      QObject::connect( timer, SIGNAL(timeout()), this, SLOT(upd()) );
  }
  ~appTimer()
  {
      delete timer;
  }
  void start()
  {
      timer->start(1000);
  }
private slots:
  void upd()
  {
      qApp->processEvents();
      cout << "ok" << endl;
  }
protected:
  QTimer *timer;
  
};

#endif // QT_APP_H
