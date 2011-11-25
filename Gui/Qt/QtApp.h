/*
 * Smil
 * Copyright (c) 2010 Matthieu Faessel
 *
 * This file is part of Smil.
 *
 * Smil is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * Smil is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Smil.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 */


#ifndef QT_APP_H
#define QT_APP_H

#include <QApplication>
#include <QThread>

class Thread : public QThread
{
public:
Thread(QApplication *a): qa(a) {}
void run() 
{ 
    while(true)
    {
      usleep(100000); 
      qa->processEvents(); /*qApp->exec();*/ 
    }
}
private:
    QApplication *qa;
};

class QtApp
{
public:
  QtApp()
  {
      if (!qApp)
      {
	cout << "created" << endl;
	  int ac = 1;
	  char **av = NULL;
	  qapp = new QApplication(ac, av);
      }
      else qapp = qApp;
      th = new Thread(qapp);
//       th->start();
  }
  void start() { th->start(); }
  void _exe() { qApp->exec(); }
  QApplication *qapp;
  Thread *th;
};

#endif // QT_APP_H
