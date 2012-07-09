
#ifdef USE_QT

#include "DTimer.h"
#include <iostream>
#include <QApplication>
#include <QWidget>

// #include <pthread.h>

// #include <X11/X.h>
// #include <X11/Xlib.h>

using namespace std;


void * fun (void * _timer) {
      timer *t = (timer*)_timer;
      while(t->running)
      {
	  usleep(10000);
// 	sleep(1);
// 	  cout << "ok" << endl;
// 	cout << QApplication::allWidgets().count() << endl;
     foreach (QWidget *widget, QApplication::allWidgets())
     {
// 	  widget->repaint();
         widget->update();
     }
	  t->app->processEvents();
      }
}


void timer::start()
{
// //   XInitThreads();
//     running = true;
//     pthread_t thread;
//     pthread_attr_t thread_attr;
//     pthread_attr_init(&thread_attr);
//     if (pthread_attr_setdetachstate (&thread_attr, PTHREAD_CREATE_DETACHED) != 0)
//       cout << "err" << endl;
//     pthread_create (&thread, &thread_attr, &fun, this);
//     end();
}
void timer::stop()
{
    running = false;
}

#endif // USE_QT
