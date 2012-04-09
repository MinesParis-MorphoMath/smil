
#include "DTimer.h"
#include <iostream>
#include <QApplication>

using namespace std;


void * fun (void * _timer) {
      timer *t = (timer*)_timer;
      while(t->running)
      {
	  sleep(1);
	  t->app->processEvents();
      }
}


void timer::start()
{
    running = true;
    pthread_attr_t thread_attr;
    pthread_attr_init(&thread_attr);
    if (pthread_attr_setdetachstate (&thread_attr, PTHREAD_CREATE_DETACHED) != 0)
      cout << "err" << endl;
    pthread_create (&thread, &thread_attr, &fun, this);
    end();
}
void timer::stop()
{
    running = false;
}
