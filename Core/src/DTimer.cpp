
#include "DTimer.h"
#include <iostream>

using namespace std;

bool timerRunning;

void * fun (void * args) {
      while(timerRunning)
      {
	  sleep(1);
	  cout << "ok" << endl;
      }
}


void timer::start()
{
    timerRunning = true;
    pthread_attr_t thread_attr;
    if (pthread_attr_setdetachstate (&thread_attr, PTHREAD_CREATE_DETACHED) != 0)
      cout << "err" << endl;
    pthread_create (&thread, &thread_attr, &fun, NULL);
    end();
}
void timer::stop()
{
    timerRunning = false;
}
