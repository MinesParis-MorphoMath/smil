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
