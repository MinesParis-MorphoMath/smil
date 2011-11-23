 
#include "DImageViewer.h"

imageViewer::imageViewer()
{
    if (!qApp)
    {
      cout << "created" << endl;
	int ac = 1;
	char **av = NULL;
	_qapp = new QApplication(ac, av);
    }
    qtViewer = new ImageViewerWidget();
//     qtViewer = new ImageViewer();
}

imageViewer::~imageViewer()
{
    delete qtViewer;
}
    
void imageViewer::show()
{
    qtViewer->show();
}

bool imageViewer::isVisible()
{
    qtViewer->isVisible();
}

void imageViewer::setName(const char* name)
{
    qtViewer->setName(name);
}

void imageViewer::loadFromData(void *pixels, UINT w, UINT h)
{
    qtViewer->loadFromData((UINT8*)pixels, w, h);
}
