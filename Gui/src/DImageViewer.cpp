 
#include "DImageViewer.h"

imageViewer::imageViewer()
{
    qtViewer = new ImageViewerWidget();
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
