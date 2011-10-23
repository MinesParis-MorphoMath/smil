#ifndef _D_IMAGE_VIEWER_H
#define _D_IMAGE_VIEWER_H

#include "DImage.h"
#include "Qt/ImageViewerWidget.h"

class imageViewer : public baseImageViewer
{
public:
    imageViewer();
    ~imageViewer();
    virtual void show();
    virtual bool isVisible();
    virtual void setName(const char* name);
    virtual void loadFromData(void *pixels, UINT w, UINT h);
private:
    ImageViewerWidget *qtViewer;
};

#endif // _D_IMAGE_VIEWER_H
