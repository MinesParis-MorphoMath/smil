 
#include "DGui.h"
#include "DImageViewer.h"

template <> 
void Image<UINT8>::show(const char* name)
{
    if (!viewer)
	viewer = new imageViewer();
    if (name)
      setName(name);
    updateViewerData();
    viewer->show();
}

template <> 
void Image<UINT16>::show(const char* name)
{
    if (!viewer)
	viewer = new imageViewer();
    if (name)
      setName(name);
    updateViewerData();
    viewer->show();
}
