 
#include "DGui.h"
#include "DImageViewer.h"

template <> 
void Image<UINT8>::show(const char* name)
{
    if (name)
      setName(name);
    if (!viewer)
	viewer = new imageViewer();
    updateViewerData();
    viewer->show();
//     qapp->exec();
}
