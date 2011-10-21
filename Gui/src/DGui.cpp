 
#include "DGui.h"
#include "DImageViewer.h"

template <> 
void Image<UINT8>::show(const char* name)
{
    
    cout << "ok there" << endl;
    if (name)
      setName(name);
    updateViewerData();
    if (!viewer)
	viewer = new imageViewer();
    viewer->show();
//     qapp->exec();
}
