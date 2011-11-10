 
#ifndef _D_GUI_H
#define _D_GUI_H

/**
 * \defgroup Gui Gui
 */
/*@{*/

#include "DImage.h"
#include "DImageViewer.h"
#include "Qt/QtApp.h"


template <> 
void Image<UINT8>::show(const char* name);

template <> 
void Image<UINT16>::show(const char* name);

/*@}*/

#endif // _D_GUI_H
