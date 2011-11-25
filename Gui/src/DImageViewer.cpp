/*
 * Smil
 * Copyright (c) 2010 Matthieu Faessel
 *
 * This file is part of Smil.
 *
 * Smil is free software: you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * Smil is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with Smil.  If not, see
 * <http://www.gnu.org/licenses/>.
 *
 */



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
