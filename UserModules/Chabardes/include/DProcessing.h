#ifndef _DPROCESSING_H_
#define _DPROCESSING_H_

#include "DChabardes.h"

namespace smil {

    void broadcastMPITypeRegistration (GlobalHeader &gh, const int inter_StoP, const int inter_PtoR) {

    }

    void broadcastEndOfTransmission (const int comm) {

    }

    template <class T>
    RES_T send (const Chunk<T> &c, const GlobalHeader &gh, const int comm) {

    }

    template <class T>
    RES_T recv (const Chunk<T> &c, const GlobalHeader &gh, const int comm) {

    }

}

#endif
