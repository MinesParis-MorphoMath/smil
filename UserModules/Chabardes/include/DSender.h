#ifndef _DSENDER_H_
#define _DSENDER_H_

#include "DChabardes.h"
#include "DSendBuffer.h"
#include "DSendStream.h"

namespace smil {
    template <class T>
    RES_T initializeChunkStyle (const int nbr_procs, const int intersect_width, Image<T> &im, GlobalHeader gh, SendChunkStream<UINT8> ss) {

    }

    template <class T>
    RES_T intializeSliceStyle (const int nbr_procs, Image<T> &im, GlobalHeader gh, SendSliceStream<UINT8> ss) {

    }

    void broadcastMPITypeRegistration (GlobalHeader gh, int comm) {

    }

    void broadCastEndOfTransmission (int comm) {

    }
}
#endif
