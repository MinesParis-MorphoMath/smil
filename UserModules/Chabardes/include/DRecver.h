#ifndef _DRECV_H_
#define _DRECV_H_

#include "DChabardes.h"
#include "DRecvBuffer.h"
#include "DRecvStream.h"

namespace smil {

    void broadcastMPITypeRegistration (GlobalHeader gh, int comm) {

    }
    
    template <class T>
    bool isEndOfTransmission (const RecvBuffer<T> &rb) {

    }

}

#endif 
