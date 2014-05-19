#ifndef _DCHABARDES_H_
#define _DCHABARDES_H_

#ifdef USE_OPEN_MP
#include <omp.h>
#endif

enum TAGS {PTOR_MPITYPEREGISTRATION_TAG, CHUNK_TAG, EOT_TAG};

#include "mpi.h"
#include "DImage.h"
#include "DMorpho.h"
#include "DChunk.h"
#include "DGlobalHeader.h"

#endif
