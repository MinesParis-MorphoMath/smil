#ifndef _DDISTRIBUTED_H_
#define _DDISTRIBUTED_H_

#ifdef USE_OPEN_MP
#include <omp.h>
#endif

enum TAGS {MPITYPEREGISTRATION_TAG, CHUNK_TAG, EOT_TAG};

#include "mpi.h"
#include "distributed/DImage.h"
#include "distributed/DMorpho.h"
#include "distributed/DChunk.h"
#include "distributed/DGlobalHeader.h"

#endif
