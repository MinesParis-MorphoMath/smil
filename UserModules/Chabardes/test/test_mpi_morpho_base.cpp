#include "DChabardes.h"
#include "DCpuID.h"

using namespace smil;

int main(int argc, char* argv[]){
    int err_code, nbr_procs, rank, nbr_chunks, nbr_intersects;
    CpuID cpu;
    Cache_Descriptors L = cpu.getCaches()[0];

    err_code = MPI_Init (&argc, &argv);
    if (err_code != MPI_SUCCESS) {
        return err_code;
    }

    MPI_Comm_size (MPI_COMM_WORLD, &nbr_procs);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);

    Chunk<UINT8>* ca; // chunks array.
    Chunk<UINT8>* ia; // array of intersections between chunks.
    Image<UINT8> im;
    int ncpd[3]; // nbr of chunks per dimensions.

//    Pre-processing the data.
    if (rank == 0) {
        im = Image<UINT8> (10,10,10);
        fill (im, UINT8(0));
        getNumberOfChunksIntersects (im, L.line_size, L.size/10, nbr_procs, nbr_chunks, nbr_intersects, ncpd);
        createChunks (im, nbr_chunks, ncpd, ca);
        createIntersects (im, ncpd, nbr_intersects, 4, ia);
    }
//    MPI_Scatter (ca, nbr_chunks, SMIL_CHUNK, c, 1, SMIL_CHUNK, 0, MPI_COMM_WORLD);
//    MPI_Scatter (ia, nbr_intersects, SMIL_CHUNK, i, 1, SMIL_CHUNK, 0, MPI_COMM_WORLD);

//    Process CHUNKS!
//    Process INTERSECTS!

//    MPI_Gather (ca, nbr_procs, SMIL_CHUNK, c, 1, SMIL_CHUNK);
//    MPI_Gather (ia, nbr_procs, SMIL_CHUNK, i, 1, SMIL_CHUNK, 0, MPI_COMM_WORLD);

//    Post-processing the data.
    if (rank == 0) {
        storeChunks (ca, im, nbr_procs); 
//        write (im, );
    }

    err_code = MPI_Finalize ();
    return err_code;
}
