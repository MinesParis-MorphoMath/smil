#include "DChabardes.h"
#include "DCpuID.h"
#include <unistd.h>

using namespace smil;

int master_proc (int argc, char* argv[], const int nbr_procs, const int rank) {
    Image<UINT8> imI, imO;
    ArraysStream<UINT8> as;
    ChunkBuffer<UINT8> cb; //TODO: Rename it as Chunk Buffer ?

    imI = Image<UINT8> (30,30,30);
    imO = Image<UINT8> (imI);
    fill (imI, UINT8(0));

    as.initialize (nbr_procs, 1, imI, imO) ;
    cb.initialize (nbr_procs, as);
    cb.nextRound (as); 
    as.mpiRegisterMaster ();

    // Processing
    cb.scatter (as);
    cb.nextRound (as);
    while (!as.eof ()) {
        as.recv();
        cb.send (as);
        cb.next(as);
    }
    // End Processing

    as.mpiFree ();

}

int slave_proc (int nbr_procs, int rank) {
/*    IOStream<UINT8> is;
    is.mpiRegisterSlave ();
   
    Chunk<UINT8> *c;

    // Processing
    while (!is.eof ()) {
        is.recv(c);
        //TODO: Do stuff with c.
        is.send(c);
    }
    // End Processing

    is.mpiFree ();
*/}

int main(int argc, char* argv[]){
    int err_code, nbr_procs, rank;

    err_code = MPI_Init (&argc, &argv);
    if (err_code != MPI_SUCCESS) {
        return err_code;
    }

    MPI_Comm_size (MPI_COMM_WORLD, &nbr_procs);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);

//    Pre-processing the data.
    if (rank == 0) {
        err_code = master_proc (argc, argv, nbr_procs, rank);
    } else {
        err_code = slave_proc (nbr_procs, rank);
    }
    cout << rank << " : exited" << endl;

    err_code = MPI_Finalize ();
    return err_code;
}
