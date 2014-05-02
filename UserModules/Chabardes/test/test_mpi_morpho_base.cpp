#include "DChabardes.h"
#include "DCpuID.h"
#include <unistd.h>

using namespace smil;

int master_proc (int argc, char* argv[], const int nbr_procs, const int rank) {
    Image<UINT8> imI, imO;
    Arrays_Stream<UINT8> as;
//    Chunks_Array ca; //TODO: Rename it as Chunk Buffer ?

    imI = Image<UINT8> (30,30,30);
    imO = Image<UINT8> (imI);
    fill (imI, UINT8(0));
    as.initialize (nbr_procs, 1, imI, imO) ;
//    ca.initialize (is);
//    ca.nextRound (is); 
    Chunk<UINT8> c;
    void* ptr = ::operator new (30*30*30) ;
    c.setMemorySpace (ptr, 10000) ;
    while (!as.eof()) {
         as.next (c) ;
         c.print () ; 
    }
/*  is.mpiRegisterMaster ();

    // Processing
    ca.scatter (is);
    ca.nextRound (is);
    while (!is.eof ()) {
        is.recv();
        ca.send (is) ;
        ca.next(is);
    }
    // End Processing

    is.mpiFree ();
*/
}

int slave_proc (int nbr_procs, int rank) {
/*    IOStream is;
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
