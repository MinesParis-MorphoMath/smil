#include "DChabardes.h"
#include "DCpuID.h"

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
    cb.scatter ();
    cb.nextRound (as);
    while (!as.eof ()) {
        as.recv();
        cb.send ();
        cb.next(as);
    }
    // End Processing

    as.mpiFree ();
}

int slave_proc (int nbr_procs, int rank) {
    IOStream<UINT8> is;
    is.mpiRegisterSlave ();
    
    int max_dimension = is.get_chunks_len (0);
    for (int i=1; i<is.get_nbr_diff_chunks(); ++i)
         max_dimension = MAX (max_dimension, is.get_chunks_len(i));
    int memory_step = 6*sizeof(int)+ max_dimension * sizeof(UINT8);
    void* ptr = ::operator new (memory_step);
    Chunk<UINT8> *c;
    c->setMemorySpace (ptr, memory_step) ;


    cout << is.eof () << endl;

    // Processing
    while (!is.eof ()) {
        c->recv (0);
        //TODO: Do stuff with c.
        c->send (0);
    }
    // End Processing

    is.mpiFree ();
}

int main(int argc, char* argv[]){
    int err_code, nbr_masters, nbr_slaves=8, nbr_procs, rank;
    int inter_comm, intra_comm, master_rank=1;
    bool flag = false;

    err_code = MPI_Init (&argc, &argv);
    if (err_code != MPI_SUCCESS) {
        return err_code;
    }

    MPI_Comm_size (MPI_COMM_WORLD, &nbr_master);
    // Activate the slave processes
    MPI_Comm_spawn ("test_mpi_slave", MPI_ARGV_NULL, nb_slaves, MPI_INFO_NULL, &master_rank, MPI_COMM_WORLD, inter_comm, MPI_ERRCODES_IGNORE);
    // Merge communicators associated to inter_comm. In intra_comm, process ranks are sorted according to the value of the flag argument
    MPI_Intercomm_merge (inter_comm, flag, &intra_comm) ;
    MPI_Comm_size (intra_comm, &nb_procs)
    MPI_Comm_rank (intra_comm, &rank);

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
