#include "DChabardes.h"
#include "DCpuID.h"
#include <unistd.h>

using namespace smil;

RES_T processChunk (UINT8 value, Chunk<UINT8> &c) {
    int size = c.getSize(0)*c.getSize(1)*c.getSize(2);
    UINT8* data = c.getData () ;
    for (int i=0; i<size; ++i) {
        data[i] = value; 
    }
    return RES_OK;
} 

int master_proc (int argc, char* argv[], const int nbr_procs, const int rank) {
    int err_code;
    int nbr_chunks_processed = 0;

    CpuID cpu;
    Cache_Descriptors L1;// = cpu.getCaches().front();
    L1.size = 6;
    Cache_Descriptors LL;// = cpu.getCaches().back();
    LL.size = 541;
    Image<UINT8> im;
    Cross3DSE se;
    Chunk<UINT8>* ca;
    Chunks_Header ch; 

    im = Image<UINT8> (30,30,30);
    fill (im, UINT8(0));
    generateChunks (im, L1.size, LL.size, nbr_procs, se, 3, ch, ca);
//    registerChunks (ch) ;

    // Processing.
    // Scatter
    /*
    MPI_Status status;
    Chunk<UINT8> c;
    int max_dimension = MAX (ch.size_borders, ch.size_intersects);
    c.data = new UINT8[max_dimension];
    while (nbr_chunks_processed < ch.nbr_chunks+ch.nbr_borders+ch.nbr_intersects) {
        recvChunk (ch, MPI_ANY_SOURCE, MPI_ANY_TAG, c, &status) ;
        //cout << "[0] receive chunk from " << status.MPI_SOURCE << endl;
        storeChunk (ch, c, im);
        if (nbr_chunks_processed < ch.nbr_chunks+ch.nbr_borders+ch.nbr_intersects) {
            sendChunk (ch, ca[nbr_chunks_processed++], status.MPI_SOURCE, status.MPI_TAG) ;
            //cout << "[0] send chunk to " << status.MPI_SOURCE << endl;
        }
    }
    */
//    im.save ( "/tmp/test.png" );
//    unregisterChunks (ch) ;
    delete ca;
}

int slave_proc (int nbr_procs, int rank) {
/*    int err_code;

    Chunks_Header ch; 
    registerChunks (ch) ;

    // Processing.
    Chunk<UINT8> c;
    int max_dimension = MAX (ch.size_borders, ch.size_intersects);
    c.allocate (max_dimension) ; 
    while (1) {
        recvChunk (ch, 0, rank*10, c, MPI_STATUS_IGNORE);
        sleep (1);
        //cout << "[" << rank << "] receive chunk from " << 0 << endl;
        processChunk (rank*10, c);
        sendChunk (ch, c, 0, rank*10);
        //cout << "[" << rank << "] send chunk to " << 0 << endl;
    }
    */
}

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
