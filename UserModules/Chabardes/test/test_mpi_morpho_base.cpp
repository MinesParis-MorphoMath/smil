#include "DChabardes.h"
#include "DCpuID.h"

using namespace smil;

int master_proc (int argc, char* argv[], int nbr_procs, int rank) {
    int err_code;
    MPI_Datatype mpi_chunk_type, mpi_border_type, mpi_intersect_type;

    CpuID cpu;
    Cache_Descriptors L1 = cpu.getCaches().front();
    Cache_Descriptors LL = cpu.getCaches().back();
    Image<UINT8> im;
    Cross3DSE se;
    Chunks_Data<UINT8> cd;
    Chunks_Header ch; 

    im = Image<UINT8> (300,300,300);
    fill (im, UINT8(0));
    generateChunks (im, L1.size, LL.size, nbr_procs, se, 4, ch, cd);

/*        
    cout << "CHUNKS" << endl;
    for (int i=0; i<cd.nbr_chunks; ++i) {
        printChunk (cd.ca[i]);
    }
    cout << "BORDERS" << endl;
    for (int i=0; i<cd.nbr_borders; ++i) {
        printChunk (cd.ba[i]);
    }
    cout << "INTERSECTS" << endl;
    for (int i=0; i<cd.nbr_intersects; ++i) {
        printChunk (cd.ia[i]);
    }
*/

    registerChunks (ch, mpi_chunk_type, mpi_border_type, mpi_intersect_type);

    // Processing.

//    write (im, );
    deleteChunks (ch, cd);

    unregisterChunks (ch, mpi_chunk_type, mpi_border_type, mpi_intersect_type);


}

int slave_proc (int nbr_procs, int rank) {
    int err_code;
    MPI_Datatype mpi_chunk_type, mpi_border_type, mpi_intersect_type;

    Chunks_Header ch; 

    registerChunks (ch, mpi_chunk_type, mpi_border_type, mpi_intersect_type);

    // Processing.

    unregisterChunks (ch, mpi_chunk_type, mpi_border_type, mpi_intersect_type);
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

    err_code = MPI_Finalize ();
    return err_code;
}
