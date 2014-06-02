#include <DSender.h>

using namespace smil;

int main (int argc, char* argv[]) {
    if (argc != 4) {
        cerr << "usage : mpiexec <bin> <path> <min_nbr_blocs> <address>" << endl;
        return -1;
    } 

    MPI_Init (&argc, &argv);

    Image<UINT8> im = Image<UINT8> (argv[1]);
    sender<UINT8> ps(true);
    ps.connect (argv[3]);
    ps.run (im, atoi(argv[2]), 1);
    ps.disconnect ();

    MPI_Finalize ();

    return 0;
}
