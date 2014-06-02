#include <DSender.h>

using namespace smil;

int main (int argc, char* argv[]) {
    if (argc != 5) {
        cerr << "usage : mpiexec <bin> <path> <min_nbr_blocs> <ip_address> <ip_port>" << endl;
        return -1;
    } 

    MPI_Init (&argc, &argv);

    Image<UINT8> im = Image<UINT8> (argv[1]);
    sender<UINT8> ps(true);
    ps.connect (argv[3], argv[4]);
    ps.run (im, atoi(argv[2]), 1);
    ps.disconnect ();

    MPI_Finalize ();

    return 0;
}
