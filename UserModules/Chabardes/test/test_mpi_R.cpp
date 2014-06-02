#include <DRecver.h>

using namespace smil;

int main (int argc, char* argv[]) {
    if (argc != 4) {
        cerr << "usage : mpiexec <bin> <path> <ip_address> <ip_port>" << endl;
        return -1;
    } 

    MPI_Init (&argc, &argv);

    Image<UINT8> im;
    recver<UINT8> pr(true);
    pr.connect (argv[2], argv[3]);
    pr.run (im);
    pr.disconnect ();
    write (im, argv[1]) ;

    MPI_Finalize ();

    return 0;
}
