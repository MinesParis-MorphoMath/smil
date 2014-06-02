#include <DRecver.h>

using namespace smil;

int main (int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "usage : mpiexec <bin> <path> <address>" << endl;
        return -1;
    } 

    MPI_Init (&argc, &argv);

    Image<UINT8> im;
    recver<UINT8> pr(true);
    pr.connect (argv[2]);
    pr.run (im);
    pr.disconnect ();
    write (im, argv[1]) ;

    MPI_Finalize ();

    return 0;
}
