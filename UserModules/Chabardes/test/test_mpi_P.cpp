#include <DProcessing.h>
#include <DSharedImage.hpp>

using namespace smil;


int main (int argc, char* argv[]) {

    if (argc != 1) {
        cerr << "usage : mpiexec <bin>" << endl;
        return -1;
    }

    MPI_Init (&argc, &argv);

    list <chunkFunctor<UINT8> > fl;
    fl.push_front (chunkGradient<UINT8>());
    cpu<UINT8> pc(true);
    pc.open_ports ();
    pc.accept_connection ();
    pc.run(fl);
    pc.disconnect();

    MPI_Finalize ();

    return 0;
} 
