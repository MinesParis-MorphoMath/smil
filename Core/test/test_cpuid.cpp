#include "DCore.h"

using namespace smil;

int main () {

    CpuID id; 

    cout << "Vendor: " << id.getVendor () << endl;
    cout << "Cores: " << id.getCores () << endl;
    cout << "Logical: " << id.getLogical () << endl;
    cout << "HyperThreading: " << id.isHyperThreated () << endl;

    SIMD_Instructions si = id.getSimdInstructions () ;

    cout << "MMX: " << si.MMX << endl;
    cout << "SSE: " << si.SSE << endl;
    cout << "SSE2: " << si.SSE2 << endl;
    cout << "SSE2: " << si.SSE3 << endl;
    cout << "SSSE3: " << si.SSSE3 << endl;
    cout << "SSE41: " << si.SSE41 << endl;
    cout << "SSE42: " << si.SSE42 << endl;
    cout << "AES: " << si.AES << endl;
    cout << "AVX: " << si.AVX << endl;

    vector<Cache_Descriptors> L = id.getCaches () ;

    for (int i=0; i<L.size(); ++i) {
        cout << "cache at level: " << i << endl;
        cout << "type: " << L[i].type << endl;
        cout << "size: " << L[i].size << endl;
        cout << "set: " << L[i].sets << endl;
        cout << "associativity: " << L[i].associativity << endl;
        cout << "lines_per_tag: " << L[i].lines_per_tag << endl;
        cout << "line_size: " << L[i].line_size << endl;
    }

    return 0;
}
