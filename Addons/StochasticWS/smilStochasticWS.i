%include smilCommon.i

SMIL_MODULE(smilStochasticWS)

%import smilMorpho.i

%{
/* Includes .hppe.hppeader in .hppe wrapper code */
#include "DStochasticWatershed.hpp"
%}

#include <jansson.hpp>

%import smilCore.i

%include"DStochasticWatershed.hpp"
void stochasticWatershed (const Image<UINT8>& primary, const Image<UINT8>& gradient, Image<UINT8> &out, const size_t& n_seeds, const StrElt& se);
void stochasticWatershed (const Image<UINT16>& primary, const Image<UINT16>& gradient, Image<UINT16> &out, const size_t& n_seeds, const StrElt& se);
size_t stochasticFlatZones (const Image<UINT8>& primary, const Image<UINT8>& gradient, Image<UINT8> &out, const size_t& n_seeds, const double& t0, const StrElt& s);
size_t stochasticFlatZones (const Image<UINT16>& primary, const Image<UINT16>& gradient, Image<UINT16> &out, const size_t& n_seeds, const double& t0, const StrElt& s);
size_t overSegmentationCorrection (const Image<UINT8>& primary, const Image<UINT8>& gradient, Image<UINT8> &out, const size_t& n_seeds, const double& r0, const StrElt& s);
size_t overSegmentationCorrection (const Image<UINT16>& primary, const Image<UINT16>& gradient, Image<UINT16> &out, const size_t& n_seeds, const double& r0, const StrElt& s);
void stochasticWatershedParallel (const Image<UINT8>& primary, const Image<UINT8>& gradient, Image<UINT8> &out, const size_t& n_seeds, const StrElt& se);
void stochasticWatershedParallel (const Image<UINT16>& primary, const Image<UINT16>& gradient, Image<UINT16> &out, const size_t& n_seeds, const StrElt& se);
size_t stochasticFlatZonesParallel (const Image<UINT8>& primary, const Image<UINT8>& gradient, Image<UINT8> &out, const size_t& n_seeds, const double& t0, const StrElt& s);
size_t stochasticFlatZonesParallel (const Image<UINT16>& primary, const Image<UINT16>& gradient, Image<UINT16> &out, const size_t& n_seeds, const double& t0, const StrElt& s);

