#ifndef MOSAIC_GEOCUTSALGO_IMPL_T_HPP
#define MOSAIC_GEOCUTSALGO_IMPL_T_HPP

#include <time.h>

#include <boost/config.hpp>
// for boost::tie
#include <boost/utility.hpp>
// for boost::graph_traits
#include <boost/graph/graph_traits.hpp> 
#include <boost/graph/adjacency_list.hpp>
//#include <boost/graph/graphviz.hpp>

#include <boost/version.hpp>
#if BOOST_VERSION >= 104700
#include <boost/graph/boykov_kolmogorov_max_flow.hpp>
#elif BOOST_VERSION >= 103500
#include <boost/graph/kolmogorov_max_flow.hpp>
#endif


//#include <morphee/selement/include/selementNeighborList.hpp>
//#include <morphee/selement/include/private/selementNeighborhood_T.hpp>
//#include <morphee/stats/include/private/statsMeasure_T.hpp>
//#include <morphee/image/include/private/imagePixelwise_T.hpp>

#include <vector>

#define MORPHEE_ENTER_FUNCTION(a) 
#define MORPHEE_REGISTER_ERROR(a) 

typedef off_t offset_t;

namespace smil
{
  using namespace boost;
#if 0

