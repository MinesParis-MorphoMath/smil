/* __HEAD__
 * Copyright (c) 2011-2016, Matthieu FAESSEL and ARMINES
 * Copyright (c) 2017-2019, Centre de Morphologie Mathematique
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of Matthieu FAESSEL, or ARMINES nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS'' 
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS AND CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF 
 * THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Description :
 *   Porting GeoCuts module from Morph-M - This is the MinSurfaces part
 *
 * History :
 *   - 20/03/2019 - by Jose-Marcio Martins da Cruz
 *     Just created it...
 *   - XX/XX/XXXX - 
 * 
 *
 * __HEAD__ - Stop here !
 */


#ifndef _D_GEOCUTS_MINSURFACE_HPP_ 
#define _D_GEOCUTS_MINSURFACE_HPP_

#include <morphee/selement/include/selementNeighborList.hpp>
#include <morphee/selement/include/private/selementNeighborhood_T.hpp>
#include <morphee/image/include/private/image_T.hpp>
#include <morphee/image/include/imageUtils.hpp>

#include <boost/config.hpp>
// for boost::tie
#include <boost/utility.hpp>            
// for boost::graph_traits
#include <boost/graph/graph_traits.hpp> 
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/prim_minimum_spanning_tree.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/johnson_all_pairs_shortest.hpp>

#include <boost/version.hpp>
#if BOOST_VERSION >= 104700
#include <boost/graph/boykov_kolmogorov_max_flow.hpp>
#elif BOOST_VERSION >= 103500
#include <boost/graph/kolmogorov_max_flow.hpp>
#else
#include "../boost_ext/kolmogorov_max_flow.hpp"
#endif

// FROM STAWIASKI JAN 2012
#include "../boost_ext/kolmogorov_max_flow_min_cost.hpp"
//#include "../boost_ext/maximum_spanning_tree.hpp"
//STAWIASKI JAN2012 commented, why?
//#include "../boost_ext/boost_compare.hpp"  
#include <boost/graph/connected_components.hpp>

#include <morphee/image/include/private/image_T.hpp>
#include <morphee/image/include/private/imageManipulation_T.hpp>
#include <morphee/selement/include/selementNeighborList.hpp>
#include <morphee/selement/include/private/selementNeighborhood_T.hpp>
#include <morphee/morphoBase/include/private/morphoLabel_T.hpp>
#include <morphee/morphoBase/include/private/morphoLabel2_T.hpp>
#include <morphee/morphoBase/include/private/morphoGraphs_T.hpp>
#include <morphee/morphoBase/include/private/morphoHierarch_T.hpp>
#include <morphee/graph/include/private/graphProjection_T.hpp>
#include <morphee/graph/include/graph.hpp>

// Required for t_Order_Edges_Weights
#include <graphs/MorphoGraph/include/Morpho_Graph_T.hpp> 

#include <vector>

// ##################################################
// BEGIN FROM STAWIASKI JAN 2012
// ##################################################

#include <math.h>
#define M_PI 3.14159265358979323846
#define INFINI_POSITIF std::numeric_limits<double>::max)()
#define _SECURE_SCL 0
#include <stdio.h>

typedef struct {
  float x;
  float y;
  float p;
} morceau;

typedef std::list<morceau> affine_par_morceaux;

// ##################################################
// END FROM STAWIASKI JAN 2012
// ##################################################
//#include <morphee/common/include/commonTypesOperator.hpp>

namespace morphee
{

} // namespace smil

#endif // _D_GEOCUTS_MINSURFACE_HPP_
