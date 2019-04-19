// ImageLabel and ImageMarker should be unsigned integers
template <class ImageLabel, class ImageVal, class ImageMarker, class SE,
          class ImageOut>
RES_T t_GeoCuts_MinSurfaces_with_steps_vGradient(
    const ImageLabel &imLabel, const ImageVal &imVal,
    const ImageMarker &imMarker, const SE &nl, F_SIMPLE step_x, F_SIMPLE step_y,
    F_SIMPLE step_z, ImageOut &imOut)
{
  MORPHEE_ENTER_FUNCTION(
      "t_GeoCuts_MinSurfaces_with_steps_vGradient (multi-valued version)");

  // std::cout << "Enter function t_GeoCuts_MinSurfaces_with_steps
  // (multi_valued version)" << std::endl;

  if (!imOut.isAllocated() || !imLabel.isAllocated() || !imVal.isAllocated() ||
      !imMarker.isAllocated()) {
    MORPHEE_REGISTER_ERROR("Image not allocated");
    return RES_NOT_ALLOCATED;
  }

  // common image iterator
  typename ImageLabel::const_iterator it, iend;
  morphee::selement::Neighborhood<SE, ImageLabel> neighb(imLabel, nl);
  typename morphee::selement::Neighborhood<SE, ImageLabel>::iterator nit, nend;
  offset_t o0;
  offset_t o1;

  // needed for max flow: capacit map, rev_capacity map, etc.
  typedef boost::adjacency_list_traits<boost::vecS, boost::vecS,
                                       boost::directedS>
      Traits;
  typedef boost::adjacency_list<
      boost::vecS, boost::vecS, boost::directedS,
      boost::property<boost::vertex_name_t, std::string>,
      boost::property<
          boost::edge_capacity_t, F_DOUBLE,
          boost::property<
              boost::edge_residual_capacity_t, F_DOUBLE,
              boost::property<boost::edge_reverse_t, Traits::edge_descriptor>>>>
      Graph_d;

  // if we had computed the number of vertices before, we could directly
  // initialize g with it
  Graph_d g;

  boost::property_map<Graph_d, boost::edge_capacity_t>::type capacity =
      boost::get(boost::edge_capacity, g);
  boost::property_map<Graph_d, boost::edge_reverse_t>::type rev =
      get(boost::edge_reverse, g);
  boost::property_map<Graph_d, boost::edge_residual_capacity_t>::type
      residual_capacity = get(boost::edge_residual_capacity, g);

  bool in1;
  Graph_d::edge_descriptor e1, e2, e3, e4, e5;
  Graph_d::vertex_descriptor vSource, vSink;
  int numVert  = 0;
  int numEdges = 0;

  // Vertices creation
  // std::cout<<"build Region Adjacency Graph Vertices"<<std::endl;
  clock_t t1 = clock();
  int max, not_used;
  morphee::stats::t_measMinMax(imLabel, not_used, max);
  numVert = max;
  // std::cout<<"number of Vertices (without source and sink):
  // "<<numVert<<std::endl;
  // Warning : numVert+1 nodes created, but node 0 is not used (in order to
  // simplify correspondance between labels and nodes)
  for (int i = 0; i <= numVert; i++) {
    boost::add_vertex(g);
  }
  vSource = boost::add_vertex(g);
  vSink   = boost::add_vertex(g);

  clock_t tt_marker2 = 0, tt_marker3 = 0, tt_new_edge = 0, tt_old_edge = 0;
  clock_t t2 = clock();
  // std::cout << "Nodes creation time : " << F_DOUBLE(t2-t1) /
  // CLOCKS_PER_SEC << " seconds\n" ;

  // Edges creation
  // std::cout<<"Building Region Adjacency Graph Edges"<<std::endl;
  t1 = clock();
  for (it = imLabel.begin(), iend = imLabel.end(); it != iend; ++it) {
    o1                                           = it.getOffset();
    typename ImageLabel::value_type label        = imLabel.pixelFromOffset(o1),
                                    label_prec   = 0;
    typename ImageMarker::value_type marker      = imMarker.pixelFromOffset(o1),
                                     marker_prec = 0;

    if (label > 0) {
      if (marker == 2 && marker_prec != marker &&
          label_prec != label) // add edge to Source
      {
        clock_t temps_marker2 = clock();
        boost::tie(e4, in1)   = boost::edge(vSource, label, g);
        if (in1 == 0) // if in1 == 0 : edge had not yet been added
        {
          boost::tie(e4, in1) = boost::add_edge(vSource, label, g);
          boost::tie(e3, in1) = boost::add_edge(label, vSource, g);
          capacity[e4]        = (std::numeric_limits<F_DOUBLE>::max)();
          capacity[e3]        = (std::numeric_limits<F_DOUBLE>::max)();
          rev[e4]             = e3;
          rev[e3]             = e4;
        }
        tt_marker2 += clock() - temps_marker2;
      } else if (marker == 3 && marker_prec != marker &&
                 label_prec != label) // add edge to Sink
      {
        clock_t temps_marker3 = clock();
        boost::tie(e3, in1)   = boost::edge(vSink, label, g);
        if (in1 == 0) // if in1 == 0 : edge had not yet been added
        {
          // std::cout<<"Add new edge marker 3"<<std::endl;
          boost::tie(e4, in1) = boost::add_edge(label, vSink, g);
          boost::tie(e3, in1) = boost::add_edge(vSink, label, g);
          capacity[e4]        = (std::numeric_limits<F_DOUBLE>::max)();
          capacity[e3]        = (std::numeric_limits<F_DOUBLE>::max)();
          rev[e4]             = e3;
          rev[e3]             = e4;
        }
        tt_marker3 += clock() - temps_marker3;
      }

      neighb.setCenter(o1);
      typename ImageVal::value_type val_o1 = imVal.pixelFromOffset(o1);
      typename ImageLabel::value_type label2_prec =
          label; // val de label2 precedente

      for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
        const offset_t o2 = nit.getOffset();

        if (o2 > o1) {
          typename ImageLabel::value_type label2 = imLabel.pixelFromOffset(o2);
          if (label != label2) {
            typename ImageVal::value_type val_o2 = imVal.pixelFromOffset(o2);
            F_SIMPLE diff                        = std::max(val_o1, val_o2);

            INT8 dx, dy, dz;
            dx = std::abs(it.getX() - nit.getX());
            dy = std::abs(it.getY() - nit.getY());
            dz = std::abs(it.getZ() - nit.getZ());

            // 		      F_SIMPLE dist ;
            // 		      dist = std::sqrt(std::pow(step_x*dx,2) +
            // std::pow(step_y*dy,2) + std::pow(step_z*dz,2)); if(dist == 0)
            // 			{
            // 			  std::cout << "ERROR : Distance between pixels equal to
            // zero! Setting it to 1.\n" ; 			  dist = 1 ;
            // 			}

            F_SIMPLE surf = 1; // TODO : this only works with 4-connexity
                               // (in 2d) or 6-connexity (in 3d)
            if (dx == 0)
              surf *= step_x;
            if (dy == 0)
              surf *= step_y;
            if (dz == 0)
              surf *= step_z;
            if (surf == 0) {
              // std::cout << "ERROR : Surface between pixels equal to zero!
              // Setting it to 1.\n" ;
              surf = 1;
            }

            // 		      if (o2%1000 == 0)
            // 			{
            // 			  std::cout << " surf : " << surf ;
            // 			}

            // 		      F_SIMPLE grad = diff/dist ;
            // 		      F_SIMPLE weighted_surf = grad * surf ;

            // Cette fonction devrait être remplacée par une fonction
            // paramètre
            // F_DOUBLE cost = 10000.0/(1.0+std::pow(diff/dist,4));
            // F_DOUBLE cost = dist/(1+diff);
            // F_DOUBLE cost = leak * surf ;
            F_DOUBLE cost = surf / (1 + diff);
            // 		      if (o2%1000 == 0)
            // 			{
            // 			  std::cout << " cost : " << cost << "\n";
            // 			}

            // std::cout <<  "dx: " << (double)dx << " dy: " <<  (double)dy
            // << " dz: " <<  (double)dz << " dist: " <<  (double)dist << "
            // surf: " <<  (double)surf << " grad: " <<  (double)grad << "
            // w_s: " <<  (double)weighted_surf << " cost: " << (double)cost
            // << "\n";

            if (label2_prec == label2) // same label2 means same edge (thus,
                                       // keep e3 and e4)
            {
              capacity[e4] = capacity[e4] + cost;
              capacity[e3] = capacity[e3] + cost;
            } else {
              boost::tie(e5, in1) = boost::edge(label, label2, g);
              if (in1 == 0) {
                clock_t temps_new_edge = clock();
                // std::cout<<"Add new edge "<< label<<" --
                // "<<label2<<std::endl;
                numEdges++;
                boost::tie(e4, in1) = boost::add_edge(label, label2, g);
                boost::tie(e3, in1) = boost::add_edge(label2, label, g);
                capacity[e4]        = cost;
                capacity[e3]        = cost;
                rev[e4]             = e3;
                rev[e3]             = e4;
                tt_new_edge += clock() - temps_new_edge;

              } else {
                clock_t temps_old_edge = clock();
                // std::cout<<"existing edge"<<std::endl;
                boost::tie(e4, in1) = boost::edge(label, label2, g);
                boost::tie(e3, in1) = boost::edge(label2, label, g);
                capacity[e4]        = capacity[e4] + cost;
                capacity[e3]        = capacity[e3] + cost;
                tt_old_edge += clock() - temps_old_edge;
              }
              label2_prec = label2;
            }
          }
        }
      }
      label_prec  = label;
      marker_prec = marker;
    }
  }

  t2 = clock();
  // std::cout << "Number of initial edges : " << numEdges << std::endl;
  //     std::cout << "Edges creation time : " << F_DOUBLE(t2-t1) /
  //     CLOCKS_PER_SEC << " seconds\n" ; std::cout << "Marker2   : " <<
  //     F_DOUBLE(tt_marker2)  / CLOCKS_PER_SEC << " seconds\n"; std::cout
  //     << "Marker3   : " << F_DOUBLE(tt_marker3)  / CLOCKS_PER_SEC << "
  //     seconds\n"; std::cout << "New edges : " << F_DOUBLE(tt_new_edge) /
  //     CLOCKS_PER_SEC << " seconds\n"; std::cout << "Old edges : " <<
  //     F_DOUBLE(tt_old_edge) / CLOCKS_PER_SEC << " seconds\n";

  // We should test that the same region node is not connected
  // simultaneously to source and sink :
  // * iterate on sink neighbors ;
  // * if neigbhbor is linked to source, remove edges to source and sink (in
  // fact, set its capacity to 0, to avoid the modification of the graph
  // structure)
  //
  t1 = clock();
  Graph_d::vertex_descriptor sink_neighbour;
  typename boost::graph_traits<Graph_d>::adjacency_iterator ai, ai_end;
  UINT32 rem_edges = 0;
  for (boost::tie(ai, ai_end) = adjacent_vertices(vSink, g); ai != ai_end;
       ++ai) {
    sink_neighbour = *ai;
    tie(e1, in1)   = edge(sink_neighbour, vSource, g);
    if (in1) {
      // remove_edge(vSource, sink_neighbour, g);
      // remove_edge(sink_neighbour, vSource, g);
      capacity[e1]      = 0;
      capacity[rev[e1]] = 0;
      rem_edges++;
      tie(e2, in1)      = edge(vSink, sink_neighbour, g);
      capacity[e2]      = 0;
      capacity[rev[e2]] = 0;
      rem_edges++;
    }
  }
  t2 = clock();
  // std::cout << "Graph post-processing : Removal of " << rem_edges << "
  // edges in  : " << F_DOUBLE(t2-t1) / CLOCKS_PER_SEC << " seconds\n" ;

  // Prepare to run the max-flow algorithm
  boost::property_map<Graph_d, boost::vertex_index_t>::type indexmap =
      boost::get(boost::vertex_index, g);
  std::vector<boost::default_color_type> color(boost::num_vertices(g));
  // std::cout << "Compute Max flow" << std::endl;
  t1 = clock();
#if BOOST_VERSION >= 104700
  F_DOUBLE flow = boykov_kolmogorov_max_flow(
      g, capacity, residual_capacity, rev, &color[0], indexmap, vSource, vSink);
#else
  F_DOUBLE flow = kolmogorov_max_flow(g, capacity, residual_capacity, rev,
                                      &color[0], indexmap, vSource, vSink);
#endif
  // std::cout << "c  The total flow:" << std::endl;
  // std::cout << "s " << flow << std::endl;
  t2 = clock();
  // std::cout << "Flow computation time : " << F_DOUBLE(t2-t1) /
  // CLOCKS_PER_SEC << " seconds\n" ;

  t1 = clock();
  for (it = imLabel.begin(), iend = imLabel.end(); it != iend; ++it) {
    o1                                    = it.getOffset();
    typename ImageLabel::value_type label = imLabel.pixelFromOffset(o1);

    if (label == 0) {
      imOut.setPixel(o1, 0);
    } else // if color[label] == 0 : source node ; else, sink node (accord
           // to boost graph doc)
    {
      if (color[label] == color[vSource])
        imOut.setPixel(o1, 2);
      else
        imOut.setPixel(o1, 3);
    }
  }
  t2 = clock();
  // std::cout << "Computing imOut took : " << F_DOUBLE(t2-t1) /
  // CLOCKS_PER_SEC << " seconds\n" ;

  // std::cout << "\n";

  return RES_OK;
}
