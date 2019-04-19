template <class ImageIn, class ImageGrad, class ImageMarker, class SE,
          class ImageOut>
RES_T GeoCuts_MinSurfaces_with_steps_old(const ImageIn &imIn,
                                           const ImageGrad &imGrad,
                                           const ImageMarker &imMarker,
                                           const SE &nl, F_SIMPLE step_x,
                                           F_SIMPLE step_y, F_SIMPLE step_z,
                                           ImageOut &imOut)
{
  MORPHEE_ENTER_FUNCTION("t_GeoCuts_MinSurfaces_with_steps");

  std::cout << "Enter function t_GeoCuts_MinSurfaces" << std::endl;

  if ((!imOut.isAllocated())) {
    MORPHEE_REGISTER_ERROR("Not allocated");
    return RES_NOT_ALLOCATED;
  }

  if ((!imIn.isAllocated())) {
    MORPHEE_REGISTER_ERROR("Not allocated");
    return RES_NOT_ALLOCATED;
  }

  if ((!imGrad.isAllocated())) {
    MORPHEE_REGISTER_ERROR("Not allocated");
    return RES_NOT_ALLOCATED;
  }

  if ((!imMarker.isAllocated())) {
    MORPHEE_REGISTER_ERROR("Not allocated");
    return RES_NOT_ALLOCATED;
  }

  // common image iterator
  typename ImageIn::const_iterator it, iend;
  morphee::selement::Neighborhood<SE, ImageIn> neighb(imIn, nl);
  typename morphee::selement::Neighborhood<SE, ImageIn>::iterator nit, nend;
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
          boost::edge_capacity_t, double,
          boost::property<
              boost::edge_residual_capacity_t, double,
              boost::property<boost::edge_reverse_t, Traits::edge_descriptor>>>>
      Graph_d;

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
  int max, not_used;

  std::cout << "build Region Adjacency Graph" << std::endl;

  std::cout << "build Region Adjacency Graph Vertices" << std::endl;

  clock_t t1 = clock();

  morphee::stats::t_measMinMax(imIn, not_used, max);
  numVert = max;
  std::cout << "number of Vertices : " << numVert << std::endl;

  for (int i = 1; i <= numVert; i++) {
    boost::add_vertex(g);
  }

  vSource = boost::add_vertex(g);
  vSink   = boost::add_vertex(g);

  clock_t tt_marker2 = 0, tt_marker3 = 0, tt_new_edge = 0, tt_old_edge = 0;
  clock_t t2 = clock();
  std::cout << "Nodes creation time : " << double(t2 - t1) / CLOCKS_PER_SEC
            << " seconds\n";

  std::cout << "Building Region Adjacency Graph Edges" << std::endl;
  t1 = clock();

  for (it = imIn.begin(), iend = imIn.end(); it != iend; ++it) {
    o1 = it.getOffset();
    // pas de raison que ce soit int ; il faut rendre ceci générique
    int val      = imIn.pixelFromOffset(o1);
    int marker   = imMarker.pixelFromOffset(o1);
    int val_prec = 0, marker_prec = 0;

    if (val > 0) {
      if (marker == 2 && marker_prec != marker && val_prec != val) {
        clock_t temps_marker2 = clock();
        boost::tie(e4, in1)   = boost::edge(vSource, val, g);

        if (in1 == 0) {
          // std::cout<<"Add new edge marker 2"<<std::endl;
          boost::tie(e4, in1) = boost::add_edge(vSource, val, g);
          boost::tie(e3, in1) = boost::add_edge(val, vSource, g);
          capacity[e4]        = (std::numeric_limits<double>::max)();
          capacity[e3]        = (std::numeric_limits<double>::max)();
          rev[e4]             = e3;
          rev[e3]             = e4;
        }
        tt_marker2 += clock() - temps_marker2;
      } else if (marker == 3 && marker_prec != marker && val_prec != val) {
        clock_t temps_marker3 = clock();
        boost::tie(e3, in1)   = boost::edge(vSink, val, g);
        if (in1 == 0) {
          // std::cout<<"Add new edge marker 3"<<std::endl;
          boost::tie(e4, in1) = boost::add_edge(val, vSink, g);
          boost::tie(e3, in1) = boost::add_edge(vSink, val, g);
          capacity[e4]        = (std::numeric_limits<double>::max)();
          capacity[e3]        = (std::numeric_limits<double>::max)();
          rev[e4]             = e3;
          rev[e3]             = e4;
        }
        tt_marker3 += clock() - temps_marker3;
      }

      neighb.setCenter(o1);
      // Enlever double et int ; prendre types génériques
      double val_grad_o1 = imGrad.pixelFromOffset(o1);
      // typename mageGrad::value_type val_grad_o1 =
      // imGrad.pixelFromOffset(o1);
      int val2_prec = val; // val de val2 precedente

      for (nit = neighb.begin(), nend = neighb.end(); nit != nend; ++nit) {
        const offset_t o2 = nit.getOffset();

        if (o2 > o1) {
          // enlever int ; prendre type générique
          int val2 = imIn.pixelFromOffset(o2);
          if (val != val2) {
            // enlever double ; prendre type générique
            double val_grad_o2 = imGrad.pixelFromOffset(o2);
            // prendre la distance L1, remplacer double et F_SIMPLE par
            // valeur générique
            double diff = t_Distance_L1(val_grad_o1, val_grad_o2);
            // double diff = std::abs(val_grad_o1 - val_grad_o2);

            F_SIMPLE dist =
                std::sqrt(std::pow(step_x * (it.getX() - nit.getX()), 2) +
                          std::pow(step_y * (it.getY() - nit.getY()), 2) +
                          std::pow(step_z * (it.getZ() - nit.getZ()), 2));

            // Cette fonction devrait être remplacée par une fonction
            // paramètre
            // double cost = 10000.0/(1.0+std::pow(diff/dist,4));
            double cost = dist / (1 + diff);

            if (val2_prec ==
                val2) // same val2 means same edge (thus, keep e3 and e4)
            {
              capacity[e4] = capacity[e4] + cost;
              capacity[e3] = capacity[e3] + cost;
            } else {
              boost::tie(e5, in1) = boost::edge(val, val2, g);

              if (in1 == 0) {
                clock_t temps_new_edge = clock();
                // std::cout<<"Add new edge "<< val<<" --
                // "<<val2<<std::endl;
                numEdges++;
                boost::tie(e4, in1) = boost::add_edge(val, val2, g);
                boost::tie(e3, in1) = boost::add_edge(val2, val, g);
                capacity[e4]        = cost;
                capacity[e3]        = cost;
                rev[e4]             = e3;
                rev[e3]             = e4;
                tt_new_edge += clock() - temps_new_edge;

              } else {
                clock_t temps_old_edge = clock();
                // std::cout<<"existing edge"<<std::endl;
                boost::tie(e4, in1) = boost::edge(val, val2, g);
                boost::tie(e3, in1) = boost::edge(val2, val, g);
                capacity[e4]        = capacity[e4] + cost;
                capacity[e3]        = capacity[e3] + cost;
                tt_old_edge += clock() - temps_old_edge;
              }
              val2_prec = val2;
            }
          }
        }
      }
      val_prec    = val;
      marker_prec = marker;
    }
  }

  std::cout << "Number of edges : " << numEdges << std::endl;
  t2 = clock();
  std::cout << "Edges creation time : " << double(t2 - t1) / CLOCKS_PER_SEC
            << " seconds\n";
  std::cout << "Marker2   : " << double(tt_marker2) / CLOCKS_PER_SEC
            << " seconds\n";
  std::cout << "Marker3   : " << double(tt_marker3) / CLOCKS_PER_SEC
            << " seconds\n";
  std::cout << "New edges : " << double(tt_new_edge) / CLOCKS_PER_SEC
            << " seconds\n";
  std::cout << "Old edges : " << double(tt_old_edge) / CLOCKS_PER_SEC
            << " seconds\n";

  boost::property_map<Graph_d, boost::vertex_index_t>::type indexmap =
      boost::get(boost::vertex_index, g);
  std::vector<boost::default_color_type> color(boost::num_vertices(g));

  std::cout << "Compute Max flow" << std::endl;
  t1 = clock();
#if BOOST_VERSION >= 104700
  double flow = boykov_kolmogorov_max_flow(g, capacity, residual_capacity, rev,
                                           &color[0], indexmap, vSource, vSink);
#else
  double flow = kolmogorov_max_flow(g, capacity, residual_capacity, rev,
                                    &color[0], indexmap, vSource, vSink);
#endif
  std::cout << "c  The total flow:" << std::endl;
  std::cout << "s " << flow << std::endl;
  t2 = clock();
  std::cout << "Flow computation time : " << double(t2 - t1) / CLOCKS_PER_SEC
            << " seconds\n";

  t1       = clock();
  int miss = 0;
  for (it = imIn.begin(), iend = imIn.end(); it != iend; ++it) {
    o1 = it.getOffset();
    // enlever int ; prendre type générique
    int val = imIn.pixelFromOffset(o1);

    if (val == 0) {
      imOut.setPixel(o1, 0);
    } else {
      if (color[val] == color[vSource])
        imOut.setPixel(o1, 2);
      else if (color[val] == color[vSink])
        imOut.setPixel(o1, 3);
      else {
        imOut.setPixel(o1, 20);
        miss++;
      }
    }
  }
  t2 = clock();
  std::cout << "Computing imOut took : " << double(t2 - t1) / CLOCKS_PER_SEC
            << " seconds\n";

  if (miss > 0)
    std::cout << "WARNING : Missclassified nodes : " << miss << "\n";

  return RES_OK;
}
