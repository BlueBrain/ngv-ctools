#pragma once
#include <cmath> // NAN
#include <limits> // numeric_limits<T>::infinity

#include <pybind11/stl.h> // conversio of stl to python types
#include <pybind11/numpy.h> // py::array_t

#include "types.hpp" // Float
#include "priority_heap.hpp" // MinPriorityHeap


namespace py = pybind11;

namespace ngv {

const Float EPSILON = 1e-6;
using Point = std::array<Float, 3>;

/**
 * Calculate the Dot product between the x, y, z components
 * of two vectors.
 */
inline Float Dot(const Point& v1, const Point& v2) noexcept {
  return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

/**
 * Calculate the euclidean Norm of the x, y, z components of a vector
 */
inline Float Norm(const Point& v) noexcept {
  return std::sqrt(Dot(v, v));
}

/**
 * Calculate the squared distance between two vectors
 */
inline Float DistSquared(const Point &xyz1, const Point &xyz2) noexcept {
  const Point diff = {xyz2[0] - xyz1[0], xyz2[1] - xyz1[1], xyz2[2] - xyz1[2]};
  return Dot(diff, diff);
}

/**
 * Find the real roots of the quadratic equation a * x ^ 2 + b * x + c
 *
 * Notes:
 *   If a is close to 0, then the solution to the first order equation is given
 * as the first root with the second one being NAN.
 *
 *   If the discriminant is 0, then instead of two identical solution, the first
 * one is given and the second one is NAN.
 *
 *   Depending if b is >= 0 or not, the numerically stable method of finding the
 * roots is used as presented in:
 * https://people.csail.mit.edu/bkph/articles/Quadratics.pdf
 *
 */
inline std::array<Float, 2> SecondOrderSolutions(Float a, Float b,
                                                 Float c) noexcept {

  // round close to zero values
  b = std::abs(b) < EPSILON ? 0.0 : b;

  std::array<Float, 2> roots = {NAN, NAN};

  // first order equation if a is zero
  if (std::abs(a) < EPSILON) {
    roots[0] = b != 0.0 ? -c / b : NAN;
    return roots;
  }

  // second order discriminant
  const Float delta = b * b - 4. * a * c;

  if (std::abs(delta) < EPSILON) {
    roots[0] = -b / (2.0 * a);
    return roots;
  }

  // Numerically Stable Method for Solving Quadratic Equation
  if (delta > 0.) {
    const Float sqrt_delta = std::sqrt(delta);
    if (b >= 0) {
        roots[0] = 0.5 * (-b - sqrt_delta) / a;
        roots[1] = 2.0 * c / (-b - sqrt_delta);
    } else {
        roots[0] = 0.5 * (-b + sqrt_delta) / a;
        roots[1] = 2.0 * c / (-b + sqrt_delta);
    }
  }
  return roots;
}

/**
 * Solve for travel_time in triangle when two vertice's times and the geometry
 * are known
 *
 * Following: A FAST ITERATIVE METHOD FOR SOLVING THE EIKONAL EQUATION ON
 * TRIANGULATED SURFACES doi: 10.1137/100788951
 *
 * Update the travel time at C taking into account
 * the upwind neighbors A and B.
 *
 *         C (v3)
 *        / \
 *       /   \
 *      /     \
 *     /       \
 *    /         \
 *    - - - - - -
 *  A (v1)       B (v2)
 *
 *  TA is the travel time of the wavefront to A
 *  TB is the travel time of the wavefront to B
 *
 */
inline Float LocalSolver2D(Float Ax, Float Ay, Float Az, Float Bx, Float By,
                           Float Bz, Float Cx, Float Cy, Float Cz, Float TA,
                           Float TB) {

  const Point AC = {Cx - Ax, Cy - Ay, Cz - Az};

  // AB vector
  const Point AB = {Bx - Ax, By - Ay, Bz - Az};

  // Dot products
  const Float Caa = Dot(AC, AC);
  const Float Cab = Dot(AC, AB);
  const Float Cbb = Dot(AB, AB);

  // travel time from A to B
  const Float TAB = TB - TA;

  auto tfunc = [=](Float l) {
    return TA + l * TAB + Norm(Point({AC[0] - l * AB[0], AC[1] - l * AB[1], AC[2] - l * AB[2]}));
  };

  if (std::abs(TAB) < EPSILON) {
    return tfunc(Cab / Cbb);
  }

  const Float inv_TAB_sq = 1.0 / (TAB * TAB);

  const std::array<Float, 2> roots = SecondOrderSolutions(
    Cbb * (1. - Cbb * inv_TAB_sq),
    2. * Cab * (-1. + Cbb * inv_TAB_sq),
    Caa - Cab * Cab * inv_TAB_sq
  );

  const bool is_root1_valid = not std::isnan(roots[0]);
  const bool is_root2_valid = not std::isnan(roots[1]);

  // solutions can be symmetric. In that case pick the one that gives the
  // shortest travel time (Fermat's principle).
  if (is_root1_valid && is_root2_valid) {
    return std::fmin(tfunc(roots[0]), tfunc(roots[1]));
  }

  if (is_root1_valid) {
    return tfunc(roots[0]);
  }

  if (is_root2_valid) {
    return tfunc(roots[1]);
  }

  // if no solution is found the characteristic of the gradient is outside
  // the triangle. In that case give the smallest travel time through the
  // edges of the triangle
  return std::fmin(TA + std::sqrt(Caa), TB + Norm(Point({Cx - Bx, Cy - By, Cz - Bz})));
}

// Fast marching method vertex visited status. All vertices start as FAR and
// while processed by the fast marching method, they become TRIAL or KNOWN.
enum Status { FAR = -1, TRIAL = 0, KNOWN = 1 };

// Vertices hold all the information required by the fast marching method.
struct Vertex {

  // See Status above. All Vertices in FMM start as FAR.
  Status status = Status::FAR;

  // Travel time to the closest seed, used in the priority heap.
  // All vertices start at infinite distance before they are updated
  Float travel_time = std::numeric_limits<Float>::infinity();

  // The id of the closest seed to each vertex. Starts as unassigned (-1)
  int32_t group_id = -1;

  // The one ring neighbors of the vertex.
  std::vector<size_t> neighbors;

  // The euclidean coordinates of the vertex.
  Point xyz = {NAN, NAN, NAN};
};

/**
 * Fast marching method struct
 *
 */
struct FastMarchingMethod {

  // The vertices of the 3D surface mesh
  std::vector<ngv::Vertex> vertices;

  // Starting from each seed the wavefronts will not update vertices the squared
  // distance of which to the former is greater than the cutoff one.
  const Float squared_cutoff_distance_;

  // The vertex indices which will act as the starting points of the wavefront
  // propagation.
  const std::vector<uint32_t> seed_ids;

  /*
   * Fast marching method constructor for initializing the vertices vector.
   *
   * @param neighbors 1D numpy array containing a flat list of vertex indices.
   * @param nn_offsets 1D numpy array containing the neighbors offsets. The
   * neighbors of vertex i are accessed as neighbors[nn_offsets[i]:
   * nn_offsets[i+1]].
   * @param coordinates 2D array containing the coordinates of each vertex.
   * @param seed_vertices 1D array of the vertex indices.
   * @param squared_cutoff_distance The squared distance at which vertices
   * belonging to the wavefront that starts from a seed vertex stop being
   * updated.
   */
  explicit FastMarchingMethod(py::array_t<size_t> neighbors,
                              py::array_t<size_t> nn_offsets,
                              py::array_t<float, 3> coordinates,
                              py::array_t<uint32_t> seed_vertices,
                              Float squared_cutoff_distance)
      : squared_cutoff_distance_(squared_cutoff_distance),
        seed_ids({seed_vertices.data(),
                  seed_vertices.data() + seed_vertices.size()}) {

    // unchecked direct access proxies to access data
    // without bounds and dimension checks on every access.
    auto proxy_offsets = nn_offsets.unchecked<1>();
    auto proxy_coordinates = coordinates.unchecked<2>();

    const size_t n_vertices = nn_offsets.size() - 1;
    vertices.reserve(n_vertices);
    for (size_t i = 0; i != n_vertices; ++i) {

      Vertex vertex;

      size_t beg_offset = proxy_offsets(i);
      size_t end_offset = proxy_offsets(i + 1);

      // copy the subvector to the struct neighbors
      vertex.neighbors = {neighbors.data() + beg_offset,
                          neighbors.data() + end_offset};

      vertex.xyz = {proxy_coordinates(i, 0), proxy_coordinates(i, 1),
                    proxy_coordinates(i, 2)};

      vertices.push_back(std::move(vertex));
    }
  }

  void Solve() {

    utils::MinPriorityHeap trial_heap(vertices.size());

    // initialize the seed vertices
    for (size_t i = 0; i != seed_ids.size(); ++i) {
      Vertex &vertex = vertices[seed_ids[i]];
      vertex.status = Status::KNOWN;
      vertex.travel_time = 0.0;
      vertex.group_id = i;
    }

    // update the 1-ring neighborhood of the seed vertices
    // and calculate the travel time to their neighbors
    for (size_t index : seed_ids) {
      UpdateNeighbors(vertices[index], trial_heap);
    }

    // expand in a breadth first manner from the smallest
    // distance node and update the travel times for the
    // propagation of the wavefront
    while (not trial_heap.empty()) {

      // min travel time vertex
      const auto &record = trial_heap.top();
      Vertex &vertex = vertices[record.id];

      if (vertex.status != Status::KNOWN) {
        vertex.status = Status::KNOWN;
        vertex.travel_time = record.value;
      }

      // after finishing using the record from top
      trial_heap.pop();

      UpdateNeighbors(vertex, trial_heap);
    }
  }

  inline void UpdateNeighbors(Vertex &vertex,
                              utils::MinPriorityHeap &trial_heap) {

    for (size_t neighbor_index : vertex.neighbors) {

      Vertex &neighbor = vertices[neighbor_index];

      // if the neighbor value has not been finalized (FMM_FAR, FMM_TRIAL)
      if (neighbor.status == Status::KNOWN) {
        continue;
      }

      // find the travel time of the wave to the neighbor vertex in the
      // ring
      neighbor.travel_time = ComputeTravelTime(neighbor);

      // otherwise add in the priority queue with the travel time
      // as priority. It starts as a trial vertex.
      const Vertex &group_vertex = vertices[seed_ids[vertex.group_id]];
      if (neighbor.status == Status::FAR &&
          DistSquared(neighbor.xyz, group_vertex.xyz) <
              squared_cutoff_distance_) {
        neighbor.group_id = vertex.group_id;
        neighbor.status = Status::TRIAL;
        trial_heap.emplace(neighbor_index, neighbor.travel_time);
      }
    }
  }

  /**
   */
  inline Float ComputeTravelTime(const Vertex &vertex) noexcept {

    // find a triangle with nodes of known values and update the
    // traveling time at vertex C

    Float min_value = vertex.travel_time;
    const auto &neighbors = vertex.neighbors;
    const size_t n_neighbors = neighbors.size();

    for (size_t i = 0; i != n_neighbors; ++i) {

      // consecutive ring neighbors
      // the last pair is the last and the first vertices in the ring
      size_t nb1 = neighbors[i];
      size_t nb2 = neighbors[i == n_neighbors - 1 ? 0 : i + 1];

      const Float TA = vertices[nb1].travel_time;
      const Float TB = vertices[nb2].travel_time;

      if (std::isinf(TA) && std::isinf(TB)) {
        continue;
      }

      if (TB < TA) {
        std::swap(nb1, nb2);
      }

      min_value = std::min(
          min_value, TriangleTravelTime(vertex, vertices[nb1], vertices[nb2]));
    }
    return min_value;
  }

  /**
   *
   */
  inline Float TriangleTravelTime(const Vertex &v0, const Vertex &v1,
                                  const Vertex &v2) const {
    return LocalSolver2D(v1.xyz[0], v1.xyz[1], v1.xyz[2], v2.xyz[0], v2.xyz[1],
                         v2.xyz[2], v0.xyz[0], v0.xyz[1], v0.xyz[2],
                         v1.travel_time, v2.travel_time);
  }
};

inline void BindFastMarchingMethodModule(py::module &m) {

  // create a submodule for the c
  py::module fast_marching_method = m.def_submodule("fast_marching_method");

  fast_marching_method.def(
      "grow_waves_on_triangulated_surface",
      [](py::array_t<size_t> neighbors, py::array_t<size_t> nn_offsets,
         py::array_t<float, 3> coordinates, py::array_t<size_t> seeds,
         double squared_cutoff_distance) {
        FastMarchingMethod fmm(neighbors, nn_offsets, coordinates, seeds,
                               squared_cutoff_distance);

        fmm.Solve();

        // array of the group that each vertex belongs to
        py::array_t<int32_t> group_indices(fmm.vertices.size());
        auto proxy_group_indices = group_indices.mutable_unchecked<1>();

        // array of the travel time to the closest seed for each vertex
        py::array_t<Float> travel_times(fmm.vertices.size());
        auto proxy_travel_times = travel_times.mutable_unchecked<1>();

        // array of the visited status for each vertex
        py::array_t<int8_t> statuses(fmm.vertices.size());
        auto proxy_statuses = statuses.mutable_unchecked<1>();

        size_t n = 0;
        for (const auto &vertex : fmm.vertices) {

          proxy_statuses[n] = vertex.status;
          proxy_group_indices[n] = vertex.group_id;
          proxy_travel_times[n] = vertex.travel_time;
          ++n;
        }

        return py::make_tuple(group_indices, travel_times, statuses);
      },
      R"(
        Given a triangulation of N vertices, where the connectivity of the i-th vertex can be extracted
        via the neighbors and nn_offsets, and with coordinates v_xyz, Solve the eikonal equation setting
        as starting points of the wave propagation the seed_vertices. If the wave propagation exceeds the
        squared_cutoff_distance then the spreading of that wave stops letting the neighboring seeds expand if any.

        Args:
            neighbors:
                Array of vertices stored so that the neighbors of the i-th vertex can be extracted
                neighbors[nn_offsets[i]: nn_offsets[i + 1]]

            nn_offsets:
                The offsets for extracting the neighbors for the i-th vertex

            v_xyz:
                The coordinates of the vertices

            seed_vertices:
                The vertex ids that will play the role of starting points in the simulation

            squared_cutoff_distance:
                The distance that the neighbor updating will stop for each vertex

        Returns:
            v_group_indices:
                The array of the indices to the seed_vertices or -1 the unassigned group

            v_travel_times:
                The travel times of the wavefronts for each vertex

            v_status:
                The fast marching method status, e.g. FMM_KNOWN, FMM_TRIAL, FMM_FAR for each
                vertex
        )",
      py::arg("neighbors"), py::arg("nn_offsets"), py::arg("coordinates"),
      py::arg("seeds"), py::arg("squared_cutoff_distance"));

  // for testing
  py::class_<utils::MinPriorityHeap>(fast_marching_method, "MinPriorityHeap")
      .def(py::init<size_t>(), "capacity")
      .def_property_readonly("size", &utils::MinPriorityHeap::size)
      .def_property_readonly("capacity", &utils::MinPriorityHeap::capacity)
      .def("empty", &utils::MinPriorityHeap::empty)
      .def("push", [](utils::MinPriorityHeap &self, size_t index,
                      Float value) { self.emplace(index, value); })
      .def("pop", &utils::MinPriorityHeap::pop)
      .def("top", [](utils::MinPriorityHeap *heap) {
        auto record = heap->top();
        return py::make_tuple(record.id, record.value);
      });

  py::class_<Vertex>(fast_marching_method, "Vertex")
      .def(py::init<>())
      .def_readonly("status", &Vertex::status)
      .def_readonly("travel_time", &Vertex::travel_time)
      .def_readonly("group_id", &Vertex::group_id)
      .def_readonly("neighbors", &Vertex::neighbors)
      .def_readonly("xyz", &Vertex::xyz);

  py::class_<FastMarchingMethod>(fast_marching_method, "FastMarchingMethod")
      .def(py::init<py::array_t<size_t>, py::array_t<size_t>,
                    py::array_t<float, 3>, py::array_t<size_t>, double>(),
           py::arg("neighbors"), py::arg("nn_offsets"), py::arg("coordinates"),
           py::arg("seeds"), py::arg("squared_cutoff_distance"))
      .def("solve", &FastMarchingMethod::Solve)
      .def_readonly("seeds", &FastMarchingMethod::seed_ids)
      .def_readonly("vertices", &FastMarchingMethod::vertices);

  fast_marching_method.def("dot", [](Float x1, Float y1, Float z1, Float x2, Float y2, Float z2){
    return Dot(Point({x1, y1, z1}), Point({x2, y2, z2}));
  });
  fast_marching_method.def("norm", [](Float x, Float y, Float z){ return Norm(Point({x, y, z})); });
  fast_marching_method.def("dist_squared", &DistSquared);
  fast_marching_method.def("second_order_solutions", &SecondOrderSolutions);
  fast_marching_method.def("local_solver_2D", &LocalSolver2D);
};

} // namespace ngv
