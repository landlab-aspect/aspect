/*
  Copyright (C) 2025 - 2025 by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.

  ASPECT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ASPECT; see the file LICENSE.  If not see
  <http://www.gnu.org/licenses/>.
 */


#include <aspect/mesh_deformation/landlab.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/patterns.h>

#include <aspect/gravity_model/interface.h>
#include <aspect/geometry_model/box.h>
#include <deal.II/base/array_view.h>

#include <cfenv>

#include <aspect/python_helper.h>

#ifdef ASPECT_WITH_PYTHON

using namespace dealii;

/**
 * Call a Python function from module @p pModule with name @p func_name
 * Returns the result (caller must Py_DECREF). Throws on error.
 */
PyObject *call_python_function(PyObject *pModule, const char *func_name, PyObject *pArgs = nullptr)
{
  PyObject *pFunc = PyObject_GetAttrString(pModule, func_name);
  if (!pFunc || !PyCallable_Check(pFunc))
    AssertThrow(false, ExcMessage(std::string("Failed to load function: ") + func_name));

  PyObject *pValue = PyObject_CallObject(pFunc, pArgs);
  Py_DECREF(pFunc);

  if (pValue == nullptr)
    {
      if (PyErr_Occurred())
        PyErr_Print();
      AssertThrow(false, ExcMessage(std::string(func_name) + " returned NULL"));
    }
  return pValue;
}

#endif




namespace aspect
{

  namespace MeshDeformation
  {
    template <int dim>
    Landlab<dim>::Landlab()
    {
    }



    template <int dim>
    void
    Landlab<dim>::initialize ()
    {
      AssertThrow(Plugins::plugin_type_matches<GeometryModel::Box<dim>>(this->get_geometry_model()),
                  ExcMessage("The surface diffusion mesh deformation plugin only works for Box geometries."));

      unsigned int rank = Utilities::MPI::this_mpi_process(this->get_mpi_communicator());
      if (rank >= n_landlab_ranks)
        return;

#ifdef ASPECT_WITH_PYTHON
      // Append script dirs so env packages (venv site-packages, PYTHONPATH) are found first
      // for "import landlab":
      PyRun_SimpleString("import sys");
      PyRun_SimpleString("sys.path.append(\"" ASPECT_SOURCE_DIR "/tests\")");
      PyRun_SimpleString("sys.path.append(\".\")");

      // avoid floating point exceptions in Landlab Python code:
#ifdef ASPECT_USE_FP_EXCEPTIONS
      fedisableexcept(FE_DIVBYZERO|FE_INVALID);
#endif

      std::cout << "importing '" << script_module_name << "' ..." << std::endl;
      pModule = PyImport_ImportModule(script_module_name.c_str());
      if (PyErr_Occurred())
        PyErr_Print();
      AssertThrow(pModule, ExcMessage("Failed to load Python module"));

      // Call Python initialize() function with communicator handle
      PyObject *pArgs;
      if (n_landlab_ranks == 1)
        pArgs = PyTuple_Pack(1, Py_None);
      else
        pArgs = PyTuple_Pack(1, PyLong_FromLong(MPI_Comm_c2f(this->get_mpi_communicator())));
      PyObject *pValue = call_python_function(pModule, "initialize", pArgs);

      Py_DECREF(pArgs);
      Py_DECREF(pValue);

#else
      AssertThrow(false, ExcMessage("ASPECT needs to be configure with Python support "
                                    "(ASPECT_WITH_PYTHON=ON in CMake) to be able to use the Landlab mesh deformation model."));
#endif
    }



    template <int dim>
    void
    Landlab<dim>::update ()
    {

#ifdef ASPECT_WITH_PYTHON
      if (!this->remote_point_evaluator)
        {
          unsigned int rank = Utilities::MPI::this_mpi_process(this->get_mpi_communicator());
          if (rank >= n_landlab_ranks)
            {
              // This rank does not participate, so we don't own any evaluation points:
              std::vector<Point<dim>> surface_points;
              this->set_evaluation_points(surface_points);
              return;
            }

          {
            // set_mesh_information: call with None
            PyObject *pDict = PyDict_New();
            PyDict_SetItemString(pDict, "Mesh X extent", PyFloat_FromDouble(landlab_x_extent));
            PyDict_SetItemString(pDict, "Mesh Y extent", PyFloat_FromDouble(landlab_y_extent));
            PyDict_SetItemString(pDict, "Mesh Spacing", PyFloat_FromDouble(landlab_spacing));

            PyObject *pArgs = PyTuple_Pack(1, pDict);
            PyObject *pValue = call_python_function(pModule, "set_mesh_information", pArgs);
            Py_DECREF(pArgs);
            Py_DECREF(pValue);
          }

          {
            // Initialize landlab components
            // If Use years instead of seconds is false, then we want to make sure we scale the 
            // parameters in landlab to seconds, since surface processes parameters are reported
            // typically in years.
            double time_scaling_factor = 1.0;
            if (parameters_in_years)
              time_scaling_factor = year_in_seconds;

            PyObject *pDict = PyDict_New();
            PyDict_SetItemString(pDict, "Hillslope coefficient", PyFloat_FromDouble(hillslope_diffusion_coefficient / time_scaling_factor));
            PyDict_SetItemString(pDict, "Stream power erodibility coefficient", PyFloat_FromDouble(stream_power_erodibility_coefficient / time_scaling_factor));

            PyObject *pArgs = PyTuple_Pack(1,
                                           pDict
                                          );
            Py_DECREF(pDict);

            PyObject *pValue = call_python_function(pModule, "initalize_landlab_components", pArgs);
            Py_DECREF(pArgs);
            Py_DECREF(pValue);
          }
          
          {
            // get grid points from the landlab mesh:
            PyObject *pArgs = PyTuple_Pack(1, PyLong_FromLong(-1L));
            PyObject *pgrid_x = call_python_function(pModule, "get_grid_x", pArgs);
            Py_DECREF(pArgs);

            PyObject *pgrid_y = nullptr;
            if (dim == 3)
              {
                PyObject *pArgs = PyTuple_Pack(1, PyLong_FromLong(-1L));
                pgrid_y = call_python_function(pModule, "get_grid_y", pArgs);
                Py_DECREF(pArgs);
              }

            const ArrayView<double> data_x = PythonHelper::numpy_to_array_view(pgrid_x);
            const ArrayView<double> data_y = (dim == 3)
                                             ? PythonHelper::numpy_to_array_view(pgrid_y)
                                             : ArrayView<double>(nullptr, 0);
            if (dim == 3)
              AssertThrow(data_x.size() == data_y.size(), ExcMessage("get_grid_x and get_grid_y returned different sizes"));

            std::vector<Point<dim>> surface_points(data_x.size());
            for (size_t i = 0; i < data_x.size(); i++)
              {
                Point<dim> point;
                point(0) = data_x[i];
                if (dim == 3)
                  point(1) = data_y[i];
                point(dim-1) = this->get_geometry_model().representative_point(0.0)[dim-1];
                surface_points[i] = point;
              }

            // Clean up Python objects
            Py_DECREF(pgrid_x);
            if (pgrid_y)
              Py_DECREF(pgrid_y);

            this->set_evaluation_points(surface_points);
          }
        }
#endif
    }



    template <int dim>
    std::vector<Tensor<1,dim>>
    Landlab<dim>::compute_updated_velocities_at_points (const std::vector<std::vector<double>> &current_solution_at_points) const
    {
#ifdef ASPECT_WITH_PYTHON
      Assert(current_solution_at_points.size() == this->evaluation_points.size(), ExcInternalError());
      std::vector<Tensor<1,dim>> velocities(current_solution_at_points.size(), Tensor<1,dim>());

      if (pModule)
        {
          // Build a dictionary with solution values for each variable to pass to Python:
          PyObject *pDict = PyDict_New();
          // variable names must match those expected by the Landlab script:
          const std::vector<std::string> variable_names = { "x velocity", "y velocity", "z velocity" };
          // Create a vector<vector> that is of the shape (n_variables, n_points)
          std::vector<std::vector<double>> variable_data(variable_names.size(),  std::vector<double>(current_solution_at_points.size(), 0.0));

          // Fill variable_data (which corresponds to the LandLab nodal points) with the values from ASPECT.
          for (unsigned int i=0; i<variable_names.size(); ++i)
            {
              for (unsigned int j=0; j<current_solution_at_points.size(); ++j)
                {
                  variable_data[i][j] = current_solution_at_points[j][i];
                }
            }
          for (unsigned int i=0; i<variable_names.size(); ++i)
            {
              auto pValue = PythonHelper::vector_to_numpy_object(variable_data[i]);
              PyDict_SetItemString(pDict, variable_names[i].c_str(), pValue.get());
            }

          // Call update_until(). update_until() returns deposition_erosion. This is what eventually 
          // gets converted to velocities to deform the ASPECT mesh.
          PyObject *pArgs = PyTuple_Pack(2, PyFloat_FromDouble(this->get_time()), pDict);
          PyObject *pValue = call_python_function(pModule, "update_until", pArgs);
          Py_DECREF(pDict);
          Py_DECREF(pArgs);

          const ArrayView<const double> data = PythonHelper::numpy_to_array_view(pValue);
          for (size_t i=0; i<data.size(); ++i)
            velocities[i][dim-1] = data[i] / (this->get_timestep() > 0.0 ? this->get_timestep() : 1.0);

          Py_DECREF(pValue);
        }

      // Produce debug output as a vtu file
#if 0
      {
        static unsigned int output_no = 0;

        DataOutPoints<dim, dim> out;
        //const auto &mapping = this->get_mapping();
        std::vector<Point<dim>> real_evaluation_points(this->evaluation_points.size());
        std::vector<std::vector<double>> data(this->evaluation_points.size(), std::vector<double>(dim, 0.0));
        for (unsigned int i=0; i<this->evaluation_points.size(); ++i)
          {
            real_evaluation_points[i] = this->evaluation_points[i];  // TODO: use mapping to compute real position
            for (unsigned int c=0; c<dim; ++c)
              data[i][c] = velocities[i][c] / (this->get_timestep() > 0.0 ? this->get_timestep() : 1.0);
          }

        const std::vector<std::string> data_component_names(dim, "velocity");
        const std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretations(dim, DataComponentInterpretation::component_is_part_of_vector);

        out.build_patches(real_evaluation_points, 0, data, data_component_names, data_component_interpretations);

        out.write_vtu_with_pvtu_record(this->get_output_directory(), "surf_points", output_no, this->get_mpi_communicator(), 4, 0);

        ++output_no;
      }
#endif

      return velocities;
#else
      (void)current_solution_at_points;
      return std::vector<Tensor<1,dim>>();
#endif
    }



    template <int dim>
    void Landlab<dim>::
    compute_initial_deformation_as_constraints(const Mapping<dim> &/*mapping*/,
                                               const DoFHandler<dim> &mesh_deformation_dof_handler,
                                               const types::boundary_id boundary_indicator,
                                               AffineConstraints<double> &constraints) const
    {
#ifdef ASPECT_WITH_PYTHON
      // hack: cast away const and call update() so we get the evaluation point
      // datastructures populated:
      const_cast<Landlab<dim>*>(this)->update();

      // 1. Grab initial deformation from Landlab:
      std::vector<Tensor<1,dim>> initial_deformation(this->evaluation_points.size(), Tensor<1,dim>());
      if (pModule)
        {
          Tensor<1,dim> topography_direction;
          topography_direction[dim-1] = 1.0;

          PyObject *pArgs = PyTuple_Pack(1, PyLong_FromLong(-1L));
          PyObject *pValue = call_python_function(pModule, "get_initial_topography", pArgs);
          Py_DECREF(pArgs);
          ArrayView<double> data = PythonHelper::numpy_to_array_view(pValue);
          for (size_t i=0; i<data.size(); ++i)
            initial_deformation[i] = data[i] * topography_direction;
          Py_DECREF(pValue);
        }

      // 2. Interpolate deformation into a DoF vector:
      LinearAlgebra::Vector initial_deformation_dof_vector = this->interpolate_external_velocities_to_surface_support_points(initial_deformation);
      const DoFHandler<dim> &mesh_dof_handler = this->get_mesh_deformation_handler().get_mesh_deformation_dof_handler();
      const IndexSet mesh_locally_relevant = DoFTools::extract_locally_relevant_dofs (mesh_dof_handler);
      LinearAlgebra::Vector initial_deformation_ghosted(mesh_dof_handler.locally_owned_dofs(),
                                                        mesh_locally_relevant,
                                                        this->get_mpi_communicator());
      initial_deformation_ghosted = initial_deformation_dof_vector;

      const IndexSet constrained_dofs =
        DoFTools::extract_boundary_dofs(mesh_deformation_dof_handler,
                                        ComponentMask(dim, true),
      {boundary_indicator});

      // 3. Add constraints from DoF values:
      for (const types::global_dof_index index : constrained_dofs)
        {
          if (constraints.can_store_line(index))
            if (constraints.is_constrained(index)==false)
              {
#if DEAL_II_VERSION_GTE(9,6,0)
                constraints.add_constraint(index,
                                           {},
                                           initial_deformation_ghosted(index));
#else
                constraints.add_line(index);
                constraints.set_inhomogeneity(index, initial_deformation_ghosted(index));
#endif
              }
        }
#else
      (void)mesh_deformation_dof_handler;
      (void)boundary_indicator;
      (void)constraints;
#endif
    }



    template <int dim>
    void Landlab<dim>::declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Mesh deformation");
      {
        prm.enter_subsection("Landlab");
        {
          prm.declare_entry("MPI ranks for Landlab", "1",
                            Patterns::Integer(1),
                            "Number of ranks to use for the Landlab simulation. If set to 1, the Landlab simulation will run sequentially without MPI.");
          prm.declare_entry("Script path", "",
                            Patterns::Anything(),
                            "Path to the Python script to execute. Relative paths and the placeholders "
                            "ASPECT_SOURCE_DIR and ASPECT_BINARY_DIR are allowed.");
          prm.declare_entry("Script name", "",
                            Patterns::Anything(),
                            "Name of the Python module to load (without .py extension).");
          prm.declare_entry("Script argument", "",
                            Patterns::Anything(),
                            "An arbitrary string to be passed to the initialize() function in the "
                            "Python script. Can be used to specify a configuration file or other option.");

          prm.declare_entry("Define LandLab parameters using years", "unspecified",
                            Patterns::Selection("true|false|unspecified"),
                            "If true, the LandLab parameters (diffusion coefficient, erodibility, uplift rate) "
                            "are defined in units per year. If false, they are defined in SI units (per second).");
          prm.declare_entry("Hillslope diffusion coefficient", "0.01",
                            Patterns::Double(0),
                            "Hillslope diffusion coefficient used to compute the mesh deformation. "
                            "Units: \\si{\\meter\\squared\\per\\year}.");
          prm.declare_entry("Stream power erodibility coefficient", "1e-4",
                            Patterns::Double(0),
                            "Stream power erodibility coefficient used to compute the mesh deformation. "
                            "Units: \\si{\\meter\\raisedto{-1}\\per\\year}.");

          prm.enter_subsection("Mesh information");
           {
              prm.declare_entry("X extent", "100e3",
                                Patterns::Double(0),
                                "The extent of the Landlab grid in the x direction. Units: \\si{\\meter}.");
              prm.declare_entry("Y extent", "100e3",
                                Patterns::Double(0),
                                "The extent of the Landlab grid in the y direction. Units: \\si{\\meter}.");
              prm.declare_entry("Spacing", "1000",
                                Patterns::Double(0),
                                "The spacing of the Landlab grid. Units: \\si{\\meter}.");
           }
          prm.leave_subsection();
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    template <int dim>
    void Landlab<dim>::parse_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection ("Mesh deformation");
      {
        prm.enter_subsection ("Landlab");
        {
          n_landlab_ranks    = prm.get_integer("MPI ranks for Landlab");
          script_path        = prm.get("Script path");
          script_module_name = prm.get("Script name");
          script_argument    = prm.get("Script argument");

          hillslope_diffusion_coefficient      = prm.get_double("Hillslope diffusion coefficient");
          stream_power_erodibility_coefficient = prm.get_double("Stream power erodibility coefficient");

          if (prm.get ("Define LandLab parameters using years") == "true")
            parameters_in_years = true;
          else if (prm.get ("Define LandLab parameters using years") == "false")
            parameters_in_years = false;
          else
            AssertThrow(false, ExcMessage("The parameter 'Define LandLab parameters using years' must be set to 'true' or 'false'"));

            prm.enter_subsection("Mesh information");
              {
                landlab_x_extent = prm.get_double("X extent");
                landlab_y_extent = prm.get_double("Y extent");
                landlab_spacing = prm.get_double("Spacing");
              }
          prm.leave_subsection();
        }
        prm.leave_subsection ();
      }
      prm.leave_subsection ();
    }
  }
}


// explicit instantiation of the functions we implement in this file
namespace aspect
{
  namespace MeshDeformation
  {
    ASPECT_REGISTER_MESH_DEFORMATION_MODEL(Landlab,
                                           "Landlab",
                                           "A mesh deformation plugin that lets a Python script control the "
                                           "deformation of the surface. It is meant for coupling with the landscape evolution "
                                           "code Landlab, but any other script that provides the necessary functions can be used. "
                                           "It is necessary to have Python and numpy with their C APIs installed and that "
                                           "ASPECT_WITH_PYTHON is enabled when ASPECT is configured with CMake.")
  }
}
