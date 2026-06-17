from mpi4py import MPI
import importlib.util
import os
import sys

import numpy as np
np.set_printoptions(suppress=False, precision=12)
import landlab
from landlab.components import LinearDiffuser, AdvectionSolverTVD

import json
from landlab.io.legacy_vtk import write_legacy_vtk


from landlab_template import LandLabTemplate


class MyAspectLandlabModel(LandLabTemplate):



    def update_until(self, end_time, ASPECT_dim, ASPECT_fields_at_Landlab_nodes_dict):

        dt = end_time - self.current_time
        self.timestep += 1

        deposition_erosion = np.zeros(self.model_grid.number_of_nodes)


        # =====================================================
        # TIME STEPPING
        # =====================================================
        if dt > 0:
            n_substeps = 10
            sub_dt = dt / n_substeps

            for _ in range(n_substeps):
                elevation_before = self.elevation.copy()

                # Diffusion (Landlab physics)
                self.linear_diffuser.run_one_step(sub_dt)

                
                deposition_erosion += self.elevation - elevation_before

        self.current_time = end_time


        print("Max elevation:", np.max(self.elevation),
            "Min elevation:", np.min(self.elevation))



        dimensional_deposition_erosion = self.dimensional_deposition_erosion(
            ASPECT_dim,
            deposition_erosion
        )

        return dimensional_deposition_erosion
    

    def set_mesh_information(self, grid_dictionary):
        print("grid_dictionary =", grid_dictionary)

        if grid_dictionary is not None:
            print(grid_dictionary.keys())

        if self.model_grid is not None:
            return
        
        if self.model_grid is not None:
            return

        print("* Creating RasterModelGrid ...")

        # =====================================================
        # FIXED DOMAIN (MATCH ASPECT 3D TOP SURFACE)
        # =====================================================
        x_extent = 1
        y_extent = 1
        spacing  = 0.015625

        

        nrows = int(y_extent / spacing) + 3
        ncols = int(x_extent / spacing) + 3

        self.model_grid = landlab.RasterModelGrid(
            (nrows, ncols),
            xy_spacing=(spacing, spacing),
            xy_of_lower_left=(-spacing, -spacing),
        )

        print("* Creating topographic elevation ...")



        A = 0.075
        L = 1.0

        z = A * np.sin(np.pi * self.model_grid.node_x / L)

        self.elevation = self.model_grid.add_field(
            "topographic__elevation",
            z,
            at="node"
        )

        self.elevation = z


       
        self.model_grid.set_closed_boundaries_at_grid_edges(
            right_is_closed=True,
            left_is_closed=True,
            top_is_closed=True,
            bottom_is_closed=True,
        )

        

        print("\tnumber of nodes:", self.model_grid.number_of_nodes)

        self.initialize_landlab_components()
        print("* Done")

        return self.model_grid

    def initialize_landlab_components(self):
     
        D = 0.25 # m2/second
        # D = 0.25 /self.s2yr # 7*1e-9 m2/second

        self.Diffusivity = self.model_grid.add_zeros(
            "linear_diffusivity",
            at="node"
        )

        self.Diffusivity += D

        self.linear_diffuser = LinearDiffuser(
            self.model_grid,
            linear_diffusivity=self.Diffusivity
        )


model = MyAspectLandlabModel()
model.export_aspect_callbacks(model, globals())