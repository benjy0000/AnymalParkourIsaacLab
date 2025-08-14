"""Terrain importer for the parkour environment."""

from __future__ import annotations

import torch
import omni.log

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG, POSITION_GOAL_MARKER_CFG

from isaaclab.terrains import TerrainImporter, TerrainImporterCfg


class ParkourTerrainImporter(TerrainImporter):
    """Terrain importer for the parkour environment."""

    def __init__(self, cfg: ParkourTerrainImporterCfg):
        """Initialize the terrain importer.

        Args:
            cfg: The configuration for the terrain importer.

        Raises:
            ValueError: If input terrain type is not supported.
            ValueError: If terrain type is 'generator' and no configuration provided for ``terrain_generator``.
            ValueError: If terrain type is 'usd' and no configuration provided for ``usd_path``.
            ValueError: If terrain type is 'usd' or 'plane' and no configuration provided for ``env_spacing``.
        """
        # check that the config is valid
        cfg.validate()
        # store inputs
        self.cfg = cfg
        self.device = sim_utils.SimulationContext.instance().device  # type: ignore

        # create buffers for the terrains
        self.terrain_prim_paths = list()
        self.terrain_origins = None
        self.env_origins = None  # assigned later when `configure_env_origins` is called
        # private variables
        self._terrain_flat_patches = dict()

        # auto-import the terrain based on the config
        if self.cfg.terrain_type == "generator":
            # check config is provided
            if self.cfg.terrain_generator is None:
                raise ValueError("Input terrain type is 'generator' but no value provided for 'terrain_generator'.")
            # generate the terrain
            terrain_generator = self.cfg.terrain_generator.class_type(
                cfg=self.cfg.terrain_generator, device=self.device
            )
            self.import_mesh("terrain", terrain_generator.terrain_mesh)
            # retrieve the goals
            self.terrain_goals = terrain_generator.goals
            # configure the terrain origins based on the terrain generator
            self.configure_env_origins(terrain_generator.terrain_origins)
            # refer to the flat patches
            self._terrain_flat_patches = terrain_generator.flat_patches
        elif self.cfg.terrain_type == "usd":
            # check if config is provided
            if self.cfg.usd_path is None:
                raise ValueError("Input terrain type is 'usd' but no value provided for 'usd_path'.")
            # import the terrain
            self.import_usd("terrain", self.cfg.usd_path)
            # configure the origins in a grid
            self.configure_env_origins()
        elif self.cfg.terrain_type == "plane":
            # load the plane
            self.import_ground_plane("terrain")
            # configure the origins in a grid
            self.configure_env_origins()
        else:
            raise ValueError(f"Terrain type '{self.cfg.terrain_type}' not available.")

        # set initial state of debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

    def fetch_goals_from_env(self, env_id: torch.Tensor) -> torch.Tensor:

        # Check if the terrain origins are configured for curriculum learning
        # This is always the case for a terrain generator
        if self.terrain_origins is None or not hasattr(self, "terrain_levels"):
            omni.log.warn(
                "Cannot get terrain origin from environment ID. This function is only"
                " available when using curriculum-based terrain generation."
            )
            return torch.empty((0, 2), device=self.device, dtype=torch.long)

        # Get the row and column indices from the stored mapping
        rows = self.terrain_levels[env_id]
        cols = self.terrain_types[env_id]

        # Stack them into a (N, 2) tensor
        return self.terrain_goals[rows, cols]
    

    def set_debug_vis(self, debug_vis: bool) -> bool:
        """Set the debug visualization of the terrain importer.

        Args:
            debug_vis: Whether to visualize the terrain origins and goals.

        Returns:
            Whether the debug visualization was successfully set. False if the terrain
            importer does not support debug visualization.

        Raises:
            RuntimeError: If terrain origins are not configured.
        """
        # create a marker if necessary
        if debug_vis:
            if not hasattr(self, "origin_visualizer"):
                self.origin_visualizer = VisualizationMarkers(
                    cfg=FRAME_MARKER_CFG.replace(prim_path="/Visuals/TerrainOrigin")
                )
                if self.terrain_origins is not None:
                    self.origin_visualizer.visualize(self.terrain_origins.reshape(-1, 3))
                elif self.env_origins is not None:
                    self.origin_visualizer.visualize(self.env_origins.reshape(-1, 3))
                else:
                    raise RuntimeError("Terrain origins are not configured.")
            # set visibility
            self.origin_visualizer.set_visibility(True)

            if not hasattr(self, "goal_visualizer"):
                goal_marker_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/TerrainGoals",
                    markers={
                        "marker1": sim_utils.SphereCfg(
                            radius=0.05,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)))
                    }
                )
                self.goal_visualizer = VisualizationMarkers(cfg=goal_marker_cfg)
                if self.terrain_goals is not None:
                    terrain_goals_w = self.terrain_goals + self.terrain_origins.unsqueeze(2)
                    self.goal_visualizer.visualize(terrain_goals_w.reshape(-1, 3))
            # set visibility
            self.goal_visualizer.set_visibility(True)

        else:
            if hasattr(self, "origin_visualizer"):
                self.origin_visualizer.set_visibility(False)
            if hasattr(self, "goal_visualizer"):
                self.goal_visualizer.set_visibility(False)
        # report success
        return True


@configclass
class ParkourTerrainImporterCfg(TerrainImporterCfg):
    """Configuration for the parkour terrain importer."""

    class_type: type = ParkourTerrainImporter

    test: int = 0
