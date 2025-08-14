
"""Configuration for the Anymal Parkour terrain."""


from . import parkour_mesh_terrains_cfg as terrain_gen

from .parkour_terrain_generator import ParkourTerrainGeneratorCfg

NUM_GOALS = 8  # Number of goals to be placed in the terrain

PARKOUR_TERRAINS_CFG = ParkourTerrainGeneratorCfg(
    size=(4.0, 20.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    num_goals=NUM_GOALS,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "large_steps": terrain_gen.MeshLargeStepsTerrainCfg(
            proportion=1.0,
            num_goals=NUM_GOALS,
            step_height_range=(0.1, 0.8),
            step_length_range=(0.5, 1.5),
            step_width_range=(1.4, 2.0),
            step_mismatch_range=(-0.4, 0.4)
        ),
        "boxes": terrain_gen.MeshBoxesTerrainCfg(
            proportion=1.0,
            num_goals=NUM_GOALS,
            box_width=2.0,
            box_length=1.0,
            box_height_range=(0.1, 1.0),
            box_x_offset_range=(0, 0.15),
        ),
        "stairs": terrain_gen.MeshStairsTerrainCfg(
            proportion=0.0,
            num_goals=NUM_GOALS,
            num_steps=6,
            tread_range=(0.3, 0.5),
            riser_range=(0.05, 0.25),
        ),
        "gaps": terrain_gen.MeshGapsTerrainCfg(
            proportion=0.0,
            num_goals=NUM_GOALS,
            gap_length_range=(0.1, 1.0),
            stone_x_offset_range=(0.0, 0.15),
            stone_length=1.2
        ),
        "inclined_boxes": terrain_gen.MeshInclinedBoxesTerrainCfg(
            proportion=1.0,
            num_goals=NUM_GOALS,
            platform_length=2.0,
            pit_depth=(0.2, 0.95)
        ),
        "weave_poles": terrain_gen.MeshWeavePoleTerrainCfg(
            proportion=1.0,
            num_goals=NUM_GOALS,
            robot_x_clearance_range=(0.2, 0.4),
            pole_radius=0.12,
            pole_x_range=(0.15, 0.65),
            pole_x_noise=0.05,
            pole_y_range=(1.1, 1.7),
            pole_height_range=(0.75, 1.5)
        ),
        "flat_terrain": terrain_gen.MeshFlatTerrainCfg(
            proportion=1.0,
            num_goals=NUM_GOALS,
            x_range=1.2,
            surface_roughness=False
        )
    }
)