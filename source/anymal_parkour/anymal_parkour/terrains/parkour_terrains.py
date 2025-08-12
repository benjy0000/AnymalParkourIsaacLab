
"""Configuration for the Anymal Parkour terrain."""


from . import parkour_mesh_terrains_cfg as terrain_gen

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

PARKOUR_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(4.0, 18.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "boxes": terrain_gen.MeshBoxTerrainCfg(
            proportion=0, box_height_range=(0.05, 1.0), platform_width=2.0, double_box=False
        ),
        "large_steps": terrain_gen.MeshLargeStepsTerrainCfg(
            proportion=0,
            num_steps=6,
            step_height_range=(0.1, 0.8),
            platform_length=2.0,
            step_length_range=(0.5, 1.5),
            step_width_range=(1.4, 2.0),
            step_mismatch_range=(-0.4, 0.4)
        ),
        "boxes": terrain_gen.MeshBoxesTerrainCfg(
            proportion=0.4,
            num_boxes=4,
            box_width=2.0,
            box_length=1.0,
            box_height_range=(0.1, 1.0),
            box_x_offset_range=(0, 0.15),
        ),
        "stairs": terrain_gen.MeshStairsTerrainCfg(
            proportion=0.2,
            num_steps=6,
            tread_range=(0.3, 0.5),
            riser_range=(0.05, 0.25),
            platform_length=2.0
        ),
        "gaps": terrain_gen.MeshGapsTerrainCfg(
            proportion=0.4,
            num_gaps=7,
            gap_length_range=(0.1, 1.0),
            platform_length=2.0,
            stone_x_offset_range=(0.0, 0.15),
            stone_length=1.2
        ),
        "inclined_boxes": terrain_gen.MeshInclinedBoxesTerrainCfg(
            proportion=0.4,
            num_stones=6,
            platform_length=2.0,
            pit_depth=(0.2, 0.95)
        ),
        "weave_poles": terrain_gen.MeshWeavePoleTerrainCfg(
            proportion=0.0,
            num_poles=8,
            pole_radius=0.12,
            platform_length=2.0,
            pole_x_range=(0.15, 0.65),
            pole_x_noise=0.05,
            pole_y_range=(1.1, 1.7),
            pole_height_range=(0.75, 1.5)
        ),
        "flat_terrain": terrain_gen.MeshFlatTerrainCfg(
            proportion=0.0,
            platform_length=2.0
        )
    }
)