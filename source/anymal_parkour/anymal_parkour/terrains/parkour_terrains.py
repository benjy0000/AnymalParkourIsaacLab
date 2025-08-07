
"""Configuration for the Anymal Parkour terrain."""


from . import parkour_mesh_terrains_cfg as terrain_gen

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

PARKOUR_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "boxes": terrain_gen.MeshBoxTerrainCfg(
            proportion=1.0, box_height_range=(0.05, 1.0), platform_width=2.0, double_box=False
        )
    }
)