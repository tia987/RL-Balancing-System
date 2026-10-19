import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

TWB_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,          # difficulty levels (curriculum moves along rows)
    num_cols=20,          # variations per level
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.3),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.4, noise_range=(0.0, 0.03), noise_step=0.005, border_width=0.25
        ),
        "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.3, slope_range=(0.0, 0.25), platform_width=2.0, border_width=0.25
        ),
        # "slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
        #     proportion=0.2, slope_range=(0.0, 0.25), platform_width=2.0, border_width=0.25
        # ),
        # "obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
        #     proportion=0.7,
        #     obstacle_height_mode="choice",
        #     obstacle_width_range=(0.2, 0.8),
        #     obstacle_height_range=(0.01, 0.06),
        #     num_obstacles=30,
        #     platform_width=2.0,
        # ),
    },
)
