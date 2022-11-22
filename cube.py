from utils import *
from ray import *
from cli import render

oak = Material(vec([0, 0, 0]), texture=load_box_textures_simple(
  "log_oak_top.jpg", "log_oak.jpg"), k_a=vec([0, 0, 0]), normal_map=load_box_textures_simple("log_oak_top_n.jpg", "log_oak_n.jpg"))
gray = Material(vec([0.2, 0.2, 0.0]), 0.6, k_m=0.4)
diamond = Material(vec([0.2, 0.2, 0.2]), 0.6, k_m=0.5, texture=load_box_textures_simple(
  "diamond_block.jpg", "diamond_block.jpg"), normal_map=load_box_textures_simple("diamond_block_n.jpg", "diamond_block_n.jpg"))
dirt = Material(vec([0.2, 0.2, 0.0]), texture=load_box_textures_simple(
  "coarse_dirt.jpg", "coarse_dirt.jpg"), normal_map=load_box_textures_simple("coarse_dirt_n.jpg", "coarse_dirt_n.jpg"))
ice = Material(vec([0.2, 0.2, 0.2]), 0.6, k_m=0.2, texture=load_box_textures_simple("frosted_ice_0.jpg", "frosted_ice_0.jpg"), normal_map=load_box_textures_simple(
  "frosted_ice_0_n.jpg", "frosted_ice_0_n.jpg"), s_map=load_box_textures_simple("frosted_ice_0_s.jpg", "frosted_ice_0_s.jpg"))
redstone = Material(vec([1, 0, 0]), texture=load_box_textures_simple(
  "redstone_block.jpg", "redstone_block.jpg"),  k_a=vec([0, 0, 0]), normal_map=load_box_textures_simple("redstone_block_n.jpg", "redstone_block_n.jpg"), s_map=load_box_textures_simple("redstone_block_s.jpg", "redstone_block_s.jpg"))

# Read the triangle mesh for a 2x2x2 cube, and scale it down to 1x1x1 to fit the scene.
vs_list = 0.5 * read_obj_triangles(open("cube.obj"))

scene = Scene([
  # Make a big sphere for the floor
  # Sphere(vec([0,-40,0]), 39.5, gray),
  # Make a cube for the floor
  Cube(vec([1, 0, 0]), 1, oak),
  Cube(vec([-1, 0, 0]), 1, diamond),
  #Cube(vec([-0.25, -0.25, -0.25]), 0.5, dirt),
  Cube(vec([0, 0, 1]), 1, redstone),
] + [
  # Make triangle objects from the vertex coordinates
  # Triangle(vs, tan) for vs in vs_list
  Cube(vec([0, -10.5, 0]), 20, gray),
])

lights = [
  PointLight(vec([12, 10, 5]), vec([300, 300, 300])),
  AmbientLight(0.4),
]

camera = Camera(vec([3, 1.7, 5]), target=vec([0, 0, 0]),
                vfov=25, aspect=16/9)

render(camera, scene, lights)
