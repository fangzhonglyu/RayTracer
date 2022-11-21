from utils import *
from ray import *
from cli import render

tan = Material(vec([0.7, 0.7, 0.4]), 0.1, p=20.,
               texture=load_texture("mignon.jpg"), k_m=0.3)
blue = Material(vec([0.2, 0.2, 0.5]), k_m=0.5)
gray = Material(vec([0.2, 0.2, 0.2]), k_m=0.6)

scene = Scene([
  Sphere(vec([0.7, 0, 0]), 0.5, tan),
  Sphere(vec([0, 0, 0.7]), 0.5, blue),
  Sphere(vec([0, -40, 0]), 39.5, gray),
])

lights = [
  PointLight(vec([-12, 10, -5]), vec([300, 300, 300])),
  AmbientLight(0.1),
]

camera = Camera(vec([-4, 1.7, -1.5]), target=vec([0, 0, 0]),
                vfov=25, aspect=16/9)

render(camera, scene, lights)
