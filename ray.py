import numpy as np

from utils import *

"""
Core implementation of the ray tracer.  This module contains the classes (Sphere, Mesh, etc.)
that define the contents of scenes, as well as classes (Ray, Hit) and functions (shade) used in
the rendering algorithm, and the main entry point `render_image`.

In the documentation of these classes, we indicate the expected types of arguments with a
colon, and use the convention that just writing a tuple means that the expected type is a
NumPy array of that shape.  Implementations can assume these types are preconditions that
are met, and if they fail for other type inputs it's an error of the caller.  (This might
not be the best way to handle such validation in industrial-strength code but we are adopting
this rule to keep things simple and efficient.)
"""


class Ray:

  def __init__(self, origin, direction, start=0., end=np.inf):
    """Create a ray with the given origin and direction.

    Parameters:
      origin : (3,) -- the start point of the ray, a 3D point
      direction : (3,) -- the direction of the ray, a 3D vector (not necessarily normalized)
      start, end : float -- the minimum and maximum t values for intersections
    """
    # Convert these vectors to double to help ensure intersection
    # computations will be done in double precision
    self.origin = np.array(origin, np.float64)
    self.direction = np.array(direction, np.float64)
    self.start = start
    self.end = end


class Material:

  def __init__(self, k_d, k_s=0., p=20., k_m=0., k_a=None, texture=None):
    """Create a new material with the given parameters.

    Parameters:
      k_d : (3,) -- the diffuse coefficient
      k_s : (3,) or float -- the specular coefficient
      p : float -- the specular exponent
      k_m : (3,) or float -- the mirror reflection coefficient
      k_a : (3,) -- the ambient coefficient (defaults to match diffuse color)
      texture : Texture -- a texture to apply to the material
    """
    self.k_d = k_d
    self.k_s = k_s
    self.p = p
    self.k_m = k_m
    self.k_a = k_a if k_a is not None else k_d
    self.texture = texture


class Hit:

  def __init__(self, t, point=None, normal=None, material=None):
    """Create a Hit with the given data.

    Parameters:
      t : float -- the t value of the intersection along the ray
      point : (3,) -- the 3D point where the intersection happens
      normal : (3,) -- the 3D outward-facing unit normal to the surface at the hit point
      material : (Material) -- the material of the surface
    """
    self.t = t
    self.point = point
    self.normal = normal
    self.material = material


# Value to represent absence of an intersection
no_hit = Hit(np.inf)


class Sphere:

  def __init__(self, center, radius, material):
    """Create a sphere with the given center and radius.

    Parameters:
      center : (3,) -- a 3D point specifying the sphere's center
      radius : float -- a Python float specifying the sphere's radius
      material : Material -- the material of the surface
    """
    self.center = center
    self.radius = radius
    self.material = material

  def intersect(self, ray):
    """Computes the first (smallest t) intersection between a ray and this sphere.

    Parameters:
      ray : Ray -- the ray to intersect with the sphere
    Return:
      Hit -- the hit data
    """
    p = self.center - ray.origin
    d = ray.direction
    tc = np.dot(p, d) / np.dot(d, d)
    lm = p-tc*d
    lm2 = np.dot(lm, lm)
    if lm2 > self.radius * self.radius:
      return no_hit

    dt = np.sqrt(self.radius ** 2 - lm2) / np.linalg.norm(d)
    t0 = tc - dt
    t1 = tc + dt

    if t0 > ray.start and t0 < ray.end:
      hit_point = ray.origin + t0 * d
      nvec = (hit_point - self.center) / self.radius
      # if np.dot(nvec, d) > 0:
      #     nvec = -nvec
      mat = self.material
      if self.material.texture is not None:
        mat = Material(self.spherical_texture(nvec), k_s=self.material.k_s,
                       p=self.material.p, k_m=self.material.k_m, k_a=self.material.k_a, texture=self.material.texture)
      return Hit(t0, hit_point, nvec, mat)
    elif t1 > ray.start and t1 < ray.end:
      hit_point = ray.origin + t1 * d
      nvec = (hit_point - self.center) / self.radius
      mat = self.material
      if self.material.texture is not None:
        mat = Material(self.spherical_texture(nvec), k_s=self.material.k_s,
                       p=self.material.p, k_m=self.material.k_m, k_a=self.material.k_a, texture=self.material.texture)
      # if np.dot(nvec, d) > 0:
      #     nvec = -nvec
      return Hit(t1, hit_point, nvec, mat)
    return no_hit

  def spherical_texture(self, normal):
    """Computes the spherical texture coordinates for a given hit point.

    Parameters:
      hit : Hit -- the hit data
    Return:
      (u, v) : (float, float) -- the texture coordinates
    """
    n = normal
    u = 0.5 - np.arctan2(n[0], n[2]) / (2 * np.pi)
    v = 0.5 - np.arcsin(n[1]) / np.pi
    u, v = int(
      u*self.material.texture.shape[1]), int(v*self.material.texture.shape[0])
    return self.material.texture[v, u]/255


class Cube:
  def __init__(self, center, size, material):
    self.center = center
    self.size = size
    self.material = material

  def intersect(self, ray):
    """Computes the intersection between a ray and this triangle, if it exists.

    Parameters:
      ray : Ray -- the ray to intersect with the triangle
    Return:
      Hit -- the hit data
    """
    tmin = (self.center[0] - self.size/2 - ray.origin[0]) / ray.direction[0]
    tmax = (self.center[0] + self.size/2 - ray.origin[0]) / ray.direction[0]

    if tmin > tmax:
      tmin, tmax = tmax, tmin

    tymin = (self.center[1] - self.size/2 - ray.origin[1]) / ray.direction[1]
    tymax = (self.center[1] + self.size/2 - ray.origin[1]) / ray.direction[1]

    if tymin > tymax:
      tymin, tymax = tymax, tymin

    if (tmin > tymax) or (tymin > tmax):
      return no_hit

    if tymin > tmin:
      tmin = tymin

    if tymax < tmax:
      tmax = tymax

    tzmin = (self.center[2] - self.size/2 - ray.origin[2]) / ray.direction[2]
    tzmax = (self.center[2] + self.size/2 - ray.origin[2]) / ray.direction[2]

    if tzmin > tzmax:
      tzmin, tzmax = tzmax, tzmin

    if (tmin > tzmax) or (tzmin > tmax):
      return no_hit

    if tzmin > tmin:
      tmin = tzmin

    if tzmax < tmax:
      tmax = tzmax

    if tmin < ray.start or tmin > ray.end:
      return no_hit

    hit_point = ray.origin + tmin * ray.direction
    kd = np.zeros(3).astype(np.float32)
    i, u, v = 0, 0, 0
    nvec = np.array([0, 0, 0])
    if abs(hit_point[0] - (self.center[0] - self.size/2)) < 0.0001:
      nvec = np.array([-1, 0, 0])
      v = (hit_point[2] - (self.center[2] - self.size/2)) / self.size
      u = (hit_point[1] - (self.center[1] - self.size/2)) / self.size
      i = 0
    elif abs(hit_point[0] - (self.center[0] + self.size/2)) < 0.0001:
      nvec = np.array([1, 0, 0])
      v = (hit_point[1] - (self.center[1] - self.size/2)) / self.size
      u = (hit_point[2] - (self.center[2] - self.size/2)) / self.size
      i = 1
    elif abs(hit_point[1] - (self.center[1] - self.size/2)) < 0.0001:
      nvec = np.array([0, -1, 0])
      u = (hit_point[0] - (self.center[0] - self.size/2)) / self.size
      v = (hit_point[2] - (self.center[2] - self.size/2)) / self.size
      i = 2
    elif abs(hit_point[1] - (self.center[1] + self.size/2)) < 0.0001:
      nvec = np.array([0, 1, 0])
      u = (hit_point[0] - (self.center[0] - self.size/2)) / self.size
      v = (hit_point[2] - (self.center[2] - self.size/2)) / self.size
      i = 3
    elif abs(hit_point[2] - (self.center[2] - self.size/2)) < 0.0001:
      nvec = np.array([0, 0, -1])
      u = (hit_point[0] - (self.center[0] - self.size/2)) / self.size
      v = (hit_point[1] - (self.center[1] - self.size/2)) / self.size
      i = 4
    elif abs(hit_point[2] - (self.center[2] + self.size/2)) < 0.0001:
      nvec = np.array([0, 0, 1])
      u = (hit_point[0] - (self.center[0] - self.size/2)) / self.size
      v = (hit_point[1] - (self.center[1] - self.size/2)) / self.size
      i = 5

    if self.material.texture is not None:
      u, v = int(
        u*self.material.texture[0].shape[1]), int(v*self.material.texture[0].shape[0])
      # clamp u,v
      u = np.clip(u, 0, self.material.texture[0].shape[1]-1)
      v = np.clip(v, 0, self.material.texture[0].shape[0]-1)
      kd = self.material.texture[i][v, u]/255
    else:
      kd = self.material.k_d

    mat = Material(kd, k_s=self.material.k_s,
                   p=self.material.p, k_m=self.material.k_m, k_a=self.material.k_a, texture=self.material.texture)
    return Hit(tmin, hit_point, nvec, mat)


class Triangle:

  def __init__(self, vs, material):
    """Create a triangle from the given vertices.

    Parameters:
      vs (3,3) -- an arry of 3 3D points that are the vertices (CCW order)
      material : Material -- the material of the surface
    """
    self.vs = vs
    self.material = material

  def intersect(self, ray):
    """Computes the intersection between a ray and this triangle, if it exists.

    Parameters:
      ray : Ray -- the ray to intersect with the triangle
    Return:
      Hit -- the hit data
    """
    # TODO A4 implement this function
    # return no_hit
    p = ray.origin
    d = ray.direction
    v0 = self.vs[0]
    v1 = self.vs[1]
    v2 = self.vs[2]
    n = np.cross(v1 - v0, v0 - v2)
    t = np.dot(v0 - p, n) / np.dot(d, n)
    if t < ray.start or t > ray.end:
      return no_hit
    hitpoint = p + t * d
    s1 = np.dot(np.cross(v1 - v0, hitpoint - v0), n)
    s2 = np.dot(np.cross(v2 - v1, hitpoint - v1), n)
    s3 = np.dot(np.cross(v0 - v2, hitpoint - v2), n)
    if ((s1 >= 0) and (s2 >= 0) and (s3 >= 0)) or ((s1 <= 0) and (s2 <= 0) and (s3 <= 0)):
      nvec = n / np.linalg.norm(n)
      if np.dot(nvec, ray.direction) > 0:
        nvec = -nvec
      return Hit(t, hitpoint, nvec, self.material)
    return no_hit


class Camera:

  def __init__(self, eye=vec([0, 0, 0]), target=vec([0, 0, -1]), up=vec([0, 1, 0]),
               vfov=90.0, aspect=1.0):
    """Create a camera with given viewing parameters.

    Parameters:
      eye : (3,) -- the camera's location, aka viewpoint (a 3D point)
      target : (3,) -- where the camera is looking: a 3D point that appears centered in the view
      up : (3,) -- the camera's orientation: a 3D vector that appears straight up in the view
      vfov : float -- the full vertical field of view in degrees
      aspect : float -- the aspect ratio of the camera's view (ratio of width to height)
    """
    self.eye = eye
    self.aspect = aspect
    self.target = target
    self.up = up
    self.vfov = vfov
    # TODO A4 implement this constructor to store whatever you need for ray generation
    cvec = (self.target - self.eye) / np.linalg.norm(self.target - self.eye)
    upv = self.up / np.linalg.norm(self.up)
    rvec = np.cross(cvec, upv) / np.linalg.norm(np.cross(cvec, upv))
    uvec = np.cross(rvec, cvec) / np.linalg.norm(np.cross(rvec, cvec))

    left_buttom = cvec - uvec * \
      np.tan(self.vfov * np.pi / 180 / 2) - rvec * \
      np.tan(self.vfov * np.pi / 180 / 2) * self.aspect
    up_range = 2 * np.tan(self.vfov * np.pi / 180 / 2) * uvec
    right_range = 2 * np.tan(self.vfov * np.pi / 180 / 2) * rvec * self.aspect

    self.left_buttom = left_buttom
    self.up_range = up_range
    self.right_range = right_range

  def generate_ray(self, img_point):
    """Compute the ray corresponding to a point in the image.

    Parameters:
      img_point : (2,) -- a 2D point in [0,1] x [0,1], where (0,0) is the lower left
                  corner of the image and (1,1) is the upper right
    Return:
      Ray -- The ray corresponding to that image location (not necessarily normalized)
    """
    # TODO A4 implement this function
    # return Ray(vec([0, 0, 0]), vec([0, 0, 1]))
    p = self.left_buttom + img_point[1] * \
      self.up_range + img_point[0] * self.right_range
    return Ray(self.eye, p, 1e-6)


class PointLight:

  def __init__(self, position, intensity):
    """Create a point light at given position and with given intensity

    Parameters:
      position : (3,) -- 3D point giving the light source location in scene
      intensity : (3,) or float -- RGB or scalar intensity of the source
    """
    self.position = position
    self.intensity = intensity

  def illuminate(self, ray, hit, scene):
    """Compute the shading at a surface point due to this light.

    Parameters:
      ray : Ray -- the ray that hit the surface
      hit : Hit -- the hit data
      scene : Scene -- the scene, for shadow rays
    Return:
      (3,) -- the light reflected from the surface
    """
    # TODO A4 implement this function
    # return vec([0, 0, 0])
    surfs = scene.surfs
    iL = self.position - hit.point  # inverse light vector
    n = normalize(hit.normal)
    point_ray = Ray(self.position, -iL)
    distance = np.linalg.norm(iL)
    for surf in surfs:
      hit2 = surf.intersect(point_ray)
      if (hit2.t != np.inf) and (np.linalg.norm(hit2.point - self.position) < distance - 1e-6):
        return vec([0, 0, 0])
    cosd = np.dot(n, iL) / distance
    # I_d = self.intensity / (distance * distance) * hit.material.k_d * max(0, cosd)
    h = normalize(normalize(iL) + normalize(-ray.direction))
    coss = np.dot(n, h)
    Lr = (hit.material.k_d + hit.material.k_s * (coss**hit.material.p)) * \
      self.intensity / (distance * distance) * max(0, cosd)
    return Lr


class AmbientLight:

  def __init__(self, intensity):
    """Create an ambient light of given intensity

    Parameters:
      intensity (3,) or float: the intensity of the ambient light
    """
    self.intensity = intensity

  def illuminate(self, ray, hit, scene):
    """Compute the shading at a surface point due to this light.

    Parameters:
      ray : Ray -- the ray that hit the surface
      hit : Hit -- the hit data
      scene : Scene -- the scene, for shadow rays
    Return:
      (3,) -- the light reflected from the surface
    """
    # TODO A4 implement this function
    # return vec([0, 0, 0])
    return self.intensity * hit.material.k_a


class Scene:

  def __init__(self, surfs, bg_color=vec([0.2, 0.3, 0.5])):
    """Create a scene containing the given objects.

    Parameters:
      surfs : [Sph ere, Triangle] -- list of the surfaces in the scene
      bg_color : (3,) -- RGB color that is seen where no objects appear
    """
    self.surfs = surfs
    self.bg_color = bg_color

  def intersect(self, ray):
    """Computes the first (smallest t) intersection between a ray and the scene.

    Parameters:
      ray : Ray -- the ray to intersect with the scene
    Return:
      Hit -- the hit data
    """
    # TODO A4 implement this function
    # return no_hit

    surfs = self.surfs
    hit = no_hit
    for surf in surfs:
      h = surf.intersect(ray)
      if h.t < hit.t:
        hit = h
    return hit


MAX_DEPTH = 4


def shade(ray, hit, scene, lights, depth=0):
  """Compute shading for a ray-surface intersection.

  Parameters:
    ray : Ray -- the ray that hit the surface
    hit : Hit -- the hit data
    scene : Scene -- the scene
    lights : [PointLight or AmbientLight] -- the lights
    depth : int -- the recursion depth so far
  Return:
    (3,) -- the color seen along this ray
  When mirror reflection is being computed, recursion will only proceed to a depth
  of MAX_DEPTH, with zero contribution beyond that depth.
  """
  # TODO A4 implement this function
  # return vec([0, 0, 0])
  if hit.t == np.inf:
    return scene.bg_color
  L_ar = vec([0, 0, 0])
  for light in lights:
    L_ar += light.illuminate(ray, hit, scene)
  if hit.material.k_m > 0 and depth > 0:
    ray_out = Ray(hit.point, ray.direction - 2 *
                  np.dot(ray.direction, hit.normal) * hit.normal, 1e-6)
    hit2 = scene.intersect(ray_out)
    # if hit2.t != np.inf:
    L_ar += hit.material.k_m * shade(ray_out, hit2, scene, lights, depth - 1)
    # else:
    #     L_ar += hit.material.k_m * scene.bg_color
  return L_ar


def render_image(camera, scene, lights, nx, ny):
  """Render a ray traced image.

  Parameters:
    camera : Camera -- the camera defining the view
    scene : Scene -- the scene to be rendered
    lights : Lights -- the lights illuminating the scene
    nx, ny : int -- the dimensions of the rendered image
  Returns:
    (ny, nx, 3) float32 -- the RGB image
  """
  # TODO A4 implement this function
  # return np.zeros((ny, nx, 3), np.float32)
  image = np.zeros((ny, nx, 3), dtype=np.float32)
  offset_x = 1 / (2 * nx)
  offset_y = 1 / (2 * ny)
  offset_x2 = offset_x / 2
  offset_y2 = offset_y / 2
  num_pixel = nx * ny

  for i in range(nx):
    for j in range(ny):
      # ray = camera.generate_ray([i / nx + offset_x, j / ny + offset_y])
        # image[j, i] = shade(ray, scene.intersect(ray), scene, lights, MAX_DEPTH)
      if(i*ny+j) % 1000 == 0:
        print("Rendered: %f" % ((i*ny+j)/num_pixel), end="\r")
      ray1 = camera.generate_ray(
        [i / nx + offset_x - offset_x2, j / ny + offset_y - offset_y2])
      ray2 = camera.generate_ray(
        [i / nx + offset_x + offset_x2, j / ny + offset_y - offset_y2])
      ray3 = camera.generate_ray(
        [i / nx + offset_x - offset_x2, j / ny + offset_y + offset_y2])
      ray4 = camera.generate_ray(
        [i / nx + offset_x + offset_x2, j / ny + offset_y + offset_y2])
      s1 = shade(ray1, scene.intersect(ray1), scene, lights, MAX_DEPTH)
      s2 = shade(ray2, scene.intersect(ray2), scene, lights, MAX_DEPTH)
      s3 = shade(ray3, scene.intersect(ray3), scene, lights, MAX_DEPTH)
      s4 = shade(ray4, scene.intersect(ray4), scene, lights, MAX_DEPTH)
      image[j, i] = (s1 + s2 + s3 + s4) / 4
  return np.clip(image, 0, 1)
