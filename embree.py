import open3d as o3d
import matplotlib.pyplot as plt

# Create scene and add a cube
cube = o3d.t.geometry.TriangleMesh.from_legacy(o3d.geometry.TriangleMesh.create_box())
scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(cube)

# Use a helper function to create rays for a pinhole camera.
rays = scene.create_rays_pinhole(fov_deg=60, center=[0.5,0.5,0.5], eye=[-1,-1,-1], up=[0,0,1],
                               width_px=320, height_px=240)
print(rays.numpy().shape) # (240, 320, 6)
print(type(rays)) # <class 'open3d.cpu.pybind.core.Tensor'>

rays_numpy = rays.numpy() # to numpy ...
rays = o3d.core.Tensor(rays_numpy) # ... and back
print(type(rays)) # <class 'open3d.cpu.pybind.core.Tensor'>, again - great!

# Compute the ray intersections and visualize the hit distance (depth)
ans = scene.cast_rays(rays)
print(ans['primitive_uvs'].numpy().shape) # (240, 320, 2)
print(ans['primitive_uvs'].numpy().dtype) # float32
print(ans['primitive_ids'].numpy().shape) # (240, 320)
print(ans['primitive_ids'].numpy().dtype) # uint32
print(ans['geometry_ids'].numpy().shape) # (240, 320)
print(ans['geometry_ids'].numpy().dtype) # uint32
print(ans['primitive_normals'].numpy().shape) # (240, 320, 3)
print(ans['primitive_normals'].numpy().dtype) # float32
print(ans['t_hit'].numpy().shape) # (240, 320)
print(ans['t_hit'].numpy().dtype) # float32

plt.imshow(ans['t_hit'].numpy())
plt.show()