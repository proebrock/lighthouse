import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

# pip install trimesh "pyglet<2" rtree
import trimesh

sys.path.append(os.path.abspath('../'))
from common.mesh_utils import mesh_load, pcl_load



def o3d_mesh_to_trimesh(mesh):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if mesh.has_triangle_normals():
        triangle_normals = np.asarray(mesh.triangle_normals)
    else:
        triangle_normals = None
    if mesh.has_vertex_normals():
        vertex_normals = np.asarray(mesh.vertex_normals)
    else:
        vertex_normals = None
    if mesh.has_vertex_colors():
        vertex_colors = np.asarray(mesh.vertex_colors)
    else:
        vertex_colors = None
    tmesh = trimesh.Trimesh(vertices=vertices, faces=triangles,
        face_normals=triangle_normals, vertex_normals=vertex_normals,
        vertex_colors=vertex_colors)
    return tmesh



def trimesh_visualize(mesh):
    # requires: pip install "pyglet<2"
    scene = trimesh.Scene([ mesh ])
    scene.show()



def o3d_visualize(objects):
    cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for o in objects:
        vis.add_geometry(o)
    vis.get_render_option().background_color = (0.5, 0.5 ,0.5)
    vis.run()
    vis.destroy_window()



def o3d_mesh_pcl_signed_distance(mesh, pcl):
    tmesh = o3d_mesh_to_trimesh(mesh)
    #print(tmesh)
    #trimesh_visualize(tmesh)
    query = trimesh.proximity.ProximityQuery(tmesh)
    dists = query.signed_distance(np.asarray(pcl.points))
    return dists



def colorize_point_cloud_by_scalar(pcl, values, min_max=None, nan_color=(1, 0, 0)):
    assert values.ndim == 1
    assert np.asarray(pcl.points).shape[0] == values.size
    cm = plt.get_cmap('hot')
    isvalid = ~np.isnan(values)
    if min_max is None:
        min_max = (np.min(values[isvalid]), np.max(values[isvalid]))
    values_norm = np.clip((values[isvalid] - min_max[0]) / (min_max[1] - min_max[0]), 0, 1)
    colors = np.empty((np.asarray(pcl.points).shape[0], 3))
    colors[isvalid, :] = cm(values_norm)[:, 0:3]
    colors[~isvalid, :] = nan_color
    pcl.colors = o3d.utility.Vector3dVector(colors)



def colorize_point_cloud_by_scalar_bicolor(pcl, values, vmax=None):
    mask_pos = values >= 0
    neg = values[~mask_pos]
    pos = values[mask_pos]
    if vmax is None:
        if np.sum(~mask_pos) == 0:
            vmax = np.max(pos)
        elif np.sum(mask_pos) == 0:
            vmax = -np.min(neg)
        else:
            vmax = np.max((np.max(pos), -np.min(neg)))
    neg = neg / vmax
    pos = pos / vmax
    # [ -1 .. 0 ] -> [ 0 .. 0.5 ]
    neg = (1.0 + neg) / 2.0
    # [ 0 .. 1 ] -> [ 0.5 .. 1 ]
    pos = 0.5 + pos / 2.0
    cm = plt.get_cmap('seismic_r') # We want red=negative, blue=positive
    colors = np.empty((np.asarray(pcl.points).shape[0], 3))
    colors[~mask_pos] = cm(neg)[:, 0:3]
    colors[mask_pos] = cm(pos)[:, 0:3]
    pcl.colors = o3d.utility.Vector3dVector(colors)



if __name__ == "__main__":
    # Random but reproducible
    np.random.seed(42)
    # Get data path
    data_path_env_var = 'LIGHTHOUSE_DATA_DIR'
    if data_path_env_var in os.environ:
        data_dir = os.environ[data_path_env_var]
        data_dir = os.path.join(data_dir, '2d_calibrate_extrinsics')
    else:
        data_dir = 'data'
    data_dir = os.path.abspath(data_dir)
    print(f'Using data from "{data_dir}"')

    # Load mesh and point cloud
    filename = os.path.join(data_dir, 'mesh.ply')
    mesh = mesh_load(filename)
    filename = os.path.join(data_dir, 'point_cloud.ply')
    pcl = pcl_load(filename)

    dists = o3d_mesh_pcl_signed_distance(mesh, pcl)
    # Trimesh counts points "outside" of mesh negative,
    # we want that the other way around!
    dists = -dists
    print(f'Distances of {np.min(dists):e}..{np.max(dists):e} found')
    dist_rms = np.sqrt(np.mean(np.square(dists)))
    print(f'Distance RMS is {dist_rms:.3f}mm')

    #colorize_point_cloud_by_scalar(pcl, np.abs(dists))
    colorize_point_cloud_by_scalar_bicolor(pcl, dists)
    o3d_visualize([pcl])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.boxplot([dists], labels=['original'])
    ax.set_ylabel('point to nearest point distance')
    plt.show()
