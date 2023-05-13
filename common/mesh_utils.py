import h5py
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

import cv2



def mesh_transform(mesh, trafo):
    mesh.transform(trafo.get_homogeneous_matrix())



def mesh_generate_cs(trafo, size=1.0):
    coordinate_system = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    coordinate_system.transform(trafo.get_homogeneous_matrix())
    return coordinate_system



def mesh_visualize_with_normals(mesh, normal_scale=1.0):
    """ Visualize a mesh with its normals

    We extract a point cloud from the vertices and normals
    of the mesh and display the point cloud and the mesh along
    with each other. This way we can display the mesh's normals
    which is not directly possible in Open3D.

    :param normal_scale: Scaling factor for normals
    """
    # Generate Point cloud
    pcl = o3d.geometry.PointCloud()
    points = np.asarray(mesh.vertices)
    pcl.points = o3d.utility.Vector3dVector(points)
    normals = np.asarray(mesh.vertex_normals)
    normals = normal_scale * normals
    pcl.normals = o3d.utility.Vector3dVector(normals)
    # Visualize mesh and point cloud
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcl)
    vis.add_geometry(mesh)
    vis.get_render_option().point_show_normal = True
#    vis.get_render_option().point_color_option = o3d.visualization.PointColorOption.Normal
    vis.run()
    vis.destroy_window()



def mesh_generate_plane(shape, color=(0, 0, 0)):
    """ Generate plane

    The plane is represented by two triangles

    Y
       /\
       |
       |-----------
       |          |
       |  Plane   |
       |          |
       .------------->
    Z                  X

    """
    vertices = np.array([
        [0, 0, 0],
        [shape[0], 0, 0],
        [shape[0], shape[1], 0],
        [0, shape[1], 0]])
    vertex_normals = np.zeros((4, 3))
    vertex_normals[:, 2] = 1.0
    vertex_colors = np.array([color, color, color, color])
    triangles = np.array([[0, 1, 3], [2, 3, 1]], dtype=int)
    triangle_normals = np.zeros((2, 3))
    triangle_normals[:, 2] = 1.0

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.vertex_normals = o3d.utility.Vector3dVector(vertex_normals)
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.triangle_normals = o3d.utility.Vector3dVector(triangle_normals)
    return mesh



def mesh_generate_image_file(filename, pixel_size=1.0, scale=1.0):
    """ Load image file from disk and generate mesh from it
    :param filename: Filename (and path) of image file
    :param pixel_size: Size of one pixel in millimeter; can be single value for square
        pixels or a tupel of two value, first pixel size in x, second pixel size in y;
        unit is millimeters per pixel
    :param scale: Scale image with this factor before converting into mesh
    """
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    if img is None:
        raise Exception(f'Unable to read image file "{filename}"')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return mesh_generate_image(img, pixel_size)



def mesh_generate_image(img, pixel_size=1.0):
    """ Convert raster image into mesh

    Each pixel of the image is encoded as four vertices and two triangles
    that creates a square region in the mesh of size pixel_size x pixel_size.
    This creates duplicate vertices, but we need to use the vertex colors
    to encode the colors, hence 4 unique vertices for each pixel.
    The image is placed in the X/Y plane.

    Z                  X
       X------------->
       |           |
       |  Image    |
       |           |
       |------------
       |
    Y  V

    :param img: Input image
    :param pixel_size: Size of one pixel in millimeter; can be single value for square
        pixels or a tupel of two values, first pixel size in x, second pixel size in y;
        unit is millimeters per pixel
    """
    if isinstance(pixel_size, float):
        pixel_sizes = np.array((pixel_size, pixel_size))
    else:
        pixel_sizes = np.asarray(pixel_size)
        assert pixel_sizes.size == 2
    vertices = np.zeros((4 * img.shape[0] * img.shape[1], 3))
    vertex_normals = np.zeros((4 * img.shape[0] * img.shape[1], 3))
    vertex_normals[:, 2] = -1.0
    vertex_colors = np.zeros((4 * img.shape[0] * img.shape[1], 3))
    triangles = np.zeros((2 * img.shape[0] * img.shape[1], 3), dtype=int)
    triangle_normals = np.zeros((2 * img.shape[0] * img.shape[1], 3))
    triangle_normals[:, 2] = -1.0
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            i = 4 * (r * img.shape[1] + c)
            vertices[i, 0:2]   = pixel_sizes[0] * c,     pixel_sizes[1] * r
            vertices[i+1, 0:2] = pixel_sizes[0] * c,     pixel_sizes[1] * (r+1)
            vertices[i+2, 0:2] = pixel_sizes[0] * (c+1), pixel_sizes[1] * r
            vertices[i+3, 0:2] = pixel_sizes[0] * (c+1), pixel_sizes[1] * (r+1)
            vertex_colors[i:i+4, :] = img[r, c, :] / 255.0
            j = 2 * (r * img.shape[1] + c)
            triangles[j, :]   = i, i+1,   i+3
            triangles[j+1, :] = i+3, i+2, i
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.vertex_normals = o3d.utility.Vector3dVector(vertex_normals)
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.triangle_normals = o3d.utility.Vector3dVector(triangle_normals)
    return mesh



def mesh_generate_surface(fun, xrange, yrange, num, scale):
    """ Use 3D function to generate a patch of 3D surface

    Y
       /\
       |
       |------------
       |           |
       |  Surface  |
       |           |
       .------------->
    Z                  X

    xrange and yrange determine the domain the fun is evalued in.
    The resulting surface is scaled in X and Y to [0, 1] and then
    finally scaled with scale.

    :param fun: Callable function, input X, Y in shape (n, 2),
        output Z in shape (n, )
    :param xrange: Real range in X: (xmin, xmax)
    :param yrange: Real range in Y: (ymin, ymax)
    :param num: Number of pixels (num_x, num_y)
    :param scale: Scale to final size (xscale, yscale, zscale)
    """
    mesh = o3d.geometry.TriangleMesh()
    # Generate vertices
    x = np.linspace(xrange[0], xrange[1], num[0])
    y = np.linspace(yrange[0], yrange[1], num[1])
    x, y = np.meshgrid(x, y, indexing='ij')
    vertices = np.zeros((num[0] * num[1], 3))
    vertices[:, 0] = x.ravel()
    vertices[:, 1] = y.ravel()
    vertices[:, 2] = scale[2] * fun(vertices[:, 0:2])
    vertices[:, 0] = (scale[0] * (vertices[:, 0] - xrange[0])) / (xrange[1] - xrange[0])
    vertices[:, 1] = (scale[1] * (vertices[:, 1] - yrange[0])) / (yrange[1] - yrange[0])
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    # Generate triangles
    triangles = np.zeros((2 * (num[0] - 1) * (num[1] - 1), 3), dtype=int)
    tindex = 0
    for iy in range(num[1] - 1):
        for ix in range(num[0] - 1):
            i = iy * num[0] + ix
            triangles[tindex, :] = i+1, i, i+num[0]
            tindex += 1
            triangles[tindex, :] = i+1, i+num[0], i+num[0]+1
            tindex += 1
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    # Default color is "white"
    vertex_colors = np.ones_like(vertices)
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    # Calculate normals
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh
