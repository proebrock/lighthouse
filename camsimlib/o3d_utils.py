import cv2
import cv2.aruco as aruco
import h5py
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d



def mesh_transform(mesh, T):
    mesh.rotate(T.get_rotation_matrix(), center=(0,0,0))
    mesh.translate(T.get_translation())



def mesh_generate_cs(T, size=1.0):
    cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    cs.rotate(T.get_rotation_matrix(), center=(0, 0, 0))
    cs.translate(T.get_translation())
    return cs



def mesh_generate_plane(shape, color=(0,0,0)):
    """ Generate plane
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



def mesh_generate_image_file(filename, pixel_size=1.0):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img is None:
        raise Exception(f'Unable to read image file "{filename}"')
    return mesh_generate_image(img, pixel_size)



def mesh_generate_image(img, pixel_size=1.0):
    """ Convert raster image into mesh

    Each pixel of the image is encoded as four vertices and two triangles
    that create a square region in the mesh of size pixel_size x pixel_size.
    Colors are encoded using vertex colors.
    The image is placed in the X/Y plane.
    Y
       /\
       |
       |-----------
       |          |
       |  Image   |
       |          |
       .------------->
    Z                  X

    :param img: Input image
    :param pixel_size: Side length of square in mesh that represents one pixel of img
    """
    vertices = np.zeros((4 * img.shape[0] * img.shape[1], 3))
    vertex_normals = np.zeros((4 * img.shape[0] * img.shape[1], 3))
    vertex_normals[:,2] = 1.0
    vertex_colors = np.zeros((4 * img.shape[0] * img.shape[1], 3))
    triangles = np.zeros((2 * img.shape[0] * img.shape[1], 3), dtype=int)
    triangle_normals = np.zeros((2 * img.shape[0] * img.shape[1], 3))
    triangle_normals[:,2] = 1.0
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            i = 4 * (r * img.shape[1] + c)
            vertices[i  ,0:2] = pixel_size * c,     pixel_size * r
            vertices[i+1,0:2] = pixel_size * c,     pixel_size * (r+1)
            vertices[i+2,0:2] = pixel_size * (c+1), pixel_size * r
            vertices[i+3,0:2] = pixel_size * (c+1), pixel_size * (r+1)
            vertex_colors[i:i+4,:] = img[img.shape[0]-r-1,c,:] / 255.0
            j = 2 * (r * img.shape[1] + c)
            triangles[j  ,:] = i+1, i, i+3
            triangles[j+1,:] = i+2, i+3, i
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.vertex_normals = o3d.utility.Vector3dVector(vertex_normals)
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.triangle_normals = o3d.utility.Vector3dVector(triangle_normals)
    return mesh



def mesh_generate_charuco_board(squares, square_length):
    # Generate ChArUco board
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    marker_length = square_length / 2.0
    board = aruco.CharucoBoard_create(squares[0], squares[1],
        square_length, marker_length, aruco_dict)
    # Draw image Marker in aruco_dict is 4x4, with margin 6x6;
    # marker length is half of square length, so total square has size of 12
    side_pixels = 12
    img_bw = board.draw((squares[0] * side_pixels, squares[1] * side_pixels))
    # According to documentation: "As in the GridBoard, the coordinate
    # system of the CharucoBoard is placed in the board plane with
    # the Z axis pointing out, and centered in the bottom left corner
    # of the board." This fits perfectly the requirements of generateFromImage.
    if False:
        img = cv2.resize(img_bw, (0,0), fx=5.0, fy=5.0)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # Convert grayscale image to color image
    img = np.zeros((img_bw.shape[0], img_bw.shape[1], 3))
    img[:,:,:] = img_bw[:,:,np.newaxis]
    return mesh_generate_image(img, square_length/side_pixels)



def mesh_generate_rays(origin, pcl, color=None):
    line_set = o3d.geometry.LineSet()
    points = np.vstack((origin, np.asarray(pcl.points)))
    line_set.points = o3d.utility.Vector3dVector(points)
    lines = np.zeros((points.shape[0]-1, 2))
    lines[:, 1] = np.arange(1, lines.shape[0]+1)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    if color is None:
        colors = np.asarray(pcl.colors)
    else:
        colors = np.tile(color, (lines.shape[0], 1))
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set



def show_images(depth_image, color_image, cbar_enabled=False):
    # Color of invalid pixels
    nan_color = (0, 0, 1.0)
    fig = plt.figure()
    # Depth image
    ax = fig.add_subplot(121)
    cmap = plt.cm.viridis_r
    cmap.set_bad(color=nan_color, alpha=1.0)
    im = ax.imshow(depth_image, cmap=cmap)
    if cbar_enabled:
        fig.colorbar(im, ax=ax)
    ax.set_axis_off()
    ax.set_title('Depth')
    ax.set_aspect('equal')
    # Color image
    idx = np.where(np.isnan(color_image))
    img = color_image.copy()
    img[idx[0], idx[1], :] = nan_color
    ax = fig.add_subplot(122)
    ax.imshow(img)
    ax.set_axis_off()
    ax.set_title('Color')
    ax.set_aspect('equal')
    # Show
    plt.show()



def save_depth_image(filename, depth_image, nan_color=(0, 0, 255)):
    nanidx = np.where(np.isnan(depth_image))
    img = (depth_image - np.nanmin(depth_image)) / \
        (np.nanmax(depth_image) - np.nanmin(depth_image))
    img = (255 * (1.0 - img)).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img[nanidx[0],nanidx[1]] = nan_color
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)



def save_color_image(filename, color_image, nan_color=(0, 0, 255)):
    nanidx = np.where(np.isnan(color_image))
    img = (255.0 * color_image).astype(np.uint8)
    img[nanidx[0],nanidx[1]] = nan_color
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img)



def save_shot(basename, depth_image, color_image, pcl):
    # Write all raw data
    h5f = h5py.File(basename + '.h5', 'w')
    c = 'gzip'
    co = 9
    h5f.create_dataset('depth_image', data=depth_image,
                       compression=c, compression_opts=co)
    h5f.create_dataset('color_image', data=color_image,
                       compression=c, compression_opts=co)
    h5f.create_dataset('pcl.points', data=np.asarray(pcl.points),
                       compression=c, compression_opts=co)
    h5f.create_dataset('pcl.colors', data=np.asarray(pcl.colors),
                       compression=c, compression_opts=co)
    h5f.close()
    # Write additional files
    save_depth_image(basename + '_depth.png', depth_image)
    save_color_image(basename + '_color.png', color_image)
    o3d.io.write_point_cloud(basename + '.ply', pcl)



def load_shot(basename):
    h5f = h5py.File(basename + '.h5', 'r')
    depth_image = h5f['depth_image'][:]
    color_image = h5f['color_image'][:]
    pcl = o3d.geometry.PointCloud()
    points = h5f['pcl.points'][:]
    pcl.points = o3d.utility.Vector3dVector(points)
    colors = h5f['pcl.colors'][:]
    pcl.colors = o3d.utility.Vector3dVector(colors)
    h5f.close()
    return depth_image, color_image, pcl
