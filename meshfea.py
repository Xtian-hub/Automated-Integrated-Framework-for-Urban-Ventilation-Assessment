import open3d as o3d 
from collections import defaultdict
import CSF
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
import tempfile
import os
import subprocess
import cv2
from scipy.stats import skew, kurtosis
from scipy.spatial import KDTree
from tqdm import tqdm
import shutil
# from tqdm.notebook import tqdm
"""
Author: Xiaotian Geng(James Gordon)
Date: 2025-01-25
Description: 此文件用于计算网格模型的特征，包含几何特征、高程特征以及颜色特征
"""
class MeshFeature_Extraction:
    def __init__(self, mesh_objFilePath):
        """
        网格模型读取格式为obj
        对象属性包括：
        mesh: 三角网格对象
        vertices: 网格顶点坐标
        triangles: 三角面片索引
        vertex_normals: 顶点法向量
        triangle_normals: 三角面片法向量
        vertex_colors: 顶点颜色
        triangle_colors: 三角面片颜色
        vertex_features: 顶点特征
        triangle_features: 三角面片特征
        """
        """
        计算特征前，请对子特征进行实例化，如：geometric_features = MeshFeature_Extraction.Geometric_Features(mesh_features)
        特征方法使用如下：
        几何特征(Geometric_Features):        
        area = geometric_features.mesh_area()
        area_mean, area_variance = geometric_features.calculate_triangle_area_variance_and_mean()
        flatness = geometric_features.mesh_flatness()
        density = geometric_features.mesh_density()
        verticality = geometric_features.mesh_verticality()
        masb_radius = geometric_features.triangleCenter_masb_raduis() # 当半径设置过大时会提前停止计算
        mean,var = geometric_features.masb_radius_mean_var()
        高程特征(Elevation_Features):
        height = elevation_features.pointHeight2Ground()
        颜色特征(Color_Features):
        hvi = color_features.calculate_triangle_hvi()
        hsv_histograms = color_features.calculate_triangle_hsv_histograms()
        hsv_statistics = color_features.calculate_hsv_statistics()
        
        """
        # read triangle mesh
        textured_mesh = o3d.io.read_triangle_mesh(mesh_objFilePath)
        if textured_mesh.is_empty():
            raise ValueError(f"{mesh_objFilePath}is empty")
        # extract mesh properties
        self.mesh = o3d.geometry.TriangleMesh()
        self.mesh.vertices = textured_mesh.vertices
        self.mesh.triangles = textured_mesh.triangles
        self.mesh.triangle_uvs = textured_mesh.triangle_uvs
        textured_mesh.compute_triangle_normals()
        self.mesh.triangle_normals = textured_mesh.triangle_normals
        self.mesh.triangle_material_ids = textured_mesh.triangle_material_ids
        self.mesh.textures = textured_mesh.textures
        # record mesh properties
        self.vertices = np.asarray(self.mesh.vertices)
        self.triangles = np.asarray(self.mesh.triangles)
        self.triangle_uvs = np.asarray(self.mesh.triangle_uvs)
        self.triangle_material_ids = np.asarray(self.mesh.triangle_material_ids)
        self.triangle_normals = np.asarray(self.mesh.triangle_normals)
        self.textures = self.mesh.textures
        self.triangle_indices = np.arange(len(self.triangles))
    class Geometric_Features:
        def __init__(self,parent):
            """
            parent: MeshFeature_Extraction, the parent class of the geometric features.
            """
            self.parent = parent
            self.vertices = parent.vertices
            self.triangles = parent.triangles
            self.triangle_uvs = parent.triangle_uvs
            self.triangle_normals = parent.triangle_normals
            self.triangle_material_ids = parent.triangle_material_ids
            self.triangle_indices = parent.triangle_indices

            triangle_centers = np.mean(self.vertices[self.triangles], axis=1)
            self.point_cloud = o3d.geometry.PointCloud()
            self.point_cloud.points = o3d.utility.Vector3dVector(triangle_centers)
            self.point_cloud.normals = o3d.utility.Vector3dVector(self.triangle_normals)
            self.pcl_indices = self.triangle_indices
            # extract xyz coordinates of the point cloud based centroid of traingles
            self.xyz = np.asarray(self.point_cloud.points)
            self.normals = np.asarray(self.point_cloud.normals)
            # extract xyz coordinates of the point cloud based centroid of traingles
            self.xyz = np.asarray(self.point_cloud.points)
            self.normals = np.asarray(self.point_cloud.normals)
            # construct mesh neighbor information
            self.neighbor = self._construct_mesh_neighborhood()
            # calculate the straight-line distance between the centroids of a triangle and its neighboring triangles
            self.centroid_distances = self._calculate_centroid_distances()
            # calculate the eigenvalues of the covariance matrix of the displacements between the centroids of a triangle and its neighboring triangles
            self.eigenvalues = self._calculate_triangle_eigenvalues()
        def _construct_mesh_neighborhood(self):
            """
            Calculate the neighborhood information of the mesh.
            Construct the neighborhood relationships of triangles based on shared vertex relationships.
            
            - PARAMS:
                vertices: np.ndarray, shape=(n, 3), dtype=float32, the vertices of the mesh.
                triangles: np.ndarray, shape=(m, 3), dtype=int32, the triangles of the mesh.
                triangle_indices: np.ndarray, shape=(m,), dtype=int32, the indices of the triangles.
            - RETURNS:
                neighborhood: dict, the neighborhood information of the mesh.
            """
            vertices = self.vertices
            triangles = self.triangles
            triangle_indices = self.triangle_indices

            vertex_triangles = defaultdict(list)
            # construct vertex to triangle mapping
            for i, triangle in enumerate(triangles):
                for vertex in triangle:
                    vertex_coord = tuple(vertices[vertex])
                    vertex_triangles[vertex_coord].append(triangle_indices[i])
            # construct neighborhood information
            neighborhood = defaultdict(list)

            for i, triangle in enumerate(triangles):
                neighbors = set()
                for vertex in triangle:
                    vertex_coord = tuple(vertices[vertex])
                    neighbors.update(vertex_triangles[vertex_coord])
                neighbors.discard(triangle_indices[i]) # remove self
                neighborhood[triangle_indices[i]] = list(neighbors)

            return neighborhood
        def _calculate_centroid_distances(self):
            """
            Calculate the straight-line distance between the centroids of a triangle and its neighboring triangles.
            - PARAMS:
                vertices: np.ndarray, shape=(n, 3), dtype=float32, the vertices of the mesh.
                triangles: np.ndarray, shape=(m, 3), dtype=int32, the triangles of the mesh.
                neighbor: dict, the neighborhood information of the mesh.
            - RETURNS:
                centroid_distances: dict, the straight-line distance between the centroids of a triangle and its neighboring triangles
            """
            vertices = self.vertices
            triangles = self.triangles

            if len(vertices) == 0 or len(triangles) == 0:
                return ValueError("The mesh is empty. Ensure the mesh data is correctly loaded.")
            
            centroids = np.mean(vertices[triangles], axis=1)

            centroid_distances = {}

            for trangle_index, neighbors in self.neighbor.items():
                distances = []
                for neighbor_index in neighbors:
                    distance = np.linalg.norm(centroids[trangle_index] - centroids[neighbor_index])
                    distances.append(distance)
                centroid_distances[trangle_index] = distances
            
            return centroid_distances
        def _calculate_triangle_eigenvalues(self):
            """
            Calculate the eigenvalues of the covariance matrix of the displacements between the centroids of a triangle and its neighboring triangles.
            - PARAMS:
                vertices: np.ndarray, shape=(n, 3), dtype=float32, the vertices of the mesh.
                triangles: np.ndarray, shape=(m, 3), dtype=int32, the triangles of the mesh.
                neighbor: dict, the neighborhood information of the mesh.
            - RETURNS:
                eigenvalues_dict: dict, the eigenvalues of the covariance matrix of the displacements between the centroids of a triangle and its neighboring triangles.
            """
            vertices = self.vertices
            triangles = self.triangles

            centroids = np.mean(vertices[triangles], axis=1)
            eigenvalues_dict = {}

            for triangle_index, neighbors in self.neighbor.items():
                if len(neighbors) < 2:
                    eigenvalues_dict[triangle_index] = np.zeros(3)
                    continue

                neighbor_centroids = centroids[neighbors]
                current_centroid = centroids[triangle_index]
                displacements = neighbor_centroids - current_centroid
                if np.allclose(displacements, 0):
                    eigenvalues_dict[triangle_index] = np.zeros(3)
                    continue

                covariance_matrix = np.cov(displacements.T)
                eigenvalues = np.linalg.eigvals(covariance_matrix)
                eigenvalues = np.real_if_close(eigenvalues, tol=1e-8)
                eigenvalues_dict[triangle_index] = tuple(sorted(eigenvalues, reverse=True))
            
            return eigenvalues_dict
        def mesh_area(self):
            """
            使用海伦公式计算所有三角形的面积(矢量化实现):            :param mesh: Open3D TriangleMesh 对象
            :return: 每个三角形面积的数组
            """
            
            # 获取所有三角形的三个顶点坐标
            v0 = self.vertices[self.triangles[:, 0]]
            v1 = self.vertices[self.triangles[:, 1]]
            v2 = self.vertices[self.triangles[:, 2]]

            # 计算三条边的长度
            a = np.linalg.norm(v1 - v0, axis=1)
            b = np.linalg.norm(v2 - v1, axis=1)
            c = np.linalg.norm(v0 - v2, axis=1)

            # 计算半周长
            s = (a + b + c) / 2

            # 使用海伦公式计算面积
            areas = np.sqrt(s * (s - a) * (s - b) * (s - c))

            return np.log(areas + 1e-8)
        def calculate_triangle_area_variance_and_mean(self):
            """
            计算三角形邻域的面积均值和方差
            :return: 与三角形索引对齐的面积均值和方差列表
            """
            # 计算每个三角形的面积
            triangle_areas = self.mesh_area()

            # 获取三角形邻域关系
            neighborhood = self.neighbor

            # 初始化与三角形索引对齐的面积均值和方差列表
            area_mean = np.zeros(len(self.triangle_indices), dtype=np.float32)
            area_variance = np.zeros(len(self.triangle_indices), dtype=np.float32)

            # 遍历每个三角形，计算邻域内面积的均值和方差
            for triangle_index, neighbors in neighborhood.items():
                if len(neighbors) < 2:
                    area_mean[triangle_index] = 0.0
                    area_variance[triangle_index] = 0.0
                    continue
                # 获取邻域三角形的面积
                neighbor_areas = triangle_areas[neighbors]  # 获取邻居三角形的面积  
                # 如果所有邻居面积都接近 0，也可以根据需要做进一步判断
                if np.allclose(neighbor_areas, 0):
                    area_mean[triangle_index] = 0.0
                    area_variance[triangle_index] = 0.0
                    continue

                mean_area = np.mean(neighbor_areas)  # 计算均值
                variance_area = np.var(neighbor_areas)  # 计算方差

                # 将均值和方差赋值到对应的三角形索引位置
                area_mean[triangle_index] = mean_area
                area_variance[triangle_index] = variance_area

            # 返回面积均值和方差的对齐列表
            return area_mean, area_variance
        def mesh_flatness(self):
            eigenvalues_dict = self.eigenvalues

            # 使用一个列表来存储平整度值，确保与三角形索引对齐
            mesh_flatness = np.zeros(len(self.triangles), dtype=np.float32)

            for triangle_index, eigenvalues in eigenvalues_dict.items():
                lambda1, lambda2, lambda3 = eigenvalues
                denominator = lambda1 + lambda2 + lambda3
                if denominator > 0:
                    flatness = lambda3 / denominator
                else:
                    flatness = 0
                
                # 将平整度值存储到正确的索引位置
                mesh_flatness[triangle_index] = flatness

            return  np.log(mesh_flatness + 1e-8)
        def mesh_density(self):
            """
            calculate the mesh density of each triangle in the mesh.
            - PARAMS:
                centroid_distances_dict: dict, the straight-line distance between the centroids of a triangle and its neighboring triangles.
            - RETURNS:
                mesh_density: dict, the mesh density of each triangle in the mesh.
            """
            centroid_distances_dict = self.centroid_distances
            mesh_density = np.zeros(len(self.triangle_indices), dtype=np.float32)
                # Loop through each triangle and calculate its density
            for triangle_index, neighbors in self.neighbor.items():
                num_neighbors = len(neighbors)
                # 邻居数量检查
                if num_neighbors < 2:
                    mesh_density[triangle_index] = 0.0
                    continue

                # 取出三角形与邻居间的距离
                distances = centroid_distances_dict.get(triangle_index, [])
                if len(distances) < 2:
                    # 如果实际距离列表也不足 2 个，或者为空
                    mesh_density[triangle_index] = 0.0
                    continue

                mean_distance = np.mean(distances)
                # 检查 mean_distance 是否过小
                if np.isclose(mean_distance, 0):
                    mesh_density[triangle_index] = 0.0
                    continue

                if num_neighbors > 0:
                    # Calculate the mean distance between the centroid of a triangle and its neighboring triangles
                    centroid_distances = centroid_distances_dict.get(triangle_index, [])
                    mean_distance = np.mean(centroid_distances)

                    density = num_neighbors / mean_distance
                else:
                    density = 0

                # Assign the computed density to the corresponding index
                mesh_density[triangle_index] = density

            return  np.log(mesh_density + 1e-8)
        def mesh_verticality(self):
            """
            Calculate the verticality of a triangle using the formula:
            1 - |N_normal · Z| (where N_normal is the normal of the triangle and Z is the vertical direction.)

            """
            trangle_normals, trangle_indices = self.triangle_normals, self.triangle_indices
            z_unit_vector = np.array([0, 0, 1])   

            mesh_verticality = np.zeros(len(trangle_indices), dtype=np.float32)
            for i, normal in zip(trangle_indices, trangle_normals):
                verticality = 1 - np.abs(np.dot(normal, z_unit_vector))
                mesh_verticality[i] = verticality
            
            return mesh_verticality
        def triangleCenter_masb_raduis(self):            
            def _pcl_move_to_origin(self):
                """
                move the point cloud to the origin.
                """
                min_coords = np.min(self.xyz, axis=0)
                max_coords = np.max(self.xyz, axis=0)
                center_offset = (max_coords + min_coords) / 2
                points_move = self.xyz - center_offset
                return points_move, center_offset
            def _save_to_npy(points, normals, offset):
                """
                save the point cloud to npy file.
                """
                points = points.astype(np.float32)
                normals = normals.astype(np.float32)
                offset = offset.astype(np.float32)
                # set save path 
                coords_npy_path = os.path.join(tempDir_path, "coords.npy")
                normals_npy_path = os.path.join(tempDir_path, "normals.npy")
                offset_npy_path = os.path.join(tempDir_path, "offset.npy")
                # save to npy
                np.save(coords_npy_path, points)
                np.save(normals_npy_path, normals)
                np.save(offset_npy_path, offset)
                if not os.path.exists(coords_npy_path) or not os.path.exists(normals_npy_path) or not os.path.exists(offset_npy_path):
                    raise ValueError("Failed to save the point cloud to npy file.")
            def _masbcpp(tempDir_path):

                """
                Call the MASBCPP algorithm to calculate the radius of the triangle.
                如果半径设置过大会导致masbcpp卡在计算kdtree那里，然后提前中止运行
                """
                # set the path of the MASBCPP algorithm
                masbcpp_path = "./masbcpp"
                computeMa = os.path.join(masbcpp_path, "compute_ma.exe")
                command = [
                    computeMa,
                    "-r", "3",
                    tempDir_path,
                    tempDir_path
                ]
                # run command
                try:
                    process = subprocess.Popen(
                        command,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        bufsize=1,
                        encoding="utf-8",  # 设置为 UTF-8 编码
                        errors="ignore",  # 忽略无法解码的字节
                    )
                    stdout, stderr = process.communicate(input="\n")
                    if stderr:
                        print("子进程报告了错误:")
                        print(stderr)  # 打印错误信息
                except Exception as e:
                    print("子进程异常:")
                    print(e)  # 打印异常信息
            def calculate_triangle_masb_radius(tempDir_path):
                coords_path = os.path.join(tempDir_path, "coords.npy")
                ma_coords_path = os.path.join(tempDir_path, "ma_coords_in.npy")
                qidx_path = os.path.join(tempDir_path, "ma_qidx_in.npy")       

                coords = np.load(coords_path)
                ma_coords = np.load(ma_coords_path)
                qidx = np.load(qidx_path)
                if np.max(qidx[qidx != -1]) >= coords.shape[0]:
                    raise ValueError("The values in the index file exceed the range of the point cloud coordinates.")
                valid_mask = qidx != -1
                mapped_coords = coords[qidx[valid_mask]]
                distances = np.linalg.norm(mapped_coords - ma_coords[valid_mask], axis=1)
                radii = np.zeros(coords.shape[0])
                radii[qidx[valid_mask]] = distances
                coords_radii_mapping = {
                                        tuple(coords[i]): radii[i]
                                        for i in range(coords.shape[0])
                                        }
                # print(list(coords_radii_mapping.keys())[0])
                return coords_radii_mapping
            def map_redii_to_triangle_indices(self, coords_radii_mapping, center_offset):
                """
                Map the radius to the triangle indices.
                Uses KDTree for fast nearest neighbor search.
                """
                restored_coords = self.xyz - center_offset
                triangle_masbRadii = np.zeros(len(self.pcl_indices), dtype=np.float32)

                # Convert coords_radii_mapping keys to a NumPy array for KDTree
                coords_list = np.array(list(coords_radii_mapping.keys()))
                radii_list = np.array(list(coords_radii_mapping.values()))

                # Build a KDTree for fast nearest neighbor search
                tree = KDTree(coords_list)

                # For each restored coordinate, find the nearest neighbor in coords_radii_mapping
                for i, coord in enumerate(restored_coords):
                    # Find the index of the nearest neighbor
                    dist, index = tree.query(coord, k=1)  # k=1 for the closest point
                    if dist < 0.001:  # Set a threshold for matching
                        triangle_masbRadii[i] = radii_list[index]
                    else:
                        triangle_masbRadii[i] = 0.0  # No match found within tolerance

                return triangle_masbRadii

            
            tempDir_path = './masbcppTemp'
            # tempDir_path = tempfile.mkdtemp(dir=tempDir_path)
            # 检查临时文件夹是否存在，如果不存在则创建
            if not os.path.exists(tempDir_path):
                os.makedirs(tempDir_path)  # 创建临时文件夹
            # move the point cloud to the origin
            points_moveOrigin, center_offset = _pcl_move_to_origin(self)

            # save the point cloud to npy file
            _save_to_npy(points_moveOrigin, self.normals, center_offset)
            _masbcpp(tempDir_path)
            coords_radii_mapping = calculate_triangle_masb_radius(tempDir_path)
            triangle_radii = map_redii_to_triangle_indices(self, coords_radii_mapping, center_offset)
            log_triangle_radii = np.log(triangle_radii + 1e-8)
            if os.path.exists(tempDir_path):
                shutil.rmtree(tempDir_path)  # 删除临时文件夹及其所有内容
            return log_triangle_radii
        def masb_radius_mean_var(self):
            log_triangle_radii = self.triangleCenter_masb_raduis()
            neighbor = self.neighbor
            var = np.zeros(len(self.triangle_indices), dtype=np.float32)
            mean = np.zeros(len(self.triangle_indices), dtype=np.float32)
            for i, neighbors in tqdm(neighbor.items()):
                # 若邻居数不足 2，则直接赋默认值并跳过
                if len(neighbors) < 2:
                    var[i] = 0.0
                    mean[i] = 0.0
                    continue
                radii = log_triangle_radii[neighbors]
                # 如果数组为空或者全部 0，也可视需求再做进一步判断
                if radii.size < 2:
                    var[i] = 0.0
                    mean[i] = 0.0
                    continue
                var[i] = np.var(radii)
                mean[i] = np.mean(radii)
            return mean,var
    class Elevation_Features:
        def __init__(self, parent):
            """
            input parent: MeshFeature_Extraction, the parent class of the elevation features.
            - output: 
                self.height, the height of each point to the ground plane.
                self.triangle_masbRadii, the radius between the triangle center and its medial axis transformation sphere.
            """
            vertices = parent.vertices
            triangles = parent.triangles
            triangle_indices = parent.triangle_indices
            triangle_normals = parent.triangle_normals

            triangle_centers = np.mean(vertices[triangles], axis=1)
            self.point_cloud = o3d.geometry.PointCloud()
            self.point_cloud.points = o3d.utility.Vector3dVector(triangle_centers)
            self.point_cloud.normals = o3d.utility.Vector3dVector(triangle_normals)
            self.pcl_indices = triangle_indices
            # extract xyz coordinates of the point cloud based centroid of traingles
            self.xyz = np.asarray(self.point_cloud.points)
            self.normals = np.asarray(self.point_cloud.normals)
            # extract the ground point cloud(index and coordinates)
            self.groundPCL = self._groundPCL_extraction()
            # fit a ground plane using RANSAC
            self.plane_coeff = self._fit_plane_ransac()    
        def _groundPCL_extraction(self,
                                  unground_extracted=False,
                                  cloth_resolution=0.5,
                                  rigidness=3,
                                  bSloopSmooth=False,
                                  time_step=0.65,
                                  class_threshold=0.5,
                                  iterations=500,
                                  export_cloth_nodes=False
                                  ):
            """
            Use CSF (Cloth Simulation Filter) to extract ground points from point cloud files.
            - Parameters:
                unground_extracted (bool): whether to extract non-ground points.
                cloth_resolution (float): cloth resolution, default value is 0.5.
                rigidness (int): cloth rigidness, default value is 3.
                bSloopSmooth (bool): whether to perform slope smoothing, default value is False.
                time_step (float): time step, default value is 0.65.
                class_threshold (float): classification threshold, default value is 0.5.
                iterations (int): number of iterations, default value is 500.
                export_cloth_nodes (bool): whether to export cloth nodes, default value is False.
            - Returns:
                ground_data (dict): the extracted ground points data (indices and coordinates).
            """
            # set csf parameters
            csf = CSF.CSF()
            csf.params.bSloopSmooth = bSloopSmooth
            csf.params.cloth_resolution = cloth_resolution
            csf.params.rigidness = rigidness
            csf.params.time_step = time_step
            csf.params.class_threshold = class_threshold
            csf.params.iterations = iterations

            csf.setPointCloud(self.xyz)
            ground = CSF.VecInt()
            non_ground = CSF.VecInt()
            csf.do_filtering(ground, non_ground, export_cloth_nodes)

            ground_points = self.xyz[np.array(ground)]
            ground_indices = np.array(ground)
            ground_data = {
                "indices": ground_indices,
                "coordinates": ground_points,
            }

            if unground_extracted:
                unground_points = self.xyz[np.array(non_ground)]
                unground_indices = np.array(non_ground)
                unground_data = {
                    "indices": unground_indices,
                    "coordinates": unground_points
                }
                return unground_data
            
            return ground_data
        def _fit_plane_ransac(self):
            """
            Fit a ground plane using RANSAC.
            - Parameters:
                coordinates (np.ndarray): the coordinates of the ground points.
            - Returns:
            """
            # extract ground point coordinates
            xyCoordinates = self.groundPCL["coordinates"][:, :2]
            zCoordinates = self.groundPCL["coordinates"][:, 2]

            # fit a plane using RANSAC
            ransac = RANSACRegressor(LinearRegression(), residual_threshold=1.0, max_trials=1000)
            ransac.fit(xyCoordinates, zCoordinates)

            # get the plane coefficients
            a, b = ransac.estimator_.coef_
            d = ransac.estimator_.intercept_
            c = -1 # plane equation form: ax + by + cz + d = 0

            plane_coeff = np.array([a, b, c, d])

            return plane_coeff
        def pointHeight2Ground(self):
            """
            Calculate the height of each point to the ground plane.
            """
            
            a, b, c, d = self.plane_coeff
            height = self.xyz[:, 2] - (-(a * self.xyz[:, 0] + b * self.xyz[:, 1] + d) / c)
            
            return height
    class Color_Features:
        def __init__(self, parent):
            self.parent = parent
            self.vertices = parent.vertices
            self.triangles = parent.triangles
            self.triangle_uvs = parent.triangle_uvs
            self.triangle_material_ids = parent.triangle_material_ids
            self.triangle_indices = parent.triangle_indices

            # save textures
            mesh_textures = []
            mesh_textures.append(self.parent.textures[1])
            mesh_textures.append(self.parent.textures[1])
            for i in range(2, len(self.parent.textures) - 1):
                mesh_textures.append(self.parent.textures[i])
            self.mesh_textures = mesh_textures
            # calculate hvi 
            # self.triangle_hvi = self.calculate_triangle_hvi()
        def _calculate_hvi(self, texture_np):

            texture_np_rgb = cv2.cvtColor(texture_np, cv2.COLOR_BGR2RGB)

            R = texture_np_rgb[:, :, 0].astype(np.float32)
            G = texture_np_rgb[:, :, 1].astype(np.float32)
            B = texture_np_rgb[:, :, 2].astype(np.float32)
            
            ExG = 2 * G - R - B
            hvi = np.tanh(10 * (ExG / 510))

            return hvi
        def calculate_triangle_hvi(self):            
            def _calculate_triangle_hvi_mean(hvi_textures_np, uv_coords, texture_width, texture_height):
                # uv -> pixel coordinates
                pixel_coords = (uv_coords * [texture_width, texture_height]).astype(int)
                # print(uv_coords)
                pixel_coords = np.clip(pixel_coords, 0, [texture_width - 1, texture_height - 1])
                # print(pixel_coords)

                # calculate min 
                min_x = np.min(pixel_coords[:, 0])
                max_x = np.max(pixel_coords[:, 0])
                min_y = np.min(pixel_coords[:, 1])
                max_y = np.max(pixel_coords[:, 1])

                cropped_hvi = hvi_textures_np[min_y:max_y + 1, min_x:max_x + 1]

                local_mask = np.zeros((max_y - min_y + 1, max_x - min_x + 1), dtype=np.uint8)

                local_pixel_coords = pixel_coords.copy()
                local_pixel_coords[:, 0] -= min_x
                local_pixel_coords[:, 1] -= min_y

                cv2.fillPoly(local_mask, [local_pixel_coords], 255)

                hvi_values = cropped_hvi[local_mask == 255]
                hvi_mean = np.mean(hvi_values)

                return hvi_mean
            hvi_textures = []
            for texture in self.mesh_textures:
                texture_np = np.asarray(texture)
                hvi = self._calculate_hvi(texture_np)
                hvi_normalized = np.uint8((hvi - np.min(hvi)) / (np.max(hvi) - np.min(hvi)) * 255)
                hvi_textures.append(hvi_normalized)
            self.hvi_textures = hvi_textures

            # 预处理 UV 坐标和材质 ID
            triangle_uvs = np.asarray(self.triangle_uvs).reshape(-1, 3, 2)  # 将三角形的 UV 坐标一次性处理为 NumPy 数组
            material_ids = np.asarray(self.triangle_material_ids)

            num_triangles = len(self.triangle_indices)
            triangle_hvi_means = np.zeros(num_triangles, dtype=np.float32)

            # 为每个三角形计算 HVI 均值
            for i, (uv_coords, material_id) in tqdm(enumerate(zip(triangle_uvs, material_ids)),
                                                    desc="Processing triangles",
                                                    total=num_triangles):

                # 获取对应材质的 HVI 纹理
                hvi_texture_o3d = self.hvi_textures[material_id]
                texture_height, texture_width = hvi_texture_o3d.shape[:2]

                # 计算当前三角形的 HVI 均值
                mean_hvi_value = _calculate_triangle_hvi_mean(hvi_texture_o3d, uv_coords, texture_width, texture_height)
                triangle_hvi_means[i] = mean_hvi_value
            return triangle_hvi_means
        def calculate_triangle_hsv_histograms(self):
            """
            在循环外将所有纹理转成 HSV，在循环内对三角形内部像素
            计算直方图并返回列表。
            每个三角形的直方图是 Hue(15 bins)、Saturation(5 bins)、Value(5 bins) 的拼接向量。
            """

            # ---------- a) 预先将每张纹理整体转换为 HSV，存储以备后续使用 ----------
            hsv_textures = []
            for texture in self.mesh_textures:
                texture_np = np.asarray(texture)
                # 如果你的纹理数据实际上是 RGB，请使用 cv2.COLOR_RGB2HSV
                texture_hsv = cv2.cvtColor(texture_np, cv2.COLOR_BGR2HSV)
                hsv_textures.append(texture_hsv)
            self.hsv_textures = hsv_textures

            # ---------- b) 创建一个列表用于存储每个三角形的直方图 ----------
            triangle_hsv_histograms = []

            # ---------- c) 遍历每个三角形 ----------
            triangle_uvs = np.asarray(self.triangle_uvs).reshape(-1, 3, 2)  # 转换 UV 坐标
            material_ids = np.asarray(self.triangle_material_ids)  # 转换材质 ID
            total_triangles = len(triangle_uvs)  # 或者 len(material_ids)
            for i, (uv_coords, material_id) in tqdm(
                enumerate(zip(triangle_uvs, material_ids)),
                desc="Calculating HSV Histograms",
                total=total_triangles
            ):
                
                hsv_texture = self.hsv_textures[material_id]  # 获取材质的 HSV 纹理
                texture_height, texture_width = hsv_texture.shape[:2]

                # 1) 将 UV 转为像素坐标，并裁剪
                pixel_coords = (uv_coords * [texture_width, texture_height]).astype(int)
                pixel_coords = np.clip(pixel_coords, 0, [texture_width - 1, texture_height - 1])

                # 2) 计算包围盒
                min_x = np.min(pixel_coords[:, 0])
                max_x = np.max(pixel_coords[:, 0])
                min_y = np.min(pixel_coords[:, 1])
                max_y = np.max(pixel_coords[:, 1])

                # 3) 裁剪出包围盒区域
                cropped_hsv = hsv_texture[min_y:max_y + 1, min_x:max_x + 1]

                # 4) 构造局部掩膜
                local_mask = np.zeros((max_y - min_y + 1, max_x - min_x + 1), dtype=np.uint8)
                local_pixel_coords = pixel_coords.copy()
                local_pixel_coords[:, 0] -= min_x
                local_pixel_coords[:, 1] -= min_y
                cv2.fillPoly(local_mask, [local_pixel_coords], 255)

                # 5) 提取三角形内部所有像素
                triangle_pixels_hsv = cropped_hsv[local_mask == 255]
                if len(triangle_pixels_hsv) == 0:
                    # 若没有像素，则返回全 0 的直方图
                    total_bins = 15 + 5 + 5  # hue 15 bins, sat 5 bins, val 5 bins
                    triangle_hsv_histograms.append(np.zeros(total_bins, dtype=np.float32))
                    continue

                # 6) 分离 H, S, V 通道
                H = triangle_pixels_hsv[:, 0]
                S = triangle_pixels_hsv[:, 1]
                V = triangle_pixels_hsv[:, 2]

                # 7) 分别计算直方图
                hue_hist = np.histogram(H, bins=15, range=(0, 179), density=False)[0]
                hue_hist = hue_hist / np.sum(hue_hist) if np.sum(hue_hist) > 0 else hue_hist

                sat_hist = np.histogram(S, bins=5, range=(0, 256), density=False)[0]
                sat_hist = sat_hist / np.sum(sat_hist) if np.sum(sat_hist) > 0 else sat_hist

                val_hist = np.histogram(V, bins=5, range=(0, 256), density=False)[0]
                val_hist = val_hist / np.sum(val_hist) if np.sum(val_hist) > 0 else val_hist

                # 8) 拼接直方图向量
                hist_vector = np.concatenate([hue_hist, sat_hist, val_hist])
                triangle_hsv_histograms.append(hist_vector)

            return triangle_hsv_histograms
        def calculate_triangle_hsv_mean_std(self):
            """
            直接计算三角形“内部所有像素”的 HSV 均值和标准差，使用与 HVI 相同的mask思路，
            并提前将整张纹理转为 HSV (而非每个三角形时临时转换)，以提升效率。

            返回一个 numpy 数组，形状为 (num_triangles, 6)，列分别是：
            [hue_mean, sat_mean, val_mean, hue_std, sat_std, val_std]
            """

            # ---------- a) 预先将每张纹理整体转换为 HSV，存储以备后续使用 ----------
            hsv_textures = []
            for texture in self.mesh_textures:
                texture_np = np.asarray(texture)
                # 如果你的纹理数据是 RGB 排列，这里就用 cv2.COLOR_RGB2HSV
                texture_hsv = cv2.cvtColor(texture_np, cv2.COLOR_BGR2HSV)
                hsv_textures.append(texture_hsv)
            self.hsv_textures = hsv_textures

            # ---------- b) 建立数组保存结果 (每个三角形对应6个特征) ----------
            num_triangles = len(self.triangle_indices)
            triangle_hsv_stats = np.zeros((num_triangles, 6), dtype=np.float32)

            # 将三角形的 UV 和对应材质 ID 提前做成数组
            triangle_uvs = np.asarray(self.triangle_uvs).reshape(-1, 3, 2)
            material_ids = np.asarray(self.triangle_material_ids)

            # ---------- c) 定义一个单三角形计算函数，和 HVI思路一致 ----------
            def _calculate_triangle_hsv_mean_std(hsv_texture, uv_coords, texture_width, texture_height):
                # 1) 计算三角形像素坐标
                pixel_coords = (uv_coords * [texture_width, texture_height]).astype(int)
                pixel_coords = np.clip(pixel_coords, 0, [texture_width - 1, texture_height - 1])

                min_x = np.min(pixel_coords[:, 0])
                max_x = np.max(pixel_coords[:, 0])
                min_y = np.min(pixel_coords[:, 1])
                max_y = np.max(pixel_coords[:, 1])

                # 2) 裁剪出包围盒
                cropped_hsv = hsv_texture[min_y:max_y + 1, min_x:max_x + 1]

                # 3) 在局部区域构造 mask
                local_mask = np.zeros((max_y - min_y + 1, max_x - min_x + 1), dtype=np.uint8)
                local_pixel_coords = pixel_coords.copy()
                local_pixel_coords[:, 0] -= min_x
                local_pixel_coords[:, 1] -= min_y

                cv2.fillPoly(local_mask, [local_pixel_coords], 255)

                # 4) 提取三角形内部所有像素 (Nx3)
                triangle_pixels_hsv = cropped_hsv[local_mask == 255]
                if len(triangle_pixels_hsv) == 0:
                    return [0, 0, 0, 0, 0, 0]

                # 分离 H、S、V
                H = triangle_pixels_hsv[:, 0].astype(np.float32)
                S = triangle_pixels_hsv[:, 1].astype(np.float32)
                V = triangle_pixels_hsv[:, 2].astype(np.float32)

                # 计算均值与标准差
                hue_mean, hue_std = np.mean(H), np.std(H)
                sat_mean, sat_std = np.mean(S), np.std(S)
                val_mean, val_std = np.mean(V), np.std(V)

                return [hue_mean, sat_mean, val_mean, hue_std, sat_std, val_std]

            # ---------- d) 遍历所有三角形，利用同样的局部mask提取 + 统计 ----------
            for i, (uv_coords, material_id) in tqdm(
                    enumerate(zip(triangle_uvs, material_ids)),
                    desc="Processing triangles (HSV mean/std)",
                    total=num_triangles):
                hsv_texture = self.hsv_textures[material_id]
                texture_height, texture_width = hsv_texture.shape[:2]

                stats = _calculate_triangle_hsv_mean_std(
                    hsv_texture, uv_coords,
                    texture_width, texture_height
                )
                triangle_hsv_stats[i, :] = stats

            return triangle_hsv_stats