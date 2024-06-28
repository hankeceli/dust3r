import numpy as np
# you need to install open3d
import open3d as o3d
# pip install open3d
working_directory = '/Users/tunaseckin/Desktop/TUM/TUM Practical Courses/Practical Courses/Geometric Scene Understanding/DUST3R/DUST3R/'
recovered_path = 'Recovered_Pose_3DPoints/'
image_len = 6
# STEP 6 Visualize Camera Poses T and 3D Points P
for i in range(1,image_len+1):
    Pts3D = np.load(working_directory + recovered_path+f"P{i}.npy") # read recovered PointCloud
    PoseT = np.load(working_directory + recovered_path+f"T{i}.npy")
    print(f"Pose T{i}:\n", PoseT)

    Pts3D_shaped = Pts3D.reshape(Pts3D.shape[0]*Pts3D.shape[1],Pts3D.shape[2]) # reshape it as 3D point list.
    print(f"P{i} loaded with shape", Pts3D.shape, "reshaped to", Pts3D_shaped.shape)
    pcd = o3d.geometry.PointCloud() # initialize point cloud
    pcd.points = o3d.utility.Vector3dVector(Pts3D_shaped) # set points of the point cloud.
    o3d.visualization.draw_geometries([pcd], window_name=f"Point Cloud P{i}") # display the point cloud.
    

