## Updated

import numpy as np
import open3d as o3d
import copy


class PCD():
   
    
    def np2pcd(self,numpyData):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(numpyData)
        return pcd
    

    def pcd2np(self,pcd):
        return np.asarray(pcd.points)
    

 
    def viz(self,data,color=False,mesh_size=0.2):       # Visualization for 3D point clouds
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=mesh_size, origin=[0, 0, 0])
        for x in range(len(data)):    
            if str(type(data[x])) == "<class 'numpy.ndarray'>" :
                 data[x] = self.np2pcd(data[x])
                 
        if mesh_size:
            data+=[mesh_frame]  
        
        if color:
            for x in range(len(color)):
                data[x].paint_uniform_color(color[x]) 
            
        o3d.visualization.draw_geometries(data)

    
## Converts a 2D depth image to a 3D point cloud for viewing
    def depth2pcd(self,depthFrame):
        pcd = []
        for x in range(depthFrame.shape[0]):
            for y in range(depthFrame.shape[1]):
                point = [x,y,(depthFrame[x,y])]
                pcd.append(point)
        return np.array(pcd,dtype=np.float64)

## This for 20 joints to create human skeleton. 
## Input: 3D points. Lines: connect those points
    def skeleton(self,data,skcolor,add=[]):       ## Additional data can be added through the add argument
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25, origin=[0, 0, 0])
        pcd = [0]*len(data)
        for x in range(len(data)):
            points = data[x]
            color = skcolor[x]
            lines = [           ## Defined only for skeleton obtained from kinectv2 and for 21 joints
                [0, 1],
                [1,2],
                [2,20],
                [20,19],
                
                [19,11],
                [11,13],
                [13,15],
                [15,17],
                
                [19,12],
                [12,14],
                [14,16],
                [16,18],
                
                [2,3],
                [3,5],
                [5,7],
                [7,9],
                
                [2,4],
                [4,6],
                [6,8],
                [8,10]
            ]
            colors = [color for i in range(len(lines))]
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(points),
                lines=o3d.utility.Vector2iVector(lines),
            )
            line_set.colors = o3d.utility.Vector3dVector(colors)
            data[x] = line_set            
            pcd[x] = self.np2pcd(np.array(points))
        data+=pcd
        data+=[mesh_frame]
        for x in range(len(add)):
            if str(type(add[x])) == "<class 'numpy.ndarray'>" :
                data+= [self.np2pcd(add[x])]
            else:
                data+=[add[x]]
        o3d.visualization.draw_geometries(data)
        
        
    def draw_registration_result(self, source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0, 0])
        target_temp.paint_uniform_color([0, 1, 0])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])


    
    
    def icp(self,source,target,trans_init=np.eye(4)):
        if str(type(source)) == "<class 'numpy.ndarray'>" :
            source = self.np2pcd(source)
        if str(type(target)) == "<class 'numpy.ndarray'>" :
            target = self.np2pcd(target)
        threshold = 0.02
        # trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
        #                          [-0.139, 0.967, -0.215, 0.7],
        #                          [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])
        self.draw_registration_result(source, target, trans_init)
        print("Initial alignment")
        evaluation = o3d.registration.evaluate_registration(source, target,
                                                            threshold, trans_init)
        print(evaluation)
    
        print("Apply point-to-point ICP")
        reg_p2p = o3d.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.registration.TransformationEstimationPointToPoint())
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)
        print("")
        self.draw_registration_result(source, target, reg_p2p.transformation)    
        
        
        
    def pickPoints(self,pcd):
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()  # user picks points
        vis.destroy_window()
        print("")
        return vis.get_picked_points()