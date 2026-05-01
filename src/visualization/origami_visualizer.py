# src/visualization/origami_visualizer.py

import numpy as np
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
from typing import Dict


class OrigamiVisualizer:
    """在MeshCat中可视化折纸手"""
    
    def __init__(self, zmq_url=None):
        if zmq_url:
            self.vis = meshcat.Visualizer(zmq_url=zmq_url)
        else:
            self.vis = meshcat.Visualizer()
        
        self.colors = [
            0xff9999, 0x99ff99, 0x9999ff, 0xffff99, 0xff99ff,
            0x99ffff, 0xffcc99, 0xcc99ff, 0x99ffcc, 0xff9966,
        ]
    
    def display_hand(self, fk, joint_angles, opacity=0.7):
        """显示给定关节角度下的手部姿态"""
        world_verts = fk.get_face_vertices_world(joint_angles)
        
        for face_id, vertices in world_verts.items():
            color = self.colors[face_id % len(self.colors)]
            
            mesh = self._triangulate_face(vertices)
            
            # 修复：使用正确的材质类名
            # MeshCat 0.3+ 使用 MeshBasicMaterial, MeshLambertMaterial, MeshToonMaterial 等
            # DoubleSide用整数常量，或者尝试字符串 'DoubleSide'
            try:
                # 尝试新版API
                material = g.MeshLambertMaterial(
                    color=color,
                    opacity=opacity,
                    transparent=True,
                )
            except TypeError:
                # 旧版API
                material = g.MeshLambertMaterial(
                    color=color,
                    reflectivity=0.5,
                )
            
            self.vis[f"face_{face_id}"].set_object(mesh, material)
    
    def _triangulate_face(self, vertices):
        """将多边形扇三角化"""
        n = len(vertices)
        if n < 3:
            return None
        
        triangles = []
        for i in range(1, n - 1):
            triangles.append([0, i, i + 1])
        
        return g.TriangularMeshGeometry(vertices, np.array(triangles))
    
    def animate_folding(self, fk, joint_id, start_angle=0.0, 
                        end_angle=np.pi/2, steps=30):
        import time
        
        for i in range(steps):
            t = i / (steps - 1)
            angle = start_angle + t * (end_angle - start_angle)
            self.display_hand(fk, {joint_id: angle})
            time.sleep(0.05)
    
    def open_browser(self):
        """在默认浏览器中打开可视化"""
        import webbrowser
        url = self.vis.url()
        print(f"Opening {url}")
        webbrowser.open(url)
    
    def wait_for_close(self):
        """阻塞直到用户关闭"""
        input("Press Enter to exit...")

    def set_camera_position(self, position, lookat):
        """
        设置相机位置和观察目标
        
        Args:
            position: [x, y, z] 相机位置
            lookat: [x, y, z] 观察目标
        """
        # 使用底层meshcat的命令
        self.vis["/Cameras/default/rotated"].set_property(
            "position", list(position)
        )
        self.vis["/Cameras/default"].set_property(
            "position", list(position)
        )
        # 通过设置视角来间接控制
        # meshcat的Python API可能没有直接设置lookat的方法，
        # 我们通过在场景中添加一个"视角提示"物体来辅助
        
    def auto_set_camera(self, fk, margin=1.5):
        """
        根据手部几何自动调整相机
        
        计算所有面片在展开状态下的包围盒，
        设置相机到能俯瞰全部的位置
        """
        # 获取所有面片的顶点
        all_verts = []
        for face_id, face in fk.design.faces.items():
            for v in face.vertices:
                all_verts.append([v.x, v.y, 0.0])
        
        if not all_verts:
            return
        
        all_verts = np.array(all_verts)
        center = np.mean(all_verts, axis=0)
        extent = np.max(all_verts, axis=0) - np.min(all_verts, axis=0)
        max_extent = max(extent[0], extent[1])
        
        # 相机放在中心上方，距离与物体大小成比例
        camera_distance = max_extent * margin
        camera_position = [
            center[0],
            center[1] - camera_distance * 0.3,  # 稍微偏前
            camera_distance
        ]
        lookat = [center[0], center[1], 0.0]
        
        # meshcat 设置相机
        self.vis["/Cameras/default"].set_property(
            "position", camera_position
        )