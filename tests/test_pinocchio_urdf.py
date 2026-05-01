# tests/test_pinocchio_urdf.py

import sys
sys.path.insert(0, '.')

import os
import numpy as np
import time
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer


def main():
    hand_name = "test_2"
    urdf_path = os.path.abspath(os.path.join("models", hand_name, f"{hand_name}.urdf"))
    urdf_dir = os.path.dirname(urdf_path)
    mesh_dir = os.path.join(urdf_dir, "meshes")
    
    print(f"URDF: {urdf_path}")
    print(f"Meshes: {mesh_dir}")
    print(f"Meshes exist: {os.path.exists(mesh_dir)}")
    
    if os.path.exists(mesh_dir):
        stls = os.listdir(mesh_dir)
        print(f"STL files: {stls[:3]}... ({len(stls)} total)")
    
    # 修改URDF中的mesh路径为绝对路径（临时文件）
    with open(urdf_path, 'r') as f:
        urdf_content = f.read()
    
    # 将相对路径替换为绝对路径
    urdf_content = urdf_content.replace(
        'filename="meshes/', 
        f'filename="{mesh_dir}/'
    )
    
    # 写入临时URDF
    tmp_urdf = urdf_path.replace('.urdf', '_abs.urdf')
    with open(tmp_urdf, 'w') as f:
        f.write(urdf_content)
    
    print(f"Created temp URDF with absolute mesh paths: {tmp_urdf}")
    
    try:
        # 用绝对路径加载
        model = pin.buildModelFromUrdf(tmp_urdf)
        visual_model = pin.buildGeomFromUrdf(
            model, tmp_urdf, pin.GeometryType.VISUAL
        )
        collision_model = pin.buildGeomFromUrdf(
            model, tmp_urdf, pin.GeometryType.COLLISION
        )
        
        print(f"Model loaded:")
        print(f"  Joints: {model.njoints}")
        print(f"  Position dimension: {model.nq}")
        print(f"  Visual geometries: {len(visual_model.geometryObjects)}")
        
        # 列出joint
        print("\nJoint list:")
        for i in range(1, model.njoints):
            name = model.names[i]
            joint = model.joints[i]
            parent = model.parents[i]
            
            try:
                nq = joint.nq() if callable(joint.nq) else joint.nq
            except:
                nq = joint.nq
            
            print(f"  [{i}] {name} (parent={parent}, type={joint.shortname()})")
        
        # 可视化
        viz = MeshcatVisualizer(model, collision_model, visual_model)
        viz.initViewer(open=True)
        viz.loadViewerModel()
        
        q0 = pin.neutral(model)
        print(f"\nDisplaying model in browser...")
        viz.display(q0)
        
        input("\nPress Enter to animate first joint...")
        
        # 动画
        q = q0.copy()
        for i in range(1, model.njoints):
            joint = model.joints[i]
            try:
                nq_val = joint.nq() if callable(joint.nq) else joint.nq
            except:
                nq_val = joint.nq
            
            if nq_val == 1:
                try:
                    idx_q = model.idx_qs[i]
                except:
                    idx_q = i - 1
                
                print(f"Animating joint {i} ({model.names[i]})")
                for angle in np.linspace(0, np.pi/3, 30):
                    q[idx_q] = angle
                    viz.display(q)
                    time.sleep(0.05)
                break
        
        print("Done!")
        input("Press Enter to exit...")
    
    finally:
        # 清理临时URDF
        try:
            os.remove(tmp_urdf)
        except:
            pass


if __name__ == "__main__":
    main()