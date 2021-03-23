# mj_pc
Class for rendering depth images and generating point clouds in MuJoCo

PointCloudGenerator is a Python class that can, given a mujoco_py.cymj.MjSim, render depth images from all available cameras, convert them to an Open3D point cloud, and return the cloud with its estimated normals.

# Usage

Define any number of cameras in your MuJoCo environment with:

    <body name="cam_body_name" pos="`x` `y` `z`" euler="`ax` `ay` `az`">
        <camera name="cam_name" pos="0.0 0 0" fovy="`fov`"></camera>
    </body>
    
And replacing `x`, `y`, `z`, `ax`, `ay`, `az`, and `fov` with your values.

In your Python script:

    from mujoco_py import cymj
    from mujoco_py import load_model_from_path, MjSim
    from mj_pc.mj_point_clouds import PointCloudGenerator
    
    model = load_model_from_path("path_to_your_model_with_cameras")
    sim = MjSim(model)
    pc_gen = PointCloudGenerator(sim)
    cloud_with_normals = pc_gen.generateCroppedPointCloud()
