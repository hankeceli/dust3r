import rerun as rr  # NOTE: `rerun`, not `rerun-sdk`!
import numpy as np

# ! Install Rerun first, check this link (https://rerun.io/docs/getting-started/quick-start/python)

rr.init("rerun_example_my_data", spawn=True)

mesh_file_path = "YOUR_FILE.glb"

rr.log_file_from_path(mesh_file_path)
