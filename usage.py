import sys
import trimesh
import numpy as np
import os
import torch
from scipy.spatial.transform import Rotation
from datetime import datetime
# import rerun as rr

sys.path.append("..")
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes, segment_sky


def generate_unique_filename(prefix="file", extension="txt"):
    # Get the current date and time, and format it
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create the filename
    filename = f"{prefix}_{current_time}.{extension}"
    return filename

def get_3D_model_from_scene_list(outdir, silent, scene_list, min_conf_thr=3, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05):
    """
    extract 3D_model (glb file) from a list of reconstructed scenes
    """
    scene = trimesh.Scene()
    for scene_idx, scene_item in enumerate(scene_list):
        if scene_item is None:
            continue
        # # post processes
        # if clean_depth:
        #     scene_item = scene_item.clean_pointcloud()
        # if mask_sky:
        #     scene_item = scene_item.mask_sky()

        # scene_list.append((imgs, focals, poses, pts3d_list, confidence_masks))
        # get optimized values from scene_item
        rgbimg = scene_item[0]
        focals = scene_item[1].cpu()
        cams2world = scene_item[2].cpu()
        # 3D pointcloud from depthmap, poses and intrinsics
        pts3d = to_numpy(scene_item[3])
        # scene_item.min_conf_thr = float(scene_item.conf_trf(torch.tensor(min_conf_thr)))
        msk = to_numpy(scene_item[4])

        if mask_sky:
            msk = [m & ~segment_sky(img).cpu().numpy() for m, img in zip(msk, rgbimg)]

        if as_pointcloud:
            pts = np.concatenate([p[m] for p, m in zip(pts3d, msk)])
            col = np.concatenate([p[m] for p, m in zip(rgbimg, msk)])
            pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
            scene.add_geometry(pct)
        else:
            meshes = [pts3d_to_trimesh(rgbimg[i], pts3d[i], msk[i]) for i in range(len(rgbimg))]
            mesh = trimesh.Trimesh(**cat_meshes(meshes))
            scene.add_geometry(mesh)

    outfile = os.path.join(outdir, generate_unique_filename(prefix="scene", extension="glb"))
    if not silent:
        print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj=outfile)
    return outfile

def is_valid_rotation(R):
    """Checks if a matrix is a valid rotation matrix."""
    return np.allclose(np.dot(R, R.T), np.eye(3)) and np.isclose(np.linalg.det(R), 1)

def process_images(image_list, model, device, batch_size, niter, schedule, lr):
    """Processes a list of images to generate poses and point clouds."""
    scene_list = []
    scale_factor_cur2prev = 1.0
    prev_pose = np.eye(4)
    prev_pts3d = None
    prev_trf_cur2nex = None
    # transformation matrix (original, first iamge in first window -> next)
    trf_org2cur = None

    for idx in range(len(image_list)-1):
        print(image_list[idx], image_list[idx+1])

        images = load_images([image_list[idx], image_list[idx+1]], size=512)
        pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, model, device, batch_size=batch_size)

        # at this stage, you have the raw dust3r predictions
        view1, pred1 = output['view1'], output['pred1']
        view2, pred2 = output['view2'], output['pred2']
        # here, view1, pred1, view2, pred2 are dicts of lists of len(2)
        #  -> because we symmetrize we have (im1, im2) and (im2, im1) pairs
        # in each view you have:
        # an integer image identifier: view1['idx'] and view2['idx']
        # the img: view1['img'] and view2['img']
        # the image shape: view1['true_shape'] and view2['true_shape']
        # an instance string output by the dataloader: view1['instance'] and view2['instance']
        # pred1 and pred2 contains the confidence values: pred1['conf'] and pred2['conf']
        # pred1 contains 3D points for view1['img'] in view1['img'] space: pred1['pts3d']
        # pred2 contains 3D points for view2['img'] in view1['img'] space: pred2['pts3d_in_other_view']

        # next we'll use the global_aligner to align the predictions
        # depending on your task, you may be fine with the raw output and not need it
        # with only two input images, you could use GlobalAlignerMode.PairViewer: it would just convert the output
        # if using GlobalAlignerMode.PairViewer, no need to run compute_global_alignment
        scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
        loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

        # retrieve useful values from scene:
        imgs = scene.imgs
        focals = scene.get_focals()
        poses = scene.get_im_poses()
        pts3d = scene.get_pts3d()
        confidence_masks = scene.get_masks()

        # visualize reconstruction
        #scene.show()
        print("len(pose):", len(poses))
        print("poses[0]:\n", poses[0])
        print("poses[1]:\n", poses[1])
        print("len(pts3d):", len(pts3d))

        # * pose[1] is the transformation matrix (current->next)
        trf_cur2nex = poses[1].detach().cpu().numpy()
        assert is_valid_rotation(trf_cur2nex[:3, :3]), "Invalid rotation matrix"

        current_pts3d = pts3d[0].detach().cpu().numpy()
        print("cur_pts3d shape", current_pts3d.shape)
        if idx > 0:
            # make prev_pts3d to the same coordinate system as current_pts3d
            print("shape of prev_trf_cur2nex:", prev_trf_cur2nex.shape)
            print("shape of prev_pts3d:", prev_pts3d.shape)

            # Transform from shape (144, 512, 3) to (73728, 3)
            prev_points = prev_pts3d.reshape(-1, 3)
            # Transform from shape (73728, 3) to (73728, 4) and add a column of ones at the end
            points_homogeneous = np.hstack((prev_points, np.ones((prev_points.shape[0], 1))))
            # Apply the transformation matrix
            transformed_prev_pts3d = np.dot(prev_trf_cur2nex, points_homogeneous.T).T
            # Transform from shape (73728, 4) to (73728, 3)
            transformed_prev_pts3d = transformed_prev_pts3d[:, :3].reshape(prev_pts3d.shape)
            scale_factor_cur2prev = np.median(transformed_prev_pts3d[:, :, 2]) / np.median(current_pts3d[:, :, 2])
            print("scale_factor_cur2prev:", scale_factor_cur2prev)

        # * apply scaling on the transformation matrix
        scaled_trf_cur2nex = trf_cur2nex.copy()
        scaled_trf_cur2nex[:3, 3] = scale_factor_cur2prev * trf_cur2nex[:3, 3]
        print("scaled_trf_cur2nex:\n", scaled_trf_cur2nex)

        if idx == 1: # org is the first coordinate system
            trf_org2cur = prev_trf_cur2nex
        elif idx > 1:
            # * calculate the transformation matrix (original, first iamge in first window -> current)
            trf_org2cur = prev_trf_cur2nex @ trf_org2cur
        print("trf_org2cur:\n", trf_org2cur)

        pts3d_list = []
        # * transform point cloud (pts3d)
        for i in range(len(poses)):
            if idx == 0:
                pts3d_list.append(pts3d[i])
            else:
            # transform point cloud to the original coordinate system
                tmp_pts3d = pts3d[i].detach().cpu().numpy().copy().reshape(-1, 3)
                tmp_pts3d = np.hstack((tmp_pts3d, np.ones((tmp_pts3d.shape[0], 1))))
                transformed_pts3d = np.dot(np.linalg.inv(trf_org2cur), tmp_pts3d.T).T
                transformed_pts3d = transformed_pts3d[:, :3].reshape(current_pts3d.shape)
                pts3d_list.append(transformed_pts3d)

        # * save the scene in the original coordinate system
        scene_list.append((imgs, focals, poses, pts3d_list, confidence_masks))

        prev_trf_cur2nex = scaled_trf_cur2nex
        prev_pts3d = pts3d[1].detach().cpu().numpy()

        visualize_matches(image_list, idx, imgs, confidence_masks, pts3d)

    return scene_list

def visualize_matches(image_list, idx, imgs, confidence_masks, pts3d):
    """Visualizes matches between consecutive image pairs."""
    from dust3r.utils.geometry import find_reciprocal_matches, xy_grid
    import matplotlib.pyplot as plt

    pts2d_list, pts3d_list = [], []
    for i in range(2):
        conf_i = confidence_masks[i].cpu().numpy()
        pts2d_list.append(xy_grid(*imgs[i].shape[:2][::-1])[conf_i])
        pts3d_list.append(pts3d[i].detach().cpu().numpy()[conf_i])
    reciprocal_in_P2, nn2_in_P1, num_matches = find_reciprocal_matches(*pts3d_list)
    print(f'found {num_matches} matches')
    matches_im1 = pts2d_list[1][reciprocal_in_P2]
    matches_im0 = pts2d_list[0][nn2_in_P1][reciprocal_in_P2]

    # visualize a few matches
    import numpy as np
    from matplotlib import pyplot as plt
    n_viz = 10
    match_idx_to_viz = np.round(np.linspace(0, num_matches-1, n_viz)).astype(int)
    viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

    H0, W0, H1, W1 = *imgs[0].shape[:2], *imgs[1].shape[:2]
    img0 = np.pad(imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img1 = np.pad(imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img = np.concatenate((img0, img1), axis=1)
    plt.figure()
    figure_title = image_list[idx][image_list[idx].rfind("/")+1:] + "-" + image_list[idx+1][image_list[idx+1].rfind("/")+1:]
    print(figure_title)
    plt.title(figure_title)
    plt.imshow(img)
    cmap = plt.get_cmap('jet')
    for i in range(n_viz):
        (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
        plt.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
    plt.savefig('./output/' + figure_title)  # Save the figure before showing it
    plt.show(block=True)

if __name__ == '__main__':
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 100

    model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    path = './images/'
    image_filename_ls = ['000145.png', '000146.png', '000147.png']
    image_list = [path + image_filename for image_filename in image_filename_ls]

    scene_list = process_images(image_list, model, device, batch_size, niter, schedule, lr)
    outdir = './output'
    outfile_path = get_3D_model_from_scene_list(outdir=outdir, silent=False, scene_list=scene_list, min_conf_thr=3,
                                                as_pointcloud=False, mask_sky=True, clean_depth=False, 
                                                transparent_cams=False, cam_size=0.05)

    print("final pose:")
    # print(adjusted_pose)
    print(len(scene_list))
