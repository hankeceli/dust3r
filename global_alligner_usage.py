from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

import numpy as np

"""
3D positions are provided in (m,n,3) sized points.
To apply transformation (4x4), I reshaped it to (mxn, 3)
and add column ones to make it homogeneous: (mxn, 4)
This shape format is also required by open3d point cloud visualizer
"""
def multiply_T_P_homo(P2_1, T1_2):
    # P2_2_ = T1_2 * P2_1 starts
    P2_1_reshaped = P2_1.reshape(P2_1.shape[0]*P2_1.shape[1],P2_1.shape[2])
    #print(P2_1_reshaped.shape)
    ones = np.ones((P2_1.shape[0]*P2_1.shape[1],1))
    # make homogeneous by adding 1 to 3D point
    P2_1_homo = np.hstack((P2_1_reshaped,ones))
    #print("homo", P2_1_homo.shape)
    # Apply the transformation to our point cloud
    transformed_points = np.matmul(P2_1_homo,T1_2)
    transformed_points = transformed_points[:,:-1] # convert back to cartesian coord
    P2_2_ = transformed_points.reshape(P2_1.shape[0],P2_1.shape[1],P2_1.shape[2])
    # print(P2_2_.shape)
    # P2_2_ = T1_2 * P2_1 ends
    return P2_2_


if __name__ == '__main__':
    device = 'cpu'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 100

    model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    # load_images can take a list of images or a directory
    working_directory = '/Users/tunaseckin/Desktop/TUM/TUM Practical Courses/Practical Courses/Geometric Scene Understanding/DUST3R/'
    path = working_directory+'pragueparks/KITTI(color)/sequences(color)/00/image_2/'
    image_1 = path + '000410.png'
    image_2 = path + '000411.png'
    image_3 = path + '000412.png'
    image_4 = path + '000413.png'
    image_5 = path + '000414.png'
    image_6 = path + '000415.png'
    #image_7 = path + '000416.png'
    image_list = [image_1, image_2, image_3, image_4, image_5, image_6]
    #images = load_images([image_1, image_2, image_3, image_4, image_5, image_6], size=512)
    scene_list = []

    previous_pose = np.eye(4) # start pose / transformation with Identity
    previous_transformation = np.eye(4)
    # save T1 as Identity matrix
    np.save(working_directory+"DUST3R/Recovered_Pose_3DPoints/T1.npy",previous_pose)

    for idx in range(len(image_list)-1):
        print(image_list[idx],image_list[idx+1])
        
        images = load_images([image_list[idx],image_list[idx+1]], size=512)
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
        P1_1 = pts3d[0].detach().numpy()# First 3D Point in the window, in img. coord of First
        P2_1 = pts3d[1].detach().numpy()# Second 3D Point in the window, in img. coord of First
        T1_2 = poses[0].detach().numpy()# T1->2 ? Transformation from I1 to I2 coord. sys
        T2_1 = poses[1].detach().numpy()# T2->1 ???
        
        if idx>0: # after first window, compute Scaling factor and Apply
            # STEP 3
            # find scaling factor
            S2_1 = np.median(P2_2_[:,:,2]) / np.median(P1_1[:,:,2]) # extract only z coordinates
            print(f"scaling factor from window {idx+1} to window {idx} = {S2_1}")
            # STEP 4
            # apply scaling on pose T
            t1_2 = T1_2[:3,3] # extract translation
            t1_2scaled = S2_1 * t1_2 # apply scaling
            T1_2scaled = T1_2.copy()
            T1_2scaled[:3,3]=t1_2scaled # create scaled Transformation by replacing scaled translation

        # align point to find scaling factor: STEP 1-2
        P2_2_ = multiply_T_P_homo(P2_1, T1_2)
        
        # STEP 5
        if idx==0: # first window
            T_current = T1_2 @ previous_pose
        else: # for all other windows
            T_current = T1_2scaled @ previous_pose

        # recover current Pose T
        np.save(working_directory+f"DUST3R/Recovered_Pose_3DPoints/T{idx+2}.npy",T_current)
        # update pose
        previous_pose = T_current

        if idx==0:
            P_current_1 = P1_1 #P1_1
        else: # e.g. for 4th window: (T1_2^-1 * T2_3^-1 * T3_4^-1) * P5_4 
            P_current_1 = multiply_T_P_homo(S2_1 * P1_1, previous_transformation) # previous_transformation @ (S2_1 * P1_1)
        # update applied transformation using multiple Poses
        # all in the same coord. system (Image 1)
        previous_transformation = previous_transformation @ np.linalg.inv(T1_2)
        
        # recover current 3D point P
        np.save(working_directory+f"DUST3R/Recovered_Pose_3DPoints/P{idx+1}.npy",P_current_1)
        
        if idx==len(image_list)-2: # check if this is the last window
            P_last_1 = multiply_T_P_homo(S2_1 * P2_1, previous_transformation) 
            # save last PointCloud
            np.save(working_directory+f"DUST3R/Recovered_Pose_3DPoints/P{idx+2}.npy",P_last_1)

        # find 2D-2D matches between the two images
        from dust3r.utils.geometry import find_reciprocal_matches, xy_grid
        pts2d_list, pts3d_list = [], []
        for i in range(2):
            conf_i = confidence_masks[i].cpu().numpy()
            pts2d_list.append(xy_grid(*imgs[i].shape[:2][::-1])[conf_i])  # imgs[i].shape[:2] = (H, W)
            pts3d_list.append(pts3d[i].detach().cpu().numpy()[conf_i])
        reciprocal_in_P2, nn2_in_P1, num_matches = find_reciprocal_matches(*pts3d_list)
        print(f'found {num_matches} matches')
        matches_im1 = pts2d_list[1][reciprocal_in_P2]
        matches_im0 = pts2d_list[0][nn2_in_P1][reciprocal_in_P2]

        # visualize a few matches
        import numpy as np
        from matplotlib import pyplot as pl
        n_viz = 10
        match_idx_to_viz = np.round(np.linspace(0, num_matches-1, n_viz)).astype(int)
        viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

        H0, W0, H1, W1 = *imgs[0].shape[:2], *imgs[1].shape[:2]
        img0 = np.pad(imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        img1 = np.pad(imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        img = np.concatenate((img0, img1), axis=1)
        pl.figure()
        figure_title = image_list[idx][image_list[idx].rfind("/")+1:] + "-" + image_list[idx+1][image_list[idx+1].rfind("/")+1:] 
        print(figure_title)
        pl.title(figure_title)
        pl.imshow(img)
        cmap = pl.get_cmap('jet')
        for i in range(n_viz):
            (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
            pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
        pl.show(block=True)
    
