# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Process the point-cloud and prepare it for object detection
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# general package imports
import cv2
import numpy as np
import torch
import open3d
import zlib
from numpy.lib.function_base import percentile

# add project directory to python path to enable relative imports
import os
import sys

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# waymo open dataset reader
from tools.waymo_reader.simple_waymo_open_dataset_reader import utils as waymo_utils
from tools.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2, label_pb2

# object detection tools and helper functions
import misc.objdet_tools as tools


# visualize lidar point-cloud
def show_pcl(pcl):
    ####### ID_S1_EX2 START #######
    #######
    print("student task ID_S1_EX2")
    vis_pcl = open3d.visualization.VisualizerWithKeyCallback()  # Visualizer with custom key callack capabilities
    vis_pcl.create_window('Open3D', 1920, 1080, 50, 50, True)  # Function to create a window and initialize GLFW
    pcd = open3d.geometry.PointCloud()  # PointCloud class.
    # A point cloud consists of point coordinates, and optionally point colors and point normals.

    pcd.points = open3d.utility.Vector3dVector(pcl[:, :3])

    vis_pcl.add_geometry(pcd)  # Function to add geometry to the scene and create corresponding shaders

    global id
    id = True

    def action(vis_pcl):
        global id
        print('Right arrow is pressed')
        id = False
        return 

    vis_pcl.register_key_callback(262, action)  # Function to register a callback function for a key press event

    while id:
        vis_pcl.poll_events()
        vis_pcl.update_renderer()
    #######
    ####### ID_S1_EX2 END #######

# visualize range image
def show_range_image(frame, lidar_name):
    ####### ID_S1_EX1 START #######
    #######
    print("student task ID_S1_EX1")

    # The code for steps 1-3 is borrowed from Exampple C1-5-1

    lidar = [obj for obj in frame.lasers if obj.name == lidar_name][0]  # get laser data structure from frame
   
    ri = dataset_pb2.MatrixFloat()
    ri.ParseFromString(zlib.decompress(lidar.ri_return1.range_image_compressed))
    ri = np.array(ri.data).reshape(ri.shape.dims)
  

    # The code for step 4 is borrowed from Exampple C1-5-4
    ri[ri < 0] = 0.0
    ri_range = ri[:, :, 0]
    ri_range = ri_range * 255 / (np.amax(ri_range) - np.amin(ri_range))
    img_range = ri_range.astype(np.uint8)

    # The code of steps 5-6 borrowed from Exercise: Visualizing the Intensity Channel
    # map value range to 8bit
    ri_intensity = ri[:, :, 1]
    percentile_1 = percentile(ri_intensity, 1)
    percentile_99 = percentile(ri_intensity, 99)
    clipped_normalised_intensity = np.clip(ri_intensity, percentile_1, percentile_99) / percentile_99
    ri_intensity = 255 * clipped_normalised_intensity
    img_intensity = ri_intensity.astype(np.uint8)

    deg45 = int(img_intensity.shape[1] / 8)
    ri_center = int(img_intensity.shape[1] / 2)
    img_intensity = img_intensity[:, ri_center - deg45:ri_center + deg45]

    #######
    ####### ID_S1_EX1 END #######

    return img_intensity


# create birds-eye view of lidar data
def bev_from_pcl(lidar_pcl, configs):
    # remove lidar points outside detection area and with too low reflectivity
    mask = np.where((lidar_pcl[:, 0] >= configs.lim_x[0]) & (lidar_pcl[:, 0] <= configs.lim_x[1]) &
                    (lidar_pcl[:, 1] >= configs.lim_y[0]) & (lidar_pcl[:, 1] <= configs.lim_y[1]) &
                    (lidar_pcl[:, 2] >= configs.lim_z[0]) & (lidar_pcl[:, 2] <= configs.lim_z[1]))
    lidar_pcl = lidar_pcl[mask]

    # shift level of ground plane to avoid flipping from 0 to 255 for neighboring pixels
    lidar_pcl[:, 2] = lidar_pcl[:, 2] - configs.lim_z[0]

    # convert sensor coordinates to bev-map coordinates (center is bottom-middle)
    ####### ID_S2_EX1 START #######
    #######
    print("student task ID_S2_EX1")

    # The code is borrowed from "Exercise: Transform a Point Cloud into a Birds-Eye View"
    bev_map_discr = (configs.lim_x[1] - configs.lim_x[0]) / configs.bev_height
    lidar_pcl_cpy = np.copy(lidar_pcl)
    lidar_pcl_cpy[:, 0] = np.int_(np.floor(lidar_pcl_cpy[:, 0] / bev_map_discr))

    lidar_pcl_cpy[:, 1] = np.int_(np.floor(lidar_pcl_cpy[:, 1] / bev_map_discr) + (configs.bev_width + 1) / 2)
    lidar_pcl_cpy[:, 1] = np.abs(lidar_pcl_cpy[:, 1])
    show_pcl(lidar_pcl_cpy)

    #######
    ####### ID_S2_EX1 END #######

    # Compute intensity layer of the BEV map
    ####### ID_S2_EX2 START #######
    #######
    print("student task ID_S2_EX2")

    intensity_map = np.zeros((configs.bev_height, configs.bev_width))
    
    lidar_pcl_cpy[lidar_pcl_cpy[:,3]>1.0,3] = 1.0
    idx_intensity = np.lexsort((-lidar_pcl_cpy[:, 2], lidar_pcl_cpy[:, 1], lidar_pcl_cpy[:, 0]))
    lidar_pcl_top = lidar_pcl_cpy[idx_intensity]

    _, idx_intensity_unique, counts = np.unique(lidar_pcl_top[:, 0:2], axis=0, return_index=True, return_counts=True)
    lidar_pcl_top = lidar_pcl_top[idx_intensity_unique]

    intensity_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = lidar_pcl_top[:, 3] / (
                np.amax(lidar_pcl_top[:, 3]) - np.amin(lidar_pcl_top[:, 3]))

    img_intensity = intensity_map * 256
    img_intensity = img_intensity.astype(np.uint8)
    cv2.imshow('img_intensity', img_intensity)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #######
    ####### ID_S2_EX2 END #######

    # Compute height layer of the BEV map
    ####### ID_S2_EX3 START #######
    #######
    print("student task ID_S2_EX3")
    height_map = np.zeros((configs.bev_height, configs.bev_width))
    height_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = lidar_pcl_top[:, 2] / float(
        np.abs(configs.lim_z[1] - configs.lim_z[0]))
    img_height = height_map * 256
    img_height = img_height.astype(np.uint8)
    cv2.imshow('height_map', height_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #######
    ####### ID_S2_EX3 END #######

    # Compute density layer of the BEV map
    density_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    _, _, counts = np.unique(lidar_pcl_cpy[:, 0:2], axis=0, return_index=True, return_counts=True)
    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64))
    density_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = normalizedCounts

    # assemble 3-channel bev-map from individual maps
    bev_map = np.zeros((3, configs.bev_height, configs.bev_width))
    bev_map[2, :, :] = density_map[:configs.bev_height, :configs.bev_width]  # r_map
    bev_map[1, :, :] = height_map[:configs.bev_height, :configs.bev_width]  # g_map
    bev_map[0, :, :] = intensity_map[:configs.bev_height, :configs.bev_width]  # b_map

    # expand dimension of bev_map before converting into a tensor
    s1, s2, s3 = bev_map.shape
    bev_maps = np.zeros((1, s1, s2, s3))
    bev_maps[0] = bev_map

    bev_maps = torch.from_numpy(bev_maps)  # create tensor from birds-eye view
    input_bev_maps = bev_maps.to(configs.device, non_blocking=True).float()
    return input_bev_maps
