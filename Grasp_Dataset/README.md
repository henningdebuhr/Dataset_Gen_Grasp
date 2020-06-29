# Dataset-Gen-for-Grasp

Generator script see:
/catkin_ws/dataset_grasp/scripts/dataset_gen.py

one bagfile there, more to come

rosbag record -l 1 /darknet_ros/detection_image /darknet_ros/bounding_boxes /camera/color/image_raw /camera/aligned_depth_to_color/image_raw -O Alu20_cc.bag


some example dataset files in dataset folder
