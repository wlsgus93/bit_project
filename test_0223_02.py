import argparse
import pyrealsense2 as rs
import numpy as np
import cv2
import os


def main():
    if not os.path.exists(args.directory):
        os.mkdir(args.directory)
    try:
        config = rs.config()
        rs.config.enable_device_from_file(config, args.input)
        pipeline = rs.pipeline()
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        pipeline.start(config)
        i = 0
        colorizer = rs.colorizer()
        while True:
            print("Saving frame:", i)
            frames = rs.composite_frame(rs.frame())
            # Check if new frame is ready
            if pipeline.poll_for_frames(frames):
                depth_frame = frames.get_depth_frame()
                depth_image = np.asanyarray(depth_frame.get_data())
                # cv2.imwrite(args.directory + "/" + str(i).zfill(6) + ".png", depth_image)
                print(depth_frame.get_data())
                i += 1
            # wait logic until frame is ready
            else:
                print("Waiting for frame to be ready")
    finally:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", type=str, help="Path to save the images")
    parser.add_argument("-i", "--input", type=str, help="Bag file to read")
    args = parser.parse_args()
    main()