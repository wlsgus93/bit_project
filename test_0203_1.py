import pyrealsense2 as rs
pipe = rs.pipeline()
profile = pipe.start()
try:
  for i in range(0, 100):
    frames = pipe.wait_for_frames()
    for frame in frames:
      print(frame.profile)
finally:
    
    pipe.stop()
