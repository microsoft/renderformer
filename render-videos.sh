RENDERFORMER_VIDEO_DATA_PATH=./video-data  # location of your downloaded renderformer-video-data

# Animations
python3 batch_infer.py --h5_folder $RENDERFORMER_VIDEO_DATA_PATH/animations/marching-cubes/ --output_dir output/videos/marching-cubes --tone_mapper agx
python3 batch_infer.py --h5_folder $RENDERFORMER_VIDEO_DATA_PATH/animations/man/ --output_dir output/videos/man --tone_mapper agx
python3 batch_infer.py --h5_folder $RENDERFORMER_VIDEO_DATA_PATH/animations/crab/ --output_dir output/videos/crab
python3 batch_infer.py --h5_folder $RENDERFORMER_VIDEO_DATA_PATH/animations/gyroscope/ --output_dir output/videos/gyroscope
python3 batch_infer.py --h5_folder $RENDERFORMER_VIDEO_DATA_PATH/animations/robot/ --output_dir output/videos/robot
python3 batch_infer.py --h5_folder $RENDERFORMER_VIDEO_DATA_PATH/animations/cascade-cube/ --output_dir output/videos/cascade-cube --tone_mapper pbr_neutral

# Simulations
python3 batch_infer.py --h5_folder $RENDERFORMER_VIDEO_DATA_PATH/simulations/constant-width-sim/ --output_dir output/videos/constant-width-sim
python3 batch_infer.py --h5_folder $RENDERFORMER_VIDEO_DATA_PATH/simulations/box-rotation/ --output_dir output/videos/box-rotation --tone_mapper pbr_neutral
python3 batch_infer.py --h5_folder $RENDERFORMER_VIDEO_DATA_PATH/simulations/bowling/ --output_dir output/videos/bowling --tone_mapper filmic

# Teaser Scenes
python3 batch_infer.py --h5_folder $RENDERFORMER_VIDEO_DATA_PATH/teaser-scenes/cbox-bunny-roughness/ --output_dir output/videos/cbox-bunny-roughness
python3 batch_infer.py --h5_folder $RENDERFORMER_VIDEO_DATA_PATH/teaser-scenes/cbox-roughness/ --output_dir output/videos/cbox-roughness
python3 batch_infer.py --h5_folder $RENDERFORMER_VIDEO_DATA_PATH/teaser-scenes/compose-change-light/ --output_dir output/videos/compose-change-light
python3 batch_infer.py --h5_folder $RENDERFORMER_VIDEO_DATA_PATH/teaser-scenes/constant-width-fancy/ --output_dir output/videos/constant-width-fancy
python3 batch_infer.py --h5_folder $RENDERFORMER_VIDEO_DATA_PATH/teaser-scenes/tree-change-light/ --output_dir output/videos/tree-change-light
python3 batch_infer.py --h5_folder $RENDERFORMER_VIDEO_DATA_PATH/teaser-scenes/tree-rot-obj/ --output_dir output/videos/tree-rot-obj
