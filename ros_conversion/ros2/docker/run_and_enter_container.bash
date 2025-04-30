#!/usr/bin/env bash

# GLOBAL PATH to where your binarized dataset is
DATASETS_DIR="$(realpath /PATH/TO/BINARIZED)"

# GLOBAL PATH to CanyonlandsDataset/dataset_convert/ros2/conversion_scripts
SCRIPTS_DIR="$(realpath PATH_TO/CanyonlandsDataset/dataset_convert/ros2/conversion_scripts)"

# give privledge for screen sharing
xhost +local:root

# run Docker container
docker run -it -d --rm --privileged \
  --name canyonlands_dataset_ros2 \
  --net=host \
  --env="DISPLAY=$DISPLAY" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --volume="$DATASETS_DIR:/root/Datasets:rw" \
  --volume="$SCRIPTS_DIR:/root/conversion_scripts:rw" \
  canyonlands_dataset_ros2

docker exec -it canyonlands_dataset_ros2 /bin/bash

