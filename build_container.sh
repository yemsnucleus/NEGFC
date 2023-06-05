docker build -t negfc_cnn \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) .