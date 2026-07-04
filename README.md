# factr2_next

## Environment Setup

Create and activate a Python virtual environment:

```bash
uv venv --python 3.12 --prompt ros .venv
source .venv/bin/activate

uv pip install pip wheel setuptools==79.0.1
uv pip install colcon-core colcon-common-extensions
uv pip install numpy pyyaml termcolor h5py torch cffi

which python
which colcon

source /opt/ros/jazzy/setup.bash
python -m colcon build --symlink-install
source install/setup.bash
```