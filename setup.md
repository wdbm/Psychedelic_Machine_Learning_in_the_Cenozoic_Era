# setup 2024-03-25

The setup assumes Ubuntu 22.04 LTS and an Nvidia 1070 GPU with the P507 reference hardware.

- model:
    - Schenker XMG P507 (2017)
    - SKU: XMG-P507-KBL
- GPU:
    - NVIDIA GeForce GTX 1070 8GB GDDR5 with G-SYNC
    - SKU: GPU-GTX-1070-P507-GSYNC
- CPU:
    - Intel Core i7-7820HK, quad Core, 8 threads, 2.90 -- 3.90 GHz, 8 MB, 45 W
    - SKU: KCI-7820HK-P507
- RAM:
    - 32GB (2 x 16384) SO-DIMM DDR4 RAM 2133 MHz Samsung
    - SKU: KR4-2x16GB-2133-SAMSUNG
    - 32 GB (1 x 32768 MB) SO-DIMM DDR4 RAM 3200 MHz Crucial (2024-05-09)
- HD:
    - 512GB m.2 SSD Samsung SM961-NVMe via PCI-Express x4
    - SKU: APHDD-SM961-NVME-M2-512
- boot keys:
    - F2: access BIOS
    - F11: access USB boot

## the infrastructure

The Nvidia Compute Unified Device Architecture (CUDA) is a parallel programming architecture. It acts as an API to communicate with the GPU. cuDNN is a library for deep learning using CUDA. It is critical to ensure that compatible versions of the Nvidia driver, CUDA, cuDNN, Python, pip and TensorFlow are used.

## caveat

It can be challenging to ensure that the various dependencies are sufficiently compatible to result in a working installation. It remains a common occurrence to find that even official TensorFlow installation instructions result in broken installations. In order to avoid confusion and wasted time, only basic setup documentation is maintained here and only on a best-effort basis, while old setup documentation has been removed (and is accessible in past versions of the repository). The author urges users to contact the relevant system developers if there are any installation errors as this is beyond the scope of the author.

## install basic infrastructure

Ensure an appropriate Nvidia driver version is installed.

```Bash
software-properties-gtk
```

- Software & Updates > Additional Drivers

Reboot and then verify the driver version.

```Bash
$ cat /proc/driver/nvidia/version
NVRM version: NVIDIA UNIX x86_64 Kernel Module  545.29.06  Thu Nov 16 01:59:08 UTC 2023
GCC version:  gcc version 12.3.0 (Ubuntu 12.3.0-1ubuntu1~22.04)
```

A check can be made to ensure the driver is available, which should result in a listing of display hardware featuring the Nvidia hardware:

```Bash
$ sudo lshw -c display
  *-display                 
       description: VGA compatible controller
       product: GP104BM [GeForce GTX 1070 Mobile]
       vendor: NVIDIA Corporation
       physical id: 0
       bus info: pci@0000:01:00.0
       version: a1
       width: 64 bits
       clock: 33MHz
       capabilities: pm msi pciexpress vga_controller bus_master cap_list rom
       configuration: driver=nvidia latency=0
       resources: irq:144 memory:db000000-dbffffff memory:90000000-9fffffff memory:a0000000-a1ffffff ioport:e000(size=128) memory:dc000000-dc07ffff
  *-display
       description: VGA compatible controller
       product: HD Graphics 630
       vendor: Intel Corporation
       physical id: 2
       bus info: pci@0000:00:02.0
       logical name: /dev/fb0
       version: 04
       width: 64 bits
       clock: 33MHz
       capabilities: pciexpress msi pm vga_controller bus_master cap_list rom fb
       configuration: depth=32 driver=i915 latency=0 mode=1920x1080 resolution=1920,1080 visual=truecolor xres=1920 yres=1080
       resources: iomemory:2f0-2ef irq:140 memory:2ffe000000-2ffeffffff memory:80000000-8fffffff ioport:f000(size=64) memory:c0000-dffff
```

Install GCC, nvtop, Miniconda, update pip, install CUDA, cuDNN and TensorFlow, and then test that TensorFlow can access the GPU.

```Bash
sudo apt install gcc nvtop
```

```Bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod 755 Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
#rm Miniconda3-latest-Linux-x86_64.sh
```

```Bash
source ~/miniconda3/bin/activate
```

```Bash
conda create --name tf python=3.9
```

```Bash
conda activate tf
```

```Bash
conda install -c conda-forge cudatoolkit=11.8.0
pip install --upgrade pip
pip install nvidia-cudnn-cu11==8.6.0.163
#pip install tensorflow==2.13.*
pip install tensorflow==2.10
```

```Bash
python3.9 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

It is promising if the GPU is detected, but this check does not confirm that the installation has resulted in a working installation. It is prudent to test some Keras functionality to ensure that the GPU is accessible and is being used for TensorFlow calculations. This can be done by opening two terminals. In one terminal, set `nvtop` running. In the other terminal, set a script like the following running:

```Python
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

print(tf.config.list_physical_devices('GPU'))

# Generate nonsense data
num_samples = 100000  # Large number of samples for intensive computation
input_dim = 1000      # High dimensionality for each input sample
output_dim = 10       # Dimensionality of output data
X = np.random.random((num_samples, input_dim))
y = np.random.random((num_samples, output_dim))

# Define a simple sequential model
model = Sequential([
    Dense(512, activation='relu', input_shape=(input_dim,)),
    Dense(256, activation='relu'),
    Dense(output_dim, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy')

# Train the model on the data
model.fit(X, y, epochs=10, batch_size=256)  # Single epoch, large batch size

# Save the model
model.save('model.h5')
```

Hopefully the installation is functional. The environment can be accessed again using Miniconda:

```Bash
source ~/miniconda3/bin/activate
conda activate tf
```

## install other infrastructure

Other, perhaps optional, infrastructure can be installed.

```Bash
sudo apt install git-lfs
git lfs install
```

```Bash
pip install     \
    graphviz    \
    jupyter     \
    keras_tqdm  \
    livelossplot\
    matplotlib  \
    numpy       \
    pandas      \
    pydot       \
    scikit-learn\
    scipy       \
    seaborn     \
    tqdm
```

```Bash
pip install nltk
python -m nltk.downloader all # download to ~/nltk_data
```

```Bash
pip install keras-viz
```

```Bash
pip install psychedelic
```
