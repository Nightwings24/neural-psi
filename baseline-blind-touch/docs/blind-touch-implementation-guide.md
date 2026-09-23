# Blind-Touch Implementation Guide
## Homomorphic Encryption Fingerprint Auth on Fedora with Docker

**Paper:** Blind-Touch (AAAI'24) - HE-based distributed neural network inference  
**Repo:** https://github.com/hm-choi/blind-touch  
**Your setup:** Fedora host → 2 Docker containers communicating via shared volume

---

## Before You Start

### Claude model recommendation
- **Day-to-day work:** Claude Sonnet 4.6 (what you're using now) - fast, great for writing Dockerfiles, debugging Python, adapting notebooks
- **Hard build failures:** Switch to Claude Opus 4.6 with extended thinking enabled when you hit SEAL-Python compilation errors or TensorFlow version conflicts
- **Effort level:** Use normal mode for setup steps; enable extended thinking explicitly when debugging cryptographic library build failures

### What you're building
The original paper uses 5 servers (1 client + 1 main server + 3 cluster servers) with NAS storage for ciphertext transfer.  
For this local demo you'll use **2 Docker containers** on a single machine:
- `blindtouch-client` - runs CNN feature extraction and SEAL encryption
- `blindtouch-server` - runs HE inference (acts as Cluster 1)
- A **shared Docker volume** replaces NAS

The CKKS homomorphic encryption means ciphertexts can be sent over any channel - security comes from the math, not the transport.

### Hardware requirements
- 8 GB RAM minimum (HE operations are memory-hungry)
- 16 GB recommended
- No GPU required for the demo (CPU-only works, inference will be slow ~10-30s per query)
- If you want GPU: CUDA 11.4 + cuDNN 8.1 (matches TF 2.11.0 requirements)

---

## Phase 1 - Fedora Host Setup

### 1.1 Install Docker and Docker Compose

```bash
# Remove old versions
sudo dnf remove docker docker-client docker-client-latest \
    docker-common docker-latest docker-latest-logrotate \
    docker-logrotate docker-engine

# Add Docker repo
sudo dnf -y install dnf-plugins-core
sudo dnf config-manager --add-repo https://download.docker.com/linux/fedora/docker-ce.repo

# Install Docker Engine
sudo dnf install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Start and enable
sudo systemctl start docker
sudo systemctl enable docker

# Add your user to the docker group (avoids sudo every time)
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker run hello-world
```

### 1.2 Install Git and clone the repo

```bash
sudo dnf install git
git clone https://github.com/hm-choi/blind-touch.git
cd blind-touch
ls
# You should see: client/  server/  training/  README.md  requirements.txt
```

---

## Phase 2 - Project Structure Setup

### 2.1 Create your working directory

```bash
mkdir -p ~/blindtouch-demo
cd ~/blindtouch-demo

# Copy the repo contents in
cp -r ~/blind-touch/* .

# Create shared volume directory (simulates NAS)
mkdir -p shared_data/keys
mkdir -p shared_data/ciphertexts
mkdir -p shared_data/models

ls -la
```

### 2.2 Download the SOKOTO dataset

The SOKOTO dataset (6,000 fingerprint images, free to use) is the easier option since PolyU requires emailing HKPolyU for access.

```bash
# Install Kaggle CLI on your Fedora host
pip3 install kaggle

# Configure Kaggle API key:
# 1. Go to https://www.kaggle.com/settings
# 2. Click "Create New API Token" - downloads kaggle.json
mkdir -p ~/.kaggle
cp ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download SOKOTO
mkdir -p ~/blindtouch-demo/dataset
cd ~/blindtouch-demo/dataset
kaggle datasets download -d ruizgara/socofing
unzip socofing.zip
# You'll find Real/ folder with 6000 fingerprint images (96x103 px)
```

---

## Phase 3 - Dockerfile and Docker Compose

### 3.1 Create the shared base Dockerfile

Both containers share the same base image. Create `~/blindtouch-demo/Dockerfile.base`:

```dockerfile
FROM ubuntu:20.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.8 python3.8-dev python3-pip \
    git build-essential cmake \
    libgl1-mesa-glx libglib2.0-0 \
    wget curl \
    && rm -rf /var/lib/apt/lists/*

# Make python3.8 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
RUN update-alternatives --install /usr/bin/pip3 pip3 /usr/bin/pip 1

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# Install SEAL-Python (the core HE library - takes 5-10 min to build)
# This must come before TensorFlow to avoid dependency conflicts
RUN pip3 install numpy pybind11
RUN git clone https://github.com/Huelse/SEAL-Python.git /opt/SEAL-Python
WORKDIR /opt/SEAL-Python
RUN git submodule update --init --recursive
RUN cd SEAL && \
    cmake -S . -B build \
      -DSEAL_USE_MSGSL=OFF \
      -DSEAL_USE_ZLIB=OFF \
      -DSEAL_USE_ZSTD=OFF && \
    cmake --build build
RUN python3 setup.py build_ext -i
RUN cp seal.*.so /usr/local/lib/python3.8/dist-packages/ 2>/dev/null || \
    pip3 install .

# Python packages from requirements.txt
# Note: installing in this order avoids Keras version conflicts
RUN pip3 install numpy==1.24.1
RUN pip3 install tensorflow==2.11.0
RUN pip3 install keras==2.13.1
RUN pip3 install scikit-learn==1.2.1 \
                 pandas==1.5.3 \
                 imgaug==0.4.0 \
                 opencv-python-headless==4.5.3.56 \
                 requests==2.28.2

# Jupyter for running notebooks
RUN pip3 install jupyter

WORKDIR /workspace
```

**Important:** The SEAL-Python build compiles C++ code. It takes 5-10 minutes but only happens once. If the build fails, see Troubleshooting at the end of this guide.

### 3.2 Dockerfile for the client container

Create `~/blindtouch-demo/Dockerfile.client`:

```dockerfile
FROM blindtouch-base:latest

COPY client/ /workspace/client/
COPY training/ /workspace/training/
COPY shared_data/ /workspace/shared_data/

# Expose Jupyter port
EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", \
     "--no-browser", "--allow-root", "--NotebookApp.token=blindtouch"]
```

### 3.3 Dockerfile for the server container

Create `~/blindtouch-demo/Dockerfile.server`:

```dockerfile
FROM blindtouch-base:latest

COPY server/ /workspace/server/
COPY shared_data/ /workspace/shared_data/

# Expose Jupyter port on a different host port
EXPOSE 8889

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8889", \
     "--no-browser", "--allow-root", "--NotebookApp.token=blindtouch"]
```

### 3.4 Docker Compose file

Create `~/blindtouch-demo/docker-compose.yml`:

```yaml
version: "3.9"

services:
  client:
    build:
      context: .
      dockerfile: Dockerfile.client
    image: blindtouch-client:latest
    container_name: blindtouch-client
    ports:
      - "8888:8888"
    volumes:
      - shared_ckks:/workspace/shared_data
      - ./dataset:/workspace/dataset:ro
    networks:
      - blindtouch-net
    environment:
      - SERVER_HOST=server
      - SERVER_PORT=5000

  server:
    build:
      context: .
      dockerfile: Dockerfile.server
    image: blindtouch-server:latest
    container_name: blindtouch-server
    ports:
      - "8889:8889"
    volumes:
      - shared_ckks:/workspace/shared_data
    networks:
      - blindtouch-net

volumes:
  shared_ckks:
    driver: local

networks:
  blindtouch-net:
    driver: bridge
```

The `shared_ckks` Docker volume is the key part: both containers mount it, so ciphertexts and keys written by the client are immediately visible to the server - exactly like the NAS in the original paper.

---

## Phase 4 - Build and Launch

### 4.1 Build the base image first

```bash
cd ~/blindtouch-demo

# Build base (takes 10-15 min first time due to SEAL-Python compilation)
docker build -t blindtouch-base:latest -f Dockerfile.base .

# Watch for: "Successfully installed seal-python" or "build_ext successful"
# If it fails, see Troubleshooting section
```

### 4.2 Build and start both containers

```bash
# Build both service images
docker compose build

# Start both containers in the background
docker compose up -d

# Verify both are running
docker compose ps

# Check logs if something looks wrong
docker compose logs client
docker compose logs server
```

### 4.3 Access the Jupyter notebooks

Open two browser tabs:
- **Client:** http://localhost:8888  (token: `blindtouch`)
- **Server:** http://localhost:8889  (token: `blindtouch`)

You should see the workspace files in each.

---

## Phase 5 - Running the Experiment

### 5.1 Train the model (client container)

In the **client** Jupyter (port 8888):

1. Navigate to `training/`
2. Open `Blind-Touch-Training-SampleNotebook(SOKOTO).ipynb`
3. Update the dataset path at the top to point to `/workspace/dataset/SOCOFing/Real/`
4. Run all cells - this will:
   - Preprocess images (resize to 224×224)
   - Train two models: `feature_model` (outputs 16-dim vector) and `model` (full classifier)
   - Save both to `/workspace/shared_data/models/`

Training takes 10-30 minutes on CPU depending on your hardware.

### 5.2 Generate CKKS keys (client container)

In the **client** Jupyter:

1. Open `client/Blind-Touch-Client.ipynb`
2. Run the key generation cells - this creates:
   - `public_key` - safe to share with server
   - `galois_key` - needed for server-side rotations
   - `relin_key` - needed for relinearization after multiplication
   - All saved to `/workspace/shared_data/keys/`

Because the server container mounts the same `shared_ckks` volume, keys are instantly available to it.

### 5.3 Start the server (server container)

In the **server** Jupyter (port 8889):

1. Navigate to `server/`
2. Open `Blind-Touch-Server(Cluster1).ipynb` (or equivalent cluster notebook)
3. Update paths to point to `/workspace/shared_data/`
4. Run the server cells - the server will:
   - Load the keys from shared volume
   - Wait for incoming ciphertexts
   - Perform HE inference using the FC-1 layer
   - Write inference results back to shared volume

### 5.4 Run authentication (client container)

Back in the **client** Jupyter:

1. Continue in `Blind-Touch-Client.ipynb`
2. Run the ciphertext generation cells:
   - Extracts a 16-dim feature vector from a test fingerprint using `feature_model`
   - Encrypts it into CKKS ciphertext using your generated keys
   - Writes ciphertexts to `/workspace/shared_data/ciphertexts/`
3. Wait for the server to process (check server logs)
4. Run the comparison cells:
   - Reads compressed score from shared volume
   - Compares against threshold
   - Outputs: **Authenticated** or **Rejected**

---

## Phase 6 - Verifying Secure Communication

To confirm encrypted data is flowing correctly:

```bash
# Watch the shared volume for new files
docker exec blindtouch-client watch -n1 ls /workspace/shared_data/ciphertexts/

# Print the ciphertext size (should be ~300-400 KB per the paper)
docker exec blindtouch-client du -sh /workspace/shared_data/ciphertexts/*

# Confirm the server can see the same files
docker exec blindtouch-server ls /workspace/shared_data/ciphertexts/

# Check container network connectivity
docker exec blindtouch-client ping -c 3 blindtouch-server
```

To inspect network traffic between containers:

```bash
# Install tcpdump in a running container (temporary)
docker exec blindtouch-client apt-get install -y tcpdump
docker exec blindtouch-client tcpdump -i eth0 -n
```

---

## Phase 7 - Notebook-to-Script Adaptation (Optional)

The repo ships as Jupyter notebooks. If you want to run everything as Python scripts (more suitable for automation):

```bash
# Inside a container, convert a notebook to a script
jupyter nbconvert --to script client/Blind-Touch-Client.ipynb
# Outputs: client/Blind-Touch-Client.py

# Edit the script to use hardcoded paths instead of interactive input
# Then run:
python3 client/Blind-Touch-Client.py
```

---

## Troubleshooting

### SEAL-Python build fails with "C++17 required"
```bash
# Force a newer GCC in the Dockerfile:
apt-get install -y gcc-10 g++-10
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 10
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 10
```

### SEAL-Python: "undefined symbol" on import
```bash
# Build as shared library instead:
cd /opt/SEAL-Python/SEAL
cmake -S . -B build -DSEAL_USE_MSGSL=OFF -DSEAL_USE_ZLIB=OFF -DSEAL_USE_ZSTD=OFF \
      -DBUILD_SHARED_LIBS=ON
cmake --build build
sudo ln -s /opt/SEAL-Python/SEAL/build/lib/libseal*.so /usr/lib/
sudo ldconfig
```

### TensorFlow version conflicts with Keras
The requirements.txt lists both `keras` (unpinned) and `keras==2.13.1`. Install in this exact order to avoid the standalone Keras vs tf.keras conflict:
```bash
pip3 install tensorflow==2.11.0
pip3 uninstall keras -y
pip3 install keras==2.13.1
```

### "No module named 'seal'" inside a notebook
The `.so` file must be on the Python path. Check:
```python
import sys
print(sys.path)
# Add the SEAL-Python directory if needed:
sys.path.insert(0, '/opt/SEAL-Python')
import seal
print(seal.__version__)
```

### Ciphertexts not visible across containers
Verify the volume is correctly mounted:
```bash
docker inspect blindtouch-client | grep -A 10 Mounts
docker inspect blindtouch-server | grep -A 10 Mounts
# Both should show the same volume name for /workspace/shared_data
```

### Running out of memory during HE operations
CKKS with `d=16384` and depth 3 needs ~4 GB per container. If you hit OOM:
- Reduce `poly_modulus_degree` from 16384 to 8192 (fewer encrypted slots but lower memory)
- Increase Docker memory limits: `mem_limit: 6g` in docker-compose.yml

---

## Understanding What's Happening Cryptographically

The CKKS scheme used here allows arithmetic on encrypted floating-point numbers:
- **Depth limit:** Keys are generated to support exactly 3 multiplications (one per layer)
- **Slots:** With `d=16384`, each ciphertext holds 8192 values (the 16-dim feature vector gets packed with batch/rotation tricks)
- **Key sizes:** Public key ~240 KB, Galois key ~240 MB (large, transferred once at registration), ciphertext ~48 KB
- **Compression method:** The main server uses SIMD-style packing to process 8192 authentication results in a single ciphertext operation - this is the paper's key contribution

The server **never sees your fingerprint** or even your feature vector in plaintext. It operates entirely on ciphertexts, and the result it sends back is also encrypted. Only your client's secret key can decrypt the final score.

---

## Quick Reference

```bash
# Start everything
cd ~/blindtouch-demo && docker compose up -d

# Stop everything
docker compose down

# Rebuild after code changes
docker compose build --no-cache && docker compose up -d

# Open a shell in the client container
docker exec -it blindtouch-client /bin/bash

# View logs in real time
docker compose logs -f

# Clean up volumes (WARNING: deletes shared data including trained models)
docker compose down -v
```

---

*Guide based on AAAI'24 paper: "Blind-Touch: Homomorphic Encryption-Based Distributed Neural Network Inference for Privacy-Preserving Fingerprint Authentication" by Choi, Woo, and Kim.*
