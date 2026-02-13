# Use image as the base
FROM docker.io/andreadelprete/orc24:v1

# Switch to root
USER root

# 1. Install system build tools
RUN apt-get update && apt-get install -y \
        cmake \
        ninja-build \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# 2. Upgrade pip (Essential for metadata fixes)
RUN pip3 install --no-cache-dir --upgrade \
        "pip" \
        "setuptools<70.0.0" \
        "wheel"

# 3. Install PyTorch
RUN pip3 install --no-cache-dir --ignore-installed \
        torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. Install build dependencies
#    - "numpy==1.24.4": Satisfies SciPy (<1.25) and Pinocchio (<2.0)
RUN pip3 install --no-cache-dir \
        "numpy==1.24.4" \
        "packaging<24.0" \
        scikit-build \
        ninja \
        setuptools_scm \
        "casadi>=3.6.5"

# 5. Install l4casadi
RUN pip3 install --no-cache-dir \
        "numpy==1.24.4" \
        l4casadi \
        liecasadi \
        scikit-learn \
        tqdm \
        joblib \
        --no-build-isolation

# Switch back to student
USER student
WORKDIR /home/student/orc_project/src

ENV PYTHONPATH="/home/student/orc_project/src:/opt/openrobots/lib/python3.10/site-packages"
