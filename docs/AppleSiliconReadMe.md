# Apple Sillicon Podman Setup

This tutorial walks you through installing Lima, creating an x86_64 Podman VM, configuring the Podman CLI to use it, and verifying that everything is working correctly.
This is especially useful on Apple Silicon when you need x86_64 container builds (for example, if ARM builds break under QEMU).

## The Problem
Sometimes Mac cannot translate the commands in one chip architecture so we need a layer to do this for us.
Qemu is usual the software for the job, but in some cases when the process for building images is computationally 
complex the M series mac will Sig fault.

### Symptoms

You might see a failure to build the UI

```
RUN pnpm run build:vite
Failed to run 
```

You might see a failue building the API

```
STEP 9/19: COPY packages/db/ ./packages/db/
--> af95bf5b5140
STEP 10/19: RUN if [ "$TORCH_VARIANT" = "cpu" ]; then         echo "Installing PyTorch CPU version (lightweight, ~176MB)..." &&         uv pip install --python $(which python3) --system --no-cache --index-url https://download.pytorch.org/whl/cpu torch;     else         echo "Installing PyTorch CUDA version (GPU-enabled, ~800MB)..." &&         uv pip install --python $(which python3) --system --no-cache torch;     fi
Installing PyTorch CPU version (lightweight, ~176MB)...
qemu: uncaught target signal 11 (Segmentation fault) - core dumped
Error: building at STEP "RUN if [ "$TORCH_VARIANT" = "cpu" ]; then         echo "Installing PyTorch CPU version (lightweight, ~176MB)..." &&         uv pip install --python $(which python3) --system --no-cache --index-url https://download.pytorch.org/whl/cpu torch;     else         echo "Installing PyTorch CUDA version (GPU-enabled, ~800MB)..." &&         uv pip install --python $(which python3) --system --no-cache torch;     fi": while running runtime: exit status 139
make: *** [build-api] Error 139
```

## The Solution

### 1. Install Lima

```bash
brew install lima
```

### 2. Create an x86_64 Podman VM

Use Lima’s built-in Podman template:

```bash
limactl create --name podman --arch x86_64 --vm-type qemu template:podman
```

You’ll should be prompted:
Proceed with the current configuration?
Choose Yes and wait.
This part downloads the image and boots the VM — it may take a little while.

### 3. Verify Your Lima Instances

Before you continue, check which Lima VMs exist:

```bash
limactl ls
```

Example output:

NAME            STATUS     SSH                VMTYPE    ARCH       CPUS    MEMORY    DISK      DIR
podman          Running    127.0.0.1:49576    qemu      x86_64     4       4GiB      100GiB    ~/.lima/podman
podman-aarch64    Stopped    127.0.0.1:0        vz        aarch64    4       4GiB      100GiB    ~/.lima/podman-arm64


### 4. Configure Podman CLI to Use the Lima Podman VM

Replace <your username> with your macOS username:

```bash
podman system connection add lima-podman \
  "unix:///Users/<your username>/.lima/podman/sock/podman.sock"
```

### 5. Set Lima Podman as the Default Connection

```bash
podman system connection default lima-podman
```

You can confirm:

```bash
podman system connection ls
```

### 6. Test Podman with a Basic Container

```bash
podman run quay.io/podman/hello
```

If you see a greeting message, everything is working 🎉

### 7. Verify You're Running x86_64 Containers

Because the VM is x86_64, images should report x86_64 inside:
```bash
podman run fedora:latest uname -a
```

### 8. You finished!
