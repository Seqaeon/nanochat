import modal

# 1. Define the base image
# We use debian_slim and install uv first for the fastest build times.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("uv")
    # 2. Add your config files so uv can see them during build
    .add_local_file("pyproject.toml", remote_path="/root/pyproject.toml")
    # 3. Run uv sync to install everything exactly as defined
    # We use --system to install into the container's global site-packages
    .run_commands(
        "uv pip install --system --compile-bytecode -r /root/pyproject.toml",
        # Ensure the build environment knows about your custom torch index if needed
        # though uv usually finds it automatically from the [tool.uv.sources] in the toml
    )
)

app = modal.App("nanochat-env", image=image)

# This dummy function ensures the image gets built and registered
@app.function()
def dummy():
    pass
