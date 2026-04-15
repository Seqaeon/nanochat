curr_tag = None
with open("/home/seqaeon/Downloads/nanochat/sweep_p22.log", "r") as f:
    for line in f:
        line = line.strip()
        if line.startswith("║  ["):
            curr_tag = line.split("]")[1].strip()
        elif line.startswith("Running: /root/nanochat/"):
            if "BASELINE" in curr_tag or "REMIX_WEIGHT_4T_LEARNED" in curr_tag:
                import re
                dim = re.search(r"--model-dim (\d+)", line)
                if dim: print(f"{curr_tag}: model_dim={dim.group(1)}")
