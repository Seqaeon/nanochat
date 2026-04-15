import re
curr_tag = None
results = {}
with open("/home/seqaeon/Downloads/nanochat/sweep_p22.log", "r") as f:
    for line in f:
        line = line.strip()
        if line.startswith("╔═"):
            pass
        elif line.startswith("║  ["):
            curr_tag = line.split("]", 1)[1].strip()
            if curr_tag not in results:
                results[curr_tag] = {"status": "Started", "bpb": "N/A", "error": []}
        elif "COMPLETE ══" in line and curr_tag is not None:
            if results[curr_tag]["status"] == "Started":
                results[curr_tag]["status"] = "Completed"
        elif "Error training" in line and curr_tag is not None:
            results[curr_tag]["status"] = "Failed"
            results[curr_tag]["error"].append(line)
        elif "Unknown argument:" in line and curr_tag is not None:
            results[curr_tag]["status"] = "Failed"
            results[curr_tag]["error"].append(line)
        elif "AttributeError" in line and curr_tag is not None:
            results[curr_tag]["status"] = "Failed"
            results[curr_tag]["error"].append("AttributeError")
        elif "OutOfMemoryError" in line and curr_tag is not None:
            results[curr_tag]["status"] = "Failed"
            results[curr_tag]["error"].append("OOM")
        elif line.startswith("Minimum validation bpb:"):
            m = re.search(r"Minimum validation bpb: ([\d\.]+)", line)
            if m and curr_tag:
                results[curr_tag]["bpb"] = float(m.group(1))

for k, v in results.items():
    if v["status"] == "Failed":
        print(f"{k}: Failed -> {', '.join(v['error'])}")
    else:
        print(f"{k}: Completed -> BPB: {v['bpb']}")
