curr_tag = None
results = {}
capture = False
with open("/home/seqaeon/Downloads/nanochat/sweep_p22.log", "r") as f:
    for line in f:
        line = line.strip()
        if line.startswith("║  ["):
            curr_tag = line.split("]")[1].strip()
        elif "Parameter counts:" in line:
            if curr_tag not in results:
                results[curr_tag] = []
            capture = True
            continue
        
        if capture:
            if "total" in line:
                results[curr_tag].append(line)
                capture = False
            elif ":" in line:
                results[curr_tag].append(line)

for tag in results:
    if "REMIX" in tag:
        print(f"--- {tag} ---")
        for l in results[tag]:
            if "total" in l or "transformer" in l:
                print(l)
