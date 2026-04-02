import json
import glob

# Get all matching files
files = sorted(glob.glob("analysis_lab9-*.json"))

data = []

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        content = json.load(f)
        data.append(content)

# Write combined array to new file
with open("combined.json", "w", encoding="utf-8") as out:
    json.dump(data, out, indent=2)

print(f"Combined {len(data)} files into combined.json")
