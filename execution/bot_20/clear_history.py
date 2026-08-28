import os

FILES = ["cycle_history.jsonl", "last_cycle_data.json"]

for file in FILES:
    if os.path.exists(file):
        with open(file, "w") as f:
            f.write("")  # empty file
        print(f"🗑 Cleared {file}")
    else:
        print(f"⚠️ {file} does not exist")

print("✔ All history cleared.")
