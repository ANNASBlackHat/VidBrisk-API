import json

with open("output/baseline_timeline.json") as f:
    base = json.load(f)

with open("output/cinematic_timeline.json") as f:
    cine = json.load(f)

print("=== BASELINE (GREEDY) TIMELINE ===")
for track in base["tracks"]:
    if track.get("type") == "video":
        for item in track.get("items", []):
            ts = item.get("trackStart", 0.0)
            te = item.get("trackEnd", 0.0)
            aid = item.get("assetId")
            props = item.get("props")
            iid = item.get("id")
            print(f"[{ts:.2f}s - {te:.2f}s] ID={iid} Asset={aid} Props={props}")

print("\n=== CINEMATIC (VITERBI + VIRTUAL CAMERA) TIMELINE ===")
for track in cine["tracks"]:
    if track.get("type") == "video":
        for item in track.get("items", []):
            ts = item.get("trackStart", 0.0)
            te = item.get("trackEnd", 0.0)
            aid = item.get("assetId")
            props = item.get("props")
            iid = item.get("id")
            print(f"[{ts:.2f}s - {te:.2f}s] ID={iid} Asset={aid} Props={props}")
