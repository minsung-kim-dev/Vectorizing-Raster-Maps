import glob, json, os
import pandas as pd

out = "eval_outputs/osm_stage2_queens_4000pretrain_tuned_v3/test/epoch_last"                       # an eval.py output folder run with --save_pred
names = {i["id"]: i["id"] for i in json.load(open("/share/garg/jrg377/coco_osm_fixed/annotations/test.json"))["images"]}
rows = [{"image": names[int(os.path.basename(f)[:-5])], **json.load(open(f))} for f in glob.glob(f"{out}/result_jsons/*.json")]
df = pd.DataFrame(rows).set_index("image")
print(df.sort_values("all_f1", ascending=False)[["all_f1", "room_f1", "all_corner_f1"]].round(3).head(20))   # best
print(df.sort_values("all_f1")[["all_f1", "room_f1", "all_corner_f1"]].round(3).head(15))