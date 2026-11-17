import glob, json, sys
tot = {}
# iterate over all the sums
for f in glob.glob(f"{sys.argv[1]}/part*/sums.json"):
    for k, (s, n) in json.load(open(f)).items():
        # get totals
        # add sum and image count (s,n respectively)
        tot[k] = [tot.get(k, [0, 0])[0] + s, tot.get(k, [0, 0])[1] + n]

# reproduce results, averaging over images
res = {k: s / n for k, (s, n) in tot.items() if n}
res |= {p.replace("prec", "f1"): 2 * res[p] * res[p.replace("prec", "rec")] / ((res[p] + res[p.replace("prec", "rec")]) or 1)
        for p in list(res) if "prec" in p and p.replace("prec", "rec") in res}
json.dump(res, open(f"{sys.argv[1]}/results.txt", "w"), indent=1)