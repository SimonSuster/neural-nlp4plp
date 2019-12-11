from util import get_file_list, load_json

dir = "/mnt/b5320167-5dbd-4498-bf34-173ac5338c8d/Datasets/nlp4plp/data/examples/"
fs = get_file_list(dir, identifiers=[".json"], all_levels=True)
for f in fs:
    j = load_json(f)
    if "text" in j and "original_text" in j:
        if "girl" in " ".join(j["text"]):
            print(" ".join(j["text"]))
            print(" ".join(j["original_text"]))
#    else:
#        print("n/a")