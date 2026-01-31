# IEMOCAP -> common labels
iemocap_label_map = {
    "ang": "angry",
    "hap": "happy",
    "exc": "happy",
    "sad": "sad",
    "neu": "neutral",
    "fea": "other",
    "dis": "other",
    "fru": "other",
    "sur": "other",
    "oth": "other",
    "xxx": "other"
}

# CREMA-D -> common labels
cremad_label_map = {
    "ANG": "angry",
    "HAP": "happy",
    "SAD": "sad",
    "NEU": "neutral",
    "FEA": "other",
    "DIS": "other"
}

def map_label(dataset, original_label):
    dataset = dataset.lower()
    if dataset == "iemocap":
        return iemocap_label_map.get(original_label, "other")
    elif dataset == "cremad":
        return cremad_label_map.get(original_label, "other")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
