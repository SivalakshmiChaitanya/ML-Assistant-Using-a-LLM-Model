import json


class ModelMonitor:

    def __init__(self, stats_file="training_stats.json"):

        with open(stats_file, "r") as f:
            self.stats = json.load(f)

    def check_feature(self, name, value):

        mean = self.stats[f"{name}_mean"]
        std = self.stats[f"{name}_std"]

        if std == 0:
            return

        z_score = abs((value - mean) / std)

        if z_score > 3:
            print(f"WARNING: Possible drift detected in {name}")

    def check_all_features(self, features):

        self.check_feature("actual_price", features["actual_price"])
        self.check_feature("rating", features["rating"])
        self.check_feature("rating_count", features["rating_count"])