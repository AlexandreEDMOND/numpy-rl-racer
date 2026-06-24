import csv
import os


class TrainingLogger:
    def __init__(self, filepath, fieldnames=None):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.file = open(filepath, "w", newline="")
        if fieldnames is None:
            fieldnames = ["episode", "total_reward", "steps", "avg_loss", "epsilon", "avg_q_value", "elapsed_time"]
        self.fieldnames = fieldnames
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        self.writer.writeheader()
        self.file.flush()

    def log(self, episode, **metrics):
        row = {"episode": episode}
        row.update(metrics)
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        if self.file and not self.file.closed:
            self.file.close()
