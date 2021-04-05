import os
import pickle


def save_rdms(mean_rdms: dict, out_dir: str, model_name: str, epoch: int):
    # save dict object
    file_name = f"{model_name}_e{epoch:02d}.pkl"
    file_path = os.path.join(out_dir, file_name)
    with open(file_path, "wb") as f:
        pickle.dump(mean_rdms, f)


def load_rdms(in_dir, model_name, epoch):
    file_path = os.path.join(in_dir, f"{model_name}_e{epoch:02d}.pkl")
    with open(file_path, "rb") as f:
        return pickle.load(f)
