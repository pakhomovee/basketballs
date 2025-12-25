import kagglehub
import shutil
import os

# download kaggle dataset to temporary directory
path = kagglehub.dataset_download("atomscott/teamtrack")
# move it to the dataset directory
target_dir = os.path.join(os.environ.get("BUILD_WORKING_DIRECTORY", "."), "dataset")
os.makedirs(target_dir, exist_ok=True)
if os.path.isdir(path):
    for item in os.listdir(path):
        s = os.path.join(path, item)
        d = os.path.join(target_dir, item)
        if os.path.isdir(s):
            if os.path.exists(d):
                shutil.rmtree(d)
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)
