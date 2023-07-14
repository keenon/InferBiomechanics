import os
import hashlib
import shutil
import nimblephysics as nimble

def compute_hash(file_path):
    """
    Compute SHA256 hash of a file's content
    """
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        content = f.read()
        hasher.update(content)
    return int(hasher.hexdigest(), 16)

def split_files(base_dir, train_percentage, train_dir, dev_dir, expecting_dofs):
    """
    Iterate over all *.bin files in a directory hierarchy,
    compute file hash, and move to train or dev directories.
    """
    threshold = train_percentage * (2**256)
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.bin'):
                file_path = os.path.join(root, file)
                file_hash = compute_hash(file_path)
                subject = nimble.biomechanics.SubjectOnDisk(file_path)
                skel = subject.readSkel()
                dofs = skel.getNumDofs()
                if dofs != expecting_dofs:
                    print('Skipping ' + file_path + ' with unexpected number of DOFs: ' + str(dofs))
                    continue
                if file_hash < threshold:
                    if not os.path.exists(os.path.join(train_dir, file)):
                        shutil.copy(file_path, train_dir)
                else:
                    if not os.path.exists(os.path.join(dev_dir, file)):
                        shutil.copy(file_path, dev_dir)

# Modify these paths as needed
base_dir = './data/processed'
train_dir = './data/train'
dev_dir = './data/dev'

# Ensure output directories exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(dev_dir, exist_ok=True)

# Call the function
split_files(base_dir, 0.8, train_dir, dev_dir, 23)