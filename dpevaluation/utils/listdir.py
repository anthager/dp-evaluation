import os

# returns all the dirs within the dir
# ignore hidden files (like .DS_Store)
def listdir(path):
    dirs = []
    try:
        with os.scandir(path) as it:
            for entry in it:
                if not entry.name.startswith('.') and entry.is_dir():
                    dirs.append(entry.name)
        return dirs
    except:
        return []
