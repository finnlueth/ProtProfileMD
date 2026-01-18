import h5py

def h5_tree_string(val, pre=""):
    output = ""
    items = len(val)
    for key, val in val.items():
        items -= 1
        if items == 0:
            if type(val) is h5py._hl.group.Group:
                output += pre + "└── " + key + "\n"
                output += h5_tree_string(val, pre + "    ")
            else:
                try:
                    output += pre + "└── " + key + " (%d)\n" % len(val)
                except TypeError:
                    output += pre + "└── " + key + " (scalar)\n"
        else:
            if type(val) is h5py._hl.group.Group:
                output += pre + "├── " + key + "\n"
                output += h5_tree_string(val, pre + "│   ")
            else:
                try:
                    output += pre + "├── " + key + " (%d)\n" % len(val)
                except TypeError:
                    output += pre + "├── " + key + " (scalar)\n"
    return output