import os,sys,inspect
import pandas as pd

# get the os PATH's value
print("the PATH is: ", os.environ['PATH'])

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("currentdir: ", current_dir)
parent_dir = os.path.dirname(current_dir)
print("parentdir: ", parent_dir)
grandparent_dir = os.path.dirname(parent_dir)
print("grandparent dir: ", grandparent_dir)
# not encouraged to use this insert function but to use the append function sys.path.append(grandparent_dir)
# or non-ambiguous names for your files and methods
# sys.path.insert(0, grandparent_dir)

ds_path = os.path.join(grandparent_dir, 'dataset')
print("dataset dir: ", ds_path)
train_csv_path = os.path.join(ds_path, 'train.csv')
df_train_csv = pd.read_csv(train_csv_path)