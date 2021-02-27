import pathlib

from src.data_structures import DataInformation, PivotInfo
from src.explorer import Explorer
from src.cascading_bandit import CascadeLinTS, CascadeLinUCB

"""
This is an example of config. It can be treated as a starting point when 
playing with MAB.
"""

N_FEATURES = 16
N_ROUNDS = 3
UCB_C_VALUE = 2
DATA_PATH = ""

dataset_info = DataInformation(
    path=pathlib.Path(DATA_PATH),
    sep="\t",
    header=["user_id", "item_id", "rating", "timestamp"],
    create=True,
    raw_rating_column_name="rating",
    rating_column_expression=lambda x: 1 if x > 3 else 0,
    pivot_info=PivotInfo(values="rating", columns="item_id", index="user_id"),
)

explorer = Explorer(dataset_info, N_FEATURES)

bandits = [
    CascadeLinTS(features=explorer.get_features()),
    CascadeLinUCB(c=UCB_C_VALUE, features=explorer.get_features()),
]
