from typing import List, Optional, Callable
from pathlib import Path
from dataclasses import dataclass, asdict

from dill.source import getsource


@dataclass
class PivotInfo:
    index: str
    columns: str
    values: str


@dataclass
class DataInformation:
    path: Path
    sep: str
    create: bool
    header: Optional[List[str]] = None
    rating_column_name: Optional[str] = None
    raw_rating_column_name: Optional[str] = None
    rating_column_expression: Optional[Callable[[int], int]] = None
    pivot_info: Optional[PivotInfo] = None

    def to_dict(self):
        return {
            "path": self.path.as_posix(),
            "sep": self.sep,
            "create": self.create,
            "header": self.header,
            "rating_column_name": self.rating_column_name,
            "raw_rating_column_name": self.raw_rating_column_name,
            "rating_column_expression": self.get_expression(),
            "pivot_info": asdict(self.pivot_info),
        }

    def get_expression(self):
        full_expr = getsource(self.rating_column_expression)
        return full_expr.split("=")[1].replace(",\n", "")
