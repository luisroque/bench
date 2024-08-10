from codebase.load_data.m4 import M4Dataset
from codebase.load_data.m3 import M3Dataset
from codebase.load_data.tourism import TourismDataset
from codebase.load_data.m5 import M5Dataset
from codebase.load_data.labour import LabourDataset
from codebase.load_data.traffic import TrafficDataset
from codebase.load_data.wiki2 import Wiki2Dataset

DATASETS = {
    "Tourism": TourismDataset,
    "M3": M3Dataset,
    "M4": M4Dataset,
    "M5": M5Dataset,
    "Labour": LabourDataset,
    "Traffic": TrafficDataset,
    "Wiki2": Wiki2Dataset
}
