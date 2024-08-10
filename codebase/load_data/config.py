from codebase.load_data.m4 import M4Dataset
from codebase.load_data.m3 import M3Dataset
from codebase.load_data.tourism import TourismDataset
from codebase.load_data.m5 import M5Dataset
from codebase.load_data.labour import LabourDataset
from codebase.load_data.traffic import TrafficDataset
from codebase.load_data.wiki2 import Wiki2Dataset
from codebase.load_data.etth1 import ETTH1Dataset
from codebase.load_data.etth2 import ETTH2Dataset
from codebase.load_data.ettm1 import ETTm1Dataset
from codebase.load_data.ettm2 import ETTm2Dataset
from codebase.load_data.ecl import ECLDataset
from codebase.load_data.trafficl import TrafficLDataset
from codebase.load_data.weather import WeatherDataset


DATASETS = {
    "Tourism": TourismDataset,
    "M3": M3Dataset,
    "M4": M4Dataset,
    "M5": M5Dataset,
    "Labour": LabourDataset,
    "Traffic": TrafficDataset,
    "Wiki2": Wiki2Dataset,
    "ETTH1": ETTH1Dataset,
    "ETTH2": ETTH2Dataset,
    "ETTm1": ETTm1Dataset,
    "ETTm2": ETTm2Dataset,
    "ECL": ECLDataset,
    "TrafficL": TrafficLDataset,
    "Weather": WeatherDataset,
}
