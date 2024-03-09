from collections import OrderedDict
from typing import Any, Dict, Dict as OrderedDictType

from robustbench.model_zoo.imagenet import imagenet_models
from robustbench.model_zoo.enums import BenchmarkDataset, ThreatModel

ModelsDict = OrderedDictType[str, Dict[str, Any]]
ThreatModelsDict = OrderedDictType[ThreatModel, ModelsDict]
BenchmarkDict = OrderedDictType[BenchmarkDataset, ThreatModelsDict]

model_dicts: BenchmarkDict = OrderedDict([
    (BenchmarkDataset.imagenet, imagenet_models)
])
