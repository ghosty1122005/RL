from dataclasses import dataclass, field
import copy
from diseases import *

@dataclass
class HmmEntity():
    name: str
    disease: BaseDisease
    #
    num_tests: int = 0
    MAX_TESTS: int = 5

    def __init__(self, name, disease=None):
        self.name = name
        self.set_disease(disease)

    def set_disease(self, disease):
        if disease:
            self.disease=copy.deepcopy(disease)
        else:
            self.disease = None

    def disease_randomise(self):
        self.disease.randomise()

    def describe(self):
        return f"{self.name} | {self.disease}"