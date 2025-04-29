from dataclasses import dataclass, field
import diseases_symptoms_matrix

import numpy as np


@dataclass
class BaseSymptomScale:
    MIN: int
    NORMAL: int
    MAX: int
    STD_DEV: float
    value: float
    value_dict: float
    name: str

    def __init__(self, name, value=None, MIN=None, NORMAL=None, MAX=None, STD_DEV=None):
        self.name=name
        self.MIN=MIN
        self.NORMAL=NORMAL
        self.MAX=MAX
        self.STD_DEV=STD_DEV #0.0
        self.value=value
        self.value_dict=value

    def set_value(self, value):
        self.value=value
        self.value_dict=value

    def randomise(self):
        # only if has randomness
        if self.STD_DEV and self.value:
            self.value = float(np.random.normal(loc=self.value, scale=self.STD_DEV, size=1)[0])
            if self.value > self.MAX:
                self.value = self.MAX
            elif self.value < self.MIN:
                self.value = self.MIN

    def describe(self):
        return f"{self.name} | {self.value} | [{self.MIN}, {self.NORMAL}, {self.MAX}] | {self.STD_DEV}"


@dataclass
class GooDensitySymptom(BaseSymptomScale):
    def __init__(self):
        super().__init__(name="Goo Density", value=None, MIN=1, NORMAL=3, MAX=5, STD_DEV=1)

@dataclass
class GooPressureSymptom(BaseSymptomScale):
    def __init__(self):
        super().__init__(name="Goo Pressure", value=None, MIN=1, NORMAL=3, MAX=5, STD_DEV=0.5)

@dataclass
class GooTemperatureSymptom(BaseSymptomScale):
    def __init__(self):
        super().__init__(name="Goo Temperature", value=None, MIN=1, NORMAL=3, MAX=5, STD_DEV=1)

@dataclass
class GooSoundSymptom(BaseSymptomScale):
    def __init__(self):
        super().__init__(name="Goo Sound", value=None, MIN=1, NORMAL=3, MAX=3, STD_DEV=0.2)

@dataclass
class GooPainSymptom(BaseSymptomScale):
    def __init__(self):
        super().__init__(name="Goo Pain", value=None, MIN=1, NORMAL=1, MAX=5, STD_DEV=0.75)

@dataclass
class GooVibrationSymptom(BaseSymptomScale):
    def __init__(self):
        super().__init__(name="Goo Vibration", value=None, MIN=0, NORMAL=1, MAX=1, STD_DEV=0.1)

@dataclass
class GooPerspirationSymptom(BaseSymptomScale):
    def __init__(self):
        super().__init__(name="Goo Perspiration", value=None, MIN=1, NORMAL=1, MAX=3, STD_DEV=0.1)

@dataclass
class GooCommunicationSymptom(BaseSymptomScale):
    def __init__(self):
        super().__init__(name="Goo Communication", value=None, MIN=1, NORMAL=3, MAX=3, STD_DEV=1)

@dataclass
class GooSmellSymptom(BaseSymptomScale):
    def __init__(self):
        super().__init__(name="Goo Smell", value=None, MIN=1, NORMAL=3, MAX=5, STD_DEV=1)

@dataclass
class GooColourSymptom(BaseSymptomScale):
    def __init__(self):
        super().__init__(name="Goo Colour", value=None, MIN=0, NORMAL=3, MAX=3, STD_DEV=0.2)

@dataclass
class GooTransparencySymptom(BaseSymptomScale):
    def __init__(self):
        super().__init__(name="Goo Transparency", value=None, MIN=2, NORMAL=5, MAX=9, STD_DEV=3)

@dataclass
class BaseDisease:
    name: str
    symptoms: list
    symptoms_dict: dict
    def __init__(self, name, symptoms):
        self.name=name
        self.symptoms=symptoms
        self.symptoms_dict={}
        self.symptoms_dict['goo_density']=GooDensitySymptom()
        self.symptoms_dict['goo_density'].set_value(symptoms[0])
        self.symptoms_dict['goo_pressure']=GooPressureSymptom()
        self.symptoms_dict['goo_pressure'].set_value(symptoms[1])
        self.symptoms_dict['goo_temperature']=GooTemperatureSymptom()
        self.symptoms_dict['goo_temperature'].set_value(symptoms[2])
        self.symptoms_dict['goo_sound']=GooSoundSymptom()
        self.symptoms_dict['goo_sound'].set_value(symptoms[3])
        self.symptoms_dict['goo_pain']=GooPainSymptom()
        self.symptoms_dict['goo_pain'].set_value(symptoms[4])
        self.symptoms_dict['goo_vibration']=GooVibrationSymptom()
        self.symptoms_dict['goo_vibration'].set_value(symptoms[5])
        self.symptoms_dict['goo_perspiration']=GooPerspirationSymptom()
        self.symptoms_dict['goo_perspiration'].set_value(symptoms[6])
        self.symptoms_dict['goo_communication']=GooCommunicationSymptom()
        self.symptoms_dict['goo_communication'].set_value(symptoms[7])
        self.symptoms_dict['goo_smell']=GooSmellSymptom()
        self.symptoms_dict['goo_smell'].set_value(symptoms[8])
        self.symptoms_dict['goo_colour']=GooColourSymptom()
        self.symptoms_dict['goo_colour'].set_value(symptoms[9])
        self.symptoms_dict['goo_transparency']=GooTransparencySymptom()
        self.symptoms_dict['goo_transparency'].set_value(symptoms[10]) #10

    def randomise(self):
        self.symptoms=[]
        for key in self.symptoms_dict:
            self.symptoms_dict[key].randomise()
            self.symptoms.append(self.symptoms_dict[key].value)

    def describe(self):
        return f"""
            {self.name}
            symptoms: {self.symptoms}
            symptoms_dict: {self.symptoms_dict}
        """


@dataclass
class A1_Disease(BaseDisease):
    def __init__(self):
        super().__init__(name="A1_Disease", symptoms=diseases_symptoms_matrix.DISEASES_SYMPTOMS_MATRIX[0])

@dataclass
class A2_Disease(BaseDisease):
    def __init__(self):
        super().__init__(name="A2_Disease", symptoms=diseases_symptoms_matrix.DISEASES_SYMPTOMS_MATRIX[1])

@dataclass
class A3_Disease(BaseDisease):
    def __init__(self):
        super().__init__(name="A3_Disease", symptoms=diseases_symptoms_matrix.DISEASES_SYMPTOMS_MATRIX[2])

@dataclass
class B1_Disease(BaseDisease):
    def __init__(self):
        super().__init__(name="B1_Disease", symptoms=diseases_symptoms_matrix.DISEASES_SYMPTOMS_MATRIX[3])

@dataclass
class B2_Disease(BaseDisease):
    def __init__(self):
        super().__init__(name="B2_Disease", symptoms=diseases_symptoms_matrix.DISEASES_SYMPTOMS_MATRIX[4])

@dataclass
class B3_Disease(BaseDisease):
    def __init__(self):
        super().__init__(name="B3_Disease", symptoms=diseases_symptoms_matrix.DISEASES_SYMPTOMS_MATRIX[5])

DISEASES = [
    A1_Disease(),
    A2_Disease(),
    A3_Disease(),
    B1_Disease(),
    B2_Disease(),
    B3_Disease(),
]
