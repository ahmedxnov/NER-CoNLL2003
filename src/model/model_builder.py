from sklearn_crfsuite import CRF
from config import CRF_PARAMS

def build_crf_model(args : dict) -> CRF:
    model = CRF(**CRF_PARAMS)
    return model