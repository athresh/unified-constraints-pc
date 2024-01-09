from packages.pfc.models.base import TractableModel
from packages.pfc.components.spn.ExponentialFamilyArray import NormalArray, CategoricalArray, BinomialArray

class EinsumNet(TractableModel):
    def __init__(self, config):
        self.leaf_distribution = eval(config.leaf_type) #NormalArray
        super().__init__(config)