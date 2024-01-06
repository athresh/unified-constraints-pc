import sys 
import unittest
sys.path.append("./../src/")

from packages.pfc.components.spn.Graph import random_binary_trees
from packages.pfc.components.spn.EinsumNetwork import EinsumNetwork, Args, NormalArray
from packages.pfc.components.spn.FlowArray import LinearRationalSpline
import torch 

class TestEinet(unittest.TestCase):
    num_vars                = 8
    depth                   = 3
    num_repetition          = 5
    num_sums                = 6
    exponential_family      = NormalArray
    num_input_distributions = 10
    batch_size              = 5
    graph = random_binary_trees(num_vars, depth, num_repetition)
    device = torch.device('cuda')
    args = Args(
            num_classes=1,
            num_input_distributions=num_input_distributions,
            exponential_family=exponential_family,
            num_sums=num_sums,
            num_var=num_vars
    )
    einet = EinsumNetwork(graph, args)
    einet.initialize()
    einet.to(device)
    
    def test_layers(self):
        self.assertEqual(len(self.einet.einet_layers), self.depth+2)
        self.assertEqual(type(self.einet.einet_layers[0].ef_array), self.exponential_family)
    
    def test_marginalized_forward_pass(self):
        # Generate synthetic input data
        input_data = torch.randn((self.batch_size, self.num_vars)).to(self.device)
        marginalization_idx = torch.randint(0, self.num_vars, (1,)).item()
        
        # Perform forward pass
        output = self.einet(input_data, marginalization_idx=marginalization_idx)
        
        # Corrupt input data along margionalization dimension
        input_data[:, marginalization_idx] = 0
        
        # Validate that prediction does not change
        self.assertTrue(torch.allclose(output, self.einet(input_data, marginalization_idx=marginalization_idx)))
        
        # Validate output shape
        self.assertEqual(output.shape, (self.batch_size, self.args.num_classes))
        
    def test_forward_pass(self):
        # Generate synthetic input data
        input_data = torch.randn((self.batch_size, self.num_vars)).to(self.device)
        
        # Perform forward pass
        output = self.einet(input_data)
        
        # Validate output shape
        self.assertEqual(output.shape, (self.batch_size, self.args.num_classes))
    
    def test_backward_pass(self):
        # Generate synthetic input data
        input_data = torch.randn((self.batch_size, self.num_vars)).to(self.device)
        
        # Generate synthetic target data
        target_data = torch.randint(0, self.args.num_classes, (self.batch_size,)).to(self.device)
        
        # Perform forward pass
        output = self.einet(input_data)
        
        # Calculate loss
        loss = torch.nn.functional.cross_entropy(output, target_data)
        
        # Perform backward pass
        loss.backward()
        
        # Validate gradients
        for param in self.einet.parameters():
            if(param.requires_grad):
                self.assertTrue(param.grad is not None)

# class TestPFC(TestEinet):
#     num_vars                = 8
#     depth                   = 3
#     num_repetition          = 5
#     num_sums                = 6
#     exponential_family      = LinearRationalSpline
#     num_input_distributions = 10
#     batch_size              = 5
#     graph = random_binary_trees(num_vars, depth, num_repetition)
#     device = torch.device('cuda')
#     args = Args(
#             num_classes=1,
#             num_input_distributions=num_input_distributions,
#             exponential_family=exponential_family,
#             num_sums=num_sums,
#             num_var=num_vars
#     )
#     einet = EinsumNetwork(graph, args)
#     einet.initialize()
#     einet.to(device)
    
if __name__ == '__main__':
    unittest.main()