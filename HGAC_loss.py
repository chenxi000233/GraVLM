import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class HGACLoss(nn.Module):
    """
    Hierarchical Graph-Aligned Contrastive Loss (HGAC-Loss)

    Combines:
    - Global contrastive loss between scene-level representations
    - Node-level alignment using optimal transport (Hungarian)
    - Graph structure alignment via adjacency consistency
    """
    def __init__(self, temperature=0.07, lambda_node=1.0, lambda_graph=1.0):
        super(HGACLoss, self).__init__()
        self.temperature = temperature
        self.lambda_node = lambda_node
        self.lambda_graph = lambda_graph

    def compute_global_contrastive(self, v_global, t_global):
        """ Scene-level InfoNCE contrastive loss """
        v_norm = F.normalize(v_global, dim=1)
        t_norm = F.normalize(t_global, dim=1)
        sim = torch.matmul(v_norm, t_norm.T) / self.temperature
        labels = torch.arange(sim.size(0)).to(sim.device)
        loss_i2t = F.cross_entropy(sim, labels)
        loss_t2i = F.cross_entropy(sim.T, labels)
        return 0.5 * (loss_i2t + loss_t2i)

    def compute_node_alignment(self, v_nodes, t_nodes):
        """ Hungarian matching between visual and textual nodes per sample """
        B, N, D = v_nodes.size()
        loss = 0.0
        for b in range(B):
            v = F.normalize(v_nodes[b], dim=-1)  # [N, D]
            t = F.normalize(t_nodes[b], dim=-1)
            cost = 1.0 - torch.matmul(v, t.T).detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost)
            aligned_v = v_nodes[b, row_ind]
            aligned_t = t_nodes[b, col_ind]
            loss += F.mse_loss(aligned_v, aligned_t)
        return loss / B

    def compute_graph_structure(self, A_v, A_t):
        """ MSE loss between normalized adjacency matrices """
        return F.mse_loss(A_v, A_t)

    def forward(self, v_g, t_g, V, T, A_v, A_t):
        """
        Compute total HGAC loss.
        Inputs:
            v_g: [B, D]  - scene-level visual features
            t_g: [B, D]  - scene-level textual features
            V:   [B, N, D] - visual node embeddings (SOG)
            T:   [B, N, D] - textual node embeddings (ERG)
            A_v: [B, N, N] - visual adjacency
            A_t: [B, N, N] - textual adjacency
        Returns:
            scalar loss, component-wise loss dictionary
        """
        L_global = self.compute_global_contrastive(v_g, t_g)
        L_node = self.compute_node_alignment(V, T)
        L_graph = self.compute_graph_structure(A_v, A_t)

        total_loss = L_global + self.lambda_node * L_node + self.lambda_graph * L_graph
        return total_loss, {
            "L_global": L_global.item(),
            "L_node": L_node.item(),
            "L_graph": L_graph.item()
        }
