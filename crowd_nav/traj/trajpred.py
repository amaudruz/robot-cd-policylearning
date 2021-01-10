import torch

class TrajPred() :
    
    def __init__(self, position_head, decoder, position_emb, pred_length=3) -> None:
        self.pred_length = pred_length
        self.decoder = decoder 
        self.position_emb = position_emb
        self.position_head = position_head

    def loss(self, features, trajectory_truth, last_human_pos) :
        assert(trajectory_truth.size(1) >= self.pred_length)
        device = 'cpu'
        if features.is_cuda :
            device = 'cuda'

        bs, n_humans, hidden_size = features.shape
        features = features.reshape(bs*n_humans, hidden_size)
        last_human_pos = last_human_pos.reshape(bs*n_humans, 2)
        h_state = (features, torch.zeros_like(features))

        # trajectory preditcion
        trajectory_pred = torch.zeros(bs, n_humans, self.pred_length, 2)
        for i in range(self.pred_length) :
            inp = self.position_emb(last_human_pos)
            h_state = self.decoder(inp, h_state)

            last_human_pos = self.position_head(h_state[0])
            trajectory_pred[:, :, i, :] = last_human_pos.clone().view(bs, n_humans, 2)

        # bs, n_humans, hidden_size = features.shape
        # pred_pos = self.position_head(features).view(bs, n_humans, self.pred_length, 2)

        
        loss = ((trajectory_truth[:, :self.pred_length].to(device)-trajectory_pred.transpose(1, 2).to(device))**2).mean()
        return loss

    
class TrajPredFF() :
    
    def __init__(self, position_head, pred_length=3, pred_start=0) -> None:
        self.pred_length = pred_length
        self.position_head = position_head
        self.pred_start = pred_start

    def loss(self, features, trajectory_truth, last_human_pos) :
        assert(trajectory_truth.size(1) >= self.pred_length+self.pred_start)
        
        device = 'cpu'
        if features.is_cuda :
            device = 'cuda'

        bs, n_humans, hidden_size = features.shape
        trajectory_pred = self.position_head(features).view(bs, n_humans, self.pred_length, 2)
        #print(neg_seeds[:, self.pred_start:self.pred_start + self.pred_length-1].shape, self.pred_length, self.pred_start, pred_pos.shape)
        
        loss = ((trajectory_truth[:, self.pred_start:self.pred_start + self.pred_length].to(device)-trajectory_pred.transpose(1, 2).to(device))**2).mean()
        return loss

class TrajPredFFMult() :
    
    def __init__(self, position_heads, pred_length=3, pred_start=0) -> None:
        self.pred_length = pred_length
        self.position_heads = position_heads
        self.pred_start = pred_start

    def loss(self, features, trajectory_truth, last_human_pos) :
        assert(trajectory_truth.size(2) >= self.pred_length+self.pred_start)
        
        device = 'cpu'
        if features.is_cuda :
            device = 'cuda'

        bs, n_envs, n_humans, hidden_size = features.shape
        # pred_pos = self.position_head(features).view(bs, n_humans, self.pred_length, 2)
        trajectory_pred = [ph(features[:, i]).view(bs, n_humans, self.pred_length, 2) for i, ph in enumerate(self.position_heads)]
        trajectory_pred = torch.stack(trajectory_pred, dim=1)
        
        loss = ((trajectory_truth[:, :, self.pred_start:self.pred_start + self.pred_length].to(device)-trajectory_pred.transpose(2, 3).to(device))**2).mean()
        return loss



