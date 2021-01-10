
class UniPred() :

    def __init__(self, prediction_head, pred_length=3 ):
        self.pred_length = pred_length
        self.prediction_head = prediction_head 
    
    def loss(self, traj_features, neg_seeds) :
        pred_pos = self.prediction_head(traj_features)
        loss = ((neg_seeds[:, self.pred_length-1] - pred_pos)**2).mean()
        return loss
    