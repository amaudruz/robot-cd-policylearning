class ARGS() :
    def __init__(self) :
        self.policy = 'sail'
        self.policy_config = 'configs/policy.config'
        self.traj_path = None
        self.batch_size = 128
        self.max_obs = 5
        self.lr = 1e-3
        self.visibility = 'visible'
        self.contrast_sampling = 'social'
        self.contrast_weight = 0.4
        self.contrast_horizon = 4
        self.contrast_temperature= 0.2
        self.contrast_range = 2.0
        self.contrast_nboundary = 0
        self.ratio_boundary = 0.5
        self.percent_label = 0.5
        self.num_epoch = 200
        self.save_every = 5
        self.length_pred = 1
        self.skip_pred = 1
        self.model_file = ""
        self.output_dir = 'data/output/imitate'
        self.memory_dir = 'data/demonstrate'
        self.freeze = False
        self.predict = False
        self.gpu = False
        self.data_sample = 1
        
        self.traj_weight = 0.4
        self.traj_length = 3
        self.traj_start = 0
        self.uni_weight = 0.4
        self.uni_length = 1

        self.auxiliary_task=''
