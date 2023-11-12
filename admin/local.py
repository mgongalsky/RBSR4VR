class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = ''    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.pretrained_nets_dir = 'E:\VR\Experiments\python\RBSR\admin\../pretrained_networks/'    # Directory for pre-trained networks.
        self.save_data_path = ''    # Directory for saving network predictions for evaluation.
        self.zurichraw2rgb_dir = ''    # Zurich RAW 2 RGB path
        self.burstsr_dir = './my_dataset/burstsr_dataset'    # BurstSR dataset path
        self.synburstval_dir = 'E:\VR\Experiments\python\RBSR\SyntheticBurstVal'    # SyntheticBurst validation set path
