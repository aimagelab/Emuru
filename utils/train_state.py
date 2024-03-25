class TrainState:
    def __init__(self, global_step, epoch):
        self.global_step = global_step
        self.epoch = epoch

    def __eq__(self, other):
        if isinstance(other, TrainState):
            return self.global_step == other.global_step and self.epoch == other.epoch
        return False

    def __repr__(self):
        return f"TrainState(global_step={self.global_step}, epoch={self.epoch})"

    def state_dict(self):
        return {'global_step': self.global_step, 'epoch': self.epoch}

    def load_state_dict(self, state_dict):
        self.global_step = state_dict['global_step']
        self.epoch = state_dict['epoch']
