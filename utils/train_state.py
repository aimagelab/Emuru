class TrainState:
    def __init__(self, global_step, epoch):
        self.global_step = global_step
        self.epoch = epoch
        self.best_eval = 0.0
        self.last_eval = 0.0

    def __eq__(self, other):
        if isinstance(other, TrainState):
            return self.global_step == other.global_step and self.epoch == other.epoch and self.best_eval == other.best_eval and self.last_eval == other.last_eval
        return False

    def __repr__(self):
        return f"TrainState(global_step={self.global_step}, epoch={self.epoch}, best_eval={self.best_eval}, last_eval={self.last_eval})"

    def state_dict(self):
        return {'global_step': self.global_step, 'epoch': self.epoch, 'best_eval': self.best_eval, 'last_eval': self.last_eval}

    def load_state_dict(self, state_dict):
        self.global_step = state_dict['global_step']
        self.epoch = state_dict['epoch']
        self.best_eval = state_dict.get('best_eval', 0.0)
        self.last_eval = state_dict.get('last_eval', 0.0)
