import torch


class NoisyTeacherForcing:
    def __init__(self, alphabet_size: int, noise_prob: float = 0.0):
        self.noise_prob = noise_prob
        self.alphabet_size = alphabet_size

    def __call__(self, x, unpadded_text_len):
        num_extra_tokens = 3  # TODO CHANGE THIS AFTER ALPHABET REFACTORING
        noise = torch.randint(low=num_extra_tokens-1, high=self.alphabet_size, size=x.shape, device=x.device)
        prob = torch.rand(size=x.shape, device=x.device)
        prob[:, 0] = 1

        for i, eos_token_place in enumerate(unpadded_text_len):
            prob[i, eos_token_place + 1:] = 1

        return torch.where(prob > self.noise_prob, x, noise)


