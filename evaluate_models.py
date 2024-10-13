from hwd.metrics import HWDScore, FIDScore, KIDScore
from hwd.datasets import FolderDataset

real_dataset = FolderDataset('files/evaluation/iam_words', extension='png')
fake_dataset = FolderDataset('files/evaluation/iam_words_vae', extension='png')

score = HWDScore().cuda()
result = score(real_dataset, fake_dataset)
print('HWD:', result)

score = FIDScore().cuda()
result = score(real_dataset, fake_dataset)
print('FID:', result)

score = KIDScore().cuda()
result = score(real_dataset, fake_dataset)
print('KID:', result)