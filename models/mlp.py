from torch import nn

class MLP(nn.Module):
    def __init__(self, im_h: int=224, im_w: int=224, num_strokes: int=16, num_cp: int=4, innerdim: int=1000):
        super().__init__()
        self.layers_points = nn.Sequential(
            nn.Flatten(),
            nn.Linear(im_h * im_w, innerdim),
            nn.SELU(inplace=True),
            nn.Linear(innerdim, innerdim),
            nn.SELU(inplace=True),
            nn.Linear(innerdim, innerdim),
            nn.SELU(inplace=True),
            nn.Linear(innerdim, num_strokes * num_cp * 2),
        )

    def forward(self, x):
        '''Forward pass'''
        return self.layers_points(x)
        