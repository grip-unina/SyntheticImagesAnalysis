#
# Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt
#

import numpy as np


class get_denoiser:
    def __init__(self, sigma, cuda=False):
        import torch
        from DnCNN import make_net
        num_levels = 17

        out_channel = 3
        network = make_net(3, kernels=[3, ] * num_levels,
                           features=[64, ] * (num_levels - 1) + [out_channel],
                           bns=[False, ] + [True, ] *
                           (num_levels - 2) + [False, ],
                           acts=['relu', ] * (num_levels - 1) + ['linear', ],
                           dilats=[1, ] * num_levels,
                           bn_momentum=0.1, padding=0)
        self.sigma = sigma
        if sigma == 1:
            weights_path = "./DenoiserWeight/model_best.th"
        else:
            print("Sigma should be one")
            assert False
        state_dict = torch.load(weights_path, torch.device('cpu'))
        network.load_state_dict(state_dict["network"])
        self.device = 'cuda:0' if cuda else 'cpu'
        self.network = network.to(self.device).eval()

        print(weights_path)

    def __call__(self, img):
        import torch
        with torch.no_grad():
            img = torch.from_numpy(np.float32(img)).permute(2, 0, 1)[None, ...]
            res = self.network(img.to(self.device))[
                0].permute(1, 2, 0).cpu().numpy()
        return res

    def denoise(self, img):
        import torch
        with torch.no_grad():
            img = torch.from_numpy(np.float32(img)).permute(2, 0, 1)[None, ...]
            img = img[:, :, 17:-17, 17:-17] - self.sigma / \
                256.0 * self.network(img.to(self.device)).cpu()
            img = img[0].permute(1, 2, 0).numpy()
        return img
