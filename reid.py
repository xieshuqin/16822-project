import sys
sys.path.append('./deep-person-reid')  # nopep8
sys.path.append('./deep-person-reid/scripts')  # nopep8
import torch
import torchreid
from torchreid.utils import (
    Logger, check_isfile, set_random_seed, collect_env_info,
    resume_from_checkpoint, load_pretrained_weights, compute_model_complexity
)
from default_config import (
    imagedata_kwargs, optimizer_kwargs, videodata_kwargs, engine_run_kwargs,
    get_default_config, lr_scheduler_kwargs
)
from PIL import Image
from torchreid.data.transforms import build_transforms


def open_image(path):
    img = Image.open(path).convert('RGB')
    return img


class Retriever:
    def __init__(self):
        cfg = get_default_config()
        cfg.model.name = 'osnet_ain_x1_0'
        cfg.use_gpu = torch.cuda.is_available()
        model = torchreid.models.build_model(
            name=cfg.model.name,
            num_classes=0,
            loss=cfg.loss.name,
            pretrained=cfg.model.pretrained,
            use_gpu=cfg.use_gpu
        )
        load_pretrained_weights(model, 'reid_models/osnet_ain_ms_d_m.pth.tar')
        model.train(False)
        self.model = model.cuda()
        print(self.model)
        self.transform_tr, self.transform_te = build_transforms(
            cfg.data.height,
            cfg.data.width,
            transforms=cfg.data.transforms,
            norm_mean=cfg.data.norm_mean,
            norm_std=cfg.data.norm_std
        )

    def forward(self, x):
        x = self.transform_te(x).cuda()
        return torch.nn.functional.normalize(self.model(x[None]))[0].detach().cpu().numpy()


def create_reid_model():
    return Retriever()


def reid_people():
    pass


if __name__ == '__main__':
    model = create_reid_model()
    query = open_image('reid_tests/0001_c1s1_001051_00.jpg')
    pos = open_image('reid_tests/0001_c6s3_077467_00.jpg')
    neg = open_image('reid_tests/0002_c1s1_000451_00.jpg')
    query_f = model.forward(query)
    pos_f = model.forward(pos)
    neg_f = model.forward(neg)
    print(query_f@pos_f, query_f@neg_f, pos_f@neg_f)
