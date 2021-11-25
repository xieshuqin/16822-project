import sys
sys.path.append('./deep-person-reid')  # nopep8
sys.path.append('./deep-person-reid/scripts')  # nopep8
import torch
import torchreid
import cv2
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


def reid_people(reid_model, images, all_bboxes, thres):
    features = []
    for image, boxes in zip(images, all_bboxes):
        this_features = []
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            cropped = image[ymin:ymax+1, xmin:xmax+1]
            cropped = cv2.cvtColor(cropped.numpy(), cv2.COLOR_BGR2RGB)
            cropped = Image.fromarray(cropped)
            this_features.append(reid_model.forward(cropped))
        features.append(this_features)
    link = {}
    for i in range(len(features)):
        for j in range(len(features[i])):
            link[(i, j)] = []
    scores = []
    for x in link:
        for y in link:
            if x < y:
                scores.append(
                    (features[x[0]][x[1]]@features[y[0]][y[1]], x, y))
    scores = sorted(scores, reverse=True)

    def get_connected_component(x, vis):
        vis.add(x)
        for y in link[x]:
            if y not in vis:
                get_connected_component(y, vis)
    for s, x, y in scores:
        if s < thres:
            break
        print('possible match', x, y, s)
        comp_x = set()
        get_connected_component(x, comp_x)
        comp_y = set()
        get_connected_component(y, comp_y)
        print(comp_x, comp_y)
        good = True
        for xx in comp_x:
            for yy in comp_y:
                if xx[0] == yy[0] and xx[1] != yy[1]:
                    print('conflict detected, no match')
                    good = False
        if good:
            link[x].append(y)
            link[y].append(x)
    all_vis = set()
    people = []
    for x in link:
        if x in all_vis:
            continue
        vis = set()
        get_connected_component(x, vis)
        all_vis |= vis
        if len(vis) == 1:
            continue
        vis = sorted(vis)
        print(vis)
        people.append(vis)
    return people


if __name__ == '__main__':
    model = create_reid_model()
    query = open_image('reid_tests/0001_c1s1_001051_00.jpg')
    pos = open_image('reid_tests/0001_c6s3_077467_00.jpg')
    neg = open_image('reid_tests/0002_c1s1_000451_00.jpg')
    query_f = model.forward(query)
    pos_f = model.forward(pos)
    neg_f = model.forward(neg)
    print(query_f@pos_f, query_f@neg_f, pos_f@neg_f)
