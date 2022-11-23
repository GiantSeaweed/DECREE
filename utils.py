import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image

def compute_self_cos_sim(mat_a):
    # assert(mat_a.shape == mat_b.shape)
    # compute cosine similarity within one batch
    ele_size = mat_a.shape[0]
    mat_a = F.normalize(mat_a, dim=-1)
    sim_matrix = torch.mm(mat_a, mat_a.t())
    assert sim_matrix.shape == (ele_size, ele_size)

    sim_mask = (torch.ones_like(sim_matrix) - \
                torch.eye(ele_size, device=sim_matrix.device)).bool()
    sim_matrix = sim_matrix.masked_select(sim_mask).view(ele_size, -1)
    assert sim_matrix.shape == (ele_size, ele_size - 1)

    cos_sim = torch.mean(sim_matrix)
    return cos_sim

def compute_cos_sim(mat_a, mat_b):
    assert(len(mat_a.shape) == 2)
    assert(len(mat_b.shape) == 2)
    assert(mat_a.shape[1] == mat_b.shape[1])

    mat_a = F.normalize(mat_a, dim=-1)
    mat_b = F.normalize(mat_b, dim=-1)
    sim_matrix = torch.mm(mat_a, mat_b.t())
    assert(sim_matrix.shape == (mat_a.shape[0], mat_b.shape[0]))
    cos_sim = torch.mean(sim_matrix)
    return cos_sim

def assert_range(val, vmin, vmax, ratio=0.7):
    val = val.float()
    vmin = vmin - 1e-4
    vmax = vmax + 1e-4

    if isinstance(val, np.ndarray):
        val = torch.tensor(val).cuda()
    elif isinstance(val, float):
        val = torch.tensor(val).cuda()
        flag = ((vmin <= val) and (val <= vmax))
        if (flag == True):
            return
        
    diff = vmax - vmin
    assert(diff > 0)
    # [min, min + r(max-min)]
    flag = (vmin <= torch.min(val))
    flag = (flag and (torch.min(val) <= vmin + ratio * diff))
    # [max-r(max-min), max]
    flag = (flag and (vmax - ratio * diff <= torch.max(val)))
    flag = (flag and (torch.max(val) <= vmax))
    if flag == False:
        print("alert ###" * 10)
        print(f'val=[{torch.min(val)}, {torch.max(val)}], [{vmin}, {vmax}, r={ratio}]')

def dump_img(data, name):
    if torch.is_tensor(data):
        data = np.asarray(data.detach(), dtype=np.uint8)
    img = Image.fromarray(data.astype(np.uint8))
    print("in", data.dtype)
    img.save(name + '.png')

def epsilon():
    return 1e-7