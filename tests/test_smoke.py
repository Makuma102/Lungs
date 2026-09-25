import json
import os

import numpy as np
import torch

from lungseg import data as D
from lungseg.losses import DiceCELoss
from lungseg.metrics import dice, hd95, lesion_detection
from lungseg.model import SmallUNet, count_parameters
from lungseg.reconstruct3d import nodule_report, reconstruct


def test_model_shapes_and_size():
    m = SmallUNet()
    assert m(torch.zeros(2, 1, 64, 64)).shape == (2, 3, 64, 64)
    assert 1e6 < count_parameters(m) < 3e6


def test_phantom_labels_and_hu():
    hu, lab = D.synth_ct_volume((16, 64, 64), rng=np.random.default_rng(0), n_nodules=2)
    assert set(np.unique(lab)) <= {0, 1, 2} and (lab == 2).any()
    assert hu[lab == 1].mean() < -600 and hu[lab == 2].mean() > -200


def test_cxr_phantom():
    img, lab = D.synth_cxr(64, np.random.default_rng(0))
    assert img.shape == lab.shape == (64, 64) and (lab == 1).any()


def test_loss_decreases_one_batch():
    torch.manual_seed(0)
    vols = D.make_phantom_volumes(1, (8, 64, 64), seed=1)
    x = torch.from_numpy(vols[0][0][:, None]); y = torch.from_numpy(vols[0][1].astype(np.int64))
    m = SmallUNet(base=8); opt = torch.optim.Adam(m.parameters(), 3e-3); crit = DiceCELoss()
    first = crit(m(x), y).item()
    for _ in range(15):
        opt.zero_grad(); loss = crit(m(x), y); loss.backward(); opt.step()
    assert loss.item() < first


def test_metrics():
    a = np.zeros((10, 10, 10), bool); a[2:6, 2:6, 2:6] = True
    assert dice(a, a) == 1.0 and hd95(a, a) == 0.0
    assert lesion_detection(a, a) == (1, 0, 0)


def test_reconstruct_writes_mesh_and_report(tmp_path):
    vol = D.make_phantom_volumes(1, (8, 64, 64), seed=2)[0][0]
    _, rep = reconstruct(SmallUNet(base=8), vol, (2.5, 2.0, 2.0), str(tmp_path / "c"))
    assert os.path.exists(tmp_path / "c.obj") and "disclaimer" in rep
    assert json.load(open(tmp_path / "c_report.json"))["n_nodules"] >= 0


def test_nodule_report_volume():
    lab = np.zeros((20, 20, 20), np.uint8); lab[5:15, 5:15, 5:15] = 2
    r = nodule_report(lab, (1.0, 1.0, 1.0))
    assert r["n_nodules"] == 1 and abs(r["nodules"][0]["volume_ml"] - 1.0) < 1e-6


def test_train_cli_end_to_end(tmp_path):
    from lungseg.train import main
    res = main(["--epochs", "1", "--size", "64", "--depth-slices", "8", "--n-train", "2",
                "--n-val", "1", "--n-test", "1", "--batch", "4", "--base", "8", "--out", str(tmp_path)])
    assert "lung_dice" in res and os.path.exists(tmp_path / "results.json")


def test_unet3d_shapes_and_loss():
    from lungseg.model import SmallUNet3D
    m = SmallUNet3D(base=4)
    x = torch.zeros(1, 1, 16, 32, 32)
    out = m(x)
    assert out.shape == (1, 3, 16, 32, 32)
    DiceCELoss()(out, torch.zeros(1, 16, 32, 32, dtype=torch.long)).backward()


def test_patch_dataset3d():
    vols = D.make_phantom_volumes(1, (16, 64, 64), seed=3)
    x, y = D.PatchDataset3D(vols, patch=(8, 32, 32))[0]
    assert x.shape == (1, 8, 32, 32) and y.shape == (8, 32, 32)


def test_train_cli_3d(tmp_path):
    from lungseg.train import main
    res = main(["--dim", "3", "--epochs", "1", "--size", "96", "--depth-slices", "16", "--n-train", "2",
                "--n-val", "1", "--n-test", "1", "--batch", "2", "--base", "4", "--out", str(tmp_path)])
    assert "lung_dice" in res and os.path.exists(tmp_path / "case0.obj")


def test_sliding_window_matches_shape():
    from lungseg.model import SmallUNet3D
    from lungseg.reconstruct3d import _predict_3d
    m = SmallUNet3D(base=4, depth=3, anisotropic=False).eval()
    vol = np.random.default_rng(0).random((40, 50, 45)).astype(np.float32)
    lab, prob = _predict_3d(m, vol, "cpu", patch=(16, 32, 32))
    assert lab.shape == prob.shape == vol.shape and np.isfinite(prob).all()


def test_resume_continues_training(tmp_path):
    """Interrupted run + --resume must reach the same final state as the log length implies."""
    import json
    from lungseg.train import main
    common = ["--epochs", "2", "--size", "64", "--depth-slices", "8", "--n-train", "2", "--n-val", "1",
              "--n-test", "1", "--batch", "4", "--base", "8", "--out", str(tmp_path)]
    assert main(common + ["--stop-after", "0"]) is None   # "crash" after epoch 0
    main(common + ["--resume"])                     # resume, run epoch 2 only
    log = json.load(open(tmp_path / "results.json"))["log"]
    assert [r["epoch"] for r in log] == [0, 1]


def test_mesh_of_scattered_voxels_does_not_crash():
    from lungseg.reconstruct3d import mesh
    m = np.zeros((20, 20, 20), bool)
    m[::4, ::4, ::4] = True   # 125 isolated voxels: >= 8 voxels but no surface after smoothing
    assert mesh(m, (1, 1, 1)) == (None, None)
