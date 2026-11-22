from scripts import skeleton


def test_skeleton_step0_equals_last_value(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    m = skeleton.run(str(tmp_path / "results"), steps=5)
    assert m["step0_mae"] == m["last_value_mae"]
    assert (tmp_path / "results" / "test_preds.npz").exists()
