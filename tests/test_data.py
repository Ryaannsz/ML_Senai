from module_olist.config import PROJ_ROOT


def test_project_root_exists():
    assert PROJ_ROOT.is_dir()
