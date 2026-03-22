import numpy as np

from core.product_manager import ProductManager


def test_rejects_path_traversal_name(tmp_path):
    pm = ProductManager(products_dir=str(tmp_path / "produces"))
    ok, msg = pm.save_product("../escape", config_data={})
    assert ok is False
    assert "Invalid product name" in msg


def test_roundtrip_poly_json_without_pickle(tmp_path):
    pm = ProductManager(products_dir=str(tmp_path / "produces"))
    name = "pytest_demo"
    path_polys = [
        np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
        np.array([[4.0, 5.0]], dtype=np.float32),
    ]
    path_points = np.array([[8.0, 9.0]], dtype=np.float32)

    ok, msg = pm.save_product(name, config_data={}, path_points=path_points, path_polys=path_polys)
    assert ok is True, msg

    dots, polys = pm.load_product_paths(name)
    assert dots is not None
    assert polys is not None
    assert len(polys) == 2
    assert all(isinstance(seg, np.ndarray) for seg in polys)
    assert polys[0].shape == (2, 2)
