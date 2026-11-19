import math,sqlite3
from app.features import encode_product,encode_profile,neural_features,training_dataset
from app.db import Catalog

def test_vector_13_dimensions_normalized():
    repo=Catalog(':memory:')
    v=encode_product(repo.products()[0])
    assert len(v)==13
    assert math.isclose(sum(x*x for x in v),1.0,rel_tol=1e-6)
    repo.db.close()

def test_profile_and_neural_27_dimensions():
    repo=Catalog(':memory:')
    u=repo.user(1);p=repo.products()[0]; purchases=repo.purchases(1)
    assert len(encode_profile(purchases))==13
    assert len(neural_features(u,p,purchases))==27
    repo.db.close()

def test_training_examples_and_labels():
    repo=Catalog(':memory:')
    xs,ys=training_dataset(repo)
    assert len(xs)==60 and len(ys)==60
    assert sum(ys)==12
    assert set(ys)=={0.0,1.0}
    repo.db.close()

def test_purchase_add_and_remove():
    repo=Catalog(':memory:')
    assert repo.add_purchase(99,10)
    assert not repo.add_purchase(99,10)
    assert repo.remove_purchase(99,10)
    assert not repo.remove_purchase(99,10)
    repo.db.close()
