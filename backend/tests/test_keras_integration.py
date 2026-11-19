"""Teste opcional: pytest -m integration (requer tensorflow-cpu instalado)."""
import importlib.util
import pytest
pytestmark=pytest.mark.integration

def test_keras_train_and_predict(tmp_path):
    if importlib.util.find_spec('tensorflow') is None:
        pytest.skip('TensorFlow não instalado neste ambiente')
    from app.neural_worker import run
    from app.db import Catalog
    from app.features import training_dataset
    repo=Catalog(':memory:')
    xs,ys=training_dataset(repo)
    model=str(tmp_path/'model.keras')
    result=run({'action':'train','xs':xs,'ys':ys,'epochs':2,'modelPath':model})
    assert result['ok'] and len(result['history'])==2
    pred=run({'action':'predict','xs':xs[:3],'modelPath':model})
    assert len(pred['scores'])==3
    assert all(0<=s<=1 for s in pred['scores'])
    repo.db.close()
