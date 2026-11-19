"""Executado em PROCESSO separado. TensorFlow/Keras é importado somente aqui."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
import sys,json
import numpy as np
import tensorflow as tf
from pathlib import Path

# Em CPU pequena, evita criar pools grandes de threads.
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(1)

def create_model():
    model=tf.keras.Sequential([
        tf.keras.layers.Input(shape=(27,)),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(64,activation='relu'),
        tf.keras.layers.Dense(32,activation='relu'),
        tf.keras.layers.Dense(1,activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
                  loss='binary_crossentropy',metrics=['accuracy'])
    return model

def run(payload):
    action=payload['action']
    if action=='train':
        xs=np.asarray(payload['xs'],dtype=np.float32)
        ys=np.asarray(payload['ys'],dtype=np.float32).reshape(-1,1)
        epochs=int(payload['epochs'])
        if xs.ndim!=2 or xs.shape[1]!=27 or not len(xs):raise ValueError('Dataset inválido')
        tf.keras.utils.set_random_seed(42)
        model=create_model()
        history=model.fit(xs,ys,epochs=epochs,batch_size=min(16,len(xs)),shuffle=True,verbose=0)
        model_path=Path(payload['modelPath'])
        model_path.parent.mkdir(parents=True,exist_ok=True)
        temp=model_path.with_name(model_path.stem+'-pending.keras')
        model.save(str(temp))
        os.replace(temp,model_path)
        return {'ok':True,'examples':len(xs),'history':[
            {'epoch':i+1,'loss':round(float(history.history['loss'][i]),6),
             'accuracy':round(float(history.history['accuracy'][i]),4)} for i in range(epochs)]}
    if action=='predict':
        xs=np.asarray(payload['xs'],dtype=np.float32)
        if not len(xs):return {'ok':True,'scores':[]}
        model=tf.keras.models.load_model(payload['modelPath'],compile=False)
        scores=model.predict(xs,verbose=0).reshape(-1).tolist()
        return {'ok':True,'scores':[float(x) for x in scores]}
    raise ValueError('Ação desconhecida')

if __name__=='__main__':
    try:
        payload=json.load(sys.stdin)
        print(json.dumps(run(payload),ensure_ascii=False),flush=True)
    except Exception as exc:
        print(json.dumps({'ok':False,'error':str(exc)},ensure_ascii=False),flush=True)
        sys.exit(1)
