import os,asyncio,logging
from contextlib import asynccontextmanager
from fastapi import FastAPI,HTTPException,Query,Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .db import Catalog
from .vector_store import VectorStore
from .neural_service import NeuralService
from .recommendation import RecommendationService

log=logging.getLogger('vectorshop')
DB_PATH=os.getenv('DB_PATH','./data/ecommerce.sqlite')
MODEL_PATH=os.getenv('MODEL_PATH','./data/neural-model.keras')
CHROMA_URL=os.getenv('CHROMA_URL','http://localhost:8000')

@asynccontextmanager
async def lifespan(app):
    repo=Catalog(DB_PATH)
    vectors=VectorStore(CHROMA_URL)
    neural=NeuralService(repo,MODEL_PATH)
    rec=RecommendationService(repo,vectors,neural)
    app.state.repo=repo
    app.state.vectors=vectors
    app.state.neural=neural
    app.state.rec=rec
    # Treina uma vez, depois reutiliza o modelo persistido.
    if not neural.status()['trained']:
        log.warning('Treinando modelo inicial (100 épocas) pela CPU...')
        try:await neural.train(100)
        except Exception:log.exception('Treinamento inicial falhou; modo vetorial permanece disponível')
    connected=False
    for attempt in range(30):
        try:
            await asyncio.to_thread(vectors.connect)
            await asyncio.to_thread(rec.rebuild)
            connected=True
            break
        except Exception as exc:
            log.warning('Chroma indisponível (%s/30): %s',attempt+1,exc)
            await asyncio.sleep(1.5)
    if not connected:
        repo.db.close()
        raise RuntimeError('ChromaDB indisponível; verifique CHROMA_URL')
    try:yield
    finally:repo.db.close()

app=FastAPI(title='VectorShop Python + Keras + ChromaDB',version='2.0',lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=['http://localhost:8080','http://localhost:5173'],
                   allow_methods=['GET','POST','DELETE'],allow_headers=['Content-Type'])

class PurchaseInput(BaseModel):
    productId:int

class TrainInput(BaseModel):
    epochs:int=100

@app.get('/api/users')
def users():return app.state.repo.users()

@app.get('/api/products')
def products():return app.state.repo.products()

@app.get('/api/users/{uid}')
def user(uid:int):
    value=app.state.repo.user(uid)
    if value is None:raise HTTPException(404,'Usuário não encontrado')
    return dict(value,purchases=app.state.repo.purchases(uid))

@app.get('/api/users/{uid}/recommendations')
async def recommend(uid:int,limit:int=Query(5,ge=1,le=50),mode:str='hybrid'):
    try:result=await app.state.rec.recommend(uid,limit,mode)
    except ValueError as e:raise HTTPException(400,str(e))
    except RuntimeError as e:raise HTTPException(503,str(e))
    if result is None:raise HTTPException(404,'Usuário não encontrado')
    return result

@app.post('/api/users/{uid}/purchases')
async def buy(uid:int,purchase:PurchaseInput):
    repo=app.state.repo
    if not repo.user(uid):raise HTTPException(404,'Usuário não encontrado')
    if not repo.product(purchase.productId):raise HTTPException(404,'Produto não encontrado')
    if not repo.add_purchase(uid,purchase.productId):raise HTTPException(409,'Produto já comprado')
    synced=True
    try:await asyncio.to_thread(app.state.rec.refresh_user,uid)
    except Exception:log.exception('Falha ao sincronizar perfil');synced=False
    return {'userId':uid,'productId':purchase.productId,'indexSynced':synced}

@app.delete('/api/users/{uid}/purchases/{pid}')
async def remove(uid:int,pid:int):
    repo=app.state.repo
    if not repo.user(uid):raise HTTPException(404,'Usuário não encontrado')
    if not repo.remove_purchase(uid,pid):raise HTTPException(404,'Compra não encontrada')
    synced=True
    try:await asyncio.to_thread(app.state.rec.refresh_user,uid)
    except Exception:log.exception('Falha ao sincronizar perfil');synced=False
    return {'userId':uid,'productId':pid,'removed':True,'indexSynced':synced}

@app.get('/api/model/status')
def model_status():return app.state.neural.status()

@app.post('/api/model/train')
async def model_train(payload:TrainInput=Body(default=TrainInput())):
    try:return await app.state.neural.train(payload.epochs)
    except ValueError as e:raise HTTPException(400,str(e))
    except RuntimeError as e:
        if 'já em andamento' in str(e):raise HTTPException(409,str(e))
        raise HTTPException(503,str(e))

@app.get('/api/health')
async def health():
    try:chroma=await asyncio.to_thread(app.state.vectors.health)
    except Exception:chroma=False
    return {'status':'ok' if chroma else 'degraded','sqlite':'ok','chroma':chroma,
            'neural':app.state.neural.status()['trained']}
