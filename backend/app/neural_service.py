import asyncio,json,sys
from pathlib import Path
from .features import training_dataset,neural_features

class NeuralService:
    def __init__(self,repo,model_path):
        self.repo=repo
        self.path=Path(model_path)
        self.metadata=self.path.with_suffix('.json')
        self.model=None
        self.training=False
        self.lock=asyncio.Lock()
        if self.path.is_file() and self.metadata.is_file():
            try:
                data=json.loads(self.metadata.read_text(encoding='utf8'))
                if data.get('version')==1:self.model=data
            except (OSError,ValueError):pass
    def status(self):
        return {'trained':self.model is not None,'training':self.training,
                'architecture':[128,64,32,1], 'epochs':self.model['epochs'] if self.model else 0,
                'examples':self.model['examples'] if self.model else 0,
                'history':self.model['history'] if self.model else []}
    async def _worker(self,payload):
        proc=await asyncio.create_subprocess_exec(sys.executable,'-m','app.neural_worker',
            stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
        out,err=await proc.communicate(json.dumps(payload).encode('utf8'))
        try:response=json.loads(out.decode('utf8'))
        except ValueError:raise RuntimeError(f'Worker falhou: {err.decode("utf8")[-700:]}')
        if proc.returncode or not response.get('ok'):
            raise RuntimeError(response.get('error') or err.decode('utf8')[-700:])
        return response
    async def train(self,epochs=100):
        if type(epochs) is not int or not 1<=epochs<=300:raise ValueError('epochs deve estar entre 1 e 300')
        if self.lock.locked():raise RuntimeError('Treinamento já em andamento')
        async with self.lock:
            self.training=True
            try:
                xs,ys=training_dataset(self.repo)
                response=await self._worker({'action':'train','xs':xs,'ys':ys,'epochs':epochs,'modelPath':str(self.path)})
                model={'version':1,'epochs':epochs,'examples':response['examples'],'history':response['history']}
                self.metadata.write_text(json.dumps(model,ensure_ascii=False),encoding='utf8')
                self.model=model
                return self.status()
            finally:self.training=False
    async def score(self,user,products,purchases):
        if self.model is None:return None
        xs=[neural_features(user,p,purchases) for p in products]
        response=await self._worker({'action':'predict','xs':xs,'modelPath':str(self.path)})
        return response['scores']
