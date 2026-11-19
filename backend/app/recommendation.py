from .features import encode_profile

class RecommendationService:
    def __init__(self,repo,vectors,neural):
        self.repo,self.vectors,self.neural=repo,vectors,neural
    def rebuild(self):
        self.vectors.index_products(self.repo.products())
        for user in self.repo.users():self.refresh_user(user['id'])
    def refresh_user(self,uid):
        self.vectors.index_user(uid,self.repo.purchases(uid))
    async def recommend(self,uid,limit=5,mode='hybrid'):
        user=self.repo.user(uid)
        if user is None:return None
        if mode not in ('hybrid','vector','neural'):raise ValueError('mode deve ser hybrid, vector ou neural')
        purchases=self.repo.purchases(uid)
        excluded={p['id'] for p in purchases}
        all_products=[p for p in self.repo.products() if p['id'] not in excluded]
        if not purchases:
            return {'user':user,'mode':'popularidade','note':'Cold start: sem compras, fallback por popularidade.',
                    'recommendations':[dict(p,score=None) for p in self.repo.popular(limit)]}
        matches=self.vectors.nearest(encode_profile(purchases),len(self.repo.products()))
        by_id={p['id']:p for p in all_products}
        retrieved=[m for m in matches if m['id'] in by_id]
        candidates=all_products if mode=='neural' else [by_id[m['id']] for m in retrieved]
        distances={m['id']:m['distance'] for m in retrieved}
        neural_scores=None if mode=='vector' else await self.neural.score(user,candidates,purchases)
        if mode!='vector' and neural_scores is None:
            raise RuntimeError('Rede neural não treinada. Execute POST /api/model/train ou escolha mode=vector.')
        result=[]
        for i,p in enumerate(candidates):
            similarity=max(0.0,min(1.0,1-distances[p['id']])) if p['id'] in distances else None
            neural_score=neural_scores[i] if neural_scores is not None else None
            score=similarity if mode=='vector' else neural_score
            result.append(dict(p,score=score,similarity=similarity,neuralScore=neural_score))
        result.sort(key=lambda p:p['score'] if p['score'] is not None else -1,reverse=True)
        notes={'vector':'Score = similaridade; não é probabilidade.',
               'neural':'Score = saída sigmoid treinada; não é probabilidade calibrada.',
               'hybrid':'ChromaDB recupera candidatos; TensorFlow/Keras os ranqueia. Saída sigmoid não calibrada.'}
        return {'user':user,'mode':mode,'note':notes[mode],'recommendations':result[:limit]}
