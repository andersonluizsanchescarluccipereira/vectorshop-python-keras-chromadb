import asyncio
from app.db import Catalog
from app.recommendation import RecommendationService
from app.features import encode_product

class FakeVectors:
    def nearest(self,vector,count):
        return [{'id':i,'distance':(i/100)} for i in range(1,11)]

class FakeNeural:
    async def score(self,user,products,purchases):
        return [p['id']/10 for p in products]

def test_cold_start():
    repo=Catalog(':memory:')
    svc=RecommendationService(repo,FakeVectors(),FakeNeural())
    result=asyncio.run(svc.recommend(99,5,'hybrid'))
    assert result['mode']=='popularidade'
    assert len(result['recommendations'])==5
    repo.db.close()

def test_hybrid_ranks_by_neural():
    repo=Catalog(':memory:')
    svc=RecommendationService(repo,FakeVectors(),FakeNeural())
    result=asyncio.run(svc.recommend(1,5,'hybrid'))
    assert result['recommendations'][0]['id']==10
    assert all(p['id'] not in {1,2} for p in result['recommendations'])
    repo.db.close()

def test_vector_mode_without_neural():
    repo=Catalog(':memory:')
    svc=RecommendationService(repo,FakeVectors(),FakeNeural())
    result=asyncio.run(svc.recommend(1,3,'vector'))
    assert len(result['recommendations'])==3
    assert result['recommendations'][0]['id']==3
    repo.db.close()
