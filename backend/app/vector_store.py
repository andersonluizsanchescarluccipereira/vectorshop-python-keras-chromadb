import chromadb
from .features import encode_product,encode_profile

class VectorStore:
    def __init__(self,url):
        self.client=chromadb.HttpClient(host=url)
        self.products=None
        self.profiles=None
    def connect(self):
        self.client.heartbeat()
        self.products=self.client.get_or_create_collection(name='ecommerce_products_v1',metadata={'hnsw:space':'cosine'})
        self.profiles=self.client.get_or_create_collection(name='ecommerce_user_profiles_v1',metadata={'hnsw:space':'cosine'})
    def index_products(self,products):
        if not products:return
        self.products.upsert(ids=[str(p['id']) for p in products],embeddings=[encode_product(p) for p in products],
          documents=[f"{p['name']}. Categoria {p['category']}. Cor {p['color']}. Preço {p['price']}." for p in products],
          metadatas=[{'productId':p['id'],'category':p['category'],'color':p['color']} for p in products])
    def index_user(self,uid,purchases):
        vec=encode_profile(purchases)
        if vec is None:
            self.profiles.delete(ids=[str(uid)])
            return None
        self.profiles.upsert(ids=[str(uid)],embeddings=[vec],documents=[f'Perfil de compra do usuário {uid}'],
          metadatas=[{'userId':uid,'purchaseCount':len(purchases)}])
        return vec
    def nearest(self,vec,count):
        if count<1:return []
        result=self.products.query(query_embeddings=[vec],n_results=count,include=['distances','metadatas'])
        ids=result['ids'][0]
        distances=(result.get('distances') or [[]])[0]
        return [{'id':int(pid),'distance':float(distances[i])} for i,pid in enumerate(ids)]
    def health(self):
        return self.products is not None and self.products.count()>=0
