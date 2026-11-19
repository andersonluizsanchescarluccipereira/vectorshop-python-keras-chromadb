"""Vetores estruturados (NÃO embeddings semânticos gerados por LLM)."""
import math
CATEGORIES=['eletrônicos','vestuário','calçados','acessórios']
COLORS=['preto','prata','azul','branco','vermelho','bege','cinza','marrom']
DIMENSION=13
INPUT_DIMENSION=27

def normalize(values):
    norm=math.sqrt(sum(x*x for x in values))
    return [x/norm for x in values] if norm else [0.0 for _ in values]

def encode_product(product):
    category,color,price=product['category'],product['color'],product['price']
    if category not in CATEGORIES or color not in COLORS or not math.isfinite(price) or price<0:
        raise ValueError(f'Produto {product["id"]} inválido')
    vec=[0.0]*DIMENSION
    vec[CATEGORIES.index(category)]=math.sqrt(0.55)
    vec[len(CATEGORIES)+COLORS.index(color)]=math.sqrt(0.30)
    vec[-1]=math.sqrt(0.15)*min(price/200,1)
    return normalize(vec)

def encode_profile(purchases):
    if not purchases: return None
    avg=[0.0]*DIMENSION
    for product in purchases:
        vec=encode_product(product)
        for i in range(DIMENSION):avg[i]+=vec[i]/len(purchases)
    return normalize(avg)

def neural_features(user,product,purchases):
    profile=encode_profile(purchases) or [0.0]*DIMENSION
    age=max(0.0,min(1.0,(user['age']-18)/62))
    return encode_product(product)+[age]+profile

def training_dataset(repo):
    xs,ys=[],[]
    for user in repo.users():
        purchases=repo.purchases(user['id'])
        purchased={p['id'] for p in purchases}
        for product in repo.products():
            # Exclui o produto alvo do perfil para evitar vazamento do rótulo.
            context=[p for p in purchases if p['id']!=product['id']]
            xs.append(neural_features(user,product,context))
            ys.append(1.0 if product['id'] in purchased else 0.0)
    return xs,ys
