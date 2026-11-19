# VectorShop Python — TensorFlow/Keras + ChromaDB

**Migração da última versão corrigida do laboratório Node.js/TensorFlow.js, preservando a finalidade didática do módulo de Machine Learning.**

## 1. O que este laboratório demonstra

- **Rede neural supervisionada** com 128 → 64 → 32 → 1 neurônios, ReLU, sigmoid, Adam, `binary_crossentropy`, 100 épocas.
- **Busca vetorial** com ChromaDB usando vetores **estruturados de 13 dimensões** (não embeddings de LLM).
- **Recomendação em dois estágios:** ChromaDB recupera candidatos, TensorFlow/Keras os ranqueia.
- **Modos de comparação:** `vector`, `neural`, `hybrid`.
- **Persistência:** SQLite para clientes/produtos/compras; ChromaDB para índices; modelo Keras salvo em disco.
- **Cold start:** cliente sem compras recebe itens por popularidade.

> A rede neural **não é treinada pelo ChromaDB**. O treinamento usa exemplos rotulados derivados do histórico de compras. ChromaDB recupera candidatos durante a recomendação. No modo híbrido, a pontuação final vem da rede neural, e não de uma soma dos scores.

## 2. Requisitos e compatibilidade

- Linux com Podman e `podman-compose` (ou Docker Compose).
- Portas 8080, 3001 e 8000 livres.
- CPU x86_64; sem GPU. Configurado para poucos threads para respeitar o notebook de 8 GB de RAM.
- **Atenção:** a primeira construção da imagem com `tensorflow-cpu` é pesada e, em HDD, pode levar bastante tempo. O container Python pode consumir memória significativa. Não há garantia de funcionamento sem testar em sua máquina.
- Imagens usam **nomes totalmente qualificados** (`docker.io/library/...`) para evitar o erro de *short-name* do Podman.

## 3. Subir o projeto

```bash
cd vectorshop-python-keras-chromadb
podman-compose up -d --build
podman ps -a
podman-compose logs --tail=80 api
```

Abra **http://localhost:8080**. A API está em **http://localhost:3001/api** e a documentação FastAPI em **http://localhost:3001/docs**. O ChromaDB está em `127.0.0.1:8000`.

> **Evite conflito com o projeto Node anterior:** ele utiliza as mesmas portas. Pare o Compose antigo antes de iniciar este projeto, dentro da pasta antiga: `podman-compose down` (sem `-v` para preservar os dados). O novo projeto tem volumes independentes por nome de projeto.

```bash
curl http://localhost:3001/api/health
curl http://localhost:3001/api/model/status
curl 'http://localhost:3001/api/users/1/recommendations?mode=hybrid'
```

Se o frontend não abrir:

```bash
podman ps -a
podman-compose logs --tail=100 frontend
podman-compose logs --tail=100 api
curl -I http://localhost:8080
```

## 4. Arquitetura

```text
Navegador (HTML, CSS, JS)
         |
         v
Nginx :8080 -- /api/ --> FastAPI :3001
                            |      |           |
                            |      |           +--> SQLite (clientes, produtos, compras)
                            |      +--> ChromaDB :8000 (13D, cosseno)
                            +--> subprocesso Python isolado
                                      |
                                      +--> TensorFlow/Keras CPU
                                           Dense(128) -> Dense(64) -> Dense(32) -> Dense(1)
```

### Fluxo de treinamento

1. SQLite fornece pares `(cliente, produto)` e rótulos de compra.
2. A função `training_dataset()` gera **60 exemplos** (6 clientes × 10 produtos).
3. Cada exemplo é um vetor de **27 entradas**: 13 atributos do produto + idade normalizada + 13 atributos do histórico.
4. O subprocesso `app.neural_worker` treina 100 épocas e salva `neural-model.keras` e o histórico de métricas.
5. A API continua responsiva enquanto o subprocesso trabalha; o endpoint `/api/model/train` **aguarda** a conclusão da tarefa e retorna os resultados.

### Fluxo de recomendação

```text
Cliente com compras
  -> média dos vetores dos produtos comprados (13D)
  -> ChromaDB (busca por cosseno)
  -> filtra produtos já comprados
  -> TensorFlow/Keras pontua os candidatos (27D)
  -> ordena pela saída sigmoid
  -> devolve Top-K
```

No modo `neural`, o modelo avalia **todos** os produtos não comprados; no `vector`, usa somente a similaridade retornada pelo ChromaDB. O modo `hybrid` usa os candidatos do ChromaDB e o score neural. Neste pequeno catálogo, a consulta busca todos os produtos, por isso a etapa vetorial **não reduz efetivamente** o conjunto de candidatos: é uma demonstração de arquitetura, não um ganho de escala. Para escala real, limitaríamos Top-N no ChromaDB.

## 5. Como funcionam os vetores

### ChromaDB: 13 dimensões

- 4 posições de categoria: eletrônicos, vestuário, calçados, acessórios.
- 8 posições de cor: preto, prata, azul, branco, vermelho, bege, cinza, marrom.
- 1 posição para preço (normalizado até R$ 200).
- Pesos didáticos: categoria **0,55**, cor **0,30**, preço **0,15**. Os pesos são aplicados via raiz quadrada antes da normalização L2.

Esses vetores **não são embeddings semânticos de um LLM**: são *feature vectors* construídos manualmente.

### TensorFlow/Keras: 27 entradas

```python
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(27,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])
model.compile(optimizer=tf.keras.optimizers.Adam(0.003),
              loss='binary_crossentropy', metrics=['accuracy'])
```

- **ReLU** ativa neurônios com valores positivos.
- **Sigmoid** produz valor entre 0 e 1; **não** é probabilidade calibrada neste laboratório.
- **Loss** mede o erro entre predição e rótulo; o otimizador **Adam** ajusta pesos por backpropagation.
- **Época** é uma passagem pelo conjunto de treinamento; 100 épocas não garantem melhor resultado.

## 6. Endpoints

| Método | Caminho | Função |
|---|---|---|
| GET | `/api/health` | Estado dos serviços |
| GET | `/api/users` | Clientes |
| GET | `/api/products` | Catálogo |
| GET | `/api/users/{id}` | Cliente e compras |
| POST | `/api/users/{id}/purchases` | Comprar `{ "productId": 7 }` |
| DELETE | `/api/users/{id}/purchases/{productId}` | Remover compra |
| GET | `/api/users/{id}/recommendations?mode=hybrid&limit=5` | Recomendar |
| GET | `/api/model/status` | Modelo, épocas, histórico |
| POST | `/api/model/train` | Treinar `{ "epochs": 100 }` |

A interface HTML preserva os botões de compra, remoção, seleção de usuário, modo e treinamento, além do histórico `loss` e `accuracy`.

## 7. Cenários didáticos

| Cenário | Ação | Resultado esperado |
|---|---|---|
| Cold start | Selecionar Josézin (ID 99) | Popularidade, sem histórico |
| Busca vetorial | Selecionar Ana e modo `vector` | Ordenação por similaridade de cosseno |
| Rede neural | Selecionar Ana e modo `neural` | Ordenação por saída sigmoid |
| Híbrido | Selecionar Ana e modo `hybrid` | Candidatos ChromaDB, ranking neural |
| Nova compra | Comprar um produto | Atualiza SQLite e perfil vetorial |
| Retreinamento | Clicar Treinar 100 épocas | Recria o modelo com os dados atuais |
| Inspecionar overfitting | Ver histórico de `loss`/`accuracy` | Métricas do conjunto de treino (sem validação) |

## 8. Testes

Os testes de regras, vetores, persistência e ranking **não dependem de TensorFlow nem de ChromaDB real**. Para rodar localmente:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install pytest
pytest -q -m 'not integration'
```

Teste opcional de TensorFlow real (exige instalar as dependências do `requirements.txt`):

```bash
pip install -r requirements-dev.txt
pytest -q -m integration
```

## 9. Limitações importantes

1. O dataset tem só **60 pares**, e a maior parte dos negativos significa simplesmente **não houve compra** — isso não comprova rejeição.
2. A rede pode **memorizar** o conjunto pequeno. Acurácia de treino não mede generalização.
3. A versão original também usa a compra como rótulo e o histórico como característica; retiramos o **produto alvo** do perfil durante a geração do exemplo para reduzir vazamento direto.
4. Não há divisão treino/validação, calibração, avaliação offline (Precision@K, Recall@K, NDCG), nem *negative sampling* robusto. São próximos passos, não resultados demonstrados.
5. A rede é treinada **sob demanda**; registrar compra **não altera pesos automaticamente**.
6. Os arquivos Keras e SQLite devem ser considerados **dados locais de laboratório**, não infraestrutura de produção.
7. O treinamento pode ser pesado na primeira execução, pois o TensorFlow precisa carregar bibliotecas nativas.
8. O projeto não utiliza LLM, RAG, BM25, reranking cross-encoder nem embeddings semânticos.

## 10. Organização

```text
vectorshop-python-keras-chromadb/
├── compose.yaml
├── frontend/              # HTML, CSS, JS e Dockerfile Nginx
├── nginx/default.conf     # /api/ -> api:3001
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── tests/
│   └── app/
│       ├── main.py          # Controllers FastAPI
│       ├── db.py            # SQLite / Repository
│       ├── seed.py          # Mesmos produtos e usuários do Node
│       ├── features.py      # Vetorização 13D / entrada neural 27D
│       ├── vector_store.py  # ChromaDB
│       ├── recommendation.py # Recomendação vetorial/neural/híbrida
│       ├── neural_service.py # Gerencia subprocessos e modelo
│       └── neural_worker.py  # TensorFlow/Keras (treino/inferência)
└── EXEMPLOS_REQUISICOES.http
```

**Escopo:** migração didática, não substituição da rede neural. A versão JavaScript permanece intacta.
