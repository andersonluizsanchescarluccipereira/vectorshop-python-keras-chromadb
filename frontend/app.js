const $ = (id) => document.getElementById(id);
const API = ['5173','5500','4173'].includes(location.port) || location.protocol === 'file:'
  ? 'http://localhost:3001/api' : '/api';
const state = { users: [], products: [], selectedUserId: null, user: null, busy: false, selectionToken: 0 };
const formatCurrency = value => new Intl.NumberFormat('pt-BR',{style:'currency',currency:'BRL'}).format(value);
const icons = { 'eletrônicos':'🎧','vestuário':'👕','calçados':'👟','acessórios':'🎒' };
const productIcon = p => p.name.includes('Relógio')?'⌚':p.name.includes('Caixa')?'🔊':p.name.includes('Calça')?'👖':p.name.includes('Sandália')?'🩴':p.name.includes('Boné')?'🧢':p.name.includes('Óculos')?'🕶️': icons[p.category] || '📦';
const escapeHtml = text => String(text).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

async function api(path, options={}) {
  let res;
  try { res=await fetch(`${API}${path}`,{ ...options, headers:{'Content-Type':'application/json',...(options.headers||{})} }); }
  catch { throw new Error('Não foi possível conectar à API Python.'); }
  const body=await res.json().catch(()=>({}));
  if(!res.ok) throw new Error(body.error || (typeof body.detail === 'string' ? body.detail : null) || `Erro HTTP ${res.status}`);
  return body;
}

function notify(text) {
  const el=$('message'); el.textContent=text; el.hidden=false;
  clearTimeout(notify.timeout); notify.timeout=setTimeout(()=>el.hidden=true,3700);
}
function setBusy(busy) {
  state.busy=busy;
  $('userSelect').disabled=busy;
  $('refresh').disabled=busy;
  document.querySelectorAll('button.buy,button.remove').forEach(el=>el.disabled=busy || el.dataset.disabled==='true');
}

function renderProducts() {
  $('productCount').textContent=`${state.products.length} produtos`;
  const bought=new Set((state.user?.purchases||[]).map(p=>p.id));
  $('products').innerHTML=state.products.map(p=>`
    <article class="product-card">
      <div class="product-art" aria-hidden="true">${productIcon(p)}</div>
      <div class="product-body">
        <div class="category">${escapeHtml(p.category)}</div>
        <h3>${escapeHtml(p.name)}</h3>
        <p class="detail">Cor: ${escapeHtml(p.color)}</p>
        <div class="product-bottom">
          <strong class="price">${formatCurrency(p.price)}</strong>
          <button class="buy" data-product-id="${p.id}" data-disabled="${bought.has(p.id)}" ${bought.has(p.id)||state.busy?'disabled':''}>${bought.has(p.id)?'Comprado':'Comprar'}</button>
        </div>
      </div>
    </article>`).join('');
}
function renderUser() {
  const user=state.user;
  $('userName').textContent=user?.name||'Nenhum usuário';
  $('userAge').textContent=user?`${user.age} anos · ID ${user.id}`:'—';
  $('userAvatar').textContent=user?.name?.charAt(0)||'?';
  $('purchaseCount').textContent=user?.purchases.length||'0';
  $('purchases').innerHTML=user?.purchases.length ? user.purchases.map(p=>`
    <div class="purchase-item"><span class="emoji" aria-hidden="true">${productIcon(p)}</span>
      <div><strong>${escapeHtml(p.name)}</strong><small>${formatCurrency(p.price)}</small></div>
      <button class="remove" data-product-id="${p.id}" title="Remover ${escapeHtml(p.name)}" aria-label="Remover ${escapeHtml(p.name)}">×</button>
    </div>`).join('') : '<p class="placeholder">Nenhuma compra. Este perfil demonstra o problema de cold start.</p>';
  renderProducts();
}
function renderRecommendations(result) {
  $('recExplanation').textContent = result.note;
  $('recommendations').innerHTML=result.recommendations.length ? result.recommendations.map((p,i)=>`
    <div class="recommend-card">
      <span class="emoji" aria-hidden="true">${productIcon(p)}</span>
      <strong>${i+1}. ${escapeHtml(p.name)}</strong>
      <div class="meta">${escapeHtml(p.category)} · ${formatCurrency(p.price)}</div>
      <span class="score">${result.mode==='popularidade'?'Popularidade':`${(p.score*100).toFixed(1)}% ${result.mode==='vector'?'similaridade':'score neural'}`}</span>
    </div>`).join('') : '<p class="placeholder">Nenhuma nova recomendação disponível.</p>';
}
async function selectUser(userId) {
  const token=++state.selectionToken;
  state.selectedUserId=userId;
  setBusy(true);
  try {
    const [user,recommendations]=await Promise.all([
      api(`/users/${userId}`),api(`/users/${userId}/recommendations?limit=6&mode=${encodeURIComponent($('modeSelect').value)}`)
    ]);
    if(token!==state.selectionToken)return;
    state.user=user;
    renderUser(); renderRecommendations(recommendations);
  } catch(e) {if(token===state.selectionToken) notify(e.message);}
  finally {if(token===state.selectionToken)setBusy(false);}
}
async function action(method,productId) {
  if(state.busy || !state.selectedUserId)return;
  setBusy(true);
  try {
    const result=await api(`/users/${state.selectedUserId}/purchases${method==='DELETE'?`/${productId}`:''}`,{
      method, ...(method==='POST'?{body:JSON.stringify({productId})}:{})
    });
    if(!result.indexSynced) notify('Compra salva, mas sincronização vetorial pendente.');
    else notify(method==='POST'?'Compra registrada. Perfil atualizado!':'Compra removida. Perfil atualizado!');
    await selectUser(state.selectedUserId);
  } catch(e) {notify(e.message);} finally {setBusy(false);}
}
async function updateModelStatus() {
  const m=await api('/model/status');
  $('modelStatus').textContent=m.training?'Treinando...':m.trained?`Treinado · ${m.epochs} épocas · ${m.examples} exemplos`:'Não treinado';
  $('trainingHistory').textContent=(m.history||[]).map(h=>`Época ${String(h.epoch).padStart(3)} | loss=${h.loss} | accuracy=${(h.accuracy*100).toFixed(1)}%`).join('\n');
}
async function trainModel() {
  const button=$('train');button.disabled=true;
  $('modelStatus').textContent='Treinando em processo separado...';
  try {await api('/model/train',{method:'POST',body:JSON.stringify({epochs:100})});
    await updateModelStatus();await selectUser(state.selectedUserId);notify('Treinamento concluído e modelo salvo!');
  } catch(e){notify(e.message);await updateModelStatus().catch(()=>{});}
  finally{button.disabled=false;}
}
async function main() {
  try {
    const [users,products,health]=await Promise.all([api('/users'),api('/products'),api('/health')]);
    state.users=users;state.products=products;
    $('health').classList.add(health.chroma?'ok':'bad');
    $('health').innerHTML=`<span class="dot"></span> ${health.chroma?'API + ChromaDB':'Chroma indisponível'}`;
    $('userSelect').innerHTML=users.map(u=>`<option value="${u.id}">${escapeHtml(u.name)}</option>`).join('');
    $('userSelect').addEventListener('change',e=>selectUser(Number(e.target.value)));
    $('refresh').addEventListener('click',()=>selectUser(state.selectedUserId));
    $('modeSelect').addEventListener('change',()=>selectUser(state.selectedUserId));
    $('train').addEventListener('click',trainModel);
    await updateModelStatus();
    document.addEventListener('click',e=>{
      const buy=e.target.closest('button.buy');
      const remove=e.target.closest('button.remove');
      if(buy)action('POST',Number(buy.dataset.productId));
      if(remove)action('DELETE',Number(remove.dataset.productId));
    });
    renderProducts();
    await selectUser(users[0].id);
  } catch(error) {
    $('health').classList.add('bad');$('health').innerHTML='<span class="dot"></span> API offline';
    notify(error.message);
    $('recommendations').textContent='Não foi possível carregar os dados. Confira o backend e o ChromaDB.';
  }
}
main();
