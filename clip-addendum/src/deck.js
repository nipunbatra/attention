(()=>{'use strict';
const slides=[...document.querySelectorAll('.slide')];
const mode=document.querySelector('#mode'),counter=document.querySelector('#counter');
let i=Math.max(0,slides.findIndex(s=>s.id===location.hash.slice(1))),build=1;
let present=new URLSearchParams(location.search).has('present');
function max(){return Math.max(1,...[...slides[i].querySelectorAll('[data-reveal-step]')].map(e=>+e.dataset.revealStep));}
function resize(){document.documentElement.style.setProperty('--scale',Math.min(innerWidth/1280,(innerHeight-80)/720));}
function render(){document.body.classList.toggle('present',present);mode.textContent=present?'Read':'Present';slides.forEach((s,k)=>{s.classList.toggle('active',k===i);s.querySelectorAll('[data-reveal-step]').forEach(e=>{const hidden=present&&+e.dataset.revealStep>build;e.classList.toggle('reveal-hidden',hidden);if(hidden)e.setAttribute('aria-hidden','true');else e.removeAttribute('aria-hidden');});});counter.textContent=`${i+1} / ${slides.length}${present?' · '+build+'/'+max():''}`;document.querySelector('#previous').disabled=i===0&&(!present||build===1);document.querySelector('#next').disabled=i===slides.length-1&&(!present||build===max());const titles=slides[i].querySelector('[data-reveal-titles]');counter.title=titles?JSON.parse(titles.dataset.revealTitles)[build-1]||'':'';resize();}
function jump(delta){if(present&&delta>0&&build<max())build++;else if(present&&delta<0&&build>1)build--;else{const next=Math.max(0,Math.min(slides.length-1,i+delta));if(next!==i){slides[i].querySelectorAll('video').forEach(v=>v.pause());i=next;build=delta<0?max():1;history.replaceState(null,'','#'+slides[i].id);if(!present)slides[i].scrollIntoView({behavior:'smooth'});}}render();}
document.querySelector('#next').onclick=()=>jump(1);document.querySelector('#previous').onclick=()=>jump(-1);
mode.onclick=()=>{present=!present;build=1;const url=new URL(location.href);present?url.searchParams.set('present',''):url.searchParams.delete('present');history.replaceState(null,'',url);render();if(!present)slides[i].scrollIntoView();};
document.querySelector('#all').onclick=()=>{build=max();render();};
document.querySelector('#notes-toggle').onclick=()=>{slides[i].classList.toggle('show-notes');const d=slides[i].querySelector('details');d.open=!d.open;if(!present)d.scrollIntoView({behavior:'smooth'});};
addEventListener('keydown',e=>{if(e.target.closest('video,button,a,summary,input,select,textarea'))return;if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();jump(1);}if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();jump(-1);}if(e.key==='Escape'&&present){mode.click();}});
addEventListener('hashchange',()=>{const n=slides.findIndex(s=>s.id===location.hash.slice(1));if(n>=0){slides[i].querySelectorAll('video').forEach(v=>v.pause());i=n;build=1;render();}});
addEventListener('resize',resize);
document.querySelectorAll('video').forEach(v=>{v.addEventListener('error',()=>{v.closest('figure').querySelector('.video-fallback').hidden=false;});v.querySelector('source').addEventListener('error',()=>{v.closest('figure').querySelector('.video-fallback').hidden=false;});});
render();
})();
