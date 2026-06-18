let chart = null;
let canal = "rgb";
let tipoHist = "hist";
let metaOrigen = "original";
let valoracionActual = 0;

const COLORES_NOMBRE = [
  { hex:"#000000", nom:"Negro" },   { hex:"#FFFFFF", nom:"Blanco" },
  { hex:"#808080", nom:"Gris" },    { hex:"#C0C0C0", nom:"Plateado" },
  { hex:"#FF0000", nom:"Rojo" },    { hex:"#8B0000", nom:"Rojo oscuro" },
  { hex:"#FFC0CB", nom:"Rosa" },    { hex:"#FF69B4", nom:"Rosa fuerte" },
  { hex:"#FF8C00", nom:"Naranja" }, { hex:"#FFD700", nom:"Amarillo dorado" },
  { hex:"#FFFF00", nom:"Amarillo" },{ hex:"#00FF00", nom:"Verde lima" },
  { hex:"#008000", nom:"Verde" },   { hex:"#006400", nom:"Verde oscuro" },
  { hex:"#00FFFF", nom:"Cian" },    { hex:"#008B8B", nom:"Verde teal" },
  { hex:"#0000FF", nom:"Azul" },    { hex:"#000080", nom:"Azul marino" },
  { hex:"#4169E1", nom:"Azul rey" },{ hex:"#87CEEB", nom:"Azul cielo" },
  { hex:"#EE82EE", nom:"Violeta" }, { hex:"#800080", nom:"Morado" },
  { hex:"#4B0082", nom:"Índigo" },  { hex:"#A52A2A", nom:"Marrón" },
  { hex:"#D2691E", nom:"Terracota"},{ hex:"#F5DEB3", nom:"Trigo" },
  { hex:"#FFDAB9", nom:"Melocotón"},{ hex:"#808000", nom:"Oliva" },
  { hex:"#2F4F4F", nom:"Verde pizarra" },
];

function hexDist(h1, h2) {
  const p = h => ({
    r: parseInt(h.slice(1,3),16),
    g: parseInt(h.slice(3,5),16),
    b: parseInt(h.slice(5,7),16)
  });
  const a=p(h1), b=p(h2);
  return Math.sqrt((a.r-b.r)**2+(a.g-b.g)**2+(a.b-b.b)**2);
}

function nombreColor(hex) {
  let min=999, nom="Color personalizado";
  COLORES_NOMBRE.forEach(c => {
    const d = hexDist(hex, c.hex);
    if (d < min) { min=d; nom=c.nom; }
  });
  return nom;
}

function detectarColoresDominantes(imgEl, callback) {
  const canvas = document.createElement("canvas");
  const ctx    = canvas.getContext("2d");
  canvas.width = 100; canvas.height = 100;
  ctx.drawImage(imgEl, 0, 0, 100, 100);
  const data = ctx.getImageData(0, 0, 100, 100).data;

  let fR=0,fG=0,fB=0,fN=0;
  const PX=10;
  for(let y=0;y<PX;y++) for(let x=0;x<PX;x++){
    [[y,x],[y,99-x],[99-y,x],[99-y,99-x]].forEach(([cy,cx])=>{
      const i=(cy*100+cx)*4;
      fR+=data[i]; fG+=data[i+1]; fB+=data[i+2]; fN++;
    });
  }
  fR/=fN; fG/=fN; fB/=fN;

  let varS=0;
  for(let y=0;y<PX;y++) for(let x=0;x<PX;x++){
    [[y,x],[y,99-x],[99-y,x],[99-y,99-x]].forEach(([cy,cx])=>{
      const i=(cy*100+cx)*4;
      const d=Math.abs(data[i]-fR)+Math.abs(data[i+1]-fG)+Math.abs(data[i+2]-fB);
      varS+=d*d;
    });
  }
  const umbral = Math.max(35, Math.sqrt(varS/fN)*4+25);

  const bins={}; let totalPx=0;
  for(let i=0;i<data.length;i+=4){
    const r=data[i],g=data[i+1],b=data[i+2];
    const brillo=(r+g+b)/3;
    const dist=Math.abs(r-fR)+Math.abs(g-fG)+Math.abs(b-fB);
    if(dist<umbral||brillo>220||brillo<12) continue;
    totalPx++;
    const rq=Math.round(r/24)*24;
    const gq=Math.round(g/24)*24;
    const bq=Math.round(b/24)*24;
    const key=`${rq},${gq},${bq}`;
    bins[key]=(bins[key]||0)+1;
  }

  if(!totalPx){ callback(["#888888"]); return; }

  const ordenados = Object.entries(bins)
    .map(([k,f])=>({k,f,pct:f/totalPx*100}))
    .filter(e=>e.pct>=5)
    .sort((a,b)=>b.f-a.f);

  const seleccionados=[];
  const DIST_MIN=60;
  for(const {k} of ordenados){
    const [r,g,b]=k.split(",").map(Number);
    const hex="#"+[r,g,b].map(v=>Math.max(0,Math.min(255,v)).toString(16).padStart(2,"0")).join("");
    let cercano=false;
    for(const s of seleccionados){
      if(hexDist(hex,s)<DIST_MIN){ cercano=true; break; }
    }
    if(!cercano){
      seleccionados.push(hex);
      if(seleccionados.length>=4) break;
    }
  }
  callback(seleccionados.length>0 ? seleccionados : ["#888888"]);
}

function subir(){
  let archivo = document.getElementById("imagen").files[0];
  let formData = new FormData();
  formData.append("imagen", archivo);
  fetch("/upload", { method:"POST", body:formData })
  .then(res => res.json())
  .then(data => {
    document.getElementById("original").src = data.original + "?" + Date.now();
    document.getElementById("procesada").src = "";
    if(chart) chart.destroy();
  });
}

function procesar(op){
  let formData = new FormData();
  formData.append("op", op);
  fetch("/procesar", { method:"POST", body:formData })
  .then(res => res.json())
  .then(data => {
    if(data.procesada)
      document.getElementById("procesada").src = data.procesada + "?" + Date.now();
    if(data.r || data.g || data.b)
      dibujarGrafica(data);
  });
}

function obtenerHistograma(){
  let formData = new FormData();
  formData.append("op", tipoHist);
  fetch("/procesar", { method:"POST", body:formData })
  .then(res => res.json())
  .then(data => dibujarGrafica(data));
}

function mostrar(c){ canal=c; obtenerHistograma(); }

function dibujarGrafica(data){
  if(chart) chart.destroy();
  let datasets;
  if(canal==="r")      datasets=[{label:"R",   data:data.r,    borderColor:"red",   fill:false}];
  else if(canal==="g") datasets=[{label:"G",   data:data.g,    borderColor:"green", fill:false}];
  else if(canal==="b") datasets=[{label:"B",   data:data.b,    borderColor:"blue",  fill:false}];
  else if(canal==="gray") datasets=[{label:"Gris",data:data.gray,borderColor:"black",fill:false}];
  else datasets=[
    {label:"R",data:data.r,borderColor:"red",  fill:false},
    {label:"G",data:data.g,borderColor:"green",fill:false},
    {label:"B",data:data.b,borderColor:"blue", fill:false}
  ];
  chart = new Chart(document.getElementById("grafica"),{
    type:"line",
    data:{ labels:[...Array(256).keys()], datasets },
    options:{ responsive:true, scales:{
      x:{title:{display:true,text:"Intensidad"}},
      y:{title:{display:true,text:"Frecuencia"}}
    }}
  });
}

function aplicarColor(){
  let r=document.getElementById("r").value;
  let g=document.getElementById("g").value;
  let b=document.getElementById("b").value;
  let formData = new FormData();
  formData.append("op","color");
  formData.append("r",r); formData.append("g",g); formData.append("b",b);
  fetch("/procesar",{method:"POST",body:formData})
  .then(res=>res.json())
  .then(data=>{ document.getElementById("procesada").src=data.procesada+"?"+Date.now(); });
}

function abrirMeta(origen){
  metaOrigen = origen;
  const imgEl  = document.getElementById(origen);
  const imgSrc = imgEl.src;
  if(!imgSrc || imgSrc.endsWith("undefined") || imgSrc === window.location.href){
    mostrarToast("Primero carga una imagen","info"); return;
  }
  document.getElementById("metaFotoPreview").src = imgSrc;
  resetearMeta();
  fetch("/clasificar_preview",{method:"POST"})
  .then(r=>r.json())
  .then(data=>{ if(data.categoria) document.getElementById("metaCat").value=data.categoria; });
  if(imgEl.complete && imgEl.naturalWidth){
    detectarColoresDominantes(imgEl, aplicarColoresDetectados);
  } else {
    imgEl.onload = () => detectarColoresDominantes(imgEl, aplicarColoresDetectados);
  }
  document.getElementById("modalMeta").classList.add("visible");
}

let coloresActuales = [];

function aplicarColoresDetectados(lista){
  coloresActuales = [];
  const cont = document.getElementById("coloresDetectados");
  cont.innerHTML = "";

  if(lista.length >= 4){
    const badge = document.createElement("span");
    badge.className = "badge-multicolor";
    badge.textContent = "✦ Multicolor";
    cont.appendChild(badge);
    coloresActuales = lista.slice(0,3);
    document.getElementById("btnAddColor").style.display = "none";
    return;
  }

  lista.forEach(hex => agregarCirculoColor(hex));
  actualizarVisibilidadBotonAdd();
}

function agregarCirculoColor(hex){
  if(coloresActuales.length >= 3) return;
  if(coloresActuales.includes(hex)) return;
  coloresActuales.push(hex);

  const cont = document.getElementById("coloresDetectados");
  const badge = cont.querySelector(".badge-multicolor");
  if(badge) badge.remove();

  const wrap = document.createElement("div");
  wrap.className = "color-circulo";
  wrap.style.background = hex;
  wrap.dataset.hex = hex;
  wrap.title = nombreColor(hex);

  const x = document.createElement("div");
  x.className = "circulo-x";
  x.textContent = "✕";
  x.onclick = () => quitarCirculoColor(hex, wrap);
  wrap.appendChild(x);
  cont.appendChild(wrap);

  actualizarVisibilidadBotonAdd();
}

function quitarCirculoColor(hex, el){
  coloresActuales = coloresActuales.filter(c => c !== hex);
  el.remove();
  actualizarVisibilidadBotonAdd();
}

function actualizarVisibilidadBotonAdd(){
  const btn = document.getElementById("btnAddColor");
  btn.style.display = coloresActuales.length < 3 ? "flex" : "none";
}

function mostrarPickerExtra(){
  document.getElementById("pickerExtra").style.display = "flex";
  document.getElementById("btnAddColor").style.display = "none";
}

function previewExtra(hex){
  document.getElementById("extraNombre").textContent = nombreColor(hex);
}

function confirmarColorExtra(){
  const hex = document.getElementById("metaColorExtra").value;
  agregarCirculoColor(hex);
  cancelarPickerExtra();
}

function cancelarPickerExtra(){
  document.getElementById("pickerExtra").style.display = "none";
  actualizarVisibilidadBotonAdd();
}

function resetearMeta(){
  valoracionActual = 0;
  document.querySelectorAll(".estrella").forEach(e=>e.classList.remove("llena"));
  document.querySelectorAll(".meta-tag").forEach(t=>t.classList.remove("activo"));
  document.getElementById("metaMaterial").value = "";
  coloresActuales = [];
  document.getElementById("coloresDetectados").innerHTML = "";
  document.getElementById("pickerExtra").style.display = "none";
  document.getElementById("btnAddColor").style.display = "flex";
}

function cerrarMeta(){
  document.getElementById("modalMeta").classList.remove("visible");
}

function cerrarMetaSiFondo(e){
  if(e.target===document.getElementById("modalMeta")) cerrarMeta();
}

function toggleTag(el){ el.classList.toggle("activo"); }

function valorar(n){
  valoracionActual = n;
  document.querySelectorAll(".estrella").forEach((e,i)=>{
    e.classList.toggle("llena", i < n);
  });
}

function confirmarGuardar(){
  const categoria  = document.getElementById("metaCat").value;
  const color    = coloresActuales.join(",");
  const colorNom = coloresActuales.map(h => nombreColor(h)).join(", ");
  const material   = document.getElementById("metaMaterial").value;
  const estacion   = [...document.querySelectorAll(".meta-tag[data-group='estacion'].activo")]
                       .map(t=>t.dataset.val).join(",");
  const ocasion    = [...document.querySelectorAll(".meta-tag[data-group='ocasion'].activo")]
                       .map(t=>t.dataset.val).join(",");
  const fd = new FormData();
  fd.append("categoria",    categoria);
  fd.append("color",        color);
  fd.append("color_nombre", colorNom);
  fd.append("estacion",     estacion);
  fd.append("ocasion",      ocasion);
  fd.append("material",     material);
  fd.append("valoracion",   valoracionActual);
  fd.append("usar_original", metaOrigen==="original" ? "1" : "0");
  fetch("/guardar_con_meta",{method:"POST",body:fd})
  .then(r=>r.json())
  .then(data=>{
    if(data.ok){
      cerrarMeta();
      mostrarToast("Prenda guardada en "+data.categoria,"ok");
    } else {
      mostrarToast("Error: "+(data.error||"desconocido"),"error");
    }
  });
}

function mostrarToast(mensaje, tipo){
  let toast = document.getElementById("toast");
  if(!toast){
    toast = document.createElement("div");
    toast.id = "toast";
    toast.style.cssText = `position:fixed;bottom:32px;right:32px;
      background:linear-gradient(135deg,#1C1410,#3D1F2E);color:white;
      padding:14px 22px;border-radius:12px;font-family:'DM Sans',sans-serif;
      font-size:.88rem;font-weight:500;box-shadow:0 8px 32px rgba(0,0,0,.25);
      border-left:4px solid #6D2E46;z-index:9999;opacity:0;
      transform:translateY(12px);transition:opacity .3s,transform .3s;
      pointer-events:none;max-width:320px;display:flex;align-items:center;gap:10px`;
    toast.innerHTML = '<span id="toast-icon">✓</span><span id="toast-msg"></span>';
    document.body.appendChild(toast);
  }
  const colors={ok:"#6D2E46",error:"#C0392B",info:"#B5885C"};
  const icons ={ok:"✓",error:"✕",info:"ℹ"};
  toast.style.borderLeftColor=colors[tipo]||"#6D2E46";
  toast.querySelector("#toast-icon").textContent=icons[tipo]||"✓";
  toast.querySelector("#toast-msg").textContent=mensaje;
  toast.style.opacity="1"; toast.style.transform="translateY(0)";
  clearTimeout(toast._t);
  toast._t=setTimeout(()=>{
    toast.style.opacity="0"; toast.style.transform="translateY(12px)";
  },3500);
}

function irArmario(){ window.location.href="/armario"; }

window.onload = function(){
  const params = new URLSearchParams(window.location.search);
  if(params.get("editar")==="1"){
    document.getElementById("original").src="/static/entrada.bmp?"+Date.now();
  }
};

function barra(label,val,max,color){
  const pct=Math.round((val/max)*100);
  return `<div style="margin-bottom:8px">
    <div style="display:flex;justify-content:space-between;font-size:.8rem;margin-bottom:3px">
      <span>${label}</span><span>${pct}%</span>
    </div>
    <div style="background:#eee;border-radius:4px;height:8px">
      <div style="width:${pct}%;background:${color};height:8px;border-radius:4px;
                  transition:width .4s"></div>
    </div>
  </div>`;
}
