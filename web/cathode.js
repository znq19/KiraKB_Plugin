/* KiraKB // LINJI ARCHIVE — integrated WebUI application */
(() => {
  "use strict";
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
  const sleep = (ms) => new Promise(r => setTimeout(r, ms));
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const THEMES = ["violet", "pink", "blue", "amber", "sg", "green", "red", "gray"];
  const ORGS = window.LINJI_ORGS || {};
  const CUSTOM_ASSET_KEY = "linji-custom-v1";
  const PREFIXES = ["记录", "残页", "旁注", "回收", "未归档", "观测", "编号", "残响"];
  const MOTTOS = [
    "知识不会拯救任何人，它只是让痛苦有了名字。",
    "记录即收容。未被写下的东西，会反过来注视你。",
    "档案室没有出口，只有更深的分类。",
    "名字是第一条锁链。",
    "每一次检索，都是一次招魂。",
    "不要在索引里寻找自己。",
    "遗忘是免费的，记住要付利息。",
    "所有的答案都带着编号。",
    "被归档的，不等于被理解。",
    "我们保存一切，除了当时的我们。"
  ];
  const BG_SLOTS = [
    ["catalog", "主厅 / CATALOGUE"], ["versions", "版本 / VERSIONS"],
    ["search", "检索 / SEARCH"], ["access", "权限 / ACCESS"],
    ["recall", "召回 / RECALL"], ["settings", "设置 / SETTINGS"],
    ["tasks", "任务 / TASKS"], ["wiki", "临极档案 / WIKI"], ["ticker", "字幕框 / SUBTITLE"]
  ];
  const VIEW_META = {
    catalog: ["01", "CATALOGUE"], versions: ["02", "VERSIONS"],
    search: ["03", "SEARCH"], access: ["04", "ACCESS"],
    recall: ["05", "RECALL"], settings: ["06", "SETTINGS"], tasks: ["07", "TASKS"],
    wiki: ["08", "WIKI"]
  };
  const UI_DEFAULT = {
    theme: "violet", crt: "high", performance: false, splash: true,
    text_fx: "mix", fx_loop: true, fx_interval: 6,
    quote_enabled: true, quote_interval: 9, quote_kb_ids: [],
    quote_custom_lines: [], quote_custom_mottos: [],
    wall_rotate: true, wall_interval: 18, wall_parallax: true
  };

  const state = {
    kbs: [], current: null, docs: [], versions: [], tasks: [], deleted: [],
    models: [], globalRecall: {}, settings: null,
    view: "catalog", selectedDoc: null, selected: new Set(),
    docFilter: "", docFilterKind: "all",
    ui: { ...UI_DEFAULT },
    quoteTimer: null, quoteBusy: false, tickerTimer: null,
    uiSaveTimer: null, backgrounds: loadBackgrounds(),
    orgWallpaper: "", custom: loadCustomAssets(),
    wallDisabled: loadWallDisabled(), wallTimer: null, wallFxTimer: null,
    zero: (window.linjiZero ? window.linjiZero() : null),
    zeroRandomPick: false, wallVaultTheme: "violet",
    apprOpen: false, apprPool: [], apprIndex: 0, apprTimer: null,
    apprTickerTimer: null, apprForcedQuote: false, parallaxBound: false, px: 0, py: 0, tx: 0, ty: 0
  };

  /* ---------- API ---------- */
  const isSidebarMode = typeof window.PluginPageContext !== "undefined"
    && window.PluginPageContext.ready && window.parent !== window;
  const token = new URLSearchParams(location.search).get("token") || "";
  let bridgeCtx = null;
  function pluginIdFromPath() {
    const m = location.pathname.match(/^\/plugin-page\/([^/]+)\//) || location.pathname.match(/^\/page\/plugin\/([^/]+)\//);
    return m ? decodeURIComponent(m[1]) : null;
  }
  async function api(path, opts = {}) {
    const method = (opts.method || "GET").toUpperCase();
    let data;
    if (isSidebarMode) {
      if (!bridgeCtx) {
        bridgeCtx = await Promise.race([
          window.PluginPageContext.ready(),
          new Promise((_, rej) => setTimeout(() => rej(new Error("页面桥接超时")), 8000))
        ]);
      }
      if (method === "GET") data = await window.PluginPageContext.api.get(path);
      else if (method === "POST") data = await window.PluginPageContext.api.post(path, opts.body ? JSON.parse(opts.body) : {});
      else if (method === "DELETE") data = await window.PluginPageContext.api.delete(path);
      else {
        const res = await fetch("/api/plugin/" + encodeURIComponent(bridgeCtx.pluginId) + path, {
          method, credentials: "same-origin", headers: { "Content-Type": "application/json" }, body: opts.body || "{}"
        });
        data = await res.json();
      }
    } else {
      const pid = pluginIdFromPath();
      const base = pid ? "/api/plugin/" + encodeURIComponent(pid) : "/api";
      const headers = { "Content-Type": "application/json" };
      if (token) headers.Authorization = "Bearer " + token;
      const res = await fetch(base + path, { ...opts, headers });
      data = await res.json().catch(() => ({ error: "返回内容不是有效 JSON" }));
    }
    if (data && data.error && !opts.allowError) throw new Error(data.error);
    return data;
  }
  async function apiUpload(path, formData) {
    if (isSidebarMode) {
      if (!bridgeCtx) bridgeCtx = await window.PluginPageContext.ready();
      const data = await window.PluginPageContext.api.upload(path, formData.get("file"), "file");
      if (data && data.error) throw new Error(data.error);
      return data;
    }
    const pid = pluginIdFromPath();
    const base = pid ? "/api/plugin/" + encodeURIComponent(pid) : "/api";
    const headers = token ? { Authorization: "Bearer " + token } : {};
    const res = await fetch(base + path, { method: "POST", body: formData, headers });
    const data = await res.json().catch(() => ({ error: "返回内容不是有效 JSON" }));
    if (data && data.error) throw new Error(data.error);
    return data;
  }

  /* ---------- utilities ---------- */
  function esc(value) {
    return String(value == null ? "" : value).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
  function icon(name, size = 14) {
    return `<svg style="width:${size}px;height:${size}px"><use href="#i-${name}"></use></svg>`;
  }
  function fmtTime(ts) {
    if (!ts) return "—";
    const d = new Date(Number(ts) * 1000);
    if (Number.isNaN(d.getTime())) return "—";
    const p = n => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
  }
  function fmtBytes(n) {
    const v = Number(n) || 0;
    if (v < 1024) return v + " B";
    if (v < 1024 * 1024) return (v / 1024).toFixed(1) + " KB";
    return (v / 1024 / 1024).toFixed(1) + " MB";
  }
  function toast(message, kind = "") {
    const el = $("#toast");
    if (!el) return;
    el.textContent = message;
    el.className = "toast is-on " + kind;
    clearTimeout(toast.t);
    toast.t = setTimeout(() => { el.className = "toast"; }, 2800);
  }
  function currentKb() { return state.kbs.find(k => k.kb_id === state.current) || null; }
  function activeVersion() { return state.versions.find(v => v.is_active) || null; }
  function currentDoc() { return state.docs.find(d => d.doc_id === state.selectedDoc) || null; }

  /* ---------- modal ---------- */
  function openModal({ title, subtitle = "", body = "", footer = "", small = false, onMount }) {
    const root = $("#modalRoot");
    root.innerHTML = `<div class="modal-root is-open"><div class="modal-backdrop" data-close></div><div class="modal-sheet ${small ? "small" : ""}" role="dialog" aria-modal="true"><div class="modal-head"><div><h3>${esc(title)}</h3>${subtitle ? `<p>${esc(subtitle)}</p>` : ""}</div><button class="modal-x" type="button" data-close>${icon("x", 14)}</button></div><div class="modal-body">${body}</div>${footer ? `<div class="modal-foot">${footer}</div>` : ""}</div></div>`;
    const close = () => { root.innerHTML = ""; };
    $$("[data-close]", root).forEach(b => b.addEventListener("click", close));
    root.addEventListener("keydown", e => { if (e.key === "Escape") close(); });
    const first = $("input,textarea,select,button", root);
    if (first) setTimeout(() => first.focus(), 20);
    if (onMount) onMount(root, close);
    return { root, close };
  }
  function confirmDialog(title, message, okText = "确认", danger = false) {
    return new Promise(resolve => {
      const m = openModal({
        title, small: true, body: `<p>${esc(message)}</p>`,
        footer: `<button class="btn ghost" type="button" data-cancel>取消</button><button class="btn ${danger ? "danger" : "primary"}" type="button" data-ok>${esc(okText)}</button>`,
        onMount(root, close) {
          $("[data-cancel]", root).onclick = () => { close(); resolve(false); };
          $("[data-ok]", root).onclick = () => { close(); resolve(true); };
        }
      });
    });
  }

  /* ---------- organization assets / custom skin ---------- */
  function loadCustomAssets() {
    try {
      const raw = JSON.parse(localStorage.getItem(CUSTOM_ASSET_KEY) || "{}");
      return raw && typeof raw === "object" ? raw : {};
    } catch (_) { return {}; }
  }
  function saveCustomAssets() {
    try { localStorage.setItem(CUSTOM_ASSET_KEY, JSON.stringify(state.custom)); }
    catch (_) { toast("自定义素材过大，未能写入本地存储", "err"); }
  }
  function zeroData() { return state.zero || window.linjiZero?.() || {}; }
  function zeroIsEnabled() { return !!(zeroData().enabled); }
  function zeroTargets() { return Array.isArray(zeroData().targets) ? zeroData().targets : []; }
  function zeroActiveFor(theme) {
    if (!zeroIsEnabled()) return false;
    const targets = zeroTargets();
    if (targets.includes(theme)) return true;
    return theme === state.ui.theme && state.zeroRandomPick && targets.includes("random");
  }
  function zeroAsOrg() {
    const z = zeroData();
    return {
      theme: "zero", chapter: z.chapter || "00", name: z.name || "第零章",
      en: z.en || "CHAPTER ZERO", color: z.color || "#d8c7a3",
      badge: z.badge || "", walls: Array.isArray(z.walls) ? z.walls : [],
      wallMeta: (z.walls || []).map((_, i) => ({ key: "zero-" + i, title: `第零章 · 残页 ${String(i + 1).padStart(2, "0")}`, caption: z.motto || "" })),
      motto: z.motto || "", summary: z.summary || "", lore: z.lore || "",
      doctrine: z.doctrine || "", uniform: z.uniform || "", relations: z.relations || "",
      quote: z.quote || "", tags: Array.isArray(z.tags) ? z.tags : String(z.tags || "").split(/[,，]/).map(s => s.trim()).filter(Boolean),
      chronicle: [], cases: [], extraLore: "", wallCount: (z.walls || []).length
    };
  }
  function orgFor(theme) { return theme === "zero" ? zeroAsOrg() : (zeroActiveFor(theme) ? zeroAsOrg() : (ORGS[theme] || ORGS.violet || null)); }
  function orgEnabled(theme) {
    if (theme === "zero") return true;
    if (zeroActiveFor(theme)) return true;
    const item = state.custom?.[theme];
    return !item || item.enabled !== false;
  }
  function badgeFor(theme, respectEnabled = true) {
    if (theme === "zero" || zeroActiveFor(theme)) return zeroData().badge || "";
    if (respectEnabled && !orgEnabled(theme)) return "";
    const org = orgFor(theme);
    return state.custom?.[theme]?.badge || (org && org.badge) || "";
  }
  function wallsFor(theme) {
    if (theme === "zero") return (zeroData().walls || []).filter(Boolean);
    if (zeroActiveFor(theme)) {
      const zw = (zeroData().walls || []).filter(Boolean);
      return zw.length ? zw : (ORGS[theme]?.walls || []);
    }
    const custom = Array.isArray(state.custom?.[theme]?.walls) ? state.custom[theme].walls : [];
    const base = (orgFor(theme)?.walls || []);
    return [...custom, ...base].filter((v, i, a) => v && a.indexOf(v) === i);
  }
  function wallMetaFor(theme, wall, index) {
    const org = orgFor(theme);
    const meta = Array.isArray(org?.wallMeta) ? org.wallMeta[index] : null;
    return { title: meta?.title || `WALL ${String(index + 1).padStart(2, "0")}`, caption: meta?.caption || org?.motto || "" };
  }
  function loadWallDisabled() {
    try { return new Set(JSON.parse(localStorage.getItem("linji-wall-disabled-v1") || "[]")); }
    catch (_) { return new Set(); }
  }
  function persistWallDisabled() {
    try { localStorage.setItem("linji-wall-disabled-v1", JSON.stringify([...state.wallDisabled])); }
    catch (_) {}
  }
  function wallEnabled(url) { return !state.wallDisabled.has(url); }
  function enabledWallsFor(theme) { return wallsFor(theme).filter(wallEnabled); }
  function pickOrgWallpaper(theme) {
    if (!orgEnabled(theme)) return "";
    const pool = enabledWallsFor(theme);
    if (!pool.length) return "";
    const current = state.orgWallpaper;
    const choices = pool.length > 1 ? pool.filter(u => u !== current) : pool;
    return choices[Math.floor(Math.random() * choices.length)];
  }
  function renderThemeBadges() {
    const row = $("#themeBadgeRow");
    if (!row) return;
    row.innerHTML = THEMES.map(theme => {
      const org = orgFor(theme);
      const badge = badgeFor(theme, false);
      if (!org) return "";
      return `<button type="button" data-theme-set="${theme}" class="${theme === state.ui.theme ? "is-active" : ""}" title="${esc(org.name)}">
        ${badge ? `<img src="${badge}" alt="">` : ""}<span>${esc(org.name)}</span></button>`;
    }).join("");
  }
  function applyOrgIdentity() {
    const theme = state.ui.theme;
    const org = orgFor(theme);
    const badge = badgeFor(theme, true);
    state.orgWallpaper = pickOrgWallpaper(theme);
    const rail = $("#railBadge");
    if (rail) {
      rail.src = badge || "";
      rail.hidden = !badge;
    }
    const boot = $("#bootBadge");
    if (boot) {
      boot.src = badge || (orgFor(theme)?.badge || "");
      boot.hidden = !badge;
    }
    const bootOrg = $("#bootOrg");
    if (bootOrg && org) bootOrg.textContent = `CHAPTER ${org.chapter} // ${org.name}`;
    renderThemeBadges();
    applyBackgrounds();
  }

  /* ---------- theme / splash ---------- */
  function loadBackgrounds() {
    try { return JSON.parse(localStorage.getItem("kirakb-cathode-bg-v2") || "{}") || {}; }
    catch (_) { return {}; }
  }
  function persistBackgrounds() {
    try { localStorage.setItem("kirakb-cathode-bg-v2", JSON.stringify(state.backgrounds)); }
    catch (_) { toast("背景图过大，未能写入本地存储", "err"); }
  }
  function renderThemeStrip() {
    const strip = $("#themeStrip");
    if (!strip) return;
    strip.innerHTML = THEMES.map(t => {
      const org = orgFor(t);
      const badge = badgeFor(t);
      return `<button class="swatch ${t === state.ui.theme ? "is-active" : ""}" type="button" data-theme-set="${t}" title="${org ? org.name : t}" aria-label="${org ? org.name : t}" ${org ? `style="--sw:${org.color}"` : ""}>${badge ? `<img class="swatch-badge" src="${badge}" alt="">` : ""}</button>`;
    }).join("") + `<button class="swatch random" type="button" data-theme-set="random" title="随机配色" aria-label="随机配色"></button>`;
  }
  function setTheme(theme, save = true) {
    state.zeroRandomPick = false;
    if (theme === "random") {
      const pool = THEMES.filter(t => t !== state.ui.theme);
      theme = pool[Math.floor(Math.random() * pool.length)];
      const z = zeroData();
      if (z.enabled && zeroTargets().includes("random")) state.zeroRandomPick = Math.random() < 0.5;
    }
    if (!THEMES.includes(theme)) theme = "violet";
    state.ui.theme = theme;
    state.wallVaultTheme = theme;
    document.documentElement.setAttribute("data-theme", "dark");
    document.body.dataset.theme = theme;
    $$("#themeStrip [data-theme-set]").forEach(b => b.classList.toggle("is-active", b.dataset.themeSet === theme));
    const themeSelect = $("#setUiTheme");
    if (themeSelect) themeSelect.value = theme;
    applyOrgIdentity();
    const screen = $("#screen");
    if (screen && save) { screen.classList.add("theme-flash"); setTimeout(() => screen.classList.remove("theme-flash"), 500); }
    if (save) queueUiSave();
  }
  function applyCrt(level) {
    state.ui.crt = ["off", "low", "medium", "high"].includes(level) ? level : "high";
    document.body.dataset.crt = state.ui.crt;
    const q = $("#quickCrt");
    if (q) q.setAttribute("aria-pressed", String(state.ui.crt !== "off"));
    const crtSelect = $("#setUiCrt");
    if (crtSelect) crtSelect.value = state.ui.crt;
  }
  function applyPerformance() {
    document.body.classList.toggle("perf", !!state.ui.performance);
    if (window.KBTextFX) window.KBTextFX.setConfig({ enabled: !state.ui.performance && !reduced, level: state.ui.performance ? 0 : 2 });
  }
  function applyUi() {
    setTheme(state.ui.theme, false);
    applyCrt(state.ui.crt);
    applyPerformance();
    if (window.KBTextFX) {
      window.KBTextFX.setConfig({
        mode: state.ui.text_fx, idle: true, enabled: !state.ui.performance && !reduced,
        level: state.ui.performance ? 0 : 2, loop: state.ui.fx_loop, loopInterval: state.ui.fx_interval
      });
      if (!state.ui.performance && !reduced && state.ui.fx_loop) window.KBTextFX.startLoop(state.ui.fx_interval);
      else window.KBTextFX.stopLoop();
    }
    syncUiControls();
    startBrandModes();
    startWallRotation();
    if (state.ui.quote_enabled) startQuoteTimer(); else stopQuoteTimer();
  }
  function queueUiSave() {
    clearTimeout(state.uiSaveTimer);
    state.uiSaveTimer = setTimeout(() => saveUi(false), 500);
  }
  async function saveUi(explicit = false) {
    try {
      const res = await api("/settings", { method: "PUT", body: JSON.stringify({ ui: collectUi() }) });
      if (res && res.settings) state.settings = res.settings;
      if (explicit) toast("机台设置已保存。");
    } catch (e) { if (explicit) toast("保存失败：" + e.message, "err"); }
  }
  function collectUi() {
    return {
      theme: state.ui.theme, crt: state.ui.crt, performance: !!state.ui.performance,
      splash: !!state.ui.splash, text_fx: $("#setUiTextFx")?.value || state.ui.text_fx,
      fx_loop: !!$("#setUiFxLoop")?.checked, fx_interval: Number($("#setUiFxInterval")?.value) || state.ui.fx_interval,
      quote_enabled: !!$("#setUiQuoteEnabled")?.checked, quote_interval: Number($("#setUiQuoteInterval")?.value) || state.ui.quote_interval,
      quote_kb_ids: selectedQuoteKbs(), quote_custom_lines: lines("#setUiQuoteLines"),
      quote_custom_mottos: lines("#setUiQuoteMottos"),
      wall_rotate: !!$("#setUiWallRotate")?.checked, wall_interval: Number($("#setUiWallInterval")?.value) || state.ui.wall_interval,
      wall_parallax: !!$("#setUiParallax")?.checked
    };
  }
  function lines(sel) { return ($(sel)?.value || "").split("\n").map(s => s.trim()).filter(Boolean); }
  function selectedQuoteKbs() {
    const checked = $$("#setUiQuoteKbs input:checked").map(x => x.value);
    return checked.length === state.kbs.length ? [] : checked;
  }
  function syncUiControls() {
    const set = (id, v) => { const el = $("#" + id); if (el) el.value = v; };
    const chk = (id, v) => { const el = $("#" + id); if (el) el.checked = !!v; };
    set("setUiTheme", state.ui.theme); set("setUiCrt", state.ui.crt);
    chk("setUiPerf", state.ui.performance); chk("setUiSplash", state.ui.splash);
    set("setUiTextFx", state.ui.text_fx); chk("setUiFxLoop", state.ui.fx_loop);
    set("setUiFxInterval", state.ui.fx_interval); chk("setUiQuoteEnabled", state.ui.quote_enabled);
    set("setUiQuoteInterval", state.ui.quote_interval);
    chk("setUiWallRotate", state.ui.wall_rotate); set("setUiWallInterval", state.ui.wall_interval);
    chk("setUiParallax", state.ui.wall_parallax);
    const ql = $("#setUiQuoteLines"); if (ql) ql.value = (state.ui.quote_custom_lines || []).join("\n");
    const qm = $("#setUiQuoteMottos"); if (qm) qm.value = (state.ui.quote_custom_mottos || []).join("\n");
    renderQuoteKbs(); renderBgGrid(); renderWallVault();
  }
  function renderQuoteKbs() {
    const box = $("#setUiQuoteKbs");
    if (!box) return;
    const selected = new Set(state.ui.quote_kb_ids || []);
    const all = selected.size === 0;
    box.innerHTML = state.kbs.length ? state.kbs.map(kb => {
      const on = all || selected.has(kb.kb_id);
      return `<label class="${on ? "is-on" : ""}"><input type="checkbox" value="${esc(kb.kb_id)}" ${on ? "checked" : ""}>${esc(kb.display_name || kb.kb_id)}</label>`;
    }).join("") : `<span class="faint" style="font-size:10px">还没有知识库</span>`;
  }
  function renderBgGrid() {
    const grid = $("#cathodeBgGrid");
    if (!grid) return;
    grid.innerHTML = BG_SLOTS.map(([key, label]) => {
      const img = state.backgrounds[key];
      return `<div class="bg-slot" data-bg="${key}" style="background-image:${img ? "none" : "linear-gradient(135deg,rgba(181,140,255,.13),rgba(0,0,0,.35))"}">${img ? `<img src="${img}" alt="">` : ""}<span>${label}</span><div><button type="button" data-bg-upload="${key}">上传</button><button type="button" data-bg-reset="${key}">清除</button></div><input type="file" accept="image/*" data-bg-file="${key}" hidden></div>`;
    }).join("");
  }
  function renderWallVault() {
    const tabs = $("#wallVaultTabs"), grid = $("#wallVault");
    if (!tabs || !grid) return;
    const themes = Object.keys(ORGS);
    if (zeroIsEnabled() && (zeroData().walls || []).length) themes.push("zero");
    if (!themes.includes(state.wallVaultTheme)) state.wallVaultTheme = themes[0] || "violet";
    tabs.innerHTML = themes.map(t => {
      const org = t === "zero" ? zeroAsOrg() : orgFor(t);
      return `<button type="button" data-wall-tab="${t}" class="${t === state.wallVaultTheme ? "is-on" : ""}">${esc(org?.name || t)}</button>`;
    }).join("");
    const theme = state.wallVaultTheme;
    const walls = wallsFor(theme);
    grid.innerHTML = walls.length ? walls.map((url, i) => {
      const meta = wallMetaFor(theme, url, i);
      const on = wallEnabled(url);
      return `<div class="wall-cell ${on ? "is-on" : "is-off"}" data-wall-toggle="${esc(url)}" title="${esc(meta.title)}"><img src="${esc(url)}" alt="" loading="lazy"><span class="wall-check">✓</span><span class="wall-label">${esc(meta.title)}</span></div>`;
    }).join("") : `<div class="inspect-empty">这一组还没有壁纸。</div>`;
    const enabled = walls.filter(wallEnabled).length;
    const count = $("#wallVaultCount"); if (count) count.textContent = `${enabled} / ${walls.length} ENABLED`;
    const hint = $("#wallVaultHint"); if (hint) hint.textContent = walls.length ? `当前：${orgFor(theme)?.name || theme} · 点击缩略图切换` : "默认全部启用；点缩略图切换。";
  }
  function applyBackgrounds() {
    $$("[data-view]").forEach(view => {
      const key = view.dataset.view;
      const art = view.querySelector(".view-art");
      const img = state.backgrounds[key] || state.orgWallpaper;
      if (art) art.style.backgroundImage = img ? `url("${img}")` : "none";
    });
    const ticker = $("#ticker");
    if (ticker) ticker.style.setProperty("--ticker-art", state.backgrounds.ticker ? `url("${state.backgrounds.ticker}")` : "none");
  }

  /* ---------- wallpaper motion / transitions ---------- */
  function wallFxClass(name) {
    const fx = $("#wallFx"); if (!fx) return;
    const host = document.querySelector(".view.is-active");
    if (host && fx.parentElement !== host) host.appendChild(fx);
    clearTimeout(state.wallFxTimer);
    fx.style.display = "block";
    fx.className = "wall-fx is-on " + name;
    state.wallFxTimer = setTimeout(() => { fx.className = "wall-fx"; fx.style.display = "none"; }, 850);
  }
  function setOrgWallpaper(url, animate = true) {
    if (!url) return;
    state.orgWallpaper = url;
    if (!animate || state.ui.performance || reduced) { applyBackgrounds(); return; }
    const fx = ["fx-scan", "fx-tear", "fx-iris", "fx-shutter"][Math.floor(Math.random() * 4)];
    wallFxClass(fx);
    setTimeout(applyBackgrounds, 190);
    if (state.apprOpen) setTimeout(() => syncApprArt(url), 190);
  }
  function startWallRotation() {
    clearInterval(state.wallTimer);
    const secs = Math.max(5, Number(state.ui.wall_interval) || 18);
    if (!state.ui.wall_rotate) return;
    state.wallTimer = setInterval(() => {
      if (document.hidden || state.apprOpen) return;
      const pool = enabledWallsFor(state.ui.theme);
      if (pool.length < 2) return;
      const next = pool.filter(u => u !== state.orgWallpaper);
      setOrgWallpaper(next[Math.floor(Math.random() * next.length)], true);
    }, secs * 1000);
  }
  function applyParallax() {
    const enabled = !!state.ui.wall_parallax && !state.ui.performance && !reduced;
    const x = enabled ? state.px : 0, y = enabled ? state.py : 0;
    document.documentElement.style.setProperty("--wall-px", x.toFixed(2) + "px");
    document.documentElement.style.setProperty("--wall-py", y.toFixed(2) + "px");
  }
  function initParallax() {
    if (state.parallaxBound) return;
    state.parallaxBound = true;
    let t0 = performance.now();
    const tick = (now) => {
      if (typeof now !== "number") now = performance.now();
      const sec = (now - t0) / 1000;
      state.tx = Math.sin(sec / 11.3) * 16 + Math.sin(sec / 4.7) * 5;
      state.ty = Math.cos(sec / 13.7) * 12 + Math.sin(sec / 6.1) * 4;
      state.px += (state.tx - state.px) * .12;
      state.py += (state.ty - state.py) * .12;
      applyParallax();
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
    const follow = (x, y) => {
      if (!state.ui.wall_parallax) return;
      state.tx = -((x / window.innerWidth) * 2 - 1) * 34;
      state.ty = -((y / window.innerHeight) * 2 - 1) * 24;
    };
    window.addEventListener("mousemove", e => follow(e.clientX, e.clientY), { passive: true });
    window.addEventListener("touchmove", e => { const p = e.touches[0]; if (p) follow(p.clientX, p.clientY); }, { passive: true });
    window.addEventListener("mouseleave", () => { state.tx = 0; state.ty = 0; });
  }

  /* ---------- appreciation / 欣赏模式 ---------- */
  function apprAll() {
    const out = [];
    const seen = new Set();
    const add = (theme, url, i) => {
      if (!url || seen.has(url)) return;
      seen.add(url);
      out.push({ theme, url, meta: wallMetaFor(theme, url, i), org: orgFor(theme) });
    };
    Object.keys(ORGS).forEach(theme => {
      if (!orgEnabled(theme)) return;
      wallsFor(theme).filter(wallEnabled).forEach((url, i) => add(theme, url, i));
    });
    if (zeroIsEnabled()) {
      wallsFor("zero").filter(wallEnabled).forEach((url, i) => add("zero", url, i));
    }
    return out;
  }
  function apprPool() {
    const all = apprAll();
    if (!state.apprTheme || state.apprTheme === "all") return all;
    return all.filter(x => x.theme === state.apprTheme);
  }
  function renderApprThemes() {
    const box = $("#apprThemes"); if (!box) return;
    const themes = [...new Set(apprAll().map(x => x.theme))];
    box.innerHTML = `<button type="button" data-appr-theme="all" class="${state.apprTheme === "all" ? "is-on" : ""}" style="--sw:var(--accent)" title="全部"></button>` +
      themes.map(t => { const org = orgFor(t); return `<button type="button" data-appr-theme="${t}" class="${state.apprTheme === t ? "is-on" : ""}" style="--sw:${org?.color || "#fff"}" title="${esc(org?.name || t)}"></button>`; }).join("");
  }
  function apprShow(animate = true) {
    const pool = apprPool();
    if (!pool.length) return closeAppreciate();
    if (state.apprIndex < 0) state.apprIndex = pool.length - 1;
    state.apprIndex %= pool.length;
    const item = pool[state.apprIndex];
    const set = () => {
      const art = $("#apprArt"); if (art) art.style.backgroundImage = `url("${item.url}")`;
      $("#apprTitle").textContent = item.org?.name || "第零章";
      $("#apprMeta").textContent = `CHAPTER ${item.org?.chapter || "00"} // ${item.org?.en || "LINJI"} · ${String(state.apprIndex + 1).padStart(2, "0")} / ${String(pool.length).padStart(2, "0")}`;
      $("#apprCaptionKicker").textContent = item.meta.title;
      $("#apprCaption").textContent = item.meta.caption || item.org?.motto || "";
      $("#apprIndex").textContent = `${String(state.apprIndex + 1).padStart(2, "0")} / ${String(pool.length).padStart(2, "0")}`;
      document.documentElement.style.setProperty("--appr-accent", item.org?.color || "var(--accent)");
    };
    if (!animate || state.ui.performance || reduced) { set(); return; }
    const fx = $("#apprFx");
    if (fx) {
      fx.className = "appr-fx is-on " + ["fx-scan", "fx-tear", "fx-iris", "fx-shutter"][Math.floor(Math.random() * 4)];
      setTimeout(() => { fx.className = "appr-fx"; }, 850);
    }
    setTimeout(set, 170);
    setTimeout(() => { const art = $("#apprArt"); if (art) art.style.backgroundImage = `url("${item.url}")`; }, 170);
  }
  function apprMove(delta) { state.apprIndex += delta; apprShow(true); }
  function syncApprArt(url) {
    if (!state.apprOpen) return;
    const art = $("#apprArt"); if (art) art.style.backgroundImage = `url("${url}")`;
  }
  function startApprRotation() {
    clearInterval(state.apprTimer);
    if (!state.ui.wall_rotate) return;
    const secs = Math.max(5, Number(state.ui.wall_interval) || 18);
    state.apprTimer = setInterval(() => { if (!document.hidden && state.apprOpen) apprMove(1); }, secs * 1000);
  }
  function openAppreciate(url) {
    const all = apprAll();
    if (!all.length) { toast("还没有启用的壁纸。", "err"); return; }
    state.apprTheme = "all"; state.apprPool = all;
    const found = url ? all.findIndex(x => x.url === url) : all.findIndex(x => x.url === state.orgWallpaper);
    state.apprIndex = found >= 0 ? found : 0;
    state.apprOpen = true;
    if (!state.ui.quote_enabled) { state.apprForcedQuote = true; state.ui.quote_enabled = true; startQuoteTimer(); broadcastQuote(true); }
    document.body.classList.add("appreciating");
    const el = $("#appreciate");
    if (el) { el.hidden = false; el.classList.add("is-open"); el.setAttribute("aria-hidden", "false"); }
    renderApprThemes(); apprShow(false);
    startApprRotation();
    // carry the current subtitle into the appreciation ticker immediately
    const prefix = $("#tickerPrefix")?.textContent || "记录";
    const text = $("#tickerText")?.textContent || "正在接收信号……";
    const source = $("#tickerSource")?.textContent || "ARCHIVE // PENDING";
    const ap = $("#apprPrefix"), at = $("#apprText"), as = $("#apprSource");
    if (ap) ap.textContent = prefix; if (at) at.textContent = text; if (as) as.textContent = source;
  }
  function closeAppreciate() {
    state.apprOpen = false;
    if (state.apprForcedQuote) { state.apprForcedQuote = false; state.ui.quote_enabled = false; stopQuoteTimer(); }
    document.body.classList.remove("appreciating");
    clearInterval(state.apprTimer);
    const el = $("#appreciate");
    if (el) { el.classList.remove("is-open"); el.setAttribute("aria-hidden", "true"); setTimeout(() => { if (!state.apprOpen) el.hidden = true; }, 250); }
  }
  function updateApprTicker(quote) {
    if (!state.apprOpen) return;
    const ap = $("#apprPrefix"), at = $("#apprText"), as = $("#apprSource");
    if (ap) ap.textContent = quote.prefix;
    if (as) as.textContent = quote.source;
    if (at) typeInto(at, quote.text, "apprTickerTimer");
  }

  /* ---------- textfx / boot ---------- */
  function initTextFx() {
    if (!window.KBTextFX) return;
    window.KBTextFX.init();
    applyUi();
    window.KBTextFX.replayView(state.view);
    replayBrand();
  }
  function replayText() {
    if (window.KBTextFX) window.KBTextFX.replayView(state.view || "catalog");
  }
  function replayBrand() {
    if (!window.KBTextFX) return;
    window.KBTextFX.play($("#railKb"), "mix");
    window.KBTextFX.play($("#railTitle"), "mix");
    window.KBTextFX.play($("#railAlias"), "mix");
  }
  const BRAND_MODES = ["brand-breathe", "brand-sheen", "brand-orbit", "brand-flicker", "brand-sweep"];
  let brandTimer = null;
  function cycleBrandMode() {
    const row = $("#railBrandRow");
    if (!row) return;
    row.classList.remove(...BRAND_MODES);
    if (state.ui.performance || reduced) return;
    row.classList.add(BRAND_MODES[Math.floor(Math.random() * BRAND_MODES.length)]);
  }
  function scheduleBrandMode() {
    clearTimeout(brandTimer);
    const row = $("#railBrandRow");
    if (!row || state.ui.performance || reduced) {
      row?.classList.remove(...BRAND_MODES);
      return;
    }
    if (!BRAND_MODES.some(c => row.classList.contains(c))) cycleBrandMode();
    brandTimer = setTimeout(() => {
      cycleBrandMode();
      scheduleBrandMode();
    }, 4200 + Math.random() * 4200);
  }
  function startBrandModes() {
    cycleBrandMode();
    scheduleBrandMode();
  }
  function prepareBoot() {
    const boot = $("#boot");
    if (!boot) return;
    const org = orgFor(state.ui.theme);
    const badge = badgeFor(state.ui.theme, true);
    const img = $("#bootBadge");
    if (img) { img.src = badge || ""; img.hidden = !badge; }
    const label = $("#bootOrg");
    if (label && org) label.textContent = `CHAPTER ${org.chapter} // ${org.name}`;
    boot.classList.remove("boot-v1", "boot-v2", "boot-v3", "boot-v4");
    boot.classList.add(`boot-v${1 + Math.floor(Math.random() * 4)}`);
    // per-line signal theatre: logo / kicker / org / sub each get a different random effect
    const logoFx = ["radio", "pulse", "type", "mix"][Math.floor(Math.random() * 4)];
    boot.dataset.logoFx = logoFx;
    const logo = $("#bootLogo");
    if (logo) logo.innerHTML = [..."KIRAKB"].map((c, i) => `<span class="char${i > 3 ? " kb" : ""}" style="--i:${i}">${c}</span>`).join("");
    const pick = (arr) => arr[Math.floor(Math.random() * arr.length)];
    const setFx = (el, fx, delay = 0) => {
      if (!el) return;
      el.classList.remove("fx-radio", "fx-pulse", "fx-type");
      el.classList.add("fx-" + fx);
      el.style.setProperty("--fx-delay", delay + "ms");
    };
    setFx($("#bootKicker"), pick(["type", "radio"]), 60);
    setFx(label, pick(["pulse", "radio", "type"]), 180);
    setFx($("#bootSub"), pick(["radio", "type", "pulse"]), 260);
    const status = $("#bootStatus"); if (status) status.textContent = "ACQUIRING SIGNAL";
    const pct = $("#bootPct"); if (pct) pct.textContent = "00";
  }

  function runBoot() {
    const boot = $("#boot");
    if (!boot) return;
    const skip = $("#bootSkip");
    if (!state.ui.splash || reduced || new URLSearchParams(location.search).has("skipboot")) {
      boot.classList.add("is-off"); document.body.classList.remove("boot-hold"); return;
    }
    document.body.classList.add("boot-hold");
    const pct = $("#bootPct"), status = $("#bootStatus");
    const stages = ["ACQUIRING SIGNAL", "VERIFYING HASH", "MOUNTING ARCHIVE", "SIGNAL LOCKED"];
    let v = 0;
    boot.timer = setInterval(() => {
      v = Math.min(100, v + Math.max(1, Math.round((100 - v) / 10)));
      pct.textContent = String(v).padStart(2, "0");
      status.textContent = stages[Math.min(3, Math.floor(v / 28))];
      if (v >= 100) { clearInterval(boot.timer); setTimeout(finish, 350); }
    }, 95);
    skip.onclick = finish;
    function finish() {
      clearInterval(boot.timer);
      boot.classList.add("is-off"); document.body.classList.remove("boot-hold");
      setTimeout(() => { boot.style.display = "none"; }, 600);
      replayText(); replayBrand(); startQuoteTimer();
    }
  }

  /* ---------- data ---------- */
  async function loadKbs() {
    const kbs = await api("/kbs");
    state.kbs = Array.isArray(kbs) ? kbs : [];
    if (!state.current || !state.kbs.some(k => k.kb_id === state.current)) state.current = state.kbs[0]?.kb_id || null;
  }
  async function loadDocs() {
    if (!state.current) { state.docs = []; return; }
    const docs = await api(`/kbs/${encodeURIComponent(state.current)}/documents`);
    state.docs = Array.isArray(docs) ? docs : [];
    state.selected = new Set([...state.selected].filter(id => state.docs.some(d => d.doc_id === id)));
    if (state.selectedDoc && !state.docs.some(d => d.doc_id === state.selectedDoc)) state.selectedDoc = null;
  }
  async function loadVersions() {
    if (!state.current) { state.versions = []; return; }
    const versions = await api(`/kbs/${encodeURIComponent(state.current)}/versions`);
    state.versions = Array.isArray(versions) ? versions : [];
  }
  async function loadTasks() {
    try { const tasks = await api("/tasks"); state.tasks = Array.isArray(tasks) ? tasks : []; }
    catch (_) { state.tasks = []; }
  }
  async function loadGlobalRecall() {
    try { state.globalRecall = await api("/recall/global") || {}; }
    catch (_) { state.globalRecall = {}; }
  }
  async function loadSettings() {
    try { state.settings = await api("/settings"); }
    catch (_) { state.settings = null; }
  }
  async function refreshAll(render = true) {
    await loadKbs();
    await Promise.all([loadDocs(), loadVersions(), loadTasks(), loadGlobalRecall()]);
    if (render) renderAll();
  }

  /* ---------- render ---------- */
  function saveZero(patch, rerender = false) {
    state.zero = { ...zeroData(), ...patch, custom: true };
    try { localStorage.setItem("linji-zero-v1", JSON.stringify(state.zero)); } catch (_) {}
    if (rerender) { renderZeroPanel(); renderThemeStrip(); renderThemeBadges(); renderWiki(); renderWallVault(); if (zeroActiveFor(state.ui.theme)) applyOrgIdentity(); }
  }
  function renderZeroPanel() {
    const box = $("#zeroForm");
    if (!box) return;
    const z = zeroData();
    const targets = ["violet", "pink", "blue", "amber", "sg", "green", "red", "gray", "random"];
    const tset = new Set(Array.isArray(z.targets) ? z.targets : []);
    const field = (key, label, type = "text", span = false) => {
      const val = z[key] == null ? "" : z[key];
      if (type === "textarea") return `<div class="field ${span ? "span-2" : ""}"><label>${label}</label><textarea data-zero-field="${key}">${esc(val)}</textarea></div>`;
      return `<div class="field ${span ? "span-2" : ""}"><label>${label}</label><input data-zero-field="${key}" type="${type}" value="${esc(val)}"></div>`;
    };
    box.innerHTML = `
      <div class="field span-2 zero-preview"><img src="${z.badge || ""}" alt=""><div><b>${esc(z.name || "第零章")}</b><small>${esc(z.en || "CHAPTER ZERO")} · ${(z.walls || []).length} 张插画</small></div><label class="switch"><input id="zeroEnabled" type="checkbox" ${z.enabled ? "checked" : ""}><span></span><b>启用第零章</b></label></div>
      <div class="field span-2"><label>代替哪些配色 <small>选中的主题会显示第零章的徽章、文字与插画；“随机色”表示随机时有机会出现</small></label><div class="zero-targets">${targets.map(t => `<label class="${tset.has(t) ? "is-on" : ""}"><input type="checkbox" data-zero-target="${t}" ${tset.has(t) ? "checked" : ""}>${t === "random" ? "随机色" : esc(ORGS[t]?.name || t)}</label>`).join("")}</div></div>
      ${field("name", "名称")}
      ${field("en", "英文副题")}
      ${field("motto", "格言", "text", true)}
      ${field("summary", "摘要", "textarea", true)}
      ${field("lore", "正文 / 故事", "textarea", true)}
      ${field("doctrine", "信条")}
      ${field("uniform", "制服 / 外观")}
      ${field("relations", "关系")}
      ${field("quote", "台词")}
      ${field("tags", "标签 <small>逗号分隔</small>")}
      <div class="field span-2"><label>素材</label><div class="zero-upload-row"><label class="btn ghost zero-upload">上传徽章<input id="zeroBadgeFile" type="file" accept="image/*" hidden></label><label class="btn ghost zero-upload">上传插画（可多选）<input id="zeroWallsFile" type="file" accept="image/*" multiple hidden></label><button class="btn small ghost" id="zeroClearWalls" type="button">清空插画</button></div><div class="zero-thumbs">${z.badge ? `<img class="zero-badge-thumb" src="${z.badge}" alt="">` : ""}${(z.walls || []).map((w, i) => `<img src="${esc(w)}" alt="" title="插画 ${i + 1}">`).join("")}</div></div>
      <div class="field span-2"><span class="hint">第零章只保存在本机（localStorage），不会上传服务器。启用后可在主题、壁纸库和欣赏模式中看到。</span></div>`;
    const st = $("#zeroState"); if (st) st.textContent = z.enabled ? "IN USE" : "UNWRITTEN";
    const tset2 = new Set(zeroTargets());
    $$("#zeroForm .zero-targets label").forEach(l => l.classList.toggle("is-on", tset2.has(l.querySelector("input")?.dataset.zeroTarget)));
  }
  function renderWiki() {
    const grid = $("#orgGrid");
    if (!grid) return;
    renderZeroPanel();
    const tl = $("#wikiTimeline");
    if (tl && window.LINJI_STORY) tl.innerHTML = window.LINJI_STORY.map(s => `<div><b>${esc(s.year)}</b><p>${esc(s.text)}</p></div>`).join("");
    const themes = Object.keys(ORGS);
    grid.innerHTML = themes.map(theme => {
      const org = orgFor(theme);
      const badge = badgeFor(theme, false);
      const walls = wallsFor(theme);
      const enabled = orgEnabled(theme);
      return `<article class="org-card" id="org-${theme}" style="--org:${org.color}">
        <div class="org-top">
          <img class="org-badge" src="${badge}" alt="${esc(org.name)}">
          <div class="org-title">
            <div class="org-chapter">CHAPTER ${org.chapter} // ${esc(org.en)}</div>
            <h3>${esc(org.name)}</h3>
            <small>${esc(org.motto)}</small>
          </div>
          <span class="badge" style="margin-left:auto;color:${org.color};border-color:${org.color}">${enabled ? "IN USE" : "MUTED"}</span>
        </div>
        <p class="org-motto">「${esc(org.quote)}」</p>
        <p class="org-summary">${esc(org.summary)}</p>
        <div class="org-meta">
          <span>DOCTRINE / 信条<b>${esc(org.doctrine)}</b></span>
          <span>UNIFORM / 制服<b>${esc(org.uniform)}</b></span>
          <span>RELATIONS / 关系<b>${esc(org.relations)}</b></span>
          <span>TAGS / 标签<b>${org.tags.map(esc).join(" · ")}</b></span>
        </div>
        <details class="org-details"><summary>展开档案全文 / LORE</summary><p class="org-lore">${esc(org.lore)}</p>${org.extraLore ? `<p class="org-lore">${esc(org.extraLore)}</p>` : ""}${org.chronicle?.length ? `<div class="org-chronicle">${org.chronicle.map((x, i) => `<p><b>${String(i + 1).padStart(2, "0")}</b>${esc(x)}</p>`).join("")}</div>` : ""}${org.cases?.length ? `<div class="org-cases"><span class="mini-label">CASE FILES / 未结案</span>${org.cases.map(x => `<p>${esc(x)}</p>`).join("")}</div>` : ""}</details>
        <div class="org-gallery">${walls.map((wall, index) => { const meta = wallMetaFor(theme, wall, index); return `<button type="button" data-org-wall="${theme}" data-org-wall-index="${index}" title="${esc(meta.title)} — ${esc(meta.caption)}"><img src="${wall}" alt="" loading="lazy"><span>${esc(meta.title)}</span></button>`; }).join("")}</div>
        <div class="org-actions">
          <button type="button" class="btn" data-org-theme="${theme}">应用此主题</button>
          <label class="switch" title="关闭后不使用该组织的徽章与壁纸"><input type="checkbox" data-org-enabled="${theme}" ${enabled ? "checked" : ""}><span></span><b>启用组织外观</b></label>
          <label class="btn ghost org-upload">上传徽章<input type="file" accept="image/*" data-org-badge="${theme}"></label>
          <label class="btn ghost org-upload">上传壁纸<input type="file" accept="image/*" multiple data-org-walls="${theme}"></label>
        </div>
        <div class="org-note">${walls.length} 张壁纸 · 十九个场景 · 徽章与壁纸存于本机，不上传服务器</div>
      </article>`;
    }).join("");
  }

  function renderAll() {
    renderKbSelect();
    renderCatalog();
    renderVersions();
    renderAccess();
    renderRecall();
    renderSettings();
    renderTasks();
    renderWiki();
    renderThemeStrip();
    renderThemeBadges();
    syncUiControls();
    applyBackgrounds();
    replayText();
  }
  function renderKbSelect() {
    const sel = $("#kbSelect");
    if (!sel) return;
    sel.innerHTML = state.kbs.length
      ? state.kbs.map(k => `<option value="${esc(k.kb_id)}">${esc(k.display_name || k.kb_id)} · ${k.entry_count || 0}</option>`).join("")
      : `<option value="">还没有知识库</option>`;
    if (state.current) sel.value = state.current;
    const count = $("#railKbCount"); if (count) count.textContent = state.kbs.length;
  }
  function renderCatalog() {
    const kb = currentKb();
    const ver = activeVersion();
    $("#statEntries").textContent = String(state.docs.length).padStart(3, "0");
    $("#statVersions").textContent = String(state.versions.length).padStart(2, "0");
    $("#statDimension").textContent = ver ? ver.dimension : "—";
    const recall = state.globalRecall?.enabled && kb?.recall?.enabled;
    $("#statRecall").textContent = recall ? "ON" : "OFF";
    $("#activeBadgeText").textContent = ver ? `${ver.model_name} · ${ver.dimension}D` : "NO ACTIVE VERSION";
    $("#docCount").textContent = `${state.docs.length} RECORDS`;
    $("#docFoot").textContent = `${filteredDocs().length} RECORDS`;
    renderDocList();
    renderInspector();
  }
  function filteredDocs() {
    const q = state.docFilter.trim().toLowerCase();
    return state.docs.filter(d => {
      if (q && !(d.name || "").toLowerCase().includes(q)) return false;
      if (state.docFilterKind === "ai" && d.permissions?.edit !== "admin") return false;
      if (state.docFilterKind === "frozen" && d.permissions?.edit !== "off" && d.permissions?.delete !== "off") return false;
      if (state.docFilterKind === "recall" && d.recall?.enabled !== true) return false;
      return !d.deleted;
    });
  }
  function renderDocList() {
    const list = filteredDocs();
    $("#docList").innerHTML = list.length ? list.map((d, i) => `
      <div class="entry-row ${d.doc_id === state.selectedDoc ? "is-selected" : ""}" data-doc="${esc(d.doc_id)}" tabindex="0">
        <span class="entry-no"><input class="doc-check" type="checkbox" data-check="${esc(d.doc_id)}" ${state.selected.has(d.doc_id) ? "checked" : ""}>${String(i + 1).padStart(2, "0")}</span>
        <span class="entry-main"><b>${esc(d.name)}</b><small>${fmtBytes(d.size)} · ${fmtTime(d.updated_at)}</small></span>
        <span class="entry-kb">${esc(currentKb()?.display_name || state.current || "—")}</span>
        <span class="entry-status ${d.recall?.enabled === true ? "is-recall" : ""}"><i></i>${d.recall?.enabled === true ? "召回" : "已索引"}</span>
        <span class="entry-time">${esc((d.updated_at ? fmtTime(d.updated_at) : "—").slice(5, 16))}</span>
      </div>`).join("") : `<div class="inspect-empty">没有匹配的记录。上传文件，或先创建一个版本。</div>`;
    $("#selectAllDocs").checked = list.length > 0 && list.every(d => state.selected.has(d.doc_id));
  }
  async function selectDoc(id) {
    state.selectedDoc = id;
    renderDocList();
    const d = state.docs.find(x => x.doc_id === id);
    if (!d) { renderInspector(); return; }
    $("#inspectEmpty").hidden = true;
    $("#inspectBody").hidden = false;
    $("#inspectBadge").textContent = "ARCHIVED";
    $("#inspectBadge").className = "badge ok";
    $("#inspectKicker").textContent = `RECORD ${String(id).replace(/\D/g, "").padStart(3, "0")} // ${currentKb()?.display_name || "ARCHIVE"}`;
    if (window.KBTextFX) window.KBTextFX.playText($("#inspectTitle"), d.name || id, "mix");
    else $("#inspectTitle").textContent = d.name || id;
    $("#inspectTags").innerHTML = (d.recall?.enabled ? ["召回"] : []).concat(d.permissions?.edit ? ["权限覆盖"] : []).map(t => `<span>${esc(t)}</span>`).join("");
    $("#inspectContent").textContent = "正在读取正文……";
    $("#inspectKb").textContent = currentKb()?.display_name || state.current || "—";
    $("#inspectUpdated").textContent = fmtTime(d.updated_at);
    $("#inspectHash").textContent = (d.content_hash || "").slice(0, 18) || "—";
    $("#inspectStatus").textContent = d.recall?.enabled === true ? "召回开启" : "已索引";
    try {
      const full = await api(`/kbs/${encodeURIComponent(state.current)}/documents/${encodeURIComponent(id)}`);
      if (state.selectedDoc === id) $("#inspectContent").textContent = full.content || "（空正文）";
    } catch (e) { $("#inspectContent").textContent = "读取失败：" + e.message; }
  }
  function renderInspector() {
    if (!state.selectedDoc) {
      $("#inspectEmpty").hidden = false; $("#inspectBody").hidden = true;
      $("#inspectBadge").textContent = "NO SELECTION"; $("#inspectBadge").className = "badge";
      return;
    }
    $("#inspectEmpty").hidden = true; $("#inspectBody").hidden = false;
  }
  function renderVersions() {
    $("#versionGrid").innerHTML = state.versions.length ? state.versions.map((v, i) => `
      <article class="version-card ${v.is_active ? "is-active" : ""}">
        <div class="version-top"><div><div class="version-model">${esc(v.model_name)}</div><div class="version-dim">${v.dimension} DIMENSIONS · ${v.status === "ready" ? "完整" : "不完整"}</div></div><span class="badge ${v.is_active ? "ok" : ""}">${v.is_active ? "使用中" : "待用"}</span></div>
        <div class="version-date">${fmtTime(v.created_at)}</div>
        <div class="version-counts"><div><span>OUTDATED</span><b>${v.outdated || 0}</b></div><div><span>STATUS</span><b>${v.status === "ready" ? "OK" : "—"}</b></div></div>
        <div class="version-actions">
          ${v.status === "ready" ? `<button class="btn small ghost" type="button" data-activate="${esc(v.version_id)}">${icon("check", 12)}激活</button>` : ""}
          ${v.status === "ready" ? `<button class="btn small ghost" type="button" data-sync="${esc(v.version_id)}">${icon("refresh", 12)}同步</button>` : ""}
          ${!v.is_active || v.status !== "ready" ? `<button class="btn small danger" type="button" data-delver="${esc(v.version_id)}">${icon("trash", 12)}</button>` : ""}
        </div>
      </article>`).join("") : `<div class="inspect-empty" style="grid-column:1/-1">还没有索引。创建一个版本后才能检索。</div>`;
  }
  function renderAccess() {
    const kb = currentKb();
    const perms = kb?.permissions || {};
    $("#policyAdd").value = perms.add || "open";
    $("#policyEdit").value = perms.edit || "open";
    $("#policyDelete").value = perms.delete || "whitelist";
    $("#kbAllowUsers").value = (perms.allow_users || []).join("\n");
    const list = state.docs;
    $("#overrideCount").textContent = `${list.length} ENTRIES`;
    $("#entryPolicyList").innerHTML = list.length ? list.map(d => `<div class="entry-policy"><b title="${esc(d.name)}">${esc(d.name)}</b><select class="policy-select" data-entry-edit="${esc(d.doc_id)}">${["inherit", "open", "whitelist", "admin", "off"].map(v => `<option value="${v}" ${((d.permissions?.edit) || "inherit") === v ? "selected" : ""}>${v === "inherit" ? "继承" : v}</option>`).join("")}</select><select class="policy-select" data-entry-delete="${esc(d.doc_id)}">${["inherit", "open", "whitelist", "admin", "off"].map(v => `<option value="${v}" ${((d.permissions?.delete) || "inherit") === v ? "selected" : ""}>${v === "inherit" ? "继承" : v}</option>`).join("")}</select></div>`).join("") : `<div class="inspect-empty">没有条目。</div>`;
  }
  function renderRecall() {
    const kb = currentKb();
    const recall = kb?.recall || {};
    $("#globalRecallBadge").classList.toggle("is-on", !!state.globalRecall?.enabled);
    $("#globalRecallBadge").querySelector("small").textContent = state.globalRecall?.enabled ? "ON" : "OFF";
    $("#globalRecallInfo").innerHTML = `<p><b>MODE</b> ${esc(state.globalRecall?.mode || "—")}</p><p><b>TOP K</b> ${state.globalRecall?.top_k ?? "—"}</p><p><b>MIN SCORE</b> ${state.globalRecall?.min_score ?? "—"}</p><p><b>MAX CHARS</b> ${state.globalRecall?.max_chars ?? "—"}</p>`;
    $("#kbRecallEnabled").checked = !!recall.enabled;
    $("#kbRecallTopK").value = recall.top_k ?? "";
    $("#kbRecallMinScore").value = recall.min_score ?? "";
    $("#kbRecallMaxChars").value = recall.max_chars ?? "";
    $("#entryRecallList").innerHTML = state.docs.length ? state.docs.map(d => {
      const r = d.recall || {};
      return `<div class="entry-recall"><b title="${esc(d.name)}">${esc(d.name)}</b><select data-entry-recall="${esc(d.doc_id)}"><option value="inherit" ${r.enabled == null ? "selected" : ""}>继承</option><option value="on" ${r.enabled === true ? "selected" : ""}>强制</option><option value="off" ${r.enabled === false ? "selected" : ""}>排除</option></select><input data-entry-keywords="${esc(d.doc_id)}" value="${esc((r.keywords || []).join(", "))}" placeholder="关键词"><button class="btn small ghost" type="button" data-save-recall="${esc(d.doc_id)}">保存</button></div>`;
    }).join("") : `<div class="inspect-empty">没有条目。</div>`;
  }
  function renderTasks() {
    $("#taskList").innerHTML = state.tasks.length ? state.tasks.slice().reverse().map(t => `<div class="task"><div class="task-head"><span class="name">${esc(t.description)}</span><span class="st ${esc(t.status)}">${({ running: "进行中", pending: "等待中", completed: "已完成", failed: "失败" }[t.status] || t.status)}</span></div><div class="progress"><i style="width:${Number(t.progress) || 0}%"></i></div><div class="msg">${esc(t.message || "")}${t.progress ? " · " + t.progress + "%" : ""}</div></div>`).join("") : `<div class="empty">还没有后台任务。</div>`;
  }
  function renderSettings() {
    const cfg = state.settings;
    if (!cfg) return;
    const val = (id, v) => { const el = $("#" + id); if (el && v != null) el.value = v; };
    const chk = (id, v) => { const el = $("#" + id); if (el) el.checked = !!v; };
    const s = cfg.search || {}; const e = cfg.entries || {}; const p = cfg.permission || {}; const r = cfg.recall || {}; const st = cfg.storage || {}; const w = cfg.webui || {};
    val("setTopK", s.default_top_k); val("setChunkSize", s.chunk_size); val("setChunkOverlap", s.chunk_overlap);
    chk("setHybrid", s.enable_hybrid_search); chk("setStopwords", s.enable_stopwords); chk("setRerank", s.enable_rerank); chk("setLexical", s.enable_lexical_search); chk("setIgnoreTime", s.ignore_time); chk("setAutoInstallDeps", s.auto_install_dependencies); val("setLexicalMin", s.lexical_min_score);
    val("setTitleMax", e.entry_title_max_chars); val("setHistoryKeep", e.entry_history_keep); chk("setExtraTools", e.extra_tools_enabled); chk("setAutoSync", e.auto_sync_versions);
    const fill = (id, v) => { const el = $("#" + id); if (!el) return; el.innerHTML = ["open", "whitelist", "admin", "off"].map(x => `<option value="${x}" ${x === v ? "selected" : ""}>${{ open: "开放", whitelist: "名单", admin: "仅人工", off: "冻结" }[x]}</option>`).join(""); };
    fill("setAddPolicy", p.default_add_policy || "open"); fill("setEditPolicy", p.default_edit_policy || "open"); fill("setDeletePolicy", p.default_delete_policy || "whitelist");
    $("#setOwnerWhitelist").value = (p.owner_whitelist || []).join("\n");
    chk("setRecallEnabled", r.enabled); val("setRecallMode", r.mode || "hybrid"); val("setRecallTopK", r.top_k); val("setRecallMin", r.min_score); val("setRecallStrong", r.strong_score); val("setRecallChars", r.max_chars); val("setRecallTimeout", r.timeout_ms); val("setRecallDedupe", r.dedupe_turns); val("setRecallMaxKbs", r.max_kbs); val("setRecallMinQuery", r.min_query_chars);
    val("setKbDir", st.knowledge_base_dir); chk("setEnableWebui", w.enable_webui); val("setWebHost", w.webui_host); val("setWebPort", w.webui_port); val("setWebToken", w.webui_token);
  }
  function collectFunctionalSettings() {
    const num = id => { const el = $("#" + id); return el && el.value !== "" ? Number(el.value) : null; };
    return {
      search: {
        enable_hybrid_search: $("#setHybrid").checked, default_top_k: num("setTopK"), chunk_size: num("setChunkSize"), chunk_overlap: num("setChunkOverlap"),
        enable_stopwords: $("#setStopwords").checked, enable_rerank: $("#setRerank").checked, enable_lexical_search: $("#setLexical").checked,
        ignore_time: $("#setIgnoreTime").checked, auto_install_dependencies: $("#setAutoInstallDeps").checked, lexical_min_score: num("setLexicalMin")
      },
      entries: { entry_title_max_chars: num("setTitleMax"), entry_history_keep: num("setHistoryKeep"), extra_tools_enabled: $("#setExtraTools").checked, auto_sync_versions: $("#setAutoSync").checked },
      permission: { owner_whitelist: lines("#setOwnerWhitelist"), default_add_policy: $("#setAddPolicy").value, default_edit_policy: $("#setEditPolicy").value, default_delete_policy: $("#setDeletePolicy").value },
      recall: { recall_enabled: $("#setRecallEnabled").checked, recall_mode: $("#setRecallMode").value, recall_top_k: num("setRecallTopK"), recall_min_score: num("setRecallMin"), recall_strong_score: num("setRecallStrong"), recall_max_chars: num("setRecallChars"), recall_timeout_ms: num("setRecallTimeout"), recall_dedupe_turns: num("setRecallDedupe"), recall_max_kbs: num("setRecallMaxKbs"), recall_min_query_chars: num("setRecallMinQuery") },
      storage: { knowledge_base_dir: $("#setKbDir").value.trim() },
      webui: { enable_webui: $("#setEnableWebui").checked, webui_host: $("#setWebHost").value.trim(), webui_port: num("setWebPort"), webui_token: $("#setWebToken").value }
    };
  }

  /* ---------- actions: KB / docs ---------- */
  async function selectKb(id) {
    if (!id || id === state.current) return;
    state.current = id; state.selectedDoc = null; state.selected.clear();
    try { await Promise.all([loadDocs(), loadVersions()]); renderAll(); }
    catch (e) { toast("加载知识库失败：" + e.message, "err"); }
  }
  function openKbCreate() {
    openModal({
      title: "新建知识库", subtitle: "ID 只能包含字母、数字、下划线和连字符。", small: true,
      body: `<div class="field"><label for="newKbId">知识库 ID</label><input id="newKbId" placeholder="archive_01"></div>`,
      footer: `<button class="btn ghost" type="button" data-cancel>取消</button><button class="btn primary" type="button" data-ok>创建</button>`,
      onMount(root, close) {
        $("[data-cancel]", root).onclick = close;
        $("[data-ok]", root).onclick = async () => {
          const id = $("#newKbId").value.trim(); if (!id) return toast("请填写 ID", "err");
          try { await api("/kbs", { method: "POST", body: JSON.stringify({ kb_id: id }) }); close(); state.current = id; await refreshAll(); toast("知识库已创建。"); }
          catch (e) { toast("创建失败：" + e.message, "err"); }
        };
      }
    });
  }
  function openKbEdit() {
    const kb = currentKb(); if (!kb) return toast("请先创建知识库", "err");
    openModal({
      title: "编辑知识库", subtitle: kb.kb_id, small: true,
      body: `<div class="field"><label for="kbNameEdit">显示名</label><input id="kbNameEdit" value="${esc(kb.display_name || "")}"></div><div class="field"><label for="kbDescEdit">描述</label><textarea id="kbDescEdit" rows="3">${esc(kb.description || "")}</textarea></div>`,
      footer: `<button class="btn ghost" type="button" data-cancel>取消</button><button class="btn primary" type="button" data-ok>保存</button>`,
      onMount(root, close) {
        $("[data-cancel]", root).onclick = close;
        $("[data-ok]", root).onclick = async () => {
          try { await api(`/kbs/${encodeURIComponent(kb.kb_id)}/info`, { method: "PUT", body: JSON.stringify({ display_name: $("#kbNameEdit").value.trim() || kb.kb_id, description: $("#kbDescEdit").value.trim() }) }); close(); await refreshAll(); toast("已保存。"); }
          catch (e) { toast("保存失败：" + e.message, "err"); }
        };
      }
    });
  }
  async function deleteKb() {
    const kb = currentKb(); if (!kb) return;
    const ok = await confirmDialog("删除知识库", `删除「${kb.display_name || kb.kb_id}」以及全部条目和版本？此操作不可恢复。`, "删除", true);
    if (!ok) return;
    try { await api(`/kbs/${encodeURIComponent(kb.kb_id)}`, { method: "DELETE" }); state.current = null; state.selectedDoc = null; await refreshAll(); toast("知识库已删除。"); }
    catch (e) { toast("删除失败：" + e.message, "err"); }
  }
  function triggerUpload() { $("#fileInput").click(); }
  async function uploadFiles(files) {
    if (!state.current) return toast("请先创建知识库", "err");
    if (!files?.length) return;
    let ok = 0;
    for (const file of files) {
      const fd = new FormData(); fd.append("file", file);
      try { await apiUpload(`/kbs/${encodeURIComponent(state.current)}/documents`, fd); ok++; }
      catch (e) { toast(`上传 ${file.name} 失败：${e.message}`, "err"); }
    }
    $("#fileInput").value = "";
    if (ok) toast(`已上传 ${ok} 个文件。`);
    await refreshAll();
  }
  function openDocEditor(doc) {
    if (!doc) return;
    openModal({
      title: doc.name || "编辑条目", subtitle: `${fmtBytes(doc.size)} · ${fmtTime(doc.updated_at)}`, body: `<div class="field"><label for="docNameEdit">标题</label><input id="docNameEdit" value="${esc(doc.name || "")}"></div><div class="field"><label for="docContentEdit">正文</label><textarea id="docContentEdit" class="doc-text" rows="14">正在读取……</textarea></div>`,
      footer: `<button class="btn ghost" type="button" data-cancel>取消</button><button class="btn primary" type="button" data-ok>保存并重建</button>`,
      onMount(root, close) {
        $("[data-cancel]", root).onclick = close;
        api(`/kbs/${encodeURIComponent(state.current)}/documents/${encodeURIComponent(doc.doc_id)}`).then(full => { $("#docContentEdit").value = full.content || ""; }).catch(e => toast("读取失败：" + e.message, "err"));
        $("[data-ok]", root).onclick = async () => {
          const name = $("#docNameEdit").value.trim(); if (!name) return toast("标题不能为空", "err");
          try { await api(`/kbs/${encodeURIComponent(state.current)}/documents/${encodeURIComponent(doc.doc_id)}`, { method: "PUT", body: JSON.stringify({ content: $("#docContentEdit").value, name }) }); close(); await refreshAll(); toast("条目已保存。"); }
          catch (e) { toast("保存失败：" + e.message, "err"); }
        };
      }
    });
  }
  async function deleteDoc(id) {
    const doc = state.docs.find(d => d.doc_id === id); if (!doc) return;
    const ok = await confirmDialog("删除条目", `软删除「${doc.name}」？之后可以恢复。`, "删除", true);
    if (!ok) return;
    try { await api(`/kbs/${encodeURIComponent(state.current)}/documents/${encodeURIComponent(id)}`, { method: "DELETE" }); state.selectedDoc = null; await refreshAll(); toast("条目已删除。"); }
    catch (e) { toast("删除失败：" + e.message, "err"); }
  }
  function openEntrySettings(doc) {
    if (!doc) return;
    const perms = doc.permissions || {}, recall = doc.recall || {};
    openModal({
      title: doc.name || "条目设置", subtitle: "条目级权限与召回覆盖", body: `<div class="field"><label for="entryEditPolicy">编辑策略</label><select id="entryEditPolicy">${["inherit", "open", "whitelist", "admin", "off"].map(v => `<option value="${v}" ${((perms.edit) || "inherit") === v ? "selected" : ""}>${v === "inherit" ? "继承" : v}</option>`).join("")}</select></div><div class="field"><label for="entryDeletePolicy">删除策略</label><select id="entryDeletePolicy">${["inherit", "open", "whitelist", "admin", "off"].map(v => `<option value="${v}" ${((perms.delete) || "inherit") === v ? "selected" : ""}>${v === "inherit" ? "继承" : v}</option>`).join("")}</select></div><div class="field"><label for="entryAllowUsers">允许名单</label><textarea id="entryAllowUsers" rows="2">${esc((perms.allow_users || []).join("\n"))}</textarea></div><div class="field"><label for="entryRecallMode">召回</label><select id="entryRecallMode"><option value="inherit" ${recall.enabled == null ? "selected" : ""}>继承</option><option value="on" ${recall.enabled === true ? "selected" : ""}>强制开启</option><option value="off" ${recall.enabled === false ? "selected" : ""}>排除</option></select></div><div class="field"><label for="entryKeywords">召回关键词</label><input id="entryKeywords" value="${esc((recall.keywords || []).join(", "))}"></div>`,
      footer: `<button class="btn ghost" type="button" data-cancel>取消</button><button class="btn primary" type="button" data-ok>保存</button>`,
      onMount(root, close) {
        $("[data-cancel]", root).onclick = close;
        $("[data-ok]", root).onclick = async () => {
          try {
            await api(`/kbs/${encodeURIComponent(state.current)}/documents/${encodeURIComponent(doc.doc_id)}/permissions`, { method: "PUT", body: JSON.stringify({ edit: $("#entryEditPolicy").value, delete: $("#entryDeletePolicy").value, allow_users: lines("#entryAllowUsers") }) });
            await api(`/kbs/${encodeURIComponent(state.current)}/documents/${encodeURIComponent(doc.doc_id)}/recall`, { method: "PUT", body: JSON.stringify({ enabled: $("#entryRecallMode").value === "on" ? true : $("#entryRecallMode").value === "off" ? false : null, keywords: $("#entryKeywords").value.split(/[,，]/).map(s => s.trim()).filter(Boolean) }) });
            close(); await refreshAll(); toast("条目设置已保存。");
          } catch (e) { toast("保存失败：" + e.message, "err"); }
        };
      }
    });
  }
  async function openHistory(doc) {
    if (!doc) return;
    let data;
    try { data = await api(`/kbs/${encodeURIComponent(state.current)}/documents/${encodeURIComponent(doc.doc_id)}/history`); }
    catch (e) { return toast("加载历史失败：" + e.message, "err"); }
    const revs = data.revisions || [];
    openModal({
      title: doc.name || "修订历史", subtitle: `${revs.length} 个版本 · 当前 ${data.keep || 5} 份过程版本`, body: revs.length ? revs.map(r => `<div class="history-item"><b>${esc(r.rev_id)}</b><span>${esc(r.actor || "system")} · ${fmtTime(r.created_at)} · ${fmtBytes(r.size)}</span>${r.is_current ? `<span class="badge ok">当前</span>` : `<button class="btn small ghost" type="button" data-rev="${esc(r.rev_id)}">回退</button>`}</div>`).join("") : `<div class="inspect-empty">还没有修订记录。</div>`,
      footer: `<button class="btn ghost" type="button" data-close>关闭</button>`,
      onMount(root, close) {
        $$("[data-close]", root).forEach(b => b.onclick = close);
        $$("[data-rev]", root).forEach(b => b.onclick = async () => {
          try { await api(`/kbs/${encodeURIComponent(state.current)}/documents/${encodeURIComponent(doc.doc_id)}/history/${encodeURIComponent(b.dataset.rev)}/restore`, { method: "POST" }); close(); await refreshAll(); toast("已回退到该修订。"); }
          catch (e) { toast("回退失败：" + e.message, "err"); }
        });
      }
    });
  }
  async function openDeleted() {
    try { state.deleted = await api(`/kbs/${encodeURIComponent(state.current)}/documents/deleted`); }
    catch (e) { return toast("加载失败：" + e.message, "err"); }
    openModal({
      title: "已删除条目", subtitle: "恢复会回写所有可用版本。", body: state.deleted.length ? state.deleted.map(d => `<div class="deleted-row"><div></div><div><b>${esc(d.name)}</b><small>${fmtTime(d.updated_at)}</small></div><button class="btn small ghost" type="button" data-restore="${esc(d.doc_id)}">恢复</button></div>`).join("") : `<div class="inspect-empty">没有已删除的条目。</div>`,
      footer: `<button class="btn ghost" type="button" data-close>关闭</button>`,
      onMount(root, close) {
        $$("[data-close]", root).forEach(b => b.onclick = close);
        $$("[data-restore]", root).forEach(b => b.onclick = async () => {
          try { await api(`/kbs/${encodeURIComponent(state.current)}/documents/${encodeURIComponent(b.dataset.restore)}/restore`, { method: "POST" }); close(); await refreshAll(); toast("条目已恢复。"); }
          catch (e) { toast("恢复失败：" + e.message, "err"); }
        });
      }
    });
  }
  function openBulk() {
    if (!state.selected.size) return toast("请先勾选条目", "err");
    openModal({
      title: "批量权限", subtitle: `应用到 ${state.selected.size} 个条目`, small: true,
      body: `<div class="field"><label for="bulkPolicy">编辑策略</label><select id="bulkPolicy"><option value="inherit">继承</option><option value="open">开放</option><option value="whitelist">名单</option><option value="admin">仅人工</option><option value="off">冻结</option></select></div>`,
      footer: `<button class="btn ghost" type="button" data-cancel>取消</button><button class="btn primary" type="button" data-ok>应用</button>`,
      onMount(root, close) {
        $("[data-cancel]", root).onclick = close;
        $("[data-ok]", root).onclick = async () => {
          try { await api(`/kbs/${encodeURIComponent(state.current)}/documents/permissions`, { method: "PUT", body: JSON.stringify({ doc_ids: [...state.selected], permissions: { edit: $("#bulkPolicy").value } }) }); close(); state.selected.clear(); await refreshAll(); toast("批量策略已应用。"); }
          catch (e) { toast("应用失败：" + e.message, "err"); }
        };
      }
    });
  }

  /* ---------- versions / search ---------- */
  async function openVersionModal() {
    let data;
    try { data = await api("/embedding-models"); }
    catch (e) { return toast("加载模型失败：" + e.message, "err"); }
    const models = data.models || [];
    openModal({
      title: "新建版本", subtitle: "选择嵌入模型，探测维度后创建。", body: `<div class="field"><label for="versionModel">嵌入模型</label><select id="versionModel"><option value="">KiraAI 默认嵌入模型</option>${models.map(m => `<option value="${esc(m.uuid)}">${esc(m.label)}</option>`).join("")}</select><span class="hint">留空使用默认模型。</span></div><div class="field"><label for="versionName">版本标签</label><input id="versionName" placeholder="例如 bge-m3"></div><div class="field"><label for="versionDim">维度</label><input id="versionDim" type="number" min="1"><div class="display-actions"><button class="btn small ghost" type="button" id="probeBtn">探测</button><span class="hint" id="probeHint"></span></div></div><div class="field"><label>文档（留空 = 全部）</label><div class="check-list" id="versionDocs">${state.docs.map(d => `<label><input type="checkbox" value="${esc(d.doc_id)}"> ${esc(d.name)}</label>`).join("")}</div></div>`,
      footer: `<button class="btn ghost" type="button" data-cancel>取消</button><button class="btn primary" type="button" data-ok>创建版本</button>`,
      onMount(root, close) {
        $("[data-cancel]", root).onclick = close;
        $("#probeBtn", root).onclick = async () => {
          const uuid = $("#versionModel").value;
          $("#probeHint").textContent = "探测中……";
          try { const res = await api("/embedding-models/probe", { method: "POST", body: JSON.stringify({ model_uuid: uuid || null }), allowError: true }); if (res.error) throw new Error(res.error); $("#versionDim").value = res.dimension; $("#probeHint").textContent = `已探测 ${res.dimension} 维`; }
          catch (e) { $("#probeHint").textContent = "探测失败：" + e.message; }
        };
        $("[data-ok]", root).onclick = async () => {
          const name = $("#versionName").value.trim(); const dim = Number($("#versionDim").value);
          if (!name) return toast("请填写版本标签", "err");
          if (!dim) return toast("请先探测维度", "err");
          const payload = { model_name: name, dimension: dim };
          const uuid = $("#versionModel").value; if (uuid) payload.model_uuid = uuid;
          const ids = $$("#versionDocs input:checked", root).map(x => x.value); if (ids.length) payload.doc_ids = ids;
          try { const res = await api(`/kbs/${encodeURIComponent(state.current)}/versions`, { method: "POST", body: JSON.stringify(payload) }); close(); if (res.task_id) { toast("版本构建已排队。"); await loadTasks(); renderTasks(); switchView("tasks"); } else { await refreshAll(); toast("版本已创建。"); } }
          catch (e) { toast("创建失败：" + e.message, "err"); }
        };
      }
    });
  }
  async function runSearch() {
    const query = $("#searchInput").value.trim(); if (!query) return;
    $("#searchResults").innerHTML = `<div class="inspect-empty">检索中……</div>`;
    try {
      const res = await api(`/kbs/${encodeURIComponent(state.current)}/search`, { method: "POST", body: JSON.stringify({ query, top_k: 8 }), allowError: true });
      if (res.error) { $("#searchResults").innerHTML = `<div class="inspect-empty">${esc(res.error)}</div>`; return; }
      $("#searchResults").innerHTML = Array.isArray(res) && res.length ? res.map(r => `<div class="result"><div class="result-head"><span class="name">${esc(r.doc_name || r.doc_id)}</span><span class="score">${Number(r.score || 0).toFixed(2)}</span></div><div class="score-bar"><i style="width:${Math.round(Math.max(0, Math.min(1, Number(r.score) || 0)) * 100)}%"></i></div><p>${esc(r.content || "")}</p></div>`).join("") : `<div class="inspect-empty">没有找到相关内容。</div>`;
    } catch (e) { $("#searchResults").innerHTML = `<div class="inspect-empty">${esc(e.message)}</div>`; }
  }

  /* ---------- quote ---------- */
  const docCache = Object.create(null), contentCache = Object.create(null);
  function quoteKbIds() {
    const known = new Set(state.kbs.map(k => k.kb_id));
    const ids = (state.ui.quote_kb_ids || []).filter(id => known.has(id));
    return ids.length ? ids : [...known];
  }
  async function docsFor(kbId) {
    if (kbId === state.current && state.docs.length) return state.docs;
    if (docCache[kbId]) return docCache[kbId];
    const docs = await api(`/kbs/${encodeURIComponent(kbId)}/documents`);
    docCache[kbId] = Array.isArray(docs) ? docs.filter(d => !d.deleted) : [];
    return docCache[kbId];
  }
  async function contentFor(kbId, docId) {
    const key = kbId + "/" + docId;
    if (contentCache[key] != null) return contentCache[key];
    const doc = await api(`/kbs/${encodeURIComponent(kbId)}/documents/${encodeURIComponent(docId)}`);
    contentCache[key] = doc?.content || "";
    return contentCache[key];
  }
  function splitSentences(text) { return String(text || "").split(/(?<=[。！？!?；;])\s*/).map(s => s.trim()).filter(Boolean); }
  function motto() { const custom = (state.ui.quote_custom_mottos || []).filter(Boolean); const pool = custom.length ? custom : MOTTOS; return pool[Math.floor(Math.random() * pool.length)]; }
  async function pickQuote() {
    const custom = (state.ui.quote_custom_lines || []).filter(Boolean);
    if (custom.length && Math.random() < .3) return { prefix: "自定", text: custom[Math.floor(Math.random() * custom.length)], source: "CUSTOM SCRIPT / 自定义台词" };
    const ids = quoteKbIds().sort(() => Math.random() - .5).slice(0, 4);
    for (const id of ids) {
      try {
        const docs = await docsFor(id); if (!docs.length) continue;
        const doc = docs[Math.floor(Math.random() * docs.length)];
        const content = await contentFor(id, doc.doc_id);
        const ss = splitSentences(content).filter(s => s.length >= 8 && s.length <= 150);
        if (!ss.length) continue;
        let text = ss[Math.floor(Math.random() * ss.length)];
        if (text.length > 110) text = text.slice(0, 108) + "……";
        const kb = state.kbs.find(k => k.kb_id === id);
        return { prefix: PREFIXES[Math.floor(Math.random() * PREFIXES.length)], text, source: `${kb?.display_name || id} // ${doc.name || doc.doc_id}` };
      } catch (_) { /* next */ }
    }
    return { prefix: "格言", text: motto(), source: "MOTTO / 未归档格言" };
  }
  function typeInto(el, text, timerKey) {
    if (!el) return;
    clearInterval(state[timerKey]);
    if (reduced || state.ui.performance) { el.textContent = text; return; }
    el.textContent = ""; let i = 0;
    state[timerKey] = setInterval(() => { el.textContent = text.slice(0, ++i); if (i >= text.length) clearInterval(state[timerKey]); }, 18);
  }
  function typeTicker(text) {
    typeInto($("#tickerText"), text, "tickerTimer");
  }
  async function broadcastQuote(force = false) {
    if (state.quoteBusy) return; state.quoteBusy = true;
    try {
      const quote = await pickQuote();
      $("#tickerPrefix").textContent = quote.prefix; $("#tickerSource").textContent = quote.source;
      $("#tickerMode").textContent = state.ui.quote_enabled ? "AUTO" : "MANUAL";
      typeTicker(quote.text);
      updateApprTicker(quote);
      if (window.KBTextFX) window.KBTextFX.burst($("#tickerText"), state.ui.text_fx);
      const card = $("#quoteCard");
      if (card) {
        card.classList.add("is-speaking"); setTimeout(() => card.classList.remove("is-speaking"), 1700);
        $("#heroQuoteIndex").textContent = `RECORD ${String((quote.source || "").length + (quote.text || "").length).slice(0, 3)} // ${quote.source.split(" // ")[0] || "ARCHIVE"}`;
        $("#heroSource").textContent = quote.source;
        if (window.KBTextFX) window.KBTextFX.playText($("#heroQuote"), quote.text, state.ui.text_fx);
        else $("#heroQuote").textContent = quote.text;
      }
    } finally { state.quoteBusy = false; }
  }
  function startQuoteTimer() {
    stopQuoteTimer(); if (!state.ui.quote_enabled) return;
    state.quoteTimer = setInterval(() => { if (!document.hidden) broadcastQuote(); }, Math.max(4, state.ui.quote_interval) * 1000);
  }
  function stopQuoteTimer() { if (state.quoteTimer) clearInterval(state.quoteTimer); state.quoteTimer = null; }

  /* ---------- UI events ---------- */
  function switchView(view) {
    if (!VIEW_META[view] || view === state.view) return;
    const screen = $("#screen"); screen.classList.add("is-cutting");
    setTimeout(() => {
      state.view = view;
      $$(".channel").forEach(b => b.classList.toggle("is-active", b.dataset.view === view));
      $$(".view").forEach(v => v.classList.toggle("is-active", v.dataset.view === view));
      $("#hudKicker").textContent = `CH ${VIEW_META[view][0]} // ${VIEW_META[view][1]}`;
      $("#consoleNumber").textContent = VIEW_META[view][0];
      $("#viewStack").scrollTop = 0;
      if (view === "settings") { loadSettings().then(renderSettings); syncUiControls(); }
      if (view === "wiki") renderWiki();
      replayText();
      replayBrand();
      cycleBrandMode();
    }, 170);
    setTimeout(() => screen.classList.remove("is-cutting"), 380);
  }
  function bind() {
    document.addEventListener("click", async (e) => {
      const nav = e.target.closest(".channel[data-view]"); if (nav) return switchView(nav.dataset.view);
      const theme = e.target.closest("[data-theme-set]"); if (theme) return setTheme(theme.dataset.themeSet);
      if (e.target.closest("#appreciateBtn")) return openAppreciate();
      if (e.target.closest("#apprClose")) return closeAppreciate();
      if (e.target.closest("#apprPrev")) return apprMove(-1);
      if (e.target.closest("#apprNext")) return apprMove(1);
      const apprTheme = e.target.closest("[data-appr-theme]");
      if (apprTheme) { state.apprTheme = apprTheme.dataset.apprTheme; state.apprIndex = 0; renderApprThemes(); apprShow(true); return; }
      const wallToggle = e.target.closest("[data-wall-toggle]");
      if (wallToggle) {
        const url = wallToggle.dataset.wallToggle;
        if (state.wallDisabled.has(url)) state.wallDisabled.delete(url); else state.wallDisabled.add(url);
        persistWallDisabled(); renderWallVault();
        if (state.orgWallpaper === url && !wallEnabled(url)) {
          const next = pickOrgWallpaper(state.ui.theme); if (next) setOrgWallpaper(next, true);
        }
        return;
      }
      const wallTab = e.target.closest("[data-wall-tab]");
      if (wallTab) { state.wallVaultTheme = wallTab.dataset.wallTab; renderWallVault(); return; }
      if (e.target.closest("#wallAllBtn")) { wallsFor(state.wallVaultTheme).forEach(u => state.wallDisabled.delete(u)); persistWallDisabled(); renderWallVault(); return; }
      if (e.target.closest("#wallNoneBtn")) { wallsFor(state.wallVaultTheme).forEach(u => state.wallDisabled.add(u)); persistWallDisabled(); renderWallVault(); return; }
      if (e.target.closest("#zeroClearWalls")) { saveZero({ walls: [] }, true); toast("第零章插画已清空。"); return; }
      const doc = e.target.closest("[data-doc]"); if (doc && !e.target.closest("input")) return selectDoc(doc.dataset.doc);
      if (e.target.closest("#selectAllDocs")) return;
      const check = e.target.closest("[data-check]");
      if (check) { if (check.checked) state.selected.add(check.dataset.check); else state.selected.delete(check.dataset.check); return; }
      const act = e.target.closest("[data-activate]"); if (act) { try { await api(`/kbs/${encodeURIComponent(state.current)}/versions/${encodeURIComponent(act.dataset.activate)}/activate`, { method: "POST" }); await refreshAll(); toast("版本已激活。"); } catch (err) { toast("失败：" + err.message, "err"); } return; }
      const sync = e.target.closest("[data-sync]"); if (sync) { try { const res = await api(`/kbs/${encodeURIComponent(state.current)}/versions/${encodeURIComponent(sync.dataset.sync)}/sync`, { method: "POST", body: "{}" }); if (res.task_id) { toast("同步已排队。"); await loadTasks(); renderTasks(); switchView("tasks"); } else toast("同步完成。"); } catch (err) { toast("同步失败：" + err.message, "err"); } return; }
      const delver = e.target.closest("[data-delver]"); if (delver) { if (await confirmDialog("删除版本", "删除这个索引？原始条目会保留，只移除向量。", "删除", true)) { try { await api(`/kbs/${encodeURIComponent(state.current)}/versions/${encodeURIComponent(delver.dataset.delver)}`, { method: "DELETE" }); await refreshAll(); toast("版本已删除。"); } catch (err) { toast("失败：" + err.message, "err"); } } return; }
      const saveRecall = e.target.closest("[data-save-recall]"); if (saveRecall) { const id = saveRecall.dataset.saveRecall; const sel = $(`[data-entry-recall="${CSS.escape(id)}"]`); const kw = $(`[data-entry-keywords="${CSS.escape(id)}"]`); try { await api(`/kbs/${encodeURIComponent(state.current)}/documents/${encodeURIComponent(id)}/recall`, { method: "PUT", body: JSON.stringify({ enabled: sel.value === "on" ? true : sel.value === "off" ? false : null, keywords: kw.value.split(/[,，]/).map(s => s.trim()).filter(Boolean) }) }); toast("条目召回已保存。"); } catch (err) { toast("保存失败：" + err.message, "err"); } return; }
      const bgUpload = e.target.closest("[data-bg-upload]"); if (bgUpload) { bgUpload.closest(".bg-slot")?.querySelector("input[type=file]")?.click(); return; }
      const bgReset = e.target.closest("[data-bg-reset]"); if (bgReset) { delete state.backgrounds[bgReset.dataset.bgReset]; persistBackgrounds(); applyBackgrounds(); renderBgGrid(); return; }
      const orgTheme = e.target.closest("[data-org-theme]");
      if (orgTheme) { setTheme(orgTheme.dataset.orgTheme); switchView("catalog"); toast("组织外观已应用。"); return; }
      const orgWall = e.target.closest("[data-org-wall]");
      if (orgWall) {
        const theme = orgWall.dataset.orgWall, index = Number(orgWall.dataset.orgWallIndex);
        const walls = wallsFor(theme);
        if (walls[index]) { setOrgWallpaper(walls[index], true); toast("已设为当前背景。"); }
        return;
      }
      if (e.target.closest("#replayBootBtn")) {
        const boot = $("#boot"); if (!boot) return;
        boot.style.display = "flex"; boot.classList.remove("is-off");
        prepareBoot(); runBoot();
        return;
      }
      if (e.target.closest("#settingsBtn")) return switchView("settings");
      if (e.target.closest("#newKbBtn")) return openKbCreate();
      if (e.target.closest("#editKbBtn")) return openKbEdit();
      if (e.target.closest("#deleteKbBtn")) return deleteKb();
      if (e.target.closest("#uploadBtn") || e.target.closest("#heroUploadBtn") || e.target.closest("#docUploadBtn")) return triggerUpload();
      if (e.target.closest("#deletedBtn")) return openDeleted();
      if (e.target.closest("#bulkEditBtn")) return openBulk();
      if (e.target.closest("#editDocBtn")) return openDocEditor(currentDoc());
      if (e.target.closest("#historyBtn")) return openHistory(currentDoc());
      if (e.target.closest("#entrySettingsBtn")) return openEntrySettings(currentDoc());
      if (e.target.closest("#deleteDocBtn")) return deleteDoc(state.selectedDoc);
      if (e.target.closest("#newVersionBtn")) return openVersionModal();
      if (e.target.closest("#savePermBtn")) return saveKbPerms();
      if (e.target.closest("#saveKbRecallBtn")) return saveKbRecall();
      if (e.target.closest("#saveSettingsBtn")) return saveAllSettings();
      if (e.target.closest("#refreshTasksBtn")) { await loadTasks(); renderTasks(); return; }
      if (e.target.closest("#drawQuoteBtn") || e.target.closest("#tickerNext")) return broadcastQuote(true);
      if (e.target.closest("#quoteToggleBtn")) { state.ui.quote_enabled = !state.ui.quote_enabled; if (state.ui.quote_enabled) startQuoteTimer(); else stopQuoteTimer(); syncUiControls(); queueUiSave(); return; }
      if (e.target.closest("#quickCrt")) { applyCrt(state.ui.crt === "off" ? "high" : "off"); queueUiSave(); return; }
      if (e.target.closest("#searchBtn")) return runSearch();
    });
    $("#kbSelect")?.addEventListener("change", e => selectKb(e.target.value));
    $("#docFilter")?.addEventListener("input", e => { state.docFilter = e.target.value; renderCatalog(); });
    $("#docFilterKind")?.addEventListener("change", e => { state.docFilterKind = e.target.value; renderCatalog(); });
    $("#fileInput")?.addEventListener("change", e => uploadFiles(e.target.files));
    $("#searchInput")?.addEventListener("keydown", e => { if (e.key === "Enter") runSearch(); });
    $("#selectAllDocs")?.addEventListener("change", e => { const list = filteredDocs(); if (e.target.checked) list.forEach(d => state.selected.add(d.doc_id)); else state.selected.clear(); renderDocList(); });
    document.addEventListener("change", async (e) => {
      if (e.target.id === "zeroEnabled") { saveZero({ enabled: e.target.checked }, true); toast(e.target.checked ? "第零章已启用。" : "第零章已关闭。"); return; }
      const zt = e.target.closest("[data-zero-target]");
      if (zt) {
        const set = new Set(zeroTargets());
        if (zt.checked) set.add(zt.dataset.zeroTarget); else set.delete(zt.dataset.zeroTarget);
        saveZero({ targets: [...set] }, true); return;
      }
      const zf = e.target.closest("[data-zero-field]");
      if (zf) { saveZero({ [zf.dataset.zeroField]: zf.value }, false); return; }
      if (e.target.id === "zeroBadgeFile" && e.target.files?.[0]) {
        try { const data = await resizeImage(e.target.files[0], 800); saveZero({ badge: data }, true); toast("第零章徽章已更新。"); }
        catch (err) { toast("上传失败：" + err.message, "err"); }
        return;
      }
      if (e.target.id === "zeroWallsFile" && e.target.files?.length) {
        try {
          const added = [];
          for (const file of [...e.target.files].slice(0, 19)) added.push(await resizeImage(file, 900));
          saveZero({ walls: [...added, ...(zeroData().walls || [])].slice(0, 19) }, true);
          toast(`第零章已添加 ${added.length} 张插画。`);
        } catch (err) { toast("上传失败：" + err.message, "err"); }
        return;
      }
      if (e.target.matches("#setUiWallRotate, #setUiWallInterval, #setUiParallax")) {
        state.ui.wall_rotate = $("#setUiWallRotate")?.checked ?? state.ui.wall_rotate;
        state.ui.wall_interval = Number($("#setUiWallInterval")?.value) || 18;
        state.ui.wall_parallax = $("#setUiParallax")?.checked ?? state.ui.wall_parallax;
        startWallRotation(); startApprRotation(); queueUiSave(); return;
      }
      const permId = e.target.dataset.entryEdit || e.target.dataset.entryDelete;
      if (permId) {
        const doc = state.docs.find(d => d.doc_id === permId);
        if (!doc) return;
        const edit = $(`[data-entry-edit="${CSS.escape(permId)}"]`)?.value || "inherit";
        const del = $(`[data-entry-delete="${CSS.escape(permId)}"]`)?.value || "inherit";
        try {
          await api(`/kbs/${encodeURIComponent(state.current)}/documents/${encodeURIComponent(permId)}/permissions`, {
            method: "PUT", body: JSON.stringify({ edit, delete: del, allow_users: doc.permissions?.allow_users || [] })
          });
          toast("条目权限已更新。");
        } catch (err) { toast("保存失败：" + err.message, "err"); }
        return;
      }
      const orgToggle = e.target.closest("[data-org-enabled]");
      if (orgToggle) {
        const theme = orgToggle.dataset.orgEnabled;
        state.custom[theme] = state.custom[theme] || {};
        state.custom[theme].enabled = !!orgToggle.checked;
        saveCustomAssets();
        renderThemeStrip(); renderThemeBadges(); renderWiki();
        if (theme === state.ui.theme) applyOrgIdentity();
        toast(orgToggle.checked ? "组织外观已启用。" : "组织外观已关闭。");
        return;
      }
      const orgBadgeInput = e.target.closest("input[data-org-badge]");
      if (orgBadgeInput && orgBadgeInput.files?.[0]) {
        const theme = orgBadgeInput.dataset.orgBadge;
        try {
          const data = await resizeImage(orgBadgeInput.files[0], 800);
          state.custom[theme] = state.custom[theme] || {};
          state.custom[theme].badge = data;
          state.custom[theme].enabled = true;
          saveCustomAssets(); renderThemeStrip(); renderThemeBadges(); renderWiki();
          if (theme === state.ui.theme) applyOrgIdentity();
          toast("组织徽章已更新。");
        } catch (err) { toast("上传失败：" + err.message, "err"); }
        return;
      }
      const orgWallsInput = e.target.closest("input[data-org-walls]");
      if (orgWallsInput && orgWallsInput.files?.length) {
        const theme = orgWallsInput.dataset.orgWalls;
        try {
          const added = [];
          for (const file of [...orgWallsInput.files].slice(0, 5)) added.push(await resizeImage(file, 1600));
          state.custom[theme] = state.custom[theme] || {};
          state.custom[theme].walls = [...added, ...(state.custom[theme].walls || [])].slice(0, 12);
          state.custom[theme].enabled = true;
          saveCustomAssets(); renderWiki(); renderThemeStrip(); renderThemeBadges();
          if (theme === state.ui.theme) { state.orgWallpaper = added[0] || state.orgWallpaper; applyBackgrounds(); }
          toast(`已添加 ${added.length} 张组织壁纸。`);
        } catch (err) { toast("上传失败：" + err.message, "err"); }
        return;
      }
      if (e.target.matches("#setUiTheme, #setUiCrt, #setUiTextFx, #setUiPerf, #setUiSplash, #setUiFxLoop, #setUiFxInterval, #setUiQuoteEnabled, #setUiQuoteInterval")) {
        state.ui.theme = $("#setUiTheme").value; state.ui.crt = $("#setUiCrt").value;
        state.ui.performance = $("#setUiPerf").checked; state.ui.splash = $("#setUiSplash").checked;
        state.ui.text_fx = $("#setUiTextFx").value; state.ui.fx_loop = $("#setUiFxLoop").checked;
        state.ui.fx_interval = Number($("#setUiFxInterval").value) || 6;
        state.ui.quote_enabled = $("#setUiQuoteEnabled").checked; state.ui.quote_interval = Number($("#setUiQuoteInterval").value) || 9;
        applyUi(); if (state.ui.fx_loop && !state.ui.performance && !reduced) window.KBTextFX?.startLoop(state.ui.fx_interval); else window.KBTextFX?.stopLoop();
        queueUiSave();
      }
      if (e.target.matches("#setUiQuoteKbs input")) {
        const checked = $$("#setUiQuoteKbs input:checked").length;
        if (!checked) { e.target.checked = true; toast("至少保留一个字幕知识库", "err"); return; }
        state.ui.quote_kb_ids = selectedQuoteKbs(); renderQuoteKbs(); queueUiSave();
      }
      const bg = e.target.closest("input[data-bg-file]");
      if (bg && bg.files?.[0]) {
        try { const data = await resizeImage(bg.files[0]); state.backgrounds[bg.dataset.bgFile] = data; persistBackgrounds(); applyBackgrounds(); renderBgGrid(); toast("背景立绘已更新。"); }
        catch (err) { toast("上传失败：" + err.message, "err"); }
      }
    });
    document.addEventListener("input", e => {
      const zf = e.target.closest("[data-zero-field]");
      if (zf) { state.zero = { ...zeroData(), [zf.dataset.zeroField]: zf.value, custom: true }; try { localStorage.setItem("linji-zero-v1", JSON.stringify(state.zero)); } catch (_) {} return; }
      if (e.target.matches("#setUiQuoteLines, #setUiQuoteMottos")) {
        state.ui.quote_custom_lines = lines("#setUiQuoteLines"); state.ui.quote_custom_mottos = lines("#setUiQuoteMottos"); queueUiSave();
      }
      if (e.target.matches("#setUiFxInterval, #setUiQuoteInterval")) {
        state.ui.fx_interval = Number($("#setUiFxInterval").value) || 6; state.ui.quote_interval = Number($("#setUiQuoteInterval").value) || 9;
        startQuoteTimer(); queueUiSave();
      }
    });
    document.addEventListener("visibilitychange", () => { if (document.hidden) stopQuoteTimer(); else if (state.ui.quote_enabled) startQuoteTimer(); });
    document.addEventListener("keydown", e => {
      if (e.key === "Escape") {
        if (state.apprOpen) { closeAppreciate(); return; }
        const m = $(".modal-root"); if (m) m.remove(); hideBoot(); return;
      }
      if (state.apprOpen) {
        if (e.key === "ArrowLeft") { e.preventDefault(); apprMove(-1); return; }
        if (e.key === "ArrowRight") { e.preventDefault(); apprMove(1); return; }
      }
      const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName) || e.target.isContentEditable;
      if (!typing && (e.key === "f" || e.key === "F") && !e.metaKey && !e.ctrlKey) { e.preventDefault(); state.apprOpen ? closeAppreciate() : openAppreciate(); }
    });
  }
  function hideBoot() { const b = $("#boot"); if (!b) return; b.classList.add("is-off"); document.body.classList.remove("boot-hold"); setTimeout(() => b.style.display = "none", 600); }
  async function resizeImage(file, max = 1600) {
    if (!file.type.startsWith("image/")) throw new Error("不是图片");
    if (file.size > 8 * 1024 * 1024) throw new Error("图片超过 8MB");
    const url = URL.createObjectURL(file);
    try {
      const img = await new Promise((resolve, reject) => { const i = new Image(); i.onload = () => resolve(i); i.onerror = () => reject(new Error("无法读取图片")); i.src = url; });
      const scale = Math.min(1, max / Math.max(img.width, img.height));
      const c = document.createElement("canvas"); c.width = Math.max(1, Math.round(img.width * scale)); c.height = Math.max(1, Math.round(img.height * scale));
      c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
      return c.toDataURL("image/jpeg", .82);
    } finally { URL.revokeObjectURL(url); }
  }

  /* ---------- settings save ---------- */
  async function saveKbPerms() {
    try { await api(`/kbs/${encodeURIComponent(state.current)}/permissions`, { method: "PUT", body: JSON.stringify({ add: $("#policyAdd").value, edit: $("#policyEdit").value, delete: $("#policyDelete").value, allow_users: lines("#kbAllowUsers") }) }); await refreshAll(); toast("权限策略已保存。"); }
    catch (e) { toast("保存失败：" + e.message, "err"); }
  }
  async function saveKbRecall() {
    const num = id => { const el = $("#" + id); return el && el.value !== "" ? Number(el.value) : null; };
    try { await api(`/kbs/${encodeURIComponent(state.current)}/recall`, { method: "PUT", body: JSON.stringify({ enabled: $("#kbRecallEnabled").checked, top_k: num("kbRecallTopK"), min_score: num("kbRecallMinScore"), max_chars: num("kbRecallMaxChars") }) }); await refreshAll(); toast("召回设置已保存。"); }
    catch (e) { toast("保存失败：" + e.message, "err"); }
  }
  async function saveAllSettings() {
    try {
      const payload = collectFunctionalSettings(); payload.ui = collectUi();
      const res = await api("/settings", { method: "PUT", body: JSON.stringify(payload) });
      if (res.settings) state.settings = res.settings;
      state.ui = { ...state.ui, ...(res.settings?.ui || {}) };
      applyUi(); await refreshAll(); renderSettings();
      if (res.warnings?.length) toast(res.warnings.join(" "), "err"); else toast("设置已应用。");
    } catch (e) { toast("保存失败：" + e.message, "err"); }
  }

  /* ---------- init ---------- */
  async function init() {
    bind();
    renderThemeStrip();
    document.documentElement.setAttribute("data-theme", "dark");
    document.body.dataset.theme = "violet";
    document.body.dataset.crt = "high";
    applyBackgrounds();
    initParallax();
    await loadSettings();
    if (state.settings?.ui) state.ui = { ...state.ui, ...state.settings.ui };
    state.ui.theme = THEMES.includes(state.ui.theme) ? state.ui.theme : "violet";
    state.ui.text_fx = ["mix", "radio", "pulse", "type", "decode", "scan", "glitch"].includes(state.ui.text_fx) ? state.ui.text_fx : "mix";
    state.wallVaultTheme = state.ui.theme;
    applyUi();
    renderThemeStrip();
    renderSettings();
    syncUiControls();
    prepareBoot();
    runBoot();
    try {
      await refreshAll();
      renderAll();
    } catch (e) { toast("加载失败：" + e.message, "err"); }
    initTextFx();
    startBrandModes();
    if (state.ui.quote_enabled) { broadcastQuote(true); startQuoteTimer(); }
    setInterval(() => { if (!document.hidden) { loadTasks().then(renderTasks).catch(() => {}); } }, 5000);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
