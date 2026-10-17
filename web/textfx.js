/* Text effects for LINJI ARCHIVE — typewriter, decode, radio, pulse, scan.
   No canvas, no continuous layout reads. Effects are one-shot + cheap idle CSS. */
(() => {
  "use strict";
  const VARIANTS = ["radio", "pulse", "type", "decode", "scan", "glitch"];
  const GLYPHS = "档案记忆观察收容记录编号残页静默信号";
  const timers = new WeakMap();
  const conf = { mode: "mix", idle: true, level: 2, enabled: true, loop: true };
  let loopTimer = null;
  let loopInterval = 6;

  function rand(n) { return Math.floor(Math.random() * n); }
  function later(el, fn, ms) {
    const id = setTimeout(fn, ms);
    const list = timers.get(el) || [];
    list.push(id);
    timers.set(el, list);
  }
  function clearTimers(el) {
    (timers.get(el) || []).forEach(clearTimeout);
    timers.set(el, []);
  }
  function choose(el, requested) {
    if (requested && requested !== "mix") return requested;
    if (conf.mode && conf.mode !== "mix") return conf.mode;
    if (el && el.dataset && el.dataset.fx && el.dataset.fx !== "mix") return el.dataset.fx;
    return VARIANTS[rand(VARIANTS.length)];
  }

  /** Wrap every text node into .fx-ch spans while preserving <em>, <br>, etc. */
  function prepare(el) {
    if (el._fxOriginal == null) el._fxOriginal = el.innerHTML;
    else el.innerHTML = el._fxOriginal;
    const chars = [];
    const walk = (node) => {
      if (node.nodeType === 3) {
        const text = node.nodeValue;
        if (!text) return;
        const frag = document.createDocumentFragment();
        for (const ch of text) {
          if (ch === "\n") { frag.appendChild(document.createElement("br")); continue; }
          const span = document.createElement("span");
          span.className = "fx-ch";
          span.textContent = ch === " " ? "\u00a0" : ch;
          span.dataset.final = span.textContent;
          chars.push(span);
          frag.appendChild(span);
        }
        node.parentNode.replaceChild(frag, node);
      } else if (node.nodeType === 1) {
        if (node.tagName === "BR" || node.classList.contains("fx-ch")) return;
        [...node.childNodes].forEach(walk);
      }
    };
    [...el.childNodes].forEach(walk);
    return chars;
  }

  function resetClasses(el) {
    el.classList.remove(
      "fx-title", "fx-burst-radio", "fx-burst-pulse", "fx-burst-scan",
      "fx-idle-radio", "fx-idle-pulse", "fx-idle-scan", "fx-decoding"
    );
  }

  function finish(el, effect) {
    el.classList.remove("fx-burst-radio", "fx-burst-pulse", "fx-burst-scan", "fx-decoding");
    if (!conf.idle) return;
    if (el.matches && el.matches(".hero-copy h1, .page-title h1, h1[data-fx]")) {
      // Large Chinese headlines read poorly with RGB ghosting; keep a scan
      // line as the persistent state and reserve radio/pulse for smaller text.
      el.classList.add("fx-idle-scan");
    } else if (effect === "pulse") el.classList.add("fx-idle-pulse");
    else if (effect === "scan") el.classList.add("fx-idle-scan");
    else el.classList.add("fx-idle-radio");
  }

  function play(el, requested, opts = {}) {
    if (!el) return null;
    clearTimers(el);
    resetClasses(el);
    if (!conf.enabled) {
      if (el._fxOriginal != null) el.innerHTML = el._fxOriginal;
      return null;
    }
    const effect = choose(el, requested);
    const chars = prepare(el);
    el.classList.add("fx-title");
    if (effect === "radio" || effect === "glitch") el.classList.add("fx-burst-radio");
    else if (effect === "pulse") el.classList.add("fx-burst-pulse");
    else if (effect === "scan") el.classList.add("fx-burst-scan");

    const fast = opts.fast || conf.level <= 0;
    if (effect === "decode") {
      el.classList.add("fx-decoding");
      chars.forEach(c => { c.textContent = GLYPHS[rand(GLYPHS.length)]; });
      const step = Math.max(9, Math.min(34, Math.floor(520 / Math.max(1, chars.length))));
      chars.forEach((c, i) => later(el, () => {
        c.textContent = c.dataset.final;
        c.classList.add("fx-resolved");
      }, (fast ? 0 : 90) + i * step));
      later(el, () => finish(el, effect), (fast ? 0 : 90) + chars.length * step + 180);
    } else if (effect === "type") {
      chars.forEach(c => c.classList.add("fx-hidden"));
      const step = Math.max(10, Math.min(48, Math.floor(700 / Math.max(1, chars.length))));
      chars.forEach((c, i) => later(el, () => {
        c.classList.remove("fx-hidden");
        c.classList.add("fx-type-in");
      }, (fast ? 0 : 70) + i * step));
      if (!fast) {
        const caret = document.createElement("span");
        caret.className = "fx-caret";
        el.appendChild(caret);
        later(el, () => caret.remove(), 3200);
      }
      later(el, () => finish(el, effect), (fast ? 0 : 70) + chars.length * step + 120);
    } else {
      const step = Math.max(7, Math.min(24, Math.floor(380 / Math.max(1, chars.length))));
      chars.forEach((c, i) => {
        c.classList.add(
          effect === "pulse" ? "fx-pulse-in"
            : effect === "scan" ? "fx-scan-in"
            : effect === "glitch" ? "fx-glitch-in"
            : "fx-char-in"
        );
        c.style.animationDelay = `${fast ? 0 : Math.min(i * step, 420)}ms`;
      });
      later(el, () => finish(el, effect), fast ? 0 : 560);
    }
    return effect;
  }

  function playText(el, text, requested, opts = {}) {
    if (!el) return null;
    clearTimers(el);
    el.innerHTML = "";
    el.textContent = text;
    el._fxOriginal = text;
    return play(el, requested, opts);
  }

  function replayView(view) {
    const root = document.querySelector(`.view[data-view="${view}"]`) || document.querySelector(".view.is-active") || document;
    root.querySelectorAll(".hero-copy h1, .page-title h1, h1[data-fx]").forEach((el, i) => {
      if (!el.dataset.fx) el.dataset.fx = "mix";
      later(el, () => play(el, el.dataset.fx), i * 60);
    });
    root.querySelectorAll(".panel-head h2, .version-model, .version-card h3").forEach((el, i) => {
      el.classList.remove("fx-heading");
      void el.offsetWidth;
      el.classList.add("fx-heading");
      later(el, () => play(el, ["scan", "pulse", "radio"][i % 3]), 40 + i * 35);
    });
  }

  function init() {
    document.querySelectorAll(".hero-copy h1, .page-title h1, #railKb, #railTitle, #railAlias").forEach(el => {
      if (!el.dataset.fx) el.dataset.fx = "mix";
    });
    document.querySelectorAll(".panel-head h2, .version-model, .version-card h3").forEach(el => el.classList.add("fx-heading"));
    replayView("catalog");
  }

  /** Short one-shot burst for text that is already visible (ticker). */
  function burst(el, requested) {
    if (!el || !conf.enabled) return;
    const effect = choose(el, requested);
    el.classList.remove("fx-burst-radio", "fx-burst-pulse", "fx-burst-scan");
    void el.offsetWidth;
    if (effect === "pulse") el.classList.add("fx-burst-pulse");
    else if (effect === "scan") el.classList.add("fx-burst-scan");
    else el.classList.add("fx-burst-radio");
    setTimeout(() => el.classList.remove("fx-burst-radio", "fx-burst-pulse", "fx-burst-scan"), 560);
  }

  function loopTick() {
    if (!conf.enabled || !conf.loop || document.hidden) return;
    const root = document.querySelector(".view.is-active");
    if (!root) return;
    const titles = [...root.querySelectorAll(
      ".hero-copy h1, .page-title h1, .panel-head h2, .version-model"
    )];
    const extras = [...root.querySelectorAll(
      ".entry-main b, .inspect-body h3, .stat-card b, .chapter-label, .eyebrow, #heroQuote"
    )];
    const brand = [document.querySelector("#railKb"), document.querySelector("#railTitle"), document.querySelector("#railAlias")];
    const pool = (Math.random() < .72 && titles.length ? titles : titles.concat(extras))
      .concat(brand)
      .filter(el => el && el.offsetParent !== null && !el.classList.contains("fx-decoding"));
    if (!pool.length) return;
    const el = pool[rand(pool.length)];
    const effect = el.matches(".stat-card b") ? "pulse"
      : el.matches(".chapter-label, .eyebrow") ? "scan"
      : el.matches("#railKb") ? "decode"
      : el.matches("#railTitle") ? "scan"
      : ["radio", "pulse", "scan"][rand(3)];
    if (el.id === "heroQuote") {
      const panel = el.closest(".hero-quote");
      if (panel) {
        panel.classList.add("is-speaking");
        setTimeout(() => panel.classList.remove("is-speaking"), 1500);
      }
    }
    play(el, effect);
  }
  function startLoop(interval = loopInterval) {
    stopLoop();
    loopInterval = Math.max(3, Number(interval) || 6);
    if (!conf.enabled || !conf.loop) return;
    loopTimer = setInterval(loopTick, loopInterval * 1000);
  }
  function stopLoop() {
    if (loopTimer) clearInterval(loopTimer);
    loopTimer = null;
  }
  function setConfig(cfg) {
    if (!cfg) return;
    if (cfg.mode) conf.mode = cfg.mode;
    if (typeof cfg.idle === "boolean") conf.idle = cfg.idle;
    if (typeof cfg.level === "number") conf.level = cfg.level;
    if (typeof cfg.enabled === "boolean") conf.enabled = cfg.enabled;
    if (typeof cfg.loop === "boolean") conf.loop = cfg.loop;
    if (typeof cfg.loopInterval === "number") loopInterval = Math.max(3, cfg.loopInterval);
  }
  function setEnabled(value) { conf.enabled = !!value; }

  window.KBTextFX = {
    init, play, playText, replayView, burst, setConfig, setEnabled,
    startLoop, stopLoop, choose, loopTick
  };
})();
