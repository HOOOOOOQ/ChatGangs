/* ChatGangs web client — talks to the same origin, no external dependencies. */
(() => {
  "use strict";

  // Same origin by default (FastAPI serves this page). For a separate API host
  // set this before the script loads: window.CHATGANGS_API_BASE = "https://api.example.com".
  const API = (typeof window !== "undefined" && window.CHATGANGS_API_BASE) || "";
  const API_KEY = (typeof window !== "undefined" && window.CHATGANGS_API_KEY) || "";
  const $ = (sel) => document.querySelector(sel);
  const el = (tag, attrs = {}, children = []) => {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs)) {
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else if (key === "html") node.innerHTML = value;
      else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
      else if (value !== null && value !== undefined) node.setAttribute(key, value);
    }
    for (const child of [].concat(children)) {
      if (child) node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
    }
    return node;
  };

  const state = {
    sessionId: loadSessionId(),
    busy: false,
    lastSuggestions: [],
  };

  function loadSessionId() {
    let id = localStorage.getItem("chatgangs.session");
    if (!id) {
      id = "web-" + Math.random().toString(36).slice(2, 10);
      localStorage.setItem("chatgangs.session", id);
    }
    return id;
  }

  async function api(path, options = {}) {
    const headers = { "Content-Type": "application/json" };
    if (API_KEY) headers["X-API-Key"] = API_KEY;
    const response = await fetch(API + path, {
      headers,
      ...options,
    });
    const text = await response.text();
    let data;
    try { data = text ? JSON.parse(text) : {}; } catch { data = { raw: text }; }
    if (!response.ok) {
      const message = data.error || data.detail || `HTTP ${response.status}`;
      const error = new Error(typeof message === "string" ? message : JSON.stringify(message));
      error.data = data;
      error.status = response.status;
      throw error;
    }
    return data;
  }

  /* ------------------------------------------------------------- navigation */
  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".nav-item").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
      btn.classList.add("active");
      $("#view-" + btn.dataset.view).classList.add("active");
      if (btn.dataset.view === "models") loadModels();
    });
  });

  /* ---------------------------------------------------------------- health */
  async function checkHealth() {
    const dot = $("#health-dot");
    const label = $("#health-text");
    try {
      const health = await api("/api/health");
      dot.className = "dot " + (health.status === "ok" ? "ok" : "err");
      label.textContent = `${health.status} · v${health.version}`;
    } catch (err) {
      dot.className = "dot err";
      label.textContent = "tidak tersedia";
    }
  }

  /* ------------------------------------------------------------------ chat */
  const chatLog = $("#chat-log");

  function bubble(role, text, meta) {
    const isUser = role === "user";
    const msg = el("div", { class: "msg " + (isUser ? "user" : "bot") }, [
      el("div", { class: "avatar", text: isUser ? "You" : "CG" }),
      el("div", { class: "bubble" }, [
        el("div", { text }),
        meta ? el("div", { class: "meta-chips" }, meta) : null,
      ]),
    ]);
    chatLog.appendChild(msg);
    chatLog.scrollTop = chatLog.scrollHeight;
    return msg;
  }

  function intentChip(intent) {
    if (!intent) return null;
    const cls = intent.confidence >= 0.6 ? "chip" : "chip neu";
    const node = el("span", { class: cls });
    node.textContent = `intent: ${intent.name} ${(intent.confidence * 100).toFixed(0)}%`;
    node.title = `method: ${intent.method}`;
    return node;
  }

  function sentimentChip(sentiment) {
    if (!sentiment) return null;
    const map = { positive: "pos", negative: "neg", neutral: "neu" };
    const node = el("span", { class: `chip ${map[sentiment.label] || "neu"}` });
    node.textContent = `sentimen: ${sentiment.label} ${sentiment.score > 0 ? "+" : ""}${sentiment.score.toFixed(2)}`;
    return node;
  }

  function entityChips(entities) {
    return (entities || []).slice(0, 4).map((entity) => {
      const node = el("span", { class: "chip" });
      node.textContent = `${entity.label}: ${entity.text}`;
      return node;
    });
  }

  function renderSuggestions(items) {
    const box = $("#suggestions");
    box.innerHTML = "";
    (items || []).forEach((text) => {
      box.appendChild(el("button", { type: "button", text, onclick: () => sendMessage(text) }));
    });
  }

  async function sendMessage(text) {
    if (state.busy || !text.trim()) return;
    state.busy = true;
    $("#send-btn").disabled = true;
    bubble("user", text);
    const pending = bubble("bot", "", null);
    pending.querySelector(".bubble").innerHTML = '<span class="typing"><span></span><span></span><span></span></span>';

    try {
      const data = await api("/api/chat", {
        method: "POST",
        body: JSON.stringify({ message: text, session_id: state.sessionId }),
      });
      const meta = [
        intentChip(data.intent),
        sentimentChip(data.sentiment),
        el("span", { class: "chip", text: `bahasa: ${data.language}` }),
        ...entityChips(data.entities),
      ].filter(Boolean);
      pending.remove();
      bubble("bot", data.reply, meta);
      renderSuggestions(data.suggestions);
      if (data.context && data.context.is_followup) {
        setTimeout(() => { /* follow-up indicator could go here */ }, 0);
      }
    } catch (err) {
      pending.remove();
      bubble("bot", `⚠️ ${err.message}`, [el("span", { class: "chip neg", text: `status ${err.status || "?"}` })]);
    } finally {
      state.busy = false;
      $("#send-btn").disabled = false;
      $("#chat-input").focus();
    }
  }

  $("#chat-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const input = $("#chat-input");
    const text = input.value.trim();
    if (!text) return;
    input.value = "";
    input.style.height = "auto";
    sendMessage(text);
  });

  $("#chat-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      $("#chat-form").requestSubmit();
    }
  });

  $("#chat-input").addEventListener("input", (event) => {
    event.target.style.height = "auto";
    event.target.style.height = Math.min(event.target.scrollHeight, 180) + "px";
  });

  $("#btn-new-session").addEventListener("click", () => {
    state.sessionId = "web-" + Math.random().toString(36).slice(2, 10);
    localStorage.setItem("chatgangs.session", state.sessionId);
    $("#session-label").textContent = state.sessionId;
    chatLog.innerHTML = "";
    bootChat();
  });

  $("#btn-delete-session").addEventListener("click", async () => {
    if (!confirm("Hapus seluruh riwayat sesi ini?")) return;
    try {
      await api(`/api/memory/${encodeURIComponent(state.sessionId)}`, { method: "DELETE" });
      chatLog.innerHTML = "";
      bubble("bot", "Riwayat sesi ini sudah dihapus dari database dan memori konteks. Kita mulai dari awal.");
    } catch (err) {
      bubble("bot", `Gagal menghapus: ${err.message}`);
    }
  });

  function bootChat() {
    $("#session-label").textContent = state.sessionId;
    bubble(
      "bot",
      "Halo! Saya ChatGangs, engine NLP modular. Saya bisa menganalisis sentimen, mendeteksi intent, mengekstrak entitas, mengenali bahasa, dan menjawab pertanyaan tentang sistem ini. Apa yang ingin Anda coba?"
    );
    renderSuggestions([
      "Apa saja kemampuanmu?",
      "Bagaimana cara memakai API?",
      "Analisis sentimen: pelayanannya bagus tapi pengirimannya lambat",
      "Apakah data percakapan saya aman?",
    ]);
  }

  /* -------------------------------------------------------------- analyzer */
  const EXAMPLE = "Pesanan saya dari PT Maju Jaya terlambat 3 hari dan kemasannya rusak. " +
    "Pelayanannya sebenarnya ramah, tapi saya tetap kecewa. Tolong hubungi saya di budi@mail.com " +
    "sebelum 12/03/2025 pukul 10:00. Budget kompensasi sekitar Rp 250.000.";

  $("#btn-example").addEventListener("click", () => {
    $("#analyze-input").value = EXAMPLE;
  });

  function statCard(label, value) {
    return el("div", { class: "stat" }, [
      el("div", { class: "k", text: label }),
      el("div", { class: "v", text: String(value) }),
    ]);
  }

  function sentimentCard(sentiment) {
    const pct = Math.round(((sentiment.score + 1) / 2) * 100);
    const bar = el("div", { class: "bar" + (sentiment.score < 0 ? " neg" : "") }, [
      el("i", { style: `width:${pct}%` }),
    ]);
    const evidence = (sentiment.evidence || []).slice(0, 8).map((item) =>
      el("span", { class: "chip " + (item.polarity >= 0 ? "pos" : "neg") ,
        text: `${item.term} (${item.polarity > 0 ? "+" : ""}${item.polarity}${item.negated ? ", negasi" : ""})` })
    );
    return card("Sentimen", [
      el("div", { class: "kv" }, [el("span", { class: "k", text: "Label" }),
        el("strong", { text: `${sentiment.label} · ${sentiment.score > 0 ? "+" : ""}${sentiment.score.toFixed(3)}` })]),
      el("div", { class: "kv" }, [el("span", { class: "k", text: "Keyakinan" }),
        el("span", { text: (sentiment.confidence * 100).toFixed(0) + "%" })]),
      el("div", { class: "kv" }, [el("span", { class: "k", text: "Sumber" }),
        el("span", { text: sentiment.model_used ? "lexicon + model terlatih" : "lexicon (model tidak tersedia)" })]),
      bar,
      evidence.length ? el("div", { class: "badges" }, evidence) : null,
    ]);
  }

  function card(title, children) {
    return el("section", { class: "card" }, [el("h2", { text: title }), ...[].concat(children).filter(Boolean)]);
  }

  function tokenCloud(tokens) {
    return el("div", { class: "tokens" }, tokens.slice(0, 80).map((token) => {
      const classes = ["token"];
      if (token.is_stopword) classes.push("stopword");
      if (token.entity_label) classes.push("entity");
      const node = el("span", { class: classes.join(" ") }, [token.text]);
      const labelParts = [];
      if (token.pos) labelParts.push(token.pos);
      if (token.stem && token.stem !== token.normalized) labelParts.push("→" + token.stem);
      if (labelParts.length) node.appendChild(el("span", { class: "lbl", text: labelParts.join(" ") }));
      return node;
    }));
  }

  function entityTable(entities) {
    if (!entities || !entities.length) return el("p", { class: "muted", text: "Tidak ada entitas terdeteksi." });
    return el("table", {}, [
      el("thead", {}, [el("tr", {}, ["Entitas", "Label", "Sumber", "Posisi"].map((h) => el("th", { text: h })))]),
      el("tbody", {}, entities.map((entity) => el("tr", {}, [
        el("td", { text: entity.text }),
        el("td", {}, [el("span", { class: "chip", text: entity.label })]),
        el("td", { class: "muted", text: entity.source }),
        el("td", { class: "muted", text: `${entity.start}–${entity.end}` }),
      ]))),
    ]);
  }

  function renderAnalysis(data) {
    const box = $("#analyze-results");
    box.innerHTML = "";
    const stats = data.token_stats || {};
    box.appendChild(card("Ringkasan", el("div", { class: "card-grid" }, [
      statCard("Bahasa", `${data.language.name} (${(data.language.confidence * 100).toFixed(0)}%)`),
      statCard("Intent", `${data.intent.name} · ${(data.intent.confidence * 100).toFixed(0)}%`),
      statCard("Sentimen", data.sentiment.label),
      statCard("Klasifikasi", data.classification.label),
      statCard("Token / kata", `${stats.tokens} / ${stats.words}`),
      statCard("Kalimat", stats.sentences),
      statCard("Entitas", data.entities.length),
      statCard("Latensi", `${data.stats.total_ms} ms`),
    ])));

    box.appendChild(card("Teks ternormalisasi", [
      el("p", { class: "muted", text: data.normalized }),
    ]));

    box.appendChild(card("Token & morfologi", tokenCloud(data.tokens || [])));

    box.appendChild(card("Entitas (NER)", entityTable(data.entities)));

    box.appendChild(sentimentCard(data.sentiment));

    box.appendChild(card("Klasifikasi teks", [
      el("div", { class: "kv" }, [el("span", { class: "k", text: "Label" }),
        el("strong", { text: data.classification.label })]),
      el("div", { class: "kv" }, [el("span", { class: "k", text: "Metode" }),
        el("span", { text: data.classification.method })]),
      data.classification.rationale && data.classification.rationale.length
        ? el("div", { class: "badges" }, data.classification.rationale.map((r) => el("span", { class: "chip", text: r })))
        : null,
    ]));

    if (data.qa_matches && data.qa_matches.length) {
      box.appendChild(card("Knowledge base yang relevan", el("table", {}, [
        el("thead", {}, [el("tr", {}, ["Topik", "Pertanyaan", "Skor"].map((h) => el("th", { text: h })))]),
        el("tbody", {}, data.qa_matches.map((m) => el("tr", {}, [
          el("td", {}, [el("span", { class: "chip", text: m.topic })]),
          el("td", { text: m.question }),
          el("td", { text: m.score.toFixed(3) }),
        ]))),
      ])));
    }

    if (data.keywords && data.keywords.length) {
      box.appendChild(card("Kata kunci", el("div", { class: "badges" },
        data.keywords.map((k) => el("span", { class: "chip", text: `${k.term} ×${k.frequency}` })))));
    }

    box.appendChild(card("Detail JSON", el("pre", { text: JSON.stringify(data, null, 2) })));
  }

  $("#analyze-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = $("#analyze-input").value.trim();
    if (!text) return;
    const button = $("#analyze-btn");
    button.disabled = true;
    $("#analyze-meta").textContent = "menganalisis…";
    const started = performance.now();
    try {
      const data = await api("/api/nlp/analyze", {
        method: "POST",
        body: JSON.stringify({ text }),
      });
      renderAnalysis(data);
      $("#analyze-meta").textContent =
        `${(performance.now() - started).toFixed(0)} ms round-trip · ${data.stats.total_ms} ms engine`;
    } catch (err) {
      $("#analyze-results").innerHTML = "";
      $("#analyze-results").appendChild(card("Error", el("p", { text: err.message })));
      $("#analyze-meta").textContent = "";
    } finally {
      button.disabled = false;
    }
  });

  /* ---------------------------------------------------------------- models */
  async function loadModels() {
    const box = $("#models-content");
    box.innerHTML = "";
    try {
      const info = await api("/api/model/info");
      const artifacts = info.artifacts || {};
      box.appendChild(card("Identitas", [
        el("div", { class: "kv" }, [el("span", { class: "k", text: "Nama" }), el("span", { text: info.name })]),
        el("div", { class: "kv" }, [el("span", { class: "k", text: "Versi" }), el("span", { text: info.version })]),
        el("div", { class: "kv" }, [el("span", { class: "k", text: "Environment" }), el("span", { text: info.environment })]),
        el("div", { class: "kv" }, [el("span", { class: "k", text: "Python" }), el("span", { text: info.runtime.python })]),
      ]));

      const rows = Object.entries(info.engines || {}).map(([name, data]) => el("tr", {}, [
        el("td", { text: name }),
        el("td", { text: JSON.stringify(data).slice(0, 130) }),
      ]));
      box.appendChild(card("Engine", el("table", {}, [
        el("thead", {}, [el("tr", {}, [el("th", { text: "Modul" }), el("th", { text: "Backend / status" })])]),
        el("tbody", {}, rows),
      ])));

      const artifactsCard = card("Model terlatih", el("table", {}, [
        el("thead", {}, [el("tr", {}, ["Task", "Status", "Versi", "Val. accuracy"].map((h) => el("th", { text: h })))]),
        el("tbody", {}, Object.entries(artifacts).map(([task, data]) => el("tr", {}, [
          el("td", { text: task }),
          el("td", {}, [el("span", {
            class: data.trained ? "chip pos" : "chip neu",
            text: data.trained ? "terlatih" : "belum dilatih (fallback aturan)",
          })]),
          el("td", { text: data.version || "—" }),
          el("td", { text: data.validation_accuracy != null ? (data.validation_accuracy * 100).toFixed(1) + "%" : "—" }),
        ]))),
      ]));
      box.appendChild(artifactsCard);

      const train = info.training || {};
      box.appendChild(card("Pipeline training", [
        el("p", { class: "muted", text: "Jalankan: python -m scripts.train --task all" }),
        el("div", { class: "kv" }, [el("span", { class: "k", text: "Intent model" }), el("span", { text: String(train.intent_model_trained) })]),
        el("div", { class: "kv" }, [el("span", { class: "k", text: "Sentiment model" }), el("span", { text: String(train.sentiment_model_trained) })]),
        el("div", { class: "kv" }, [el("span", { class: "k", text: "Classification model" }), el("span", { text: String(train.classification_model_trained) })]),
      ]));

      box.appendChild(card("Detail JSON", el("pre", { text: JSON.stringify(info, null, 2) })));
    } catch (err) {
      box.appendChild(card("Error", el("p", { text: err.message })));
    }
  }

  $("#btn-refresh-models").addEventListener("click", loadModels);

  /* ------------------------------------------------------------------ boot */
  checkHealth();
  setInterval(checkHealth, 45000);
  bootChat();
})();
