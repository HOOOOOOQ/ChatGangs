/**
 * ChatGangs JavaScript client example (Node >= 18 or any modern browser).
 *
 *   node examples/javascript_client.js            # Node
 *   <script type="module" src="javascript_client.js"></script>
 *
 * In a browser page served by ChatGangs itself, just use `const BASE_API = "";`
 * to call the same origin.
 */

const BASE_API = "http://localhost:8000";
const API_KEY = null; // set to your key when CHATGANGS_REQUIRE_API_KEY=true

function headers() {
  const result = { "Content-Type": "application/json" };
  if (API_KEY) result["X-API-Key"] = API_KEY;
  return result;
}

async function call(path, options = {}) {
  const response = await fetch(`${BASE_API}${path}`, {
    ...options,
    headers: { ...headers(), ...(options.headers || {}) },
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(`${response.status} ${body.error || response.statusText}`);
  }
  return body;
}

export const chatgangs = {
  health: () => call("/api/health"),
  modelInfo: () => call("/api/model/info"),
  metrics: () => call("/api/metrics"),

  chat: (message, sessionId = "js-example", extra = {}) =>
    call("/api/chat", {
      method: "POST",
      body: JSON.stringify({ message, session_id: sessionId, ...extra }),
    }),

  analyze: (text) =>
    call("/api/nlp/analyze", { method: "POST", body: JSON.stringify({ text }) }),

  tokenize: (text) =>
    call("/api/nlp/tokenize", { method: "POST", body: JSON.stringify({ text }) }),

  sentiment: (text) =>
    call("/api/nlp/sentiment", { method: "POST", body: JSON.stringify({ text }) }),

  intent: (text) =>
    call("/api/nlp/intent", { method: "POST", body: JSON.stringify({ text }) }),

  entities: (text) =>
    call("/api/nlp/entities", { method: "POST", body: JSON.stringify({ text }) }),

  qa: (question, topK = 3) =>
    call("/api/nlp/qa", {
      method: "POST",
      body: JSON.stringify({ question, top_k: topK }),
    }),

  similarity: (texts) =>
    call("/api/nlp/similarity", { method: "POST", body: JSON.stringify({ texts }) }),

  feedback: (sessionId, rating, comment) =>
    call("/api/feedback", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId, rating, comment }),
    }),

  history: (sessionId, limit = 20) =>
    call(`/api/memory/${encodeURIComponent(sessionId)}?limit=${limit}`),

  deleteHistory: (sessionId) =>
    call(`/api/memory/${encodeURIComponent(sessionId)}`, { method: "DELETE" }),
};

// ---------------------------------------------------------------------------
// Demo (runs only under Node)
// ---------------------------------------------------------------------------
if (typeof process !== "undefined" && process.versions?.node) {
  const session = `js-${Date.now()}`;

  const health = await chatgangs.health();
  console.log("health:", health.status, health.version);

  const first = await chatgangs.chat("Halo, apa saja kemampuanmu?", session);
  console.log("bot :", first.reply.slice(0, 120), "...");
  console.log("intent:", first.intent.name, first.intent.confidence);

  const followUp = await chatgangs.chat("jelaskan lebih detail", session);
  console.log("follow-up:", followUp.context.is_followup, "turns:", followUp.context.turn_count);

  const analysis = await chatgangs.analyze("Layanan bagus tapi pengiriman terlambat");
  console.log("sentiment:", analysis.sentiment.label, analysis.sentiment.score);
  console.log("tokens   :", analysis.tokens.slice(0, 8).map((t) => t.text));
  console.log("entities :", analysis.entities.map((e) => [e.text, e.label]));

  const answer = await chatgangs.qa("bagaimana cara memakai api");
  console.log("qa       :", answer.answered, (answer.answer || "-").slice(0, 120));

  await chatgangs.feedback(session, 5, "helpful");
  console.log("delete   :", await chatgangs.deleteHistory(session));
}
