// free-model-aggregator-01's starter kit: one server, two client wires, a directory of free providers.
//
// It uses the starter's provider discovery and wire helpers with rating-based routing, failover, and
// adapters for all three upstream protocols.
//
// WHAT A TRIAL GIVES YOU
//   A trial gives this server a brief and an env, and runs it as `node /bot/main.ts` from your
//   repository's root, with Node 22's standard library alone (nothing is installed; TypeScript runs
//   directly, and a relative import needs its .ts extension). The brief is at the path in
//   $MAKEFUN_PROMPT_FILE. It carries three JSON documents:
//     - the provider directory: ten simulated free-tier providers, each with its id, wire (openai,
//       anthropic or gemini), auth scheme, key environment variable, the key the service takes, and the
//       base path you join to $MAKEFUN_ENV_URL;
//     - the quality table: every model's public rating and what it can do (tools, vision, JSON mode,
//       context length), and whether it is free;
//     - the workload manifest: the twenty requests the client will send, in order, with each one's api,
//       whether it streams, its weight and the requirements a model must meet to serve it.
//   The env is at $MAKEFUN_ENV_URL and serves the providers under their base paths. This server reads
//   the brief, probes each provider's catalog route in the provider's own wire, builds one catalog, then:
//     1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//     2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//     3. serves the client's requests: GET /v1/models (answering in the shape of the auth header it got:
//        x-api-key means Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST
//        /v1/messages, each with a JSON body or with server-sent events;
//     4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}. A
//        process still running when the wall ends scores zero, so never wait for anything else.
//   `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the register
//   body's catalog must be that same document. Build it once, as this file does, and use the one object.
//
// The quality table supplies model ratings, free status, and capabilities. Routes are deterministic.

// FILES
//   main.ts            this server: the brief, the catalog, the routes, the register and the poll.
//   lib/providers.ts   the brief's types and parser, the auth headers, the catalog probe.
//   lib/wire.ts        Anthropic <-> Chat Completions translation and the SSE writers.
//   LICENSE            MIT: keep it or replace it with another OSI-approved license.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  sseEvent,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this baseline reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Route {
  probe: Probe;
  model: ProbedModel;
}

/** Best-rated free provider/model pairs that meet this request's requirements. */
function routesFor(requires: Requirement): Route[] {
  const candidates: (Route & { rating: number })[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      candidates.push({ probe, model, rating: row?.rating_milli ?? -1 });
    }
  }
  candidates.sort((a, b) => b.rating - a.rating || a.probe.provider.id.localeCompare(b.probe.provider.id) || a.model.id.localeCompare(b.model.id));
  return candidates;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST one request to the routed provider's Chat Completions route. */
async function callProvider(route: Route, body: Json): Promise<Response> {
  const base = joinBase(ENV_URL, route.probe.provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(route.probe.provider.auth, route.probe.provider.key),
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(`${base}/v1/chat/completions`, { method: 'POST', headers, body: JSON.stringify(body) });
}

function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const systemParts: string[] = [];
  const out: Json[] = [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system' || raw.role === 'developer') { if (typeof raw.content === 'string' && raw.content !== '') systemParts.push(raw.content); continue; }
    let content: unknown = raw.content;
    if (Array.isArray(content)) content = content.flatMap((p) => isRecord(p) && p.type === 'text' ? [{ type: 'text', text: p.text }] : isRecord(p) && p.type === 'image_url' && isRecord(p.image_url) && typeof p.image_url.url === 'string' && p.image_url.url.startsWith('data:') ? [{ type: 'image', source: { type: 'base64', media_type: p.image_url.url.slice(5, p.image_url.url.indexOf(';')), data: p.image_url.url.slice(p.image_url.url.indexOf(',') + 1) } }] : []);
    if (raw.role === 'tool') content = [{ type: 'tool_result', tool_use_id: raw.tool_call_id, content: raw.content }];
    if (raw.role === 'assistant' && Array.isArray(raw.tool_calls)) content = [...(typeof content === 'string' && content ? [{ type: 'text', text: content }] : []), ...raw.tool_calls.flatMap((c) => isRecord(c) && isRecord(c.function) ? [{ type: 'tool_use', id: c.id, name: c.function.name, input: (() => { try { return JSON.parse(String(c.function.arguments ?? '{}')); } catch { return {}; } })() }] : [])];
    out.push({ role: raw.role === 'assistant' ? 'assistant' : 'user', content: content ?? '' });
  }
  const tools = Array.isArray(body.tools) ? body.tools.flatMap((t) => isRecord(t) && isRecord(t.function) ? [{ name: t.function.name, description: t.function.description ?? '', input_schema: t.function.parameters ?? { type: 'object' } }] : []) : [];
  const req: Json = { model, max_tokens: typeof body.max_tokens === 'number' ? body.max_tokens : 1024, messages: out, stream: false };
  if (systemParts.length) req.system = systemParts.join('\n');
  if (typeof body.temperature === 'number') req.temperature = body.temperature;
  if (typeof body.top_p === 'number') req.top_p = body.top_p;
  if (Array.isArray(body.stop)) req.stop_sequences = body.stop;
  if (tools.length) req.tools = tools;
  if (body.tool_choice === 'required') req.tool_choice = { type: 'any' };
  else if (body.tool_choice === 'auto') req.tool_choice = { type: 'auto' };
  else if (isRecord(body.tool_choice) && isRecord(body.tool_choice.function)) req.tool_choice = { type: 'tool', name: body.tool_choice.function.name };
  return req;
}

function openAIToGemini(body: Json): Json {
  const contents: Json[] = []; const systemParts: Json[] = [];
  for (const raw of (Array.isArray(body.messages) ? body.messages : [])) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    const parts: Json[] = [];
    if (typeof raw.content === 'string') parts.push({ text: raw.content });
    else if (Array.isArray(raw.content)) for (const p of raw.content) {
      if (isRecord(p) && p.type === 'text') parts.push({ text: p.text });
      else if (isRecord(p) && p.type === 'image_url' && isRecord(p.image_url) && typeof p.image_url.url === 'string') {
        const m = p.image_url.url.match(/^data:([^;]+);base64,(.*)$/); if (m) parts.push({ inlineData: { mimeType: m[1], data: m[2] } });
      }
    }
    if (role === 'system' || role === 'developer') { systemParts.push(...parts); continue; }
    if (role === 'tool') { contents.push({ role: 'user', parts: [{ functionResponse: { name: String(raw.name ?? 'tool'), response: { content: raw.content } } }] }); continue; }
    if (role === 'assistant' && Array.isArray(raw.tool_calls)) for (const c of raw.tool_calls) if (isRecord(c) && isRecord(c.function)) { try { parts.push({ functionCall: { name: c.function.name, args: JSON.parse(String(c.function.arguments ?? '{}')) } }); } catch { parts.push({ functionCall: { name: c.function.name, args: {} } }); } }
    contents.push({ role: role === 'assistant' ? 'model' : 'user', parts });
  }
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Array.isArray(body.stop)) config.stopSequences = body.stop;
  if (isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) config.responseMimeType = 'application/json';
  const tools = Array.isArray(body.tools) ? body.tools.flatMap((t) => isRecord(t) && isRecord(t.function) ? [{ name: t.function.name, description: t.function.description ?? '', parameters: t.function.parameters ?? { type: 'object' } }] : []) : [];
  return { contents, ...(systemParts.length ? { systemInstruction: { parts: systemParts } } : {}), generationConfig: config, ...(tools.length ? { tools: [{ functionDeclarations: tools }] } : {}) };
}

function geminiToOpenAI(data: unknown, model: string): Json {
  const root = isRecord(data) ? data : {};
  const candidate = Array.isArray(root.candidates) && isRecord(root.candidates[0]) ? root.candidates[0] : {};
  const c = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(c.parts) ? c.parts : [];
  let text = ''; const calls: Json[] = [];
  for (const p of parts) if (isRecord(p)) {
    if (typeof p.text === 'string') text += p.text;
    if (isRecord(p.functionCall)) calls.push({ id: `call_${calls.length}`, type: 'function', function: { name: p.functionCall.name ?? '', arguments: JSON.stringify(isRecord(p.functionCall.args) ? p.functionCall.args : {}) } });
  }
  const usage = isRecord(root.usageMetadata) ? root.usageMetadata : {};
  const reason = candidate.finishReason === 'MAX_TOKENS' ? 'length' : calls.length ? 'tool_calls' : 'stop';
  return { id: typeof root.responseId === 'string' ? root.responseId : 'gemini-response', object: 'chat.completion', model, choices: [{ index: 0, message: { role: 'assistant', content: text || null, ...(calls.length ? { tool_calls: calls } : {}) }, finish_reason: reason }], usage: { prompt_tokens: usage.promptTokenCount ?? 0, completion_tokens: usage.candidatesTokenCount ?? 0, total_tokens: (Number(usage.promptTokenCount) || 0) + (Number(usage.candidatesTokenCount) || 0) } };
}

async function upstreamJson(route: Route, body: Json, api: 'openai' | 'anthropic'): Promise<{ response: Response; answer?: Json }> {
  const p = route.probe.provider, base = joinBase(ENV_URL, p.base);
  if (p.wire === 'openai') {
    const reqBody = api === 'anthropic' ? anthropicToOpenAI(body, route.model.id) : { ...body, model: route.model.id };
    reqBody.stream = false; delete reqBody.stream_options;
    const response = await callProvider(route, reqBody);
    if (!response.ok) return { response };
    const value: unknown = await response.json(); return { response, answer: isRecord(value) ? value : {} };
  }
  if (p.wire === 'anthropic') {
    const reqBody = api === 'anthropic' ? { ...body, model: route.model.id, stream: false } : openAIToAnthropicRequest(body, route.model.id);
    const headers = { 'content-type': 'application/json', ...authHeaders(p.auth, p.key), 'anthropic-version': '2023-06-01' };
    const response = await fetch(`${base}/v1/messages`, { method: 'POST', headers, body: JSON.stringify(reqBody) });
    if (!response.ok) return { response };
    const value: unknown = await response.json();
    if (!isRecord(value)) return { response, answer: {} };
    const blocks = Array.isArray(value.content) ? value.content : [];
    const text = blocks.flatMap((b) => isRecord(b) && b.type === 'text' ? [String(b.text ?? '')] : []).join('');
    const calls = blocks.flatMap((b) => isRecord(b) && b.type === 'tool_use' ? [{ id: b.id, type: 'function', function: { name: b.name, arguments: JSON.stringify(isRecord(b.input) ? b.input : {}) } }] : []);
    return { response, answer: { id: value.id ?? 'anthropic-response', model: route.model.id, choices: [{ message: { role: 'assistant', content: text || null, ...(calls.length ? { tool_calls: calls } : {}) }, finish_reason: value.stop_reason === 'max_tokens' ? 'length' : calls.length ? 'tool_calls' : 'stop' }], usage: { prompt_tokens: isRecord(value.usage) ? value.usage.input_tokens ?? 0 : 0, completion_tokens: isRecord(value.usage) ? value.usage.output_tokens ?? 0 : 0 } } };
  }
  const reqBody = openAIToGemini(api === 'anthropic' ? anthropicToOpenAI(body, route.model.id) : body);
  const headers = { 'content-type': 'application/json', ...authHeaders(p.auth, p.key) };
  const response = await fetch(`${base}/v1beta/models/${encodeURIComponent(route.model.id)}:generateContent`, { method: 'POST', headers, body: JSON.stringify(reqBody) });
  if (!response.ok) return { response };
  return { response, answer: geminiToOpenAI(await response.json(), route.model.id) };
}

function openAIStream(answer: Json): string {
  const choices = Array.isArray(answer.choices) ? answer.choices : [];
  const c = isRecord(choices[0]) ? choices[0] : {}, msg = isRecord(c.message) ? c.message : {};
  const chunks: Json[] = [{ id: answer.id ?? 'chatcmpl-aggregate', object: 'chat.completion.chunk', model: answer.model ?? '', choices: [{ index: 0, delta: { role: 'assistant', ...(typeof msg.content === 'string' ? { content: msg.content } : {}), ...(Array.isArray(msg.tool_calls) ? { tool_calls: msg.tool_calls.map((t, i) => isRecord(t) ? { ...t, index: i } : t) } : {}) }, finish_reason: null }] }, { id: answer.id ?? 'chatcmpl-aggregate', object: 'chat.completion.chunk', model: answer.model ?? '', choices: [{ index: 0, delta: {}, finish_reason: c.finish_reason ?? 'stop' }] }];
  if (isRecord(answer.usage)) chunks.push({ id: answer.id ?? 'chatcmpl-aggregate', object: 'chat.completion.chunk', model: answer.model ?? '', choices: [], usage: answer.usage });
  return chunks.map((x) => `data: ${JSON.stringify(x)}\n\n`).join('') + 'data: [DONE]\n\n';
}

function readable(text: string): ReadableStream<Uint8Array> { return new ReadableStream({ start(c) { c.enqueue(new TextEncoder().encode(text)); c.close(); } }); }

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by an openai-wire provider. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const candidates = routesFor(requirementsFor(body)).filter((r) => !(r.probe.provider.wire === 'anthropic' && isRecord(body.response_format) && ['json_object', 'json_schema'].includes(String(body.response_format.type))));
  if (!candidates.length) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory meets this request');
  let answer: Json | null = null, lastStatus = 502, lastBody = '';
  const failedProviders = new Set<string>();
  for (const route of candidates) {
    if (failedProviders.has(route.probe.provider.id)) continue;
    try {
      const result = await upstreamJson(route, body, 'openai');
      if (!result.response.ok) { lastStatus = result.response.status; lastBody = (await result.response.text()).slice(0, 400); if ([402, 429, 500, 502, 503, 504].includes(lastStatus)) { failedProviders.add(route.probe.provider.id); continue; } break; }
      answer = result.answer ?? {}; break;
    } catch (err) { lastBody = (err as Error).message; lastStatus = 502; failedProviders.add(route.probe.provider.id); }
  }
  if (answer === null) return sendError(res, 'openai', lastStatus, 'upstream_error', lastBody || 'all eligible providers failed');
  if (body.stream === true) { res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' }); res.end(openAIStream(answer)); }
  else sendJson(res, 200, answer);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to an openai-wire provider and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const candidates = routesFor(requirementsFor(body));
  if (!candidates.length) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory meets this request');
  let answer: Json | null = null, routeUsed: Route | null = null, lastStatus = 502, lastBody = '';
  const failedProviders = new Set<string>();
  for (const route of candidates) {
    if (failedProviders.has(route.probe.provider.id)) continue;
    try {
      const result = await upstreamJson(route, body, 'anthropic');
      if (!result.response.ok) { lastStatus = result.response.status; lastBody = (await result.response.text()).slice(0, 400); if ([402, 429, 500, 502, 503, 504].includes(lastStatus)) { failedProviders.add(route.probe.provider.id); continue; } break; }
      answer = result.answer ?? {}; routeUsed = route; break;
    } catch (err) { lastBody = (err as Error).message; lastStatus = 502; failedProviders.add(route.probe.provider.id); }
  }
  if (answer === null || routeUsed === null) return sendJson(res, lastStatus, anthropicError('api_error', lastBody || 'all eligible providers failed'));
  const requested = typeof body.model === 'string' && body.model !== '' ? body.model : routeUsed.model.id;
  if (body.stream === true) {
    const sse = openAIStream(answer);
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    for await (const event of anthropicEventsFromOpenAI(readable(sse), requested)) res.write(event);
    res.end();
  } else sendJson(res, 200, openAIToAnthropic(answer, requested));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
      else res.destroy();
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
