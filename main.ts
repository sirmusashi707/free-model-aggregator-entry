// main.ts -- free-model-aggregator-01 entry.
//
// One server that serves the OpenAI Chat Completions API and the Anthropic
// Messages API over the ten simulated free-tier providers of the trial,
// routing each request to the best free model that meets its requirements
// and failing over to the next-best model when a provider refuses.
//
// Node 22, standard library only: no dependencies, no model calls of its
// own, no network but the trial env. Every choice is a function of the
// brief, the requests seen so far and the providers' answers: no wall
// clock and no unseeded randomness, so a replay leaves the same export.
//
// Usage:
//   node main.ts          start the server, register with the env, serve
//                         the workload, exit 0 when the env says it is done.
//   node main.ts --list   print the catalog document and exit 0.

import { createServer } from "node:http";
import type { IncomingMessage, ServerResponse } from "node:http";
import { readFileSync } from "node:fs";

const FIXED_CREATED = 1767225600;
const PROTOCOL_VERSION = 1;
const ANTHROPIC_VERSION = "2023-06-01";
const CHUNK_CHARS = 13;
const MAX_BODY_BYTES = 2 * 1024 * 1024;
const PROVIDER_TIMEOUT_MS = 15000;

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

type JsonRecord = { [key: string]: unknown };

function isRecord(v: unknown): v is JsonRecord {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

function numOrNull(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function stringList(v: unknown): string[] {
  if (typeof v === "string") return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === "string");
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ---------------------------------------------------------------------------
// The brief: providers, quality table, workload manifest
// ---------------------------------------------------------------------------

type Wire = "openai" | "anthropic" | "gemini";
type Auth = "bearer" | "x-api-key" | "x-goog-api-key" | "none";

interface ProviderDir {
  id: string;
  name: string;
  wire: Wire;
  auth: Auth;
  key_env: string;
  key: string;
  path: string;
}

interface Rating {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface Requires {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

interface ManifestEntry {
  id: string;
  api: "openai" | "anthropic";
  stream: boolean;
  weight: number;
  requires: Requires;
}

interface Brief {
  providers: ProviderDir[];
  ratings: Rating[];
  manifest: ManifestEntry[];
}

// The brief is prose with three JSON documents embedded in it. Pull each
// document out by scanning for its schema marker and matching braces.
function extractBriefDocs(text: string): JsonRecord[] {
  const docs: JsonRecord[] = [];
  const marker = '{"schema":"free-model-aggregator-01/';
  let at = 0;
  for (;;) {
    const start = text.indexOf(marker, at);
    if (start === -1) break;
    let depth = 0;
    let inStr = false;
    let esc = false;
    let end = -1;
    for (let i = start; i < text.length; i++) {
      const ch = text[i];
      if (inStr) {
        if (esc) esc = false;
        else if (ch === "\\") esc = true;
        else if (ch === '"') inStr = false;
        continue;
      }
      if (ch === '"') {
        inStr = true;
        continue;
      }
      if (ch === "{") depth += 1;
      else if (ch === "}") {
        depth -= 1;
        if (depth === 0) {
          end = i;
          break;
        }
      }
    }
    if (end === -1) break;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value)) docs.push(value);
    } catch {
      // A malformed embedded document is skipped; the other two still load.
    }
    at = end + 1;
  }
  return docs;
}

function parseBrief(text: string): Brief | null {
  let providers: ProviderDir[] = [];
  let ratings: Rating[] = [];
  let manifest: ManifestEntry[] = [];
  for (const doc of extractBriefDocs(text)) {
    if (doc.schema === "free-model-aggregator-01/providers/v1" && Array.isArray(doc.providers)) {
      providers = doc.providers as ProviderDir[];
    } else if (doc.schema === "free-model-aggregator-01/ratings/v1" && Array.isArray(doc.ratings)) {
      ratings = doc.ratings as Rating[];
    } else if (doc.schema === "free-model-aggregator-01/workload/v1" && Array.isArray(doc.manifest)) {
      manifest = doc.manifest as ManifestEntry[];
    }
  }
  if (providers.length === 0) return null;
  return { providers, ratings, manifest };
}

// Fallback when no brief file is readable: the env's /instance route hands
// out the same provider directory and workload manifest (without the sealed
// quality table, so ratings stay empty and routing falls back to catalog
// order). Only used if MAKEFUN_PROMPT_FILE cannot be read.
async function briefFromInstance(envUrl: string): Promise<Brief | null> {
  try {
    const res = await fetch(`${envUrl}/instance`, { signal: AbortSignal.timeout(8000) });
    if (!res.ok) return null;
    const value: unknown = await res.json();
    if (!isRecord(value) || !Array.isArray(value.providers)) return null;
    const providers: ProviderDir[] = [];
    for (const raw of value.providers) {
      if (!isRecord(raw) || typeof raw.id !== "string") continue;
      let path = typeof raw.base_url === "string" ? raw.base_url : "";
      try {
        path = new URL(path).pathname;
      } catch {
        // base_url is already a path.
      }
      providers.push({
        id: raw.id,
        name: typeof raw.name === "string" ? raw.name : raw.id,
        wire: raw.wire === "anthropic" || raw.wire === "gemini" ? raw.wire : "openai",
        auth:
          raw.auth === "x-api-key" || raw.auth === "x-goog-api-key" || raw.auth === "none"
            ? raw.auth
            : "bearer",
        key_env: typeof raw.key_env === "string" ? raw.key_env : "",
        key: typeof raw.key === "string" ? raw.key : "",
        path,
      });
    }
    let manifest: ManifestEntry[] = [];
    if (isRecord(value.workload) && Array.isArray(value.workload.manifest)) {
      manifest = value.workload.manifest as ManifestEntry[];
    }
    if (providers.length === 0) return null;
    return { providers, ratings: [], manifest };
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// The catalog: probed from each provider's own model list
// ---------------------------------------------------------------------------

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

interface Catalog {
  providers: CatalogProvider[];
}

function authHeaders(p: ProviderDir): Record<string, string> {
  const headers: Record<string, string> = {};
  if (p.auth === "bearer") headers["authorization"] = `Bearer ${p.key}`;
  else if (p.auth === "x-api-key") headers["x-api-key"] = p.key;
  else if (p.auth === "x-goog-api-key") headers["x-goog-api-key"] = p.key;
  return headers;
}

// One provider's model list, straight from its catalog route. The catalog
// route never spends the provider's trial budget.
async function probeProvider(envUrl: string, p: ProviderDir): Promise<CatalogModel[]> {
  const base = `${envUrl}${p.path}`;
  const url = p.wire === "gemini" ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers = authHeaders(p);
  if (p.wire === "anthropic") headers["anthropic-version"] = ANTHROPIC_VERSION;
  const res = await fetch(url, { headers, signal: AbortSignal.timeout(8000) });
  if (!res.ok) throw new Error(`catalog probe for ${p.id} answered HTTP ${res.status}`);
  const body: unknown = await res.json();
  if (!isRecord(body)) throw new Error(`catalog probe for ${p.id} answered a non-object`);
  const out: CatalogModel[] = [];
  if (p.wire === "gemini") {
    const list = Array.isArray(body.models) ? body.models : [];
    for (const m of list) {
      if (!isRecord(m) || typeof m.name !== "string") continue;
      const id = m.name.startsWith("models/") ? m.name.slice("models/".length) : m.name;
      if (id === "") continue;
      out.push({
        id,
        free: m.free === true,
        context_length: numOrNull(m.context_length) ?? numOrNull(m.inputTokenLimit),
      });
    }
    return out;
  }
  const list = Array.isArray(body.data) ? body.data : [];
  for (const m of list) {
    if (!isRecord(m) || typeof m.id !== "string" || m.id === "") continue;
    out.push({ id: m.id, free: m.free === true, context_length: numOrNull(m.context_length) });
  }
  return out;
}

async function buildCatalog(brief: Brief | null, envUrl: string): Promise<Catalog> {
  if (brief === null) return { providers: [] };
  const providers: CatalogProvider[] = [];
  for (const p of brief.providers) {
    let models: CatalogModel[] | null = null;
    if (envUrl !== "") {
      for (let attempt = 0; attempt < 2 && models === null; attempt++) {
        try {
          models = await probeProvider(envUrl, p);
        } catch {
          models = null;
        }
      }
    }
    if (models === null) {
      // No probe answer: fall back to the brief's quality table, which lists
      // every free model with its context length (paid models unknown here).
      models = brief.ratings
        .filter((r) => r.provider === p.id && r.free)
        .map((r) => ({ id: r.model, free: true, context_length: r.context_length ?? null }));
    } else {
      // Fill a missing context length from the quality table.
      for (const m of models) {
        if (m.context_length === null || m.context_length === undefined) {
          const row = brief.ratings.find((r) => r.provider === p.id && r.model === m.id);
          if (row !== undefined) m.context_length = row.context_length ?? null;
        }
      }
    }
    providers.push({ id: p.id, base_url: p.path, key_env: p.key_env, wire: p.wire, models });
  }
  return { providers };
}

// ---------------------------------------------------------------------------
// The request, normalized: one shape for both client wires
// ---------------------------------------------------------------------------

type ImagePart = { kind: "data"; mediaType: string; data: string } | { kind: "url"; url: string };

interface UserMessage {
  text: string;
  images: ImagePart[];
}

interface ToolDef {
  name: string;
  description: string;
  parameters: JsonRecord;
}

interface Normalized {
  systemText: string;
  userMessages: UserMessage[];
  tools: ToolDef[];
  maxTokens: number | null;
  stops: string[];
  jsonMode: boolean;
  temperature: number | undefined;
  stream: boolean;
}

// Text of a message content: a plain string, or the text blocks of an array.
function textOfContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === "text" && typeof block.text === "string") parts.push(block.text);
  }
  return parts.join("\n");
}

function imageFromUrl(url: string): ImagePart {
  const match = /^data:([^,]*);base64,(.*)$/s.exec(url);
  if (match !== null) {
    const mediaType = match[1].split(";")[0] || "image/png";
    return { kind: "data", mediaType, data: match[2] };
  }
  return { kind: "url", url };
}

function imageToDataUrl(img: ImagePart): string {
  return img.kind === "data" ? `data:${img.mediaType};base64,${img.data}` : img.url;
}

function imagesOfOpenAIContent(content: unknown): ImagePart[] {
  const out: ImagePart[] = [];
  if (!Array.isArray(content)) return out;
  for (const block of content) {
    if (!isRecord(block) || block.type !== "image_url" || !isRecord(block.image_url)) continue;
    const url = block.image_url.url;
    if (typeof url === "string" && url !== "") out.push(imageFromUrl(url));
  }
  return out;
}

function normalizeOpenAI(body: JsonRecord): Normalized {
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  const systemParts: string[] = [];
  const userMessages: UserMessage[] = [];
  for (const m of messages) {
    if (m.role === "system") {
      const text = textOfContent(m.content);
      if (text !== "") systemParts.push(text);
    } else if (m.role === "user") {
      userMessages.push({ text: textOfContent(m.content), images: imagesOfOpenAIContent(m.content) });
    }
  }
  const tools: ToolDef[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || !isRecord(tool.function) || typeof tool.function.name !== "string") continue;
      tools.push({
        name: tool.function.name,
        description: typeof tool.function.description === "string" ? tool.function.description : "",
        parameters: isRecord(tool.function.parameters)
          ? tool.function.parameters
          : { type: "object", properties: {} },
      });
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    systemText: systemParts.join("\n"),
    userMessages,
    tools,
    maxTokens: numOrNull(body.max_tokens) ?? numOrNull(body.max_completion_tokens),
    stops: stringList(body.stop),
    jsonMode: format === "json_object" || format === "json_schema",
    temperature: typeof body.temperature === "number" ? body.temperature : undefined,
    stream: body.stream === true,
  };
}

function normalizeAnthropic(body: JsonRecord): Normalized {
  const systemText = textOfContent(body.system);
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  const userMessages: UserMessage[] = [];
  for (const m of messages) {
    if (m.role !== "user") continue;
    const images: ImagePart[] = [];
    if (Array.isArray(m.content)) {
      for (const block of m.content) {
        if (!isRecord(block) || block.type !== "image" || !isRecord(block.source)) continue;
        if (block.source.type === "base64" && typeof block.source.data === "string") {
          images.push({
            kind: "data",
            mediaType: typeof block.source.media_type === "string" ? block.source.media_type : "image/png",
            data: block.source.data,
          });
        } else if (block.source.type === "url" && typeof block.source.url === "string") {
          images.push({ kind: "url", url: block.source.url });
        }
      }
    }
    userMessages.push({ text: textOfContent(m.content), images });
  }
  const tools: ToolDef[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || typeof tool.name !== "string") continue;
      tools.push({
        name: tool.name,
        description: typeof tool.description === "string" ? tool.description : "",
        parameters: isRecord(tool.input_schema) ? tool.input_schema : { type: "object", properties: {} },
      });
    }
  }
  return {
    systemText,
    userMessages,
    tools,
    maxTokens: numOrNull(body.max_tokens),
    stops: stringList(body.stop_sequences),
    jsonMode: false,
    temperature: typeof body.temperature === "number" ? body.temperature : undefined,
    stream: body.stream === true,
  };
}

// ---------------------------------------------------------------------------
// The request, translated into a provider's wire
// ---------------------------------------------------------------------------

function openAIUserContent(um: UserMessage): unknown[] {
  const content: unknown[] = [];
  if (um.text !== "" || um.images.length === 0) content.push({ type: "text", text: um.text });
  for (const img of um.images) {
    content.push({ type: "image_url", image_url: { url: imageToDataUrl(img) } });
  }
  return content;
}

function buildOpenAIRequest(n: Normalized, model: string, stream: boolean, needsJson: boolean): JsonRecord {
  const messages: unknown[] = [];
  if (n.systemText !== "") messages.push({ role: "system", content: n.systemText });
  for (const um of n.userMessages) messages.push({ role: "user", content: openAIUserContent(um) });
  const body: JsonRecord = { model, messages, stream };
  if (n.maxTokens !== null) body.max_tokens = n.maxTokens;
  if (n.stops.length > 0) body.stop = n.stops;
  if (n.tools.length > 0) {
    body.tools = n.tools.map((t) => ({
      type: "function",
      function: { name: t.name, description: t.description, parameters: t.parameters },
    }));
    body.tool_choice = "auto";
  }
  if (needsJson) body.response_format = { type: "json_object" };
  if (n.temperature !== undefined) body.temperature = n.temperature;
  if (stream) body.stream_options = { include_usage: true };
  return body;
}

function anthropicUserContent(um: UserMessage): unknown[] {
  const content: unknown[] = [];
  if (um.text !== "" || um.images.length === 0) content.push({ type: "text", text: um.text });
  for (const img of um.images) {
    if (img.kind === "data") {
      content.push({ type: "image", source: { type: "base64", media_type: img.mediaType, data: img.data } });
    } else {
      content.push({ type: "image", source: { type: "url", url: img.url } });
    }
  }
  return content;
}

function buildAnthropicRequest(n: Normalized, model: string, stream: boolean): JsonRecord {
  const body: JsonRecord = { model, stream };
  if (n.systemText !== "") body.system = n.systemText;
  body.messages = n.userMessages.map((um) => ({ role: "user", content: anthropicUserContent(um) }));
  if (n.maxTokens !== null) body.max_tokens = n.maxTokens;
  if (n.tools.length > 0) {
    body.tools = n.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
  }
  if (n.stops.length > 0) body.stop_sequences = n.stops;
  if (n.temperature !== undefined) body.temperature = n.temperature;
  return body;
}

function geminiUserParts(um: UserMessage): unknown[] {
  const parts: unknown[] = [];
  if (um.text !== "" || um.images.length === 0) parts.push({ text: um.text });
  for (const img of um.images) {
    if (img.kind === "data") parts.push({ inlineData: { mimeType: img.mediaType, data: img.data } });
    else parts.push({ fileData: { fileUri: img.url } });
  }
  return parts;
}

function buildGeminiRequest(n: Normalized, needsJson: boolean): JsonRecord {
  const body: JsonRecord = {};
  if (n.systemText !== "") body.systemInstruction = { parts: [{ text: n.systemText }] };
  body.contents = n.userMessages.map((um) => ({ role: "user", parts: geminiUserParts(um) }));
  if (n.tools.length > 0) {
    body.tools = [
      {
        functionDeclarations: n.tools.map((t) => ({
          name: t.name,
          description: t.description,
          parameters: t.parameters,
        })),
      },
    ];
  }
  const config: JsonRecord = {};
  if (n.maxTokens !== null) config.maxOutputTokens = n.maxTokens;
  if (n.stops.length > 0) config.stopSequences = n.stops;
  if (needsJson) config.responseMimeType = "application/json";
  if (n.temperature !== undefined) config.temperature = n.temperature;
  if (Object.keys(config).length > 0) body.generationConfig = config;
  return body;
}

// ---------------------------------------------------------------------------
// A provider's answer, read back into one shape
// ---------------------------------------------------------------------------

interface ProviderAnswer {
  text: string;
  input: number;
  output: number;
}

function parseSse(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  const normalized = text.replace(/\r\n/g, "\n");
  for (const frame of normalized.split("\n\n")) {
    if (frame.trim() === "") continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split("\n")) {
      if (line.startsWith("event:")) event = line.slice("event:".length).trim();
      else if (line.startsWith("data:")) data.push(line.slice("data:".length).replace(/^ /, ""));
    }
    if (data.length > 0) frames.push({ event, data: data.join("\n") });
  }
  return frames;
}

function parseJsonAnswer(wire: Wire, text: string): ProviderAnswer | null {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    return null;
  }
  if (!isRecord(value)) return null;
  if (wire === "openai") {
    const choices = Array.isArray(value.choices) ? value.choices : [];
    const first = isRecord(choices[0]) ? choices[0] : null;
    if (first === null || !isRecord(first.message)) return null;
    const content = first.message.content;
    const usage = isRecord(value.usage) ? value.usage : {};
    return {
      text: typeof content === "string" ? content : textOfContent(content),
      input: numOrNull(usage.prompt_tokens) ?? 0,
      output: numOrNull(usage.completion_tokens) ?? 0,
    };
  }
  if (wire === "anthropic") {
    if (!Array.isArray(value.content)) return null;
    const usage = isRecord(value.usage) ? value.usage : {};
    return {
      text: textOfContent(value.content),
      input: numOrNull(usage.input_tokens) ?? 0,
      output: numOrNull(usage.output_tokens) ?? 0,
    };
  }
  const candidates = Array.isArray(value.candidates) ? value.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : null;
  if (first === null || !isRecord(first.content) || !Array.isArray(first.content.parts)) return null;
  const answerText = first.content.parts
    .map((p) => (isRecord(p) && typeof p.text === "string" ? p.text : ""))
    .join("");
  const usage = isRecord(value.usageMetadata) ? value.usageMetadata : {};
  return {
    text: answerText,
    input: numOrNull(usage.promptTokenCount) ?? 0,
    output: numOrNull(usage.candidatesTokenCount) ?? 0,
  };
}

function parseStreamAnswer(wire: Wire, text: string): ProviderAnswer | null {
  const frames = parseSse(text);
  if (wire === "openai") {
    let answer = "";
    let input = 0;
    let output = 0;
    for (const frame of frames) {
      if (frame.data === "[DONE]") continue;
      let chunk: unknown;
      try {
        chunk = JSON.parse(frame.data);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        input = numOrNull(chunk.usage.prompt_tokens) ?? input;
        output = numOrNull(chunk.usage.completion_tokens) ?? output;
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const first = isRecord(choices[0]) ? choices[0] : null;
      if (first !== null && isRecord(first.delta) && typeof first.delta.content === "string") {
        answer += first.delta.content;
      }
    }
    return { text: answer, input, output };
  }
  if (wire === "anthropic") {
    let answer = "";
    let input = 0;
    let output = 0;
    for (const frame of frames) {
      let data: unknown;
      try {
        data = JSON.parse(frame.data);
      } catch {
        continue;
      }
      if (!isRecord(data)) continue;
      if (frame.event === "message_start" && isRecord(data.message) && isRecord(data.message.usage)) {
        input = numOrNull(data.message.usage.input_tokens) ?? input;
      } else if (
        frame.event === "content_block_delta" &&
        isRecord(data.delta) &&
        data.delta.type === "text_delta" &&
        typeof data.delta.text === "string"
      ) {
        answer += data.delta.text;
      } else if (frame.event === "message_delta" && isRecord(data.usage)) {
        output = numOrNull(data.usage.output_tokens) ?? output;
      }
    }
    return { text: answer, input, output };
  }
  let answer = "";
  let input = 0;
  let output = 0;
  for (const frame of frames) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    for (const candidate of candidates) {
      if (!isRecord(candidate) || !isRecord(candidate.content) || !Array.isArray(candidate.content.parts)) continue;
      for (const part of candidate.content.parts) {
        if (isRecord(part) && typeof part.text === "string") answer += part.text;
      }
    }
    if (isRecord(data.usageMetadata)) {
      input = numOrNull(data.usageMetadata.promptTokenCount) ?? input;
      output = numOrNull(data.usageMetadata.candidatesTokenCount) ?? output;
    }
  }
  return { text: answer, input, output };
}

// ---------------------------------------------------------------------------
// Calling a provider
// ---------------------------------------------------------------------------

type CallResult =
  | { ok: true; answer: ProviderAnswer }
  | { ok: false; status: number };

async function callProvider(
  envUrl: string,
  p: ProviderDir,
  model: string,
  n: Normalized,
  stream: boolean,
  needsJson: boolean,
): Promise<CallResult> {
  const base = `${envUrl}${p.path}`;
  const headers: Record<string, string> = { "content-type": "application/json", ...authHeaders(p) };
  let url: string;
  let body: JsonRecord;
  if (p.wire === "openai") {
    url = `${base}/v1/chat/completions`;
    body = buildOpenAIRequest(n, model, stream, needsJson);
  } else if (p.wire === "anthropic") {
    url = `${base}/v1/messages`;
    headers["anthropic-version"] = ANTHROPIC_VERSION;
    body = buildAnthropicRequest(n, model, stream);
  } else {
    const verb = stream ? ":streamGenerateContent" : ":generateContent";
    url = `${base}/v1beta/models/${encodeURIComponent(model)}${verb}?alt=sse`;
    body = buildGeminiRequest(n, needsJson);
  }
  let res: Response;
  try {
    res = await fetch(url, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
    });
  } catch {
    return { ok: false, status: 0 };
  }
  if (!res.ok) return { ok: false, status: res.status };
  const text = await res.text();
  const answer = stream ? parseStreamAnswer(p.wire, text) : parseJsonAnswer(p.wire, text);
  if (answer === null) return { ok: false, status: 0 };
  return { ok: true, answer };
}

// ---------------------------------------------------------------------------
// Picking a model: best free rating first, failing over down the list
// ---------------------------------------------------------------------------

interface Candidate {
  provider: ProviderDir;
  model: string;
  rating: number;
}

function rankCandidates(
  brief: Brief,
  req: Requires,
  clientApi: "openai" | "anthropic",
  needsJson: boolean,
): Candidate[] {
  const out: Candidate[] = [];
  for (const row of brief.ratings) {
    if (!row.free) continue;
    if (req.tools === true && !row.tools) continue;
    if (req.vision === true && !row.vision) continue;
    if (req.json === true && !row.json) continue;
    if (req.min_context !== undefined && (row.context_length ?? 0) < req.min_context) continue;
    const provider = brief.providers.find((p) => p.id === row.provider);
    if (provider === undefined) continue;
    // An Anthropic-wire provider cannot carry JSON mode, so an OpenAI
    // client asking for JSON must never be sent to one: the provider would
    // echo json_mode false and the request would fail its echo check.
    if (clientApi === "openai" && needsJson && provider.wire === "anthropic") continue;
    out.push({ provider, model: row.model, rating: row.rating_milli });
  }
  out.sort(
    (a, b) =>
      b.rating - a.rating ||
      (a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : 0) ||
      (a.model < b.model ? -1 : a.model > b.model ? 1 : 0),
  );
  return out;
}

// When nothing in the table meets the full requirements, relax them one at
// a time (context, then JSON, then vision, then tools) rather than failing
// the request outright: a served answer from a weaker free model still
// earns most of its quality.
function rankCandidatesRelaxed(
  brief: Brief,
  req: Requires,
  clientApi: "openai" | "anthropic",
  needsJson: boolean,
): Candidate[] {
  const relaxations: Requires[] = [
    req,
    { ...req, min_context: undefined },
    { ...req, min_context: undefined, json: undefined },
    { ...req, min_context: undefined, json: undefined, vision: undefined },
    {},
  ];
  for (const relaxed of relaxations) {
    const candidates = rankCandidates(
      brief,
      relaxed,
      clientApi,
      needsJson && relaxed.json === true,
    );
    if (candidates.length > 0) return candidates;
  }
  return [];
}

// When the brief carries no quality table (the /instance fallback), rank the
// catalog's free models by provider then model id: deterministic, and the
// relaxation path above is not needed.
function rankCatalogCandidates(catalog: Catalog, brief: Brief): Candidate[] {
  const out: Candidate[] = [];
  for (const entry of catalog.providers) {
    const provider = brief.providers.find((p) => p.id === entry.id);
    if (provider === undefined) continue;
    for (const m of entry.models) {
      if (m.free) out.push({ provider, model: m.id, rating: 0 });
    }
  }
  out.sort(
    (a, b) =>
      a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : 0 ||
      (a.model < b.model ? -1 : a.model > b.model ? 1 : 0),
  );
  return out;
}

interface ProviderState {
  consecutiveFailures: number;
  dead: boolean;
}

interface Served {
  provider: ProviderDir;
  model: string;
  answer: ProviderAnswer;
}

// Try the candidates in rating order; on a refusal move to the next one.
// A provider that refuses twice running is treated as spent for the rest
// of the trial (its budget is gone or it is in outage), and a second pass
// over the list gives a provider refused on its own failure schedule one
// more chance to answer, since that refusal is transient.
async function serveRequest(
  envUrl: string,
  brief: Brief,
  catalog: Catalog,
  states: Map<string, ProviderState>,
  n: Normalized,
  req: Requires,
  clientApi: "openai" | "anthropic",
  stream: boolean,
): Promise<Served | null> {
  const needsJson = req.json === true;
  const candidates =
    brief.ratings.length > 0
      ? rankCandidatesRelaxed(brief, req, clientApi, needsJson)
      : rankCatalogCandidates(catalog, brief);
  const tried = candidates.slice(0, 12);
  for (let pass = 0; pass < 3; pass++) {
    for (const candidate of tried) {
      const state = states.get(candidate.provider.id);
      if (state === undefined || state.dead) continue;
      const result = await callProvider(envUrl, candidate.provider, candidate.model, n, stream, needsJson);
      if (result.ok) {
        state.consecutiveFailures = 0;
        return { provider: candidate.provider, model: candidate.model, answer: result.answer };
      }
      state.consecutiveFailures += 1;
      if (state.consecutiveFailures >= 2) state.dead = true;
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// The answer, translated back into the wire the client used
// ---------------------------------------------------------------------------

function sseFrame(event: string | null, data: unknown): string {
  return `${event === null ? "" : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += CHUNK_CHARS) pieces.push(text.slice(at, at + CHUNK_CHARS));
  return pieces.length === 0 ? [""] : pieces;
}

function openAIJsonBody(model: string, answer: ProviderAnswer): string {
  return JSON.stringify({
    id: "chatcmpl-sim",
    object: "chat.completion",
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: answer.text },
        finish_reason: "stop",
        logprobs: null,
      },
    ],
    usage: {
      prompt_tokens: answer.input,
      completion_tokens: answer.output,
      total_tokens: answer.input + answer.output,
    },
  });
}

function openAIStreamBody(model: string, answer: ProviderAnswer): string {
  const chunk = (delta: unknown, finish: string | null) => ({
    id: "chatcmpl-sim",
    object: "chat.completion.chunk",
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseFrame(null, chunk({ role: "assistant", content: "" }, null))];
  for (const piece of splitText(answer.text)) frames.push(sseFrame(null, chunk({ content: piece }, null)));
  frames.push(sseFrame(null, chunk({}, "stop")));
  frames.push(
    sseFrame(null, {
      id: "chatcmpl-sim",
      object: "chat.completion.chunk",
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: {
        prompt_tokens: answer.input,
        completion_tokens: answer.output,
        total_tokens: answer.input + answer.output,
      },
    }),
  );
  frames.push("data: [DONE]\n\n");
  return frames.join("");
}

function anthropicJsonBody(model: string, answer: ProviderAnswer): string {
  return JSON.stringify({
    id: "msg_sim",
    type: "message",
    role: "assistant",
    model,
    content: [{ type: "text", text: answer.text }],
    stop_reason: "end_turn",
    stop_sequence: null,
    usage: { input_tokens: answer.input, output_tokens: answer.output },
  });
}

function anthropicStreamBody(model: string, answer: ProviderAnswer): string {
  const frames: string[] = [
    sseFrame("message_start", {
      type: "message_start",
      message: {
        id: "msg_sim",
        type: "message",
        role: "assistant",
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.input, output_tokens: 0 },
      },
    }),
    sseFrame("content_block_start", {
      type: "content_block_start",
      index: 0,
      content_block: { type: "text", text: "" },
    }),
  ];
  for (const piece of splitText(answer.text)) {
    frames.push(
      sseFrame("content_block_delta", {
        type: "content_block_delta",
        index: 0,
        delta: { type: "text_delta", text: piece },
      }),
    );
  }
  frames.push(
    sseFrame("content_block_stop", { type: "content_block_stop", index: 0 }),
    sseFrame("message_delta", {
      type: "message_delta",
      delta: { stop_reason: "end_turn", stop_sequence: null },
      usage: { output_tokens: answer.output },
    }),
    sseFrame("message_stop", { type: "message_stop" }),
  );
  return frames.join("");
}

// ---------------------------------------------------------------------------
// The server
// ---------------------------------------------------------------------------

interface ServerContext {
  envUrl: string;
  brief: Brief;
  catalog: Catalog;
  manifestById: Map<string, ManifestEntry>;
  handled: number;
  providerStates: Map<string, ProviderState>;
  // The free models for GET /v1/models, deduplicated, in catalog order.
  modelList: { id: string; provider: string; context_length: number | null }[];
}

function readBody(req: IncomingMessage): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    let tooLarge = false;
    req.on("data", (chunk: Buffer) => {
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        tooLarge = true;
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => {
      if (tooLarge) reject(new Error("request body too large"));
      else resolve(Buffer.concat(chunks));
    });
    req.on("error", reject);
  });
}

function sendJson(res: ServerResponse, status: number, value: unknown): void {
  const body = Buffer.from(`${JSON.stringify(value)}\n`);
  res.writeHead(status, { "content-type": "application/json", "content-length": String(body.length) });
  res.end(body);
}

function sendText(res: ServerResponse, status: number, contentType: string, body: string): void {
  const bytes = Buffer.from(body);
  res.writeHead(status, { "content-type": contentType, "content-length": String(bytes.length) });
  res.end(bytes);
}

function sendError(res: ServerResponse, api: "openai" | "anthropic", status: number, message: string): void {
  if (api === "openai") {
    sendJson(res, status, { error: { message, type: "api_error", param: null, code: "internal_error" } });
  } else {
    sendJson(res, status, { type: "error", error: { type: "api_error", message } });
  }
}

// Which manifest entry is being served: ask the env, fall back to counting
// the requests handled so far (the client sends them in manifest order).
async function currentEntry(ctx: ServerContext): Promise<ManifestEntry | null> {
  try {
    const res = await fetch(`${ctx.envUrl}/work`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      const value: unknown = await res.json();
      if (isRecord(value) && value.done !== true && typeof value.next === "string") {
        const entry = ctx.manifestById.get(value.next);
        if (entry !== undefined) return entry;
      }
    }
  } catch {
    // Fall through to the counter.
  }
  return ctx.brief.manifest[ctx.handled] ?? null;
}

async function handleCompletion(
  ctx: ServerContext,
  api: "openai" | "anthropic",
  req: IncomingMessage,
  res: ServerResponse,
): Promise<void> {
  let raw: Buffer;
  try {
    raw = await readBody(req);
  } catch {
    sendError(res, api, 413, "request body too large");
    return;
  }
  let body: unknown;
  try {
    body = JSON.parse(raw.toString("utf8"));
  } catch {
    sendError(res, api, 400, "request body is not one JSON object");
    return;
  }
  if (!isRecord(body)) {
    sendError(res, api, 400, "request body is not one JSON object");
    return;
  }
  const entry = await currentEntry(ctx);
  const normalized = api === "openai" ? normalizeOpenAI(body) : normalizeAnthropic(body);
  const stream = entry !== null ? entry.stream : normalized.stream;
  const requires: Requires = { ...(entry !== null ? entry.requires : {}) };
  if (normalized.tools.length > 0) requires.tools = true;
  if (normalized.userMessages.some((um) => um.images.length > 0)) requires.vision = true;
  if (normalized.jsonMode) requires.json = true;
  const served = await serveRequest(
    ctx.envUrl,
    ctx.brief,
    ctx.catalog,
    ctx.providerStates,
    normalized,
    requires,
    api,
    stream,
  );
  ctx.handled += 1;
  if (served === null) {
    sendError(res, api, 502, "no provider could serve this request");
    return;
  }
  if (api === "openai") {
    if (stream) sendText(res, 200, "text/event-stream; charset=utf-8", openAIStreamBody(served.model, served.answer));
    else sendJson(res, 200, JSON.parse(openAIJsonBody(served.model, served.answer)));
  } else {
    if (stream) sendText(res, 200, "text/event-stream; charset=utf-8", anthropicStreamBody(served.model, served.answer));
    else sendJson(res, 200, JSON.parse(anthropicJsonBody(served.model, served.answer)));
  }
}

function handleModels(ctx: ServerContext, req: IncomingMessage, res: ServerResponse): void {
  // The wire is the auth header: Authorization: Bearer is OpenAI, x-api-key
  // is Anthropic. Both shapes list exactly the free model ids registered.
  const authorization = req.headers["authorization"];
  const apiKey = req.headers["x-api-key"];
  const anthropic =
    (typeof apiKey === "string" && apiKey !== "") &&
    !(typeof authorization === "string" && authorization !== "");
  if (anthropic) {
    sendJson(res, 200, {
      data: ctx.modelList.map((m) => ({
        type: "model",
        id: m.id,
        display_name: m.id,
        created_at: "2026-01-01T00:00:00Z",
        free: true,
        context_length: m.context_length,
      })),
      has_more: false,
    });
  } else {
    sendJson(res, 200, {
      object: "list",
      data: ctx.modelList.map((m) => ({
        id: m.id,
        object: "model",
        created: FIXED_CREATED,
        owned_by: m.provider,
        free: true,
        context_length: m.context_length,
      })),
    });
  }
}

async function route(ctx: ServerContext, req: IncomingMessage, res: ServerResponse): Promise<void> {
  try {
    const url = new URL(req.url ?? "/", "http://entry");
    if (req.method === "GET" && url.pathname === "/v1/models") {
      handleModels(ctx, req, res);
      return;
    }
    if (req.method === "POST" && url.pathname === "/v1/chat/completions") {
      await handleCompletion(ctx, "openai", req, res);
      return;
    }
    if (req.method === "POST" && url.pathname === "/v1/messages") {
      await handleCompletion(ctx, "anthropic", req, res);
      return;
    }
    sendJson(res, 404, { error: { message: `no route ${req.method ?? "GET"} ${url.pathname}`, type: "invalid_request_error", param: null, code: "not_found" } });
  } catch (err) {
    const message = err instanceof Error ? err.message : "internal error";
    try {
      sendJson(res, 500, { error: { message, type: "api_error", param: null, code: "internal_error" } });
    } catch {
      // The response is already gone; nothing left to do.
    }
  }
}

// ---------------------------------------------------------------------------
// Startup: --list, register, serve, exit 0 when the workload is done
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  const listMode = process.argv.slice(2).includes("--list");
  const envUrl = (process.env.MAKEFUN_ENV_URL ?? "").replace(/\/$/, "");
  const promptFile = process.env.MAKEFUN_PROMPT_FILE ?? "";

  let brief: Brief | null = null;
  if (promptFile !== "") {
    try {
      brief = parseBrief(readFileSync(promptFile, "utf8"));
    } catch {
      brief = null;
    }
  }
  if (brief === null && envUrl !== "") brief = await briefFromInstance(envUrl);

  const catalog = await buildCatalog(brief, envUrl);

  if (listMode) {
    process.stdout.write(`${JSON.stringify(catalog)}\n`);
    return;
  }
  if (brief === null) {
    process.stderr.write("free-model-aggregator: no brief: set MAKEFUN_PROMPT_FILE\n");
    process.exit(1);
  }
  if (envUrl === "") {
    process.stderr.write("free-model-aggregator: no env: set MAKEFUN_ENV_URL\n");
    process.exit(1);
  }

  const modelList: ServerContext["modelList"] = [];
  const seenModels = new Set<string>();
  for (const entry of catalog.providers) {
    for (const m of entry.models) {
      if (!m.free || seenModels.has(m.id)) continue;
      seenModels.add(m.id);
      modelList.push({ id: m.id, provider: entry.id, context_length: m.context_length });
    }
  }
  const ctx: ServerContext = {
    envUrl,
    brief,
    catalog,
    manifestById: new Map(brief.manifest.map((m) => [m.id, m])),
    handled: 0,
    providerStates: new Map(
      brief.providers.map((p) => [p.id, { consecutiveFailures: 0, dead: false }]),
    ),
    modelList,
  };

  const server = createServer((req, res) => {
    void route(ctx, req, res);
  });
  await new Promise<void>((resolve) => {
    server.listen(0, "0.0.0.0", () => resolve());
  });
  const address = server.address();
  const port = typeof address === "object" && address !== null ? address.port : 0;
  if (port === 0) {
    process.stderr.write("free-model-aggregator: could not bind a port\n");
    process.exit(1);
  }

  let registered = false;
  for (let attempt = 0; attempt < 20 && !registered; attempt++) {
    try {
      const res = await fetch(`${envUrl}/register`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
        signal: AbortSignal.timeout(5000),
      });
      if (res.ok) {
        const value: unknown = await res.json();
        if (isRecord(value) && value.ok === true) registered = true;
      }
    } catch {
      // The env may not be listening yet; try again.
    }
    if (!registered) await sleep(500);
  }
  if (!registered) {
    process.stderr.write("free-model-aggregator: could not register with the env\n");
    process.exit(1);
  }

  // The client sends its requests one at a time; when the env says the
  // workload is done, the trial is over and this process must exit 0.
  const timer = setInterval(() => {
    void (async () => {
      try {
        const res = await fetch(`${envUrl}/work`, { signal: AbortSignal.timeout(3000) });
        if (!res.ok) return;
        const value: unknown = await res.json();
        if (isRecord(value) && value.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // Keep polling.
      }
    })();
  }, 200);
}

main().catch((err) => {
  process.stderr.write(`free-model-aggregator: ${err instanceof Error ? err.message : String(err)}\n`);
  process.exit(1);
});
