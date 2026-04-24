import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { QdrantClient } from '@qdrant/js-client-rest';

type AnyJson = Record<string, unknown>;

type AgentFiles = {
  cfg: AnyJson;
  systemMd: string;
  templateMd: string;
  domainSummaryPromptMd: string;
  selfReviewPromptMd: string;
  domainRepairPromptMd: string;
  contractRepairPromptMd: string;
  rubricMd: string;
  outputContracts: AnyJson;
};

type EvidenceHit = {
  score: number;
  collection: string;
  specialty: string;
  doc: string;
  chunkId: string;
  level: string;
  text: string;
};

function nowIso(): string {
  return new Date().toISOString().replace(/\.\d{3}Z$/, 'Z');
}

function log(message: string): void {
  // eslint-disable-next-line no-console
  console.log(`[gertrudes] ${nowIso()} ${message}`);
}

async function readText(filePath: string): Promise<string> {
  return readFile(filePath, 'utf8');
}

async function writeText(filePath: string, content: string): Promise<void> {
  await mkdir(path.dirname(filePath), { recursive: true });
  await writeFile(filePath, content, 'utf8');
}

async function readJson<T = unknown>(filePath: string): Promise<T> {
  return JSON.parse(await readFile(filePath, 'utf8')) as T;
}

async function writeJson(filePath: string, obj: unknown): Promise<void> {
  await mkdir(path.dirname(filePath), { recursive: true });
  await writeFile(filePath, JSON.stringify(obj, null, 2), 'utf8');
}

function sha256Text(text: string): string {
  return createHash('sha256').update(text, 'utf8').digest('hex');
}

function normalizeWs(text: string): string {
  return text.replace(/\r\n/g, '\n').replace(/\r/g, '\n').replace(/[ \t]+/g, ' ').replace(/\n{3,}/g, '\n\n').trim();
}

function normalizeLlmText(value: unknown): string {
  if (value == null) return '';
  if (typeof value === 'string') return value;
  if (Array.isArray(value)) return value.filter((v) => v != null).map(String).join('\n');
  if (typeof value === 'object') return JSON.stringify(value, null, 2);
  return String(value);
}

async function loadAgentFiles(agentDir: string): Promise<AgentFiles> {
  return {
    cfg: await readJson(path.join(agentDir, 'agent.json')),
    systemMd: await readText(path.join(agentDir, 'system.md')),
    templateMd: await readText(path.join(agentDir, 'template_prompt.md')),
    domainSummaryPromptMd: await readText(path.join(agentDir, 'domain_summary_prompt.md')),
    selfReviewPromptMd: await readText(path.join(agentDir, 'self_review_prompt.md')),
    domainRepairPromptMd: await readText(path.join(agentDir, 'domain_repair_prompt.md')),
    contractRepairPromptMd: await readText(path.join(agentDir, 'contract_repair_prompt.md')),
    rubricMd: await readText(path.join(agentDir, 'rubric.md')),
    outputContracts: await readJson(path.join(agentDir, 'output_contracts.json')),
  };
}

async function loadProductInputs(productRoot: string, cfg: AnyJson): Promise<{ intentionMd: string; contextJson: AnyJson }> {
  const inputs = (cfg.inputs ?? {}) as AnyJson;
  const intentRel = String(inputs.intent_markdown ?? '0-intent/intention.md');
  const contextRel = String(inputs.context_json ?? '0-intent/context.json');

  const intentionMd = await readText(path.join(productRoot, intentRel));
  const contextJson = await readJson<AnyJson>(path.join(productRoot, contextRel));
  return { intentionMd, contextJson };
}

function computeInputHash(intentionMd: string, contextJson: AnyJson, agentCfg: AnyJson): string {
  return sha256Text(
    JSON.stringify({
      intention_md: intentionMd,
      context_json: contextJson,
      agent_version: agentCfg.version,
      agent_name: agentCfg.name,
    }),
  );
}

async function ollamaGenerate(baseUrl: string, model: string, system: string, prompt: string, timeout = 1200): Promise<string> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout * 1000);
  try {
    const response = await fetch(`${baseUrl.replace(/\/$/, '')}/api/generate`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        model,
        system,
        prompt,
        stream: false,
        options: { temperature: 0.1, num_predict: 4096 },
      }),
      signal: controller.signal,
    });
    if (!response.ok) {
      throw new Error(`Ollama generate failed: ${response.status}`);
    }
    const data = (await response.json()) as { response?: string };
    return data.response ?? '';
  } finally {
    clearTimeout(timer);
  }
}

async function ollamaEmbeddings(baseUrl: string, model: string, text: string, timeout = 600): Promise<number[]> {
  const base = baseUrl.replace(/\/$/, '');
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout * 1000);

  try {
    const first = await fetch(`${base}/api/embed`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model, input: text }),
      signal: controller.signal,
    });
    if (first.ok) {
      const j = (await first.json()) as { embeddings?: number[][]; embedding?: number[] };
      if (Array.isArray(j.embeddings) && j.embeddings.length > 0 && Array.isArray(j.embeddings[0])) return j.embeddings[0];
      if (Array.isArray(j.embedding) && j.embedding.length > 0) return j.embedding;
    }

    const second = await fetch(`${base}/api/embeddings`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model, prompt: text }),
      signal: controller.signal,
    });
    if (!second.ok) throw new Error(`Ollama embedding failed: ${second.status}`);
    const j2 = (await second.json()) as { embedding?: number[] };
    if (!Array.isArray(j2.embedding) || j2.embedding.length === 0) {
      throw new Error('Embedding inválido retornado pelo Ollama');
    }
    return j2.embedding;
  } finally {
    clearTimeout(timer);
  }
}

function qdrantClientFromEnv(): QdrantClient {
  return new QdrantClient({
    host: process.env.QDRANT_HOST ?? 'qdrant',
    port: Number(process.env.QDRANT_PORT ?? '6333'),
  });
}

async function qdrantSearch(qc: QdrantClient, collection: string, queryVector: number[], limit: number): Promise<EvidenceHit[]> {
  const res = await qc.search(collection, { vector: queryVector, limit, with_payload: true });
  return res.map((p) => {
    const payload = (p.payload ?? {}) as Record<string, unknown>;
    const text = String(payload.preview ?? payload.text_preview ?? payload.text ?? payload.chunk ?? '');
    return {
      score: Number(p.score ?? 0),
      collection,
      specialty: String(payload.specialty ?? ''),
      doc: String(payload.doc ?? payload.document ?? ''),
      chunkId: String(payload.chunk_id ?? payload.id ?? ''),
      level: String(payload.level ?? ''),
      text,
    };
  });
}

function buildQueryProfiles(product: string, intentionMd: string, contextJson: AnyJson, cfg: AnyJson): Array<[string, string]> {
  const queryProfiles = ((cfg.knowledge as AnyJson)?.query_profiles as Array<AnyJson> | undefined) ?? [];
  const contextStr = JSON.stringify(contextJson);
  const queries: Array<[string, string]> = [];

  for (const qp of queryProfiles) {
    const tpl = String(qp.query_template ?? '');
    const name = String(qp.name ?? 'generic');
    let q = tpl.replaceAll('{{product}}', product);
    q = q.replaceAll('{{intent}}', intentionMd);
    q = q.replaceAll('{{context}}', contextStr);
    queries.push([name, normalizeWs(q)]);
  }

  if (queries.length === 0) {
    queries.push(['generic', normalizeWs(`${product}\n${intentionMd}\n${contextStr}`)]);
  }
  return queries;
}

const GOOD_TERMS_DOMAIN = [
  'actor', 'actors', 'entity', 'entities', 'domain', 'workflow', 'flow',
  'integration', 'state', 'states', 'event', 'events', 'business', 'process',
  'authentication', 'authorization', 'payment', 'audit', 'trace', 'role',
  'user', 'users', 'requirement', 'requirements',
];

const GOOD_TERMS_NFR = [
  'non functional', 'security', 'privacy', 'reliability', 'availability',
  'audit', 'trace', 'observability', 'performance', 'compliance', 'idempotent',
  'idempotency', 'integrity', 'legal',
];

const BAD_TERMS = [
  'weather station', 'roads', 'road authorities', 'worldcom', 'enron',
  'html5', 'french', 'flemish', 'icebreaker', 'rosa weather', 'transportation',
];

function lexicalScore(text: string, goodTerms: string[], badTerms: string[]): number {
  const textLower = text.toLowerCase();
  let score = 0;
  for (const term of goodTerms) if (textLower.includes(term)) score += 2;
  for (const term of badTerms) if (textLower.includes(term)) score -= 5;
  return score;
}

function filterHits(hits: EvidenceHit[], kind: 'domain' | 'nfr' | 'synthesis'): EvidenceHit[] {
  return hits.filter((h) => {
    const text = normalizeWs(h.text);
    if (!text) return false;

    const score = kind === 'domain'
      ? lexicalScore(text, GOOD_TERMS_DOMAIN, BAD_TERMS)
      : kind === 'nfr'
        ? lexicalScore(text, GOOD_TERMS_NFR, BAD_TERMS)
        : lexicalScore(text, [...GOOD_TERMS_DOMAIN, ...GOOD_TERMS_NFR], BAD_TERMS);

    return score >= 0 || h.score >= 0.73;
  });
}

function dedupHits(hits: EvidenceHit[]): EvidenceHit[] {
  const seen = new Map<string, EvidenceHit>();
  for (const h of [...hits].sort((a, b) => b.score - a.score)) {
    const key = `${h.collection}|${h.doc}|${h.chunkId}`;
    if (!seen.has(key)) seen.set(key, h);
  }
  return [...seen.values()];
}

function renderEvidencePack(hits: EvidenceHit[], maxChars: number): string {
  const lines: string[] = [];
  let used = 0;
  for (const h of [...hits].sort((a, b) => b.score - a.score)) {
    const snippet = normalizeWs(h.text);
    if (!snippet) continue;
    const block = `- (${h.score.toFixed(3)}) [${h.specialty}] ${h.doc} :: ${h.chunkId}\n${snippet}`;
    if (used + block.length + 2 > maxChars) break;
    lines.push(block);
    used += block.length + 2;
  }
  return lines.join('\n\n').trim();
}

async function retrieveEvidence(
  product: string,
  intentionMd: string,
  contextJson: AnyJson,
  cfg: AnyJson,
  ollamaUrl: string,
  embedModel: string,
): Promise<{ hits: EvidenceHit[]; domainPack: string; synthesisPack: string }> {
  const qc = qdrantClientFromEnv();
  const knowledge = (cfg.knowledge ?? {}) as AnyJson;
  const collections = (knowledge.collections ?? {}) as AnyJson;
  const limits = (knowledge.limits ?? {}) as AnyJson;

  const primaryCoarse = ((collections.primary_coarse as string[]) ?? []);
  const primaryFine = ((collections.primary_fine as string[]) ?? []);
  const secondaryCoarse = ((collections.secondary_coarse as string[]) ?? []);
  const secondaryFine = ((collections.secondary_fine as string[]) ?? []);

  const topkPrimaryCoarse = Number(limits.topk_primary_coarse ?? 3);
  const topkPrimaryFine = Number(limits.topk_primary_fine ?? 5);
  const topkSecondaryCoarse = Number(limits.topk_secondary_coarse ?? 1);
  const topkSecondaryFine = Number(limits.topk_secondary_fine ?? 2);
  const maxEvidenceChars = Number(limits.max_evidence_chars ?? 3500);

  const queries = buildQueryProfiles(product, intentionMd, contextJson, cfg);
  const allHits: EvidenceHit[] = [];

  for (const [profileName, query] of queries) {
    const qvec = await ollamaEmbeddings(ollamaUrl, embedModel, query);
    log(`Embedding query ok (profile=${profileName}, dim=${qvec.length})`);

    for (const col of primaryCoarse) {
      const hits = await qdrantSearch(qc, col, qvec, topkPrimaryCoarse);
      log(`Qdrant search col=${col} hits=${hits.length}`);
      allHits.push(...hits);
    }
    for (const col of primaryFine) {
      const hits = await qdrantSearch(qc, col, qvec, topkPrimaryFine);
      log(`Qdrant search col=${col} hits=${hits.length}`);
      allHits.push(...hits);
    }
    for (const col of secondaryCoarse) {
      const hits = await qdrantSearch(qc, col, qvec, topkSecondaryCoarse);
      log(`Qdrant search col=${col} hits=${hits.length}`);
      allHits.push(...hits);
    }
    for (const col of secondaryFine) {
      const hits = await qdrantSearch(qc, col, qvec, topkSecondaryFine);
      log(`Qdrant search col=${col} hits=${hits.length}`);
      allHits.push(...hits);
    }
  }

  const deduped = dedupHits(allHits);
  const domainHits = filterHits(deduped, 'domain');
  const synthesisHits = filterHits(deduped, 'synthesis');

  return {
    hits: deduped,
    domainPack: renderEvidencePack(domainHits, maxEvidenceChars),
    synthesisPack: renderEvidencePack(synthesisHits, maxEvidenceChars),
  };
}

function extractJsonObject(text: string): string | null {
  if (!text) return null;
  const fenced = text.match(/```json\s*(\{[\s\S]*?\})\s*```/i);
  if (fenced?.[1]) return fenced[1].trim();
  const candidates = text.match(/\{[\s\S]*\}/g);
  if (!candidates || candidates.length === 0) return null;
  return candidates.sort((a, b) => b.length - a.length)[0].trim();
}

function splitMarkdownBundle(raw: string): AnyJson {
  if (!raw.trim()) return {};

  const headers: Record<string, string> = {
    'requirements.md': 'requirements_md',
    'non_functional.md': 'non_functional_md',
    'glossary.md': 'glossary_md',
    'assumptions.md': 'assumptions_md',
    'handoff_to_corrinha.md': 'handoff_md',
    'handoff.md': 'handoff_md',
  };

  const pattern = /^\s*\*\*(.+?\.md)\*\*\s*$/gim;
  const matches = [...raw.matchAll(pattern)];
  if (matches.length === 0) return {};

  const out: AnyJson = {};
  for (let i = 0; i < matches.length; i += 1) {
    const current = matches[i];
    const name = String(current[1]).trim().toLowerCase();
    const key = headers[name];
    const start = (current.index ?? 0) + current[0].length;
    const end = i + 1 < matches.length ? (matches[i + 1].index ?? raw.length) : raw.length;
    if (key) out[key] = raw.slice(start, end).trim();
  }
  return out;
}

function safeJsonFromLlm(raw: string): AnyJson {
  const normalized = (raw ?? '').trim();
  if (!normalized) return { _raw: '' };

  const jsonText = extractJsonObject(normalized);
  if (jsonText) {
    try {
      const obj = JSON.parse(jsonText);
      if (obj && typeof obj === 'object' && !Array.isArray(obj)) {
        return { ...(obj as AnyJson), _raw: normalized };
      }
    } catch {
      // ignore and continue
    }
  }

  const split = splitMarkdownBundle(normalized);
  if (Object.keys(split).length > 0) return { ...split, _raw: normalized };
  return { _raw: normalized };
}

function requiredDomainTerms(cfg: AnyJson): string[] {
  return (((cfg.generation as AnyJson)?.quality_gates as AnyJson)?.required_domain_terms as string[] | undefined) ?? [];
}

function forbiddenTerms(cfg: AnyJson): string[] {
  return (((cfg.generation as AnyJson)?.quality_gates as AnyJson)?.forbidden_terms as string[] | undefined) ?? [];
}

function minimumRequiredTermsFound(cfg: AnyJson): number {
  return Number((((cfg.generation as AnyJson)?.quality_gates as AnyJson)?.minimum_required_terms_found as number | undefined) ?? 3);
}

function countRequiredTerms(text: string, terms: string[]): number {
  const lower = text.toLowerCase();
  return terms.filter((t) => lower.includes(t.toLowerCase())).length;
}

function findForbiddenTerms(text: string, terms: string[]): string[] {
  const lower = text.toLowerCase();
  return terms.filter((t) => lower.includes(t.toLowerCase()));
}

function validateDomainAdherence(text: string, cfg: AnyJson): { ok: boolean; validation: AnyJson } {
  const reqTerms = requiredDomainTerms(cfg);
  const forbTerms = forbiddenTerms(cfg);
  const minHits = minimumRequiredTermsFound(cfg);

  const hits = countRequiredTerms(text, reqTerms);
  const bad = findForbiddenTerms(text, forbTerms);

  return {
    ok: hits >= minHits && bad.length === 0,
    validation: {
      required_terms_found: hits,
      required_terms_needed: minHits,
      forbidden_terms_found: bad,
    },
  };
}

function sectionPresent(text: string, sectionName: string): boolean {
  const re = new RegExp(`^\\s*#+\\s*${sectionName.replace(/[.*+?^${}()|[\\]\\]/g, '\\$&')}\\s*$`, 'im');
  return re.test(text);
}

function markdownTableRows(text: string): number {
  const rows = text.split('\n').filter((ln) => ln.includes('|'));
  return Math.max(0, rows.length - 2);
}

function validateOutputContracts(finalDocs: Record<string, string>, contracts: AnyJson): { ok: boolean; errors: string[] } {
  const errors: string[] = [];
  const files = (contracts.files ?? {}) as Record<string, AnyJson>;

  for (const [fname, rules] of Object.entries(files)) {
    const content = (finalDocs[fname] ?? '').trim();
    const qualityRules = (contracts.quality_rules ?? {}) as AnyJson;

    if ((qualityRules.reject_if_empty_files ?? true) && !content) {
      errors.push(`${fname}: vazio`);
      continue;
    }

    const minChars = Number(rules.minimum_chars ?? 0);
    if (minChars > 0 && content.length < minChars) {
      errors.push(`${fname}: menor que minimum_chars=${minChars}`);
    }

    for (const section of ((rules.required_sections as string[]) ?? [])) {
      if (!sectionPresent(content, section)) {
        errors.push(`${fname}: seção obrigatória ausente: ${section}`);
      }
    }

    if (rules.minimum_requirements != null) {
      const prefix = String(rules.requirement_prefix ?? 'RF-');
      const re = new RegExp(`^\\s*${prefix.replace(/[.*+?^${}()|[\\]\\]/g, '\\$&')}\\d+`, 'gim');
      const qty = (content.match(re) ?? []).length;
      if (qty < Number(rules.minimum_requirements)) errors.push(`${fname}: quantidade insuficiente de requisitos com prefixo ${prefix}`);
    }

    if (rules.minimum_assumptions != null) {
      const prefix = String(rules.assumption_prefix ?? 'AS-');
      const re = new RegExp(`^\\s*${prefix.replace(/[.*+?^${}()|[\\]\\]/g, '\\$&')}\\d+`, 'gim');
      const qty = (content.match(re) ?? []).length;
      if (qty < Number(rules.minimum_assumptions)) errors.push(`${fname}: quantidade insuficiente de suposições com prefixo ${prefix}`);
    }

    if (rules.must_be_table) {
      const rows = markdownTableRows(content);
      if (rows < Number(rules.minimum_rows ?? 1)) errors.push(`${fname}: tabela markdown insuficiente`);
    }
  }

  const domainRules = (contracts.domain_rules ?? {}) as AnyJson;
  const combined = Object.values(finalDocs).join('\n\n').toLowerCase();

  const requiredTerms = ((domainRules.must_mention_terms as string[]) ?? []).map((t) => t.toLowerCase());
  const found = requiredTerms.filter((t) => combined.includes(t)).length;
  if (requiredTerms.length > 0 && found < Number(domainRules.minimum_terms_found ?? 1)) {
    errors.push('domain_rules: termos centrais insuficientes');
  }

  for (const term of ((domainRules.forbidden_terms as string[]) ?? [])) {
    if (combined.includes(term.toLowerCase())) {
      errors.push(`domain_rules: termo proibido encontrado: ${term}`);
    }
  }

  return { ok: errors.length === 0, errors };
}

function renderDomainSummaryPrompt(template: string, intentionMd: string, contextJson: AnyJson, evidencePack: string): string {
  return template
    .replaceAll('{{INTENTION_MD}}', intentionMd.trim())
    .replaceAll('{{CONTEXT_JSON}}', JSON.stringify(contextJson, null, 2).trim())
    .replaceAll('{{EVIDENCE_PACK}}', evidencePack.trim());
}

function renderFirstPassPrompt(templateMd: string, intentionMd: string, contextJson: AnyJson, evidencePack: string, domainSummaryMd: string): string {
  let prompt = templateMd
    .replaceAll('{{INTENTION_MD}}', intentionMd.trim())
    .replaceAll('{{CONTEXT_JSON}}', JSON.stringify(contextJson, null, 2).trim())
    .replaceAll('{{EVIDENCE_PACK}}', evidencePack.trim());

  prompt += '\n\n---\n\n# Domain Summary gerado\n\n';
  prompt += domainSummaryMd.trim();
  prompt += '\n\n---\n\n# Regras mínimas de qualidade\n\n';
  prompt += '- não invente outro domínio\n';
  prompt += '- responda somente em JSON válido\n';
  prompt += '- cada campo deve ser markdown completo\n';
  prompt += '- mencione explicitamente Schola, ProdOps, turmas, matrícula, pagamento e Certificare quando relevantes\n';
  return prompt;
}

function renderSelfReviewPrompt(selfReviewTemplate: string, contextJson: AnyJson, domainSummaryMd: string, rubricMd: string, firstDraftJson: AnyJson): string {
  return selfReviewTemplate
    .replaceAll('{{CONTEXT_JSON}}', JSON.stringify(contextJson, null, 2).trim())
    .replaceAll('{{DOMAIN_SUMMARY_MD}}', domainSummaryMd.trim())
    .replaceAll('{{RUBRIC_MD}}', rubricMd.trim())
    .replaceAll('{{FIRST_DRAFT_JSON}}', JSON.stringify(firstDraftJson, null, 2));
}

function renderDomainRepairPrompt(repairTemplate: string, contextJson: AnyJson, rejectedOutputJson: AnyJson, forbiddenTermsJson: string[]): string {
  return repairTemplate
    .replaceAll('{{CONTEXT_JSON}}', JSON.stringify(contextJson, null, 2).trim())
    .replaceAll('{{REJECTED_OUTPUT_JSON}}', JSON.stringify(rejectedOutputJson, null, 2))
    .replaceAll('{{FORBIDDEN_TERMS_JSON}}', JSON.stringify(forbiddenTermsJson));
}

function renderContractRepairPrompt(repairTemplate: string, contextJson: AnyJson, domainSummaryMd: string, currentOutputJson: AnyJson, contractErrors: string[]): string {
  return repairTemplate
    .replaceAll('{{CONTEXT_JSON}}', JSON.stringify(contextJson, null, 2).trim())
    .replaceAll('{{DOMAIN_SUMMARY_MD}}', domainSummaryMd.trim())
    .replaceAll('{{CURRENT_OUTPUT_JSON}}', JSON.stringify(currentOutputJson, null, 2))
    .replaceAll('{{CONTRACT_ERRORS_JSON}}', JSON.stringify(contractErrors, null, 2));
}

function parseArgs(argv: string[]): { product: string; root: string; agentRoot: string; force: boolean } {
  const args: Record<string, string | boolean> = { force: false, agentRoot: '/opt/agents' };
  for (let i = 0; i < argv.length; i += 1) {
    const item = argv[i];
    if (item === '--force') {
      args.force = true;
      continue;
    }
    if (item.startsWith('--')) {
      const key = item.slice(2);
      const value = argv[i + 1];
      if (value == null || value.startsWith('--')) throw new Error(`Missing value for --${key}`);
      args[key] = value;
      i += 1;
    }
  }

  const product = String(args.product ?? '').trim();
  const root = String(args.root ?? '').trim();
  const agentRoot = String(args.agentRoot ?? '/opt/agents').trim();
  const force = Boolean(args.force);
  if (!product || !root) {
    throw new Error('Usage: tsx src/gertrudes-run.ts --product <name> --root <path> [--agent-root <path>] [--force]');
  }
  return { product, root, agentRoot, force };
}

async function main(): Promise<void> {
  const { product, root, agentRoot, force } = parseArgs(process.argv.slice(2));
  const productRoot = path.resolve(root, product);
  const agentDir = path.resolve(agentRoot, 'gertrudes');

  const agent = await loadAgentFiles(agentDir);
  const { intentionMd, contextJson } = await loadProductInputs(productRoot, agent.cfg);

  const inputHash = computeInputHash(intentionMd, contextJson, agent.cfg);
  const stateFile = path.join(productRoot, '_state', 'gertrudes.json');

  if (!force) {
    try {
      const st = await readJson<AnyJson>(stateFile);
      if (st.input_hash === inputHash && st.status === 'ok') {
        log(`Skip: já executado com o mesmo input_hash para product=${product}`);
        return;
      }
    } catch {
      // ignore state read errors
    }
  }

  const ollamaUrl = process.env.OLLAMA_BASE_URL ?? 'http://host.docker.internal:11434';
  const models = (agent.cfg.models ?? {}) as AnyJson;
  const embedModel = String(models.embed_model ?? 'nomic-embed-text');
  const genModel = String(models.generate_model ?? 'llama3.1:8b');

  const debugDir = path.join(productRoot, '_debug', 'gertrudes');
  await mkdir(debugDir, { recursive: true });

  const evidence = await retrieveEvidence(product, intentionMd, contextJson, agent.cfg, ollamaUrl, embedModel);
  await writeText(path.join(debugDir, 'domain_evidence_pack.txt'), evidence.domainPack);
  await writeText(path.join(debugDir, 'synthesis_evidence_pack.txt'), evidence.synthesisPack);
  log(`Domain evidence chars=${evidence.domainPack.length} | Synthesis evidence chars=${evidence.synthesisPack.length} | hits=${evidence.hits.length}`);

  const genTimeout = Number(process.env.GEN_TIMEOUT_SEC ?? '1200');

  const domainPrompt = renderDomainSummaryPrompt(agent.domainSummaryPromptMd, intentionMd, contextJson, evidence.domainPack);
  await writeText(path.join(debugDir, 'domain_prompt.txt'), domainPrompt);

  const domainSummaryMd = await ollamaGenerate(ollamaUrl, genModel, agent.systemMd, domainPrompt, genTimeout);
  await writeText(path.join(debugDir, 'domain_summary_raw.txt'), domainSummaryMd);

  const firstPassPrompt = renderFirstPassPrompt(agent.templateMd, intentionMd, contextJson, evidence.synthesisPack, domainSummaryMd);
  await writeText(path.join(debugDir, 'first_pass_prompt.txt'), firstPassPrompt);

  const rawFirst = await ollamaGenerate(ollamaUrl, genModel, agent.systemMd, firstPassPrompt, genTimeout);
  await writeText(path.join(debugDir, 'raw_llm_first_pass.txt'), rawFirst);

  const firstDraft = safeJsonFromLlm(rawFirst);
  await writeJson(path.join(debugDir, 'parsed_llm_first_pass.json'), firstDraft);

  const expectedKeys = new Set(['requirements_md', 'non_functional_md', 'glossary_md', 'assumptions_md', 'handoff_md']);
  if (typeof firstDraft.error === 'string') throw new Error(`LLM retornou erro explícito na primeira passada: ${firstDraft.error}`);
  if (![...expectedKeys].some((k) => k in firstDraft)) throw new Error('LLM não retornou JSON nem bundle markdown reconhecível na primeira passada');

  const selfReviewPrompt = renderSelfReviewPrompt(agent.selfReviewPromptMd, contextJson, domainSummaryMd, agent.rubricMd, firstDraft);
  await writeText(path.join(debugDir, 'self_review_prompt.txt'), selfReviewPrompt);

  const rawReview = await ollamaGenerate(ollamaUrl, genModel, agent.systemMd, selfReviewPrompt, genTimeout);
  await writeText(path.join(debugDir, 'raw_llm_self_review.txt'), rawReview);

  let reviewed = safeJsonFromLlm(rawReview);
  await writeJson(path.join(debugDir, 'parsed_llm_self_review.json'), reviewed);

  if (typeof reviewed.error === 'string') throw new Error(`LLM retornou erro explícito no self review: ${reviewed.error}`);
  if (![...expectedKeys].some((k) => k in reviewed)) throw new Error('LLM não retornou JSON nem bundle markdown reconhecível no self review');

  let requirementsMd = normalizeLlmText(reviewed.requirements_md).trim();
  let nonFunctionalMd = normalizeLlmText(reviewed.non_functional_md).trim();
  let glossaryMd = normalizeLlmText(reviewed.glossary_md).trim();
  let assumptionsMd = normalizeLlmText(reviewed.assumptions_md).trim();
  let handoffMd = normalizeLlmText(reviewed.handoff_md).trim();
  const reviewNotesMd = normalizeLlmText(reviewed.review_notes_md).trim();

  const combined = () => [domainSummaryMd, requirementsMd, nonFunctionalMd, glossaryMd, assumptionsMd, handoffMd].join('\n\n');

  let domainValidation = validateDomainAdherence(combined(), agent.cfg);
  await writeJson(path.join(debugDir, 'validation.json'), domainValidation.validation);

  if (!domainValidation.ok) {
    const forbiddenFound = (domainValidation.validation.forbidden_terms_found as string[] | undefined) ?? [];
    if (forbiddenFound.length > 0) {
      const repairPrompt = renderDomainRepairPrompt(
        agent.domainRepairPromptMd,
        contextJson,
        {
          requirements_md: requirementsMd,
          non_functional_md: nonFunctionalMd,
          glossary_md: glossaryMd,
          assumptions_md: assumptionsMd,
          handoff_md: handoffMd,
        },
        forbiddenFound,
      );
      await writeText(path.join(debugDir, 'domain_repair_prompt.txt'), repairPrompt);

      const rawRepair = await ollamaGenerate(ollamaUrl, genModel, agent.systemMd, repairPrompt, genTimeout);
      await writeText(path.join(debugDir, 'raw_llm_domain_repair.txt'), rawRepair);

      const repaired = safeJsonFromLlm(rawRepair);
      await writeJson(path.join(debugDir, 'parsed_llm_domain_repair.json'), repaired);
      if (![...expectedKeys].some((k) => k in repaired)) throw new Error('Domain repair não retornou JSON nem bundle markdown reconhecível');

      requirementsMd = normalizeLlmText(repaired.requirements_md).trim();
      nonFunctionalMd = normalizeLlmText(repaired.non_functional_md).trim();
      glossaryMd = normalizeLlmText(repaired.glossary_md).trim();
      assumptionsMd = normalizeLlmText(repaired.assumptions_md).trim();
      handoffMd = normalizeLlmText(repaired.handoff_md).trim();

      domainValidation = validateDomainAdherence(combined(), agent.cfg);
      await writeJson(path.join(debugDir, 'validation_after_repair.json'), domainValidation.validation);
    }

    if (!domainValidation.ok) {
      throw new Error(`Saída do LLM não aderente ao domínio do produto mesmo após repair. Validação: ${JSON.stringify(domainValidation.validation)}`);
    }
  }

  const outputs = (agent.cfg.outputs ?? {}) as AnyJson;
  const outDir = path.join(productRoot, String(outputs.output_dir ?? '1-requirements'));
  await mkdir(outDir, { recursive: true });

  let finalDocs: Record<string, string> = {
    [String(outputs.domain_summary_md ?? 'domain_summary.md')]: domainSummaryMd.trim(),
    [String(outputs.requirements_md ?? 'requirements.md')]: requirementsMd,
    [String(outputs.non_functional_md ?? 'non_functional.md')]: nonFunctionalMd,
    [String(outputs.glossary_md ?? 'glossary.md')]: glossaryMd,
    [String(outputs.assumptions_md ?? 'assumptions.md')]: assumptionsMd,
    [String(outputs.handoff_md ?? 'handoff_to_corrinha.md')]: handoffMd,
  };

  let contractValidation = validateOutputContracts(finalDocs, agent.outputContracts);
  await writeJson(path.join(debugDir, 'output_contract_validation.json'), contractValidation);

  if (!contractValidation.ok) {
    const contractRepairPrompt = renderContractRepairPrompt(
      agent.contractRepairPromptMd,
      contextJson,
      domainSummaryMd,
      {
        requirements_md: requirementsMd,
        non_functional_md: nonFunctionalMd,
        glossary_md: glossaryMd,
        assumptions_md: assumptionsMd,
        handoff_md: handoffMd,
      },
      contractValidation.errors,
    );
    await writeText(path.join(debugDir, 'contract_repair_prompt.txt'), contractRepairPrompt);

    const rawContractRepair = await ollamaGenerate(ollamaUrl, genModel, agent.systemMd, contractRepairPrompt, genTimeout);
    await writeText(path.join(debugDir, 'raw_llm_contract_repair.txt'), rawContractRepair);

    const repairedContract = safeJsonFromLlm(rawContractRepair);
    await writeJson(path.join(debugDir, 'parsed_llm_contract_repair.json'), repairedContract);

    if (![...expectedKeys].some((k) => k in repairedContract)) {
      throw new Error('Contract repair não retornou JSON nem bundle markdown reconhecível');
    }

    requirementsMd = normalizeLlmText(repairedContract.requirements_md).trim();
    nonFunctionalMd = normalizeLlmText(repairedContract.non_functional_md).trim();
    glossaryMd = normalizeLlmText(repairedContract.glossary_md).trim();
    assumptionsMd = normalizeLlmText(repairedContract.assumptions_md).trim();
    handoffMd = normalizeLlmText(repairedContract.handoff_md).trim();

    finalDocs = {
      [String(outputs.domain_summary_md ?? 'domain_summary.md')]: domainSummaryMd.trim(),
      [String(outputs.requirements_md ?? 'requirements.md')]: requirementsMd,
      [String(outputs.non_functional_md ?? 'non_functional.md')]: nonFunctionalMd,
      [String(outputs.glossary_md ?? 'glossary.md')]: glossaryMd,
      [String(outputs.assumptions_md ?? 'assumptions.md')]: assumptionsMd,
      [String(outputs.handoff_md ?? 'handoff_to_corrinha.md')]: handoffMd,
    };

    contractValidation = validateOutputContracts(finalDocs, agent.outputContracts);
    await writeJson(path.join(debugDir, 'output_contract_validation_after_repair.json'), contractValidation);

    if (!contractValidation.ok) {
      throw new Error(`Output contracts inválidos mesmo após contract repair: ${contractValidation.errors.join('; ')}`);
    }
  }

  for (const [fname, content] of Object.entries(finalDocs)) {
    await writeText(path.join(outDir, fname), `${content.trim()}\n`);
  }

  await writeText(path.join(debugDir, 'review_notes.md'), `${reviewNotesMd}\n`);

  await writeJson(stateFile, {
    agent: 'gertrudes',
    version: agent.cfg.version,
    status: 'ok',
    product,
    input_hash: inputHash,
    generated_at: nowIso(),
    ollama_url: ollamaUrl,
    embed_model: embedModel,
    generate_model: genModel,
  });

  log(`OK: arquivos gerados em ${outDir} (product=${product})`);
}

main().catch((error) => {
  log(`ERROR: ${error instanceof Error ? error.message : String(error)}`);
  process.exit(1);
});
