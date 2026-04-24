import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { twoStageSearch } from './rag-two-stage.js';

function sha256(s: string): string {
  return createHash('sha256').update(s, 'utf8').digest('hex');
}

async function writeJson(filePath: string, obj: unknown): Promise<void> {
  await mkdir(path.dirname(filePath), { recursive: true });
  await writeFile(filePath, JSON.stringify(obj, null, 2), 'utf8');
}

async function writeText(filePath: string, text: string): Promise<void> {
  await mkdir(path.dirname(filePath), { recursive: true });
  await writeFile(filePath, text, 'utf8');
}

export class BaseAgent {
  readonly productRoot: string;
  readonly artifacts: string;
  readonly trace: string;

  constructor(productRoot: string) {
    this.productRoot = productRoot;
    this.artifacts = path.join(productRoot, 'artifacts');
    this.trace = path.join(this.artifacts, '99_trace');
  }

  async logEvidence(filename: string, obj: unknown): Promise<void> {
    await writeJson(path.join(this.trace, filename), obj);
  }
}

export class Gertrudes extends BaseAgent {
  async run(intentText: string, product: string): Promise<Record<string, unknown>> {
    const evidence = [];
    for (const specialty of ['product', 'requirements', 'architecture', 'testing']) {
      evidence.push(await twoStageSearch(specialty, intentText));
    }

    await this.logEvidence('rag_evidence_gertrudes.json', evidence);

    const intentHash = sha256(intentText);
    const reqs = [] as Array<Record<string, unknown>>;
    for (let i = 1; i <= 8; i += 1) {
      const requirement = {
        id: `REQ_${String(i).padStart(3, '0')}`,
        title: `Requisito ${i}`,
        type: i <= 6 ? 'functional' : 'non_functional',
        description: `Derivado da intenção: ${intentText.slice(0, 200)}`,
        priority: i <= 4 ? 'must' : 'should',
        acceptance_criteria: ['Critério observável e verificável'],
        constraints: [],
        assumptions: [],
        risks: [],
        trace: {
          sources: [
            {
              specialty: 'requirements',
              collection_stage: 'fine',
              doc_id: evidence[1]?.fine?.[0]?.payload?.doc_id ?? '',
              section: evidence[1]?.fine?.[0]?.payload?.section ?? '',
              chunk_id: evidence[1]?.fine?.[0]?.payload?.chunk_id ?? '',
            },
          ],
        },
      };
      reqs.push(requirement);
    }

    const out = {
      schema_version: 1,
      product,
      intent_hash: intentHash,
      requirements: reqs,
    };

    const reqDir = path.join(this.artifacts, '01_requirements');
    await writeJson(path.join(reqDir, 'requirements.json'), out);
    await writeText(path.join(reqDir, 'requirements.md'), reqs.map((r) => `- ${(r as { id: string }).id}: ${(r as { title: string }).title}`).join('\n'));
    return out;
  }
}

export class Corrinha extends BaseAgent {
  async run(requirementsJson: { product: string; requirements: Array<{ title: string }> }): Promise<Record<string, unknown>> {
    const product = requirementsJson.product;
    const reqs = requirementsJson.requirements;

    const evidence = await twoStageSearch('product', `User stories e use cases para ${product}`);
    await this.logEvidence('rag_evidence_corrinha.json', evidence);

    const stories: string[] = [];
    const useCases: string[] = [];
    for (const r of reqs.slice(0, 6)) {
      stories.push(`Como usuario eu quero ${r.title.toLowerCase()} para atingir o objetivo do produto`);
      useCases.push(`Use case: ${r.title} \nFluxo principal: ... \nExceções: ...`);
    }

    const specDir = path.join(this.artifacts, '02_specs');
    const contractsDir = path.join(specDir, 'contracts');
    await mkdir(contractsDir, { recursive: true });

    await writeText(path.join(specDir, 'user_stories.md'), stories.map((s) => `- ${s}`).join('\n'));
    await writeText(path.join(specDir, 'use_cases.md'), useCases.join('\n\n'));

    const openapi = `openapi: 3.0.3\ninfo:\n  title: ${product} API\n  version: 0.1.0\npaths: {}\n`;
    const asyncapi = `asyncapi: 2.6.0\ninfo:\n  title: ${product} Async API\n  version: 0.1.0\nchannels: {}\n`;
    const eventsContracts = 'schema_version: 1\nevents: []\n';

    await writeText(path.join(contractsDir, 'openapi.yaml'), openapi);
    await writeText(path.join(contractsDir, 'asyncapi.yaml'), asyncapi);
    await writeText(path.join(contractsDir, 'events_contracts.yaml'), eventsContracts);

    return { product, stories };
  }
}

export class Creuza extends BaseAgent {
  async run(requirementsJson: { product: string }): Promise<Record<string, unknown>> {
    const product = requirementsJson.product;

    const evidence = await twoStageSearch('eventing', `Event storming e eventos de dominio para ${product}`);
    await this.logEvidence('rag_evidence_creuza.json', evidence);

    const oddDir = path.join(this.artifacts, '03_odd');
    await mkdir(oddDir, { recursive: true });

    const eventStormingMd = `# Event Storming ${product}\n\n## Eventos sugeridos\n- Evento 1\n- Evento 2\n\n## Notas\nDerivado dos requisitos e evidências de eventing.\n`;
    await writeText(path.join(oddDir, 'event_storming.md'), eventStormingMd);

    const csv = [
      'event_name,bounded_context,command,aggregate,actor,system,source,notes',
      'EventoCriado,Core,ComandoCriar,AggregatePrincipal,Usuario,API,EventStorming,Inicial',
    ].join('\n');
    await writeText(path.join(oddDir, 'odd_events.csv'), csv + '\n');

    return { product, events_csv: path.join(oddDir, 'odd_events.csv') };
  }
}

export async function readText(filePath: string): Promise<string> {
  return (await readFile(filePath, 'utf8')).trim();
}

export async function readJson<T = unknown>(filePath: string): Promise<T> {
  return JSON.parse(await readFile(filePath, 'utf8')) as T;
}
