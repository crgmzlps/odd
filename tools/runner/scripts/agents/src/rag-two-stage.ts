import { QdrantClient } from '@qdrant/js-client-rest';

type SearchResult = {
  query: string;
  specialty: string;
  collections: { coarse: string; fine: string };
  coarse: Array<{ score: number; payload: Record<string, unknown> }>;
  fine: Array<{ score: number; payload: Record<string, unknown> }>;
};

const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL ?? 'http://ollama:11434';
const OLLAMA_EMBED_MODEL = process.env.OLLAMA_EMBED_MODEL ?? 'nomic-embed-text';
const QDRANT_HOST = process.env.QDRANT_HOST ?? 'qdrant';
const QDRANT_PORT = Number(process.env.QDRANT_PORT ?? '6333');

async function ollamaEmbed(text: string): Promise<number[]> {
  const response = await fetch(`${OLLAMA_BASE_URL}/api/embeddings`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model: OLLAMA_EMBED_MODEL, prompt: text }),
  });
  if (!response.ok) {
    throw new Error(`Embedding request failed: ${response.status}`);
  }
  const data = (await response.json()) as { embedding?: number[] };
  if (!Array.isArray(data.embedding)) {
    throw new Error('Invalid embedding response from Ollama');
  }
  return data.embedding;
}

function pack(points: Array<{ score?: number; payload?: Record<string, unknown> }>) {
  return points.map((p) => ({ score: Number(p.score ?? 0), payload: { ...(p.payload ?? {}) } }));
}

export async function twoStageSearch(
  specialty: string,
  query: string,
  coarseLimit = 8,
  fineLimit = 12,
): Promise<SearchResult> {
  const client = new QdrantClient({ host: QDRANT_HOST, port: QDRANT_PORT });
  const qv = await ollamaEmbed(query);

  const coarseCollection = `kb__${specialty}__coarse`;
  const fineCollection = `kb__${specialty}__fine`;

  const coarse = await client.search(coarseCollection, {
    vector: qv,
    limit: coarseLimit,
    with_payload: true,
  });

  const docIds = [...new Set(coarse.map((p) => p.payload?.doc_id).filter(Boolean))];
  const sections = [...new Set(coarse.map((p) => p.payload?.section).filter(Boolean))];

  const must: Array<Record<string, unknown>> = [];
  if (docIds.length > 0) {
    must.push({ key: 'doc_id', match: { any: docIds } });
  }
  if (sections.length > 0) {
    must.push({ key: 'section', match: { any: sections } });
  }

  const fine = await client.search(fineCollection, {
    vector: qv,
    limit: fineLimit,
    with_payload: true,
    ...(must.length > 0 ? { filter: { must } } : {}),
  });

  return {
    query,
    specialty,
    collections: { coarse: coarseCollection, fine: fineCollection },
    coarse: pack(coarse),
    fine: pack(fine),
  };
}
