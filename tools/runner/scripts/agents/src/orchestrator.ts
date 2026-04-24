import { access, mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { Corrinha, Creuza, Gertrudes, readJson, readText } from './agents.js';

async function exists(filePath: string): Promise<boolean> {
  try {
    await access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function waitFor(filePath: string, timeoutSec = 3600): Promise<void> {
  const start = Date.now();
  while (true) {
    if (await exists(filePath)) {
      return;
    }
    if ((Date.now() - start) / 1000 > timeoutSec) {
      throw new Error(`Timeout esperando ${filePath}`);
    }
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
}

async function ensureProductStructure(productRoot: string): Promise<void> {
  await mkdir(path.join(productRoot, 'artifacts', '99_trace'), { recursive: true });
  await mkdir(path.join(productRoot, 'docs'), { recursive: true });
  const contextYaml = path.join(productRoot, 'context.yaml');
  if (!(await exists(contextYaml))) {
    await writeFile(contextYaml, 'product: unknown\n', 'utf8');
  }
}

export async function runProduct(productRoot: string): Promise<void> {
  await ensureProductStructure(productRoot);
  const product = path.basename(productRoot);
  const intentionPath = path.join(productRoot, 'intention.md');
  if (!(await exists(intentionPath))) {
    throw new Error('Crie intention.md na pasta do produto');
  }

  const intentText = await readText(intentionPath);

  const g = new Gertrudes(productRoot);
  await g.run(intentText, product);

  const reqPath = path.join(productRoot, 'artifacts', '01_requirements', 'requirements.json');
  await waitFor(reqPath);
  const reqJson = await readJson<{ product: string; requirements: Array<{ title: string }> }>(reqPath);

  const c = new Corrinha(productRoot);
  await c.run(reqJson);

  const cr = new Creuza(productRoot);
  await cr.run(reqJson);

  const overview = path.join(productRoot, 'docs', 'overview.md');
  if (!(await exists(overview))) {
    await writeFile(overview, `# ${product}\n\nIntenção\n\n${intentText}\n`, 'utf8');
  }

  // eslint-disable-next-line no-console
  console.log('Pipeline concluído');
}

async function main(): Promise<void> {
  const target = process.argv[2];
  if (!target) {
    throw new Error('Uso: tsx src/orchestrator.ts /opt/products/new/meu_produto');
  }
  await runProduct(path.resolve(target));
}

main().catch((error) => {
  // eslint-disable-next-line no-console
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
