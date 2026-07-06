/**
 * Generate the field glossary (knowledge/fields.md) FROM the field dictionary,
 * so the RAG knowledge source for "what column is X / what does Y mean" stays a
 * single source of truth (schema/columns.ts). Run: npm run gen:glossary
 */
import { writeFileSync, mkdirSync } from 'node:fs';
import { FIELDS, type TableName, type FieldDef } from '../../schema/columns.js';

const TABLES: TableName[] = ['rets_property', 'california_sold', 'rets_openhouse'];

function main() {
  const lines: string[] = [
    '# MLS Field Glossary',
    '',
    'Generated from `schema/columns.ts` (single source of truth). Each semantic field',
    'maps to physical column names per table.',
    '',
  ];
  for (const [name, rawDef] of Object.entries(FIELDS)) {
    const def: FieldDef = rawDef;
    const cols = TABLES
      .filter((t) => def.columns[t])
      .map((t) => `${t}.${def.columns[t]}`)
      .join(', ');
    lines.push(`## ${def.label} (${name})`);
    lines.push(`- Type: ${def.type}`);
    lines.push(`- Columns: ${cols}`);
    if (def.note) lines.push(`- Note: ${def.note}`);
    lines.push('');
  }
  mkdirSync('knowledge', { recursive: true });
  writeFileSync('knowledge/fields.md', lines.join('\n'));
  console.log(`wrote knowledge/fields.md (${Object.keys(FIELDS).length} fields)`);
}

main();
