/**
 * Side-effect import: load .env from the project root by ABSOLUTE path, before any
 * module that reads config. The MCP server is spawned by OpenClaw with an unknown
 * cwd, so relative dotenv loading can't be relied on. Import this first.
 */
import { config } from 'dotenv';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
config({ path: path.join(root, '.env') });
