/**
 * Verify the IDX MCP server: spawn it over stdio, list tools, call the tool.
 *   npx tsx src/mcp/testClient.ts
 */
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import { fileURLToPath } from 'node:url';

const serverPath = fileURLToPath(new URL('./idxServer.ts', import.meta.url));
const transport = new StdioClientTransport({ command: 'npx', args: ['tsx', serverPath] });
const client = new Client({ name: 'idx-test', version: '1.0.0' });

await client.connect(transport);
const { tools } = await client.listTools();
console.log('tools:', tools.map((t) => t.name));

for (const message of ['Irvine 行情怎么样', 'what is DOM?', '在 Irvine 找 3 居室 250万以下']) {
  const res = await client.callTool({ name: 'ask_idx_assistant', arguments: { message, userId: '+1' } });
  const text = (res.content as Array<{ type: string; text?: string }>)[0]?.text ?? '';
  console.log(`\n👤 ${message}\n🤖 ${text.split('\n').slice(0, 3).join('\n')}`);
}
await client.close();
process.exit(0);
