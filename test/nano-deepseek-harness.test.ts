import assert from 'node:assert/strict'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import { Context } from 'cordis'
import OpenAI from 'openai'
import {
  agentLoopPlugin,
  deepSeekPlugin,
  NanoRuntime,
  readFilePlugin,
} from '../src/nano-deepseek-harness.js'

test('model -> tool -> model', async () => {
  let request = 0
  let proofPath = ''
  const client = new OpenAI({
    apiKey: 'test',
    fetch: (async (_input, init) => {
      request += 1
      if (request === 2) {
        const body = JSON.parse(String(init?.body)) as { messages: Array<{ content: string }> }
        assert.equal(body.messages.at(-1)?.content, 'proof')
      }
      const message = request === 1
        ? {
            role: 'assistant',
            content: null,
            tool_calls: [{
              id: 'call-1',
              type: 'function',
              function: { name: 'read_file', arguments: JSON.stringify({ path: proofPath }) },
            }],
          }
        : { role: 'assistant', content: 'done' }
      return new Response(JSON.stringify({
        id: 'test',
        object: 'chat.completion',
        created: 0,
        model: 'deepseek-v4-flash',
        choices: [{ index: 0, finish_reason: 'stop', message }],
      }), { headers: { 'content-type': 'application/json' } })
    }) as typeof fetch,
  })
  const workspace = await mkdtemp(join(tmpdir(), 'nanodsh-'))
  const ctx = new Context()

  try {
    proofPath = join(workspace, 'proof.txt')
    await writeFile(proofPath, 'proof', 'utf8')
    await ctx.plugin(NanoRuntime)
    await ctx.plugin(deepSeekPlugin(client))
    await ctx.plugin(readFilePlugin)
    await ctx.plugin(agentLoopPlugin)
    assert.equal(await ctx.nano.run(`read ${proofPath}`), 'done')
    assert.equal(request, 2)
  } finally {
    await ctx.fiber.dispose()
    await rm(workspace, { recursive: true, force: true })
  }
})
