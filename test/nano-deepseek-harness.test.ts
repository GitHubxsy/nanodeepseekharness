import assert from 'node:assert/strict'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import {
  agentLoopPlugin,
  deepSeekPlugin,
  definePlugin,
  NanoHarness,
  readFilePlugin,
  type Message,
  type Model,
} from '../src/nano-deepseek-harness.js'

test('the loop executes a plugin tool and returns the final model text', async () => {
  let request = 0
  const model: Model = {
    async complete(messages, tools) {
      request += 1
      assert.equal(tools[0]?.function.name, 'echo')
      if (request === 1) {
        return {
          role: 'assistant',
          content: null,
          tool_calls: [{
            id: 'call-1',
            type: 'function',
            function: { name: 'echo', arguments: '{"text":"hello"}' },
          }],
        }
      }
      const result = messages.at(-1)
      assert.deepEqual(result, { role: 'tool', tool_call_id: 'call-1', content: 'HELLO' })
      return { role: 'assistant', content: 'done' }
    },
  }

  const harness = new NanoHarness()
    .use(definePlugin('fake-model', active => active.provideModel(model)))
    .use(definePlugin('echo-tool', active => active.registerTool({
      definition: {
        type: 'function',
        function: { name: 'echo', description: 'Echo text.', parameters: { type: 'object' } },
      },
      async execute(args) {
        return String(args['text']).toUpperCase()
      },
    })))
    .use(agentLoopPlugin())

  assert.equal(await harness.run('echo hello'), 'done')
  assert.equal(request, 2)
})

test('the read-file plugin reads inside the workspace and blocks traversal', async () => {
  const workspace = await mkdtemp(join(tmpdir(), 'nanodsh-'))
  try {
    await writeFile(join(workspace, 'proof.txt'), 'proof', 'utf8')
    const harness = new NanoHarness().use(readFilePlugin(workspace))
    const tool = harness.findTool('read_file')
    assert.ok(tool)
    assert.equal(await tool.execute({ path: 'proof.txt' }), 'proof')
    await assert.rejects(tool.execute({ path: '../secret.txt' }), /escapes the workspace/)
  } finally {
    await rm(workspace, { recursive: true, force: true })
  }
})

test('the DeepSeek plugin sends the official chat-completions tool format', async () => {
  let sent: JsonBody | undefined
  interface JsonBody { model?: unknown; messages?: unknown; tools?: unknown; thinking?: unknown }
  const fetchImpl = (async (input: string | URL | Request, init?: RequestInit) => {
    assert.equal(String(input), 'https://api.deepseek.com/chat/completions')
    sent = JSON.parse(String(init?.body)) as JsonBody
    return new Response(JSON.stringify({
      choices: [{ message: { role: 'assistant', content: 'ok' } }],
    }), { status: 200, headers: { 'content-type': 'application/json' } })
  }) as typeof fetch

  const harness = new NanoHarness()
    .use(deepSeekPlugin({ apiKey: 'test-key', fetchImpl }))
    .use(agentLoopPlugin())
  assert.equal(await harness.run('hello'), 'ok')
  assert.equal(sent?.model, 'deepseek-v4-flash')
  assert.deepEqual(sent?.thinking, { type: 'disabled' })
  assert.ok(Array.isArray(sent?.messages))
  assert.ok(Array.isArray(sent?.tools))
})
