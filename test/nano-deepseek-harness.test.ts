import assert from 'node:assert/strict'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import { Context, type Plugin as CordisPlugin } from 'cordis'
import OpenAI from 'openai'
import {
  agentLoopPlugin,
  deepSeekPlugin,
  NanoRuntime,
  readFilePlugin,
  type Model,
} from '../src/nano-deepseek-harness.js'

test('Cordis composes the model, tool, and loop plugins', async () => {
  const ctx = new Context()
  let request = 0
  const model: Model = {
    async complete(messages, tools) {
      request += 1
      assert.equal(tools[0]?.function.name, 'echo')
      if (request === 1) {
        return {
          content: null,
          toolCalls: [{ id: 'call-1', name: 'echo', arguments: '{"text":"hello"}' }],
        }
      }
      assert.deepEqual(messages.at(-1), { role: 'tool', tool_call_id: 'call-1', content: 'HELLO' })
      return { content: 'done', toolCalls: [] }
    },
  }
  const fakeModel: CordisPlugin.Object<void> = {
    name: 'fake-model',
    inject: ['nano'],
    apply(active) {
      active.effect(() => active.nano.provideModel(model))
    },
  }
  const echoTool: CordisPlugin.Object<void> = {
    name: 'echo-tool',
    inject: ['nano'],
    apply(active) {
      active.effect(() => active.nano.registerTool({
        definition: {
          type: 'function',
          function: { name: 'echo', description: 'Echo text.', parameters: { type: 'object' } },
        },
        async execute(args) {
          return String(args['text']).toUpperCase()
        },
      }))
    },
  }

  try {
    await ctx.plugin(NanoRuntime)
    await ctx.plugin(fakeModel)
    await ctx.plugin(echoTool)
    await ctx.plugin(agentLoopPlugin())
    assert.equal(await ctx.nano.run('echo hello'), 'done')
    assert.equal(request, 2)
  } finally {
    await ctx.fiber.dispose()
  }
})

test('the file plugin is confined and follows the Cordis lifecycle', async () => {
  const workspace = await mkdtemp(join(tmpdir(), 'nanodsh-'))
  const ctx = new Context()
  try {
    await writeFile(join(workspace, 'proof.txt'), 'proof', 'utf8')
    await ctx.plugin(NanoRuntime)
    const fiber = await ctx.plugin(readFilePlugin(workspace))
    const tool = ctx.nano.findTool('read_file')
    assert.ok(tool)
    assert.equal(await tool.execute({ path: 'proof.txt' }), 'proof')
    await assert.rejects(tool.execute({ path: '../secret.txt' }), /escapes the workspace/)
    await fiber.dispose()
    assert.equal(ctx.nano.findTool('read_file'), undefined)
  } finally {
    await ctx.fiber.dispose()
    await rm(workspace, { recursive: true, force: true })
  }
})

test('the OpenAI SDK sends DeepSeek chat-completions tool requests', async () => {
  let sent: JsonBody | undefined
  interface JsonBody { model?: unknown; messages?: unknown; tools?: unknown }
  const fetchImpl = (async (input: string | URL | Request, init?: RequestInit) => {
    assert.equal(String(input), 'https://api.deepseek.com/chat/completions')
    sent = JSON.parse(String(init?.body)) as JsonBody
    return new Response(JSON.stringify({
      id: 'chat-test',
      object: 'chat.completion',
      created: 0,
      model: 'deepseek-v4-flash',
      choices: [{ index: 0, finish_reason: 'stop', message: { role: 'assistant', content: 'ok' } }],
    }), { status: 200, headers: { 'content-type': 'application/json' } })
  }) as typeof fetch
  const client = new OpenAI({ apiKey: 'test-key', baseURL: 'https://api.deepseek.com', fetch: fetchImpl })
  const ctx = new Context()

  try {
    await ctx.plugin(NanoRuntime)
    await ctx.plugin(deepSeekPlugin({ client }))
    await ctx.plugin(agentLoopPlugin())
    assert.equal(await ctx.nano.run('hello'), 'ok')
    assert.equal(sent?.model, 'deepseek-v4-flash')
    assert.ok(Array.isArray(sent?.messages))
    assert.ok(Array.isArray(sent?.tools))
  } finally {
    await ctx.fiber.dispose()
  }
})
