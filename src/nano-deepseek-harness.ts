import { readFile, realpath } from 'node:fs/promises'
import { resolve, sep } from 'node:path'

type JsonObject = Record<string, unknown>

export interface ToolCall {
  id: string
  type: 'function'
  function: {
    name: string
    arguments: string
  }
}

export type Message =
  | { role: 'system' | 'user'; content: string }
  | { role: 'assistant'; content: string | null; tool_calls?: ToolCall[] }
  | { role: 'tool'; content: string; tool_call_id: string }

export interface ToolDefinition {
  type: 'function'
  function: {
    name: string
    description: string
    parameters: JsonObject
  }
}

export interface Tool {
  definition: ToolDefinition
  execute(args: JsonObject): Promise<string>
}

export interface Model {
  complete(messages: Message[], tools: ToolDefinition[]): Promise<Extract<Message, { role: 'assistant' }>>
}

export interface AgentLoop {
  run(harness: NanoHarness, task: string): Promise<string>
}

export interface Plugin {
  name: string
  install(harness: NanoHarness): void
}

export function definePlugin(name: string, install: Plugin['install']): Plugin {
  return { name, install }
}

/** The only core: a registry that composes plugins. */
export class NanoHarness {
  readonly #plugins = new Set<string>()
  readonly #tools = new Map<string, Tool>()
  #model?: Model
  #loop?: AgentLoop

  use(plugin: Plugin): this {
    if (this.#plugins.has(plugin.name)) throw new Error(`Duplicate plugin: ${plugin.name}`)
    plugin.install(this)
    this.#plugins.add(plugin.name)
    return this
  }

  provideModel(model: Model): void {
    if (this.#model) throw new Error('A model plugin is already installed.')
    this.#model = model
  }

  provideLoop(loop: AgentLoop): void {
    if (this.#loop) throw new Error('An agent-loop plugin is already installed.')
    this.#loop = loop
  }

  registerTool(tool: Tool): void {
    const name = tool.definition.function.name
    if (this.#tools.has(name)) throw new Error(`Duplicate tool: ${name}`)
    this.#tools.set(name, tool)
  }

  get model(): Model {
    if (!this.#model) throw new Error('No model plugin installed.')
    return this.#model
  }

  get tools(): Tool[] {
    return [...this.#tools.values()]
  }

  findTool(name: string): Tool | undefined {
    return this.#tools.get(name)
  }

  run(task: string): Promise<string> {
    if (!this.#loop) throw new Error('No agent-loop plugin installed.')
    return this.#loop.run(this, task)
  }
}

export interface DeepSeekOptions {
  apiKey?: string
  baseUrl?: string
  model?: string
  fetchImpl?: typeof fetch
}

function asObject(value: unknown, label: string): JsonObject {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new Error(`${label} must be an object.`)
  }
  return value as JsonObject
}

function parseAssistantMessage(payload: unknown): Extract<Message, { role: 'assistant' }> {
  const root = asObject(payload, 'DeepSeek response')
  const choices = root['choices']
  if (!Array.isArray(choices) || choices.length === 0) throw new Error('DeepSeek returned no choices.')
  const choice = asObject(choices[0], 'DeepSeek choice')
  const raw = asObject(choice['message'], 'DeepSeek message')
  const content = raw['content']
  if (content !== null && typeof content !== 'string') throw new Error('DeepSeek returned invalid content.')

  const rawCalls = raw['tool_calls']
  if (rawCalls === undefined) return { role: 'assistant', content }
  if (!Array.isArray(rawCalls)) throw new Error('DeepSeek returned invalid tool calls.')

  const toolCalls = rawCalls.map((value, index): ToolCall => {
    const call = asObject(value, `tool call ${index}`)
    const fn = asObject(call['function'], `tool call ${index} function`)
    if (typeof call['id'] !== 'string' || call['type'] !== 'function'
      || typeof fn['name'] !== 'string' || typeof fn['arguments'] !== 'string') {
      throw new Error(`DeepSeek returned malformed tool call ${index}.`)
    }
    return {
      id: call['id'],
      type: 'function',
      function: { name: fn['name'], arguments: fn['arguments'] },
    }
  })
  return { role: 'assistant', content, tool_calls: toolCalls }
}

/** Official DeepSeek Chat Completions adapter, implemented with native fetch. */
export function deepSeekPlugin(options: DeepSeekOptions = {}): Plugin {
  return definePlugin('deepseek-model', (harness) => {
    harness.provideModel({
      async complete(messages, tools) {
        const apiKey = options.apiKey ?? process.env['DEEPSEEK_API_KEY']
        if (!apiKey) throw new Error('Missing DEEPSEEK_API_KEY.')
        const baseUrl = (options.baseUrl ?? process.env['DEEPSEEK_BASE_URL']
          ?? 'https://api.deepseek.com').replace(/\/$/, '')
        const request = options.fetchImpl ?? fetch
        const response = await request(`${baseUrl}/chat/completions`, {
          method: 'POST',
          headers: {
            authorization: `Bearer ${apiKey}`,
            'content-type': 'application/json',
          },
          body: JSON.stringify({
            model: options.model ?? process.env['DEEPSEEK_MODEL'] ?? 'deepseek-v4-flash',
            thinking: { type: 'disabled' },
            messages,
            tools,
            tool_choice: 'auto',
          }),
        })
        if (!response.ok) {
          throw new Error(`DeepSeek API ${response.status}: ${await response.text()}`)
        }
        return parseAssistantMessage(await response.json())
      },
    })
  })
}

export interface AgentLoopOptions {
  maxSteps?: number
  systemPrompt?: string
}

/** ReAct loop: model -> tool calls -> tool results -> model. */
export function agentLoopPlugin(options: AgentLoopOptions = {}): Plugin {
  return definePlugin('agent-loop', (harness) => {
    harness.provideLoop({
      async run(activeHarness, task) {
        const messages: Message[] = [
          { role: 'system', content: options.systemPrompt ?? 'You are a concise assistant. Use tools when needed.' },
          { role: 'user', content: task },
        ]
        const maxSteps = options.maxSteps ?? 8

        for (let step = 0; step < maxSteps; step += 1) {
          const assistant = await activeHarness.model.complete(
            messages,
            activeHarness.tools.map(tool => tool.definition),
          )
          messages.push(assistant)
          const calls = assistant.tool_calls ?? []
          if (calls.length === 0) {
            if (!assistant.content) throw new Error('Model returned no final text.')
            return assistant.content
          }

          for (const call of calls) {
            const tool = activeHarness.findTool(call.function.name)
            let content: string
            try {
              const args = asObject(JSON.parse(call.function.arguments), 'Tool arguments')
              content = tool ? await tool.execute(args) : `Error: unknown tool ${call.function.name}`
            } catch (error) {
              content = `Error: ${error instanceof Error ? error.message : String(error)}`
            }
            messages.push({ role: 'tool', tool_call_id: call.id, content })
          }
        }
        throw new Error(`Agent exceeded ${maxSteps} steps.`)
      },
    })
  })
}

/** One model-facing capability, confined to the selected workspace. */
export function readFilePlugin(workspace = process.cwd()): Plugin {
  const root = resolve(workspace)
  return definePlugin('read-file', (harness) => {
    harness.registerTool({
      definition: {
        type: 'function',
        function: {
          name: 'read_file',
          description: 'Read a UTF-8 text file inside the current workspace.',
          parameters: {
            type: 'object',
            properties: { path: { type: 'string', description: 'Workspace-relative file path.' } },
            required: ['path'],
            additionalProperties: false,
          },
        },
      },
      async execute(args) {
        if (typeof args['path'] !== 'string') throw new Error('path must be a string.')
        const candidate = resolve(root, args['path'])
        if (candidate !== root && !candidate.startsWith(`${root}${sep}`)) {
          throw new Error('Path escapes the workspace.')
        }
        const [realRoot, target] = await Promise.all([
          realpath(root),
          realpath(candidate),
        ])
        if (target !== realRoot && !target.startsWith(`${realRoot}${sep}`)) {
          throw new Error('Path escapes the workspace.')
        }
        return readFile(target, 'utf8')
      },
    })
  })
}
