import { readFile, realpath } from 'node:fs/promises'
import { resolve, sep } from 'node:path'
import { Context, Service, type Plugin as CordisPlugin } from 'cordis'
import OpenAI from 'openai'
import type {
  ChatCompletionFunctionTool,
  ChatCompletionMessageParam,
} from 'openai/resources/chat/completions'

type JsonObject = Record<string, unknown>

export interface Tool {
  definition: ChatCompletionFunctionTool
  execute(args: JsonObject): Promise<string>
}

export interface Model {
  complete(messages: ChatCompletionMessageParam[], tools: ChatCompletionFunctionTool[]): Promise<{
    content: string | null
    toolCalls: Array<{ id: string; name: string; arguments: string }>
  }>
}

export interface AgentLoop {
  run(runtime: NanoRuntime, task: string): Promise<string>
}

declare module 'cordis' {
  interface Context {
    nano: NanoRuntime
  }
}

/** The only application service: plugins contribute the model, tools, and loop. */
export class NanoRuntime extends Service {
  private readonly toolMap = new Map<string, Tool>()
  private activeModel: Model | undefined
  private activeLoop: AgentLoop | undefined

  constructor(ctx: Context) {
    super(ctx, 'nano')
  }

  provideModel(model: Model): () => void {
    if (this.activeModel) throw new Error('A model plugin is already installed.')
    this.activeModel = model
    return () => {
      if (this.activeModel === model) this.activeModel = undefined
    }
  }

  provideLoop(loop: AgentLoop): () => void {
    if (this.activeLoop) throw new Error('An agent-loop plugin is already installed.')
    this.activeLoop = loop
    return () => {
      if (this.activeLoop === loop) this.activeLoop = undefined
    }
  }

  registerTool(tool: Tool): () => void {
    const name = tool.definition.function.name
    if (this.toolMap.has(name)) throw new Error(`Duplicate tool: ${name}`)
    this.toolMap.set(name, tool)
    return () => this.toolMap.delete(name)
  }

  get model(): Model {
    if (!this.activeModel) throw new Error('No model plugin installed.')
    return this.activeModel
  }

  get tools(): Tool[] {
    return [...this.toolMap.values()]
  }

  findTool(name: string): Tool | undefined {
    return this.toolMap.get(name)
  }

  run(task: string): Promise<string> {
    if (!this.activeLoop) throw new Error('No agent-loop plugin installed.')
    return this.activeLoop.run(this, task)
  }
}

export interface DeepSeekOptions {
  apiKey?: string
  baseURL?: string
  model?: string
  client?: OpenAI
}

/** DeepSeek model plugin using the official OpenAI TypeScript SDK. */
export function deepSeekPlugin(options: DeepSeekOptions = {}): CordisPlugin.Object<void> {
  return {
    name: 'deepseek-model',
    inject: ['nano'],
    apply(ctx) {
      const client = options.client ?? new OpenAI({
        apiKey: options.apiKey ?? process.env['DEEPSEEK_API_KEY'],
        baseURL: options.baseURL ?? process.env['DEEPSEEK_BASE_URL'] ?? 'https://api.deepseek.com',
      })
      const model = options.model ?? process.env['DEEPSEEK_MODEL'] ?? 'deepseek-v4-flash'
      ctx.effect(() => ctx.nano.provideModel({
        async complete(messages, tools) {
          const response = await client.chat.completions.create({
            model,
            messages,
            tools,
            tool_choice: 'auto',
          })
          const message = response.choices[0]?.message
          if (!message) throw new Error('DeepSeek returned no choices.')
          const toolCalls = (message.tool_calls ?? []).map((call) => {
            if (call.type !== 'function') throw new Error(`Unsupported tool call type: ${call.type}`)
            return {
              id: call.id,
              name: call.function.name,
              arguments: call.function.arguments,
            }
          })
          return { content: message.content, toolCalls }
        },
      }))
    },
  }
}

export interface AgentLoopOptions {
  maxSteps?: number
  systemPrompt?: string
}

function parseObject(json: string): JsonObject {
  const value: unknown = JSON.parse(json)
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new Error('Tool arguments must be an object.')
  }
  return value as JsonObject
}

/** ReAct loop plugin: model -> tool calls -> tool results -> model. */
export function agentLoopPlugin(options: AgentLoopOptions = {}): CordisPlugin.Object<void> {
  return {
    name: 'agent-loop',
    inject: ['nano'],
    apply(ctx) {
      ctx.effect(() => ctx.nano.provideLoop({
        async run(runtime, task) {
          const messages: ChatCompletionMessageParam[] = [
            { role: 'system', content: options.systemPrompt ?? 'You are a concise assistant. Use tools when needed.' },
            { role: 'user', content: task },
          ]
          const maxSteps = options.maxSteps ?? 8

          for (let step = 0; step < maxSteps; step += 1) {
            const assistant = await runtime.model.complete(
              messages,
              runtime.tools.map(tool => tool.definition),
            )
            messages.push({
              role: 'assistant',
              content: assistant.content,
              tool_calls: assistant.toolCalls.map(call => ({
                id: call.id,
                type: 'function',
                function: { name: call.name, arguments: call.arguments },
              })),
            })
            if (assistant.toolCalls.length === 0) {
              if (!assistant.content) throw new Error('Model returned no final text.')
              return assistant.content
            }

            for (const call of assistant.toolCalls) {
              const tool = runtime.findTool(call.name)
              let content: string
              try {
                content = tool
                  ? await tool.execute(parseObject(call.arguments))
                  : `Error: unknown tool ${call.name}`
              } catch (error) {
                content = `Error: ${error instanceof Error ? error.message : String(error)}`
              }
              messages.push({ role: 'tool', tool_call_id: call.id, content })
            }
          }
          throw new Error(`Agent exceeded ${maxSteps} steps.`)
        },
      }))
    },
  }
}

/** A read-only tool plugin confined to the selected workspace. */
export function readFilePlugin(workspace = process.cwd()): CordisPlugin.Object<void> {
  const root = resolve(workspace)
  return {
    name: 'read-file',
    inject: ['nano'],
    apply(ctx) {
      ctx.effect(() => ctx.nano.registerTool({
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
          const [realRoot, target] = await Promise.all([realpath(root), realpath(candidate)])
          if (target !== realRoot && !target.startsWith(`${realRoot}${sep}`)) {
            throw new Error('Path escapes the workspace.')
          }
          return readFile(target, 'utf8')
        },
      }))
    },
  }
}
