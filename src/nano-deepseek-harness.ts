import { readFile, realpath } from 'node:fs/promises'
import { resolve, sep } from 'node:path'
import { Context, Service, type Plugin as CordisPlugin } from 'cordis'
import OpenAI from 'openai'
import type {
  ChatCompletionFunctionTool,
  ChatCompletionMessageParam,
} from 'openai/resources/chat/completions'

// ─────────────────────────────────────────────────────────────────────────────
// 1. 插件之间共享的最小协议
// ─────────────────────────────────────────────────────────────────────────────

type JsonObject = Record<string, unknown>

/** 模型请求调用一次本地工具时，我们真正关心的三个字段。 */
export interface ToolCall {
  id: string
  name: string
  arguments: string
}

/** 隔离 OpenAI SDK 的完整响应，只把循环需要的数据交给 NanoRuntime。 */
export interface ModelReply {
  content: string | null
  toolCalls: ToolCall[]
}

export interface Tool {
  /** 发送给模型看的函数说明书。 */
  definition: ChatCompletionFunctionTool
  /** 真正作用于外部世界的代码。模型本身不会执行这个函数。 */
  execute(args: JsonObject): Promise<string>
}

export interface Model {
  complete(messages: ChatCompletionMessageParam[], tools: ChatCompletionFunctionTool[]): Promise<ModelReply>
}

export interface AgentLoop {
  run(runtime: NanoRuntime, task: string): Promise<string>
}

declare module 'cordis' {
  interface Context {
    /** 让所有 Cordis 插件都能通过 ctx.nano 访问同一个能力注册表。 */
    nano: NanoRuntime
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. 最小内核：只组合能力，不实现具体能力
// ─────────────────────────────────────────────────────────────────────────────

/**
 * NanoRuntime 是唯一的 Cordis Service。
 *
 * 它不知道模型怎样请求、文件怎样读取、循环怎样运行，只保存插件注册进来的
 * Model、Tool 和 AgentLoop。这正是“一切皆插件”中的最小稳定内核。
 */
export class NanoRuntime extends Service {
  private readonly toolsByName = new Map<string, Tool>()
  private activeModel: Model | undefined
  private activeLoop: AgentLoop | undefined

  constructor(ctx: Context) {
    super(ctx, 'nano')
  }

  provideModel(model: Model): () => void {
    if (this.activeModel) throw new Error('A model plugin is already installed.')
    this.activeModel = model

    // Cordis 会在插件卸载时调用这个 disposer，避免留下失效的模型实例。
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
    if (this.toolsByName.has(name)) throw new Error(`Duplicate tool: ${name}`)
    this.toolsByName.set(name, tool)
    return () => this.toolsByName.delete(name)
  }

  get model(): Model {
    if (!this.activeModel) throw new Error('No model plugin installed.')
    return this.activeModel
  }

  get tools(): Tool[] {
    return [...this.toolsByName.values()]
  }

  findTool(name: string): Tool | undefined {
    return this.toolsByName.get(name)
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

// ─────────────────────────────────────────────────────────────────────────────
// 3. 模型插件：用 OpenAI SDK 调用 DeepSeek 兼容接口
// ─────────────────────────────────────────────────────────────────────────────

export function deepSeekPlugin(options: DeepSeekOptions = {}): CordisPlugin.Object<void> {
  return {
    name: 'deepseek-model',
    // 只有 NanoRuntime 就绪后，Cordis 才会加载这个插件。
    inject: ['nano'],
    apply(ctx) {
      // DeepSeek 与 OpenAI Chat Completions 兼容，只需替换 baseURL 和模型名。
      const client = options.client ?? new OpenAI({
        apiKey: options.apiKey ?? process.env['DEEPSEEK_API_KEY'],
        baseURL: options.baseURL ?? process.env['DEEPSEEK_BASE_URL'] ?? 'https://api.deepseek.com',
      })
      const model = options.model ?? process.env['DEEPSEEK_MODEL'] ?? 'deepseek-v4-flash'

      // effect 把“注册模型”和当前插件的生命周期绑定起来。
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

          // 外部 SDK 响应在这里收窄为 ModelReply，循环无需依赖 SDK 细节。
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

// ─────────────────────────────────────────────────────────────────────────────
// 4. 循环插件：Model → Tool → Result → Model
// ─────────────────────────────────────────────────────────────────────────────

function parseObject(json: string): JsonObject {
  const value: unknown = JSON.parse(json)
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new Error('Tool arguments must be an object.')
  }
  return value as JsonObject
}

/** 执行一个 Tool Call；失败也写回模型，让模型有机会修正，而不是让 Agent 崩溃。 */
async function executeToolCall(runtime: NanoRuntime, call: ToolCall): Promise<string> {
  const tool = runtime.findTool(call.name)
  if (!tool) return `Error: unknown tool ${call.name}`

  try {
    return await tool.execute(parseObject(call.arguments))
  } catch (error) {
    return `Error: ${error instanceof Error ? error.message : String(error)}`
  }
}

export function agentLoopPlugin(options: AgentLoopOptions = {}): CordisPlugin.Object<void> {
  return {
    name: 'agent-loop',
    inject: ['nano'],
    apply(ctx) {
      ctx.effect(() => ctx.nano.provideLoop({
        async run(runtime, task) {
          // messages 就是最小 Session：每轮模型输出和工具结果都按顺序追加。
          const messages: ChatCompletionMessageParam[] = [
            { role: 'system', content: options.systemPrompt ?? 'You are a concise assistant. Use tools when needed.' },
            { role: 'user', content: task },
          ]
          const maxSteps = options.maxSteps ?? 8

          for (let step = 0; step < maxSteps; step += 1) {
            // 第一步：把完整上下文和当前 Tool Schema 发给模型。
            const assistant = await runtime.model.complete(
              messages,
              runtime.tools.map(tool => tool.definition),
            )

            // 第二步：先记录 assistant/tool_calls，再记录对应的 tool message。
            // 这是 Chat Completions 多轮 Tool Call 的消息顺序要求。
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
              const content = await executeToolCall(runtime, call)
              messages.push({ role: 'tool', tool_call_id: call.id, content })
            }
          }
          throw new Error(`Agent exceeded ${maxSteps} steps.`)
        },
      }))
    },
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. 工具插件：只允许读取工作区内的文件
// ─────────────────────────────────────────────────────────────────────────────

function isInside(root: string, target: string): boolean {
  return target === root || target.startsWith(`${root}${sep}`)
}

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

          // 第一层检查拦截 ../ 这类直接跳出工作区的路径。
          const candidate = resolve(root, args['path'])
          if (!isInside(root, candidate)) throw new Error('Path escapes the workspace.')

          // 第二层检查解析符号链接，防止工作区内的链接指向外部文件。
          const [realRoot, target] = await Promise.all([realpath(root), realpath(candidate)])
          if (!isInside(realRoot, target)) throw new Error('Path escapes the workspace.')
          return readFile(target, 'utf8')
        },
      }))
    },
  }
}
