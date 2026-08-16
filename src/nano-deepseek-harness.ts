import { readFile } from 'node:fs/promises'
import { Context, Service, type Plugin as CordisPlugin } from 'cordis'
import OpenAI from 'openai'
import type {
  ChatCompletionFunctionTool,
  ChatCompletionMessageParam,
} from 'openai/resources/chat/completions'

type Args = Record<string, unknown>
type ToolCall = { id: string; name: string; arguments: string }
type Reply = { content: string | null; toolCalls: ToolCall[] }
type Loop = (task: string) => Promise<string>

interface Tool {
  schema: ChatCompletionFunctionTool
  execute(args: Args): Promise<string>
}

interface Model {
  complete(messages: ChatCompletionMessageParam[], tools: ChatCompletionFunctionTool[]): Promise<Reply>
}

declare module 'cordis' {
  interface Context {
    nano: NanoRuntime
  }
}

/** 最小内核：只保存插件注册进来的模型、工具和循环。 */
export class NanoRuntime extends Service {
  private model: Model | undefined
  private loop: Loop | undefined
  private readonly tools = new Map<string, Tool>()

  constructor(ctx: Context) {
    super(ctx, 'nano')
  }

  setModel(model: Model): () => void {
    this.model = model
    return () => { this.model = undefined }
  }

  setLoop(loop: Loop): () => void {
    this.loop = loop
    return () => { this.loop = undefined }
  }

  addTool(tool: Tool): () => void {
    const name = tool.schema.function.name
    this.tools.set(name, tool)
    return () => { this.tools.delete(name) }
  }

  complete(messages: ChatCompletionMessageParam[]): Promise<Reply> {
    if (!this.model) throw new Error('No model plugin installed.')
    return this.model.complete(messages, [...this.tools.values()].map(tool => tool.schema))
  }

  async execute(name: string, args: Args): Promise<string> {
    const tool = this.tools.get(name)
    if (!tool) throw new Error(`Unknown tool: ${name}`)
    return tool.execute(args)
  }

  run(task: string): Promise<string> {
    if (!this.loop) throw new Error('No agent-loop plugin installed.')
    return this.loop(task)
  }
}

/** 模型插件：OpenAI SDK 通过兼容接口调用 DeepSeek。 */
export function deepSeekPlugin(client = new OpenAI({
  apiKey: process.env['DEEPSEEK_API_KEY'],
  baseURL: process.env['DEEPSEEK_BASE_URL'] ?? 'https://api.deepseek.com',
})): CordisPlugin.Object<void> {
  return {
    name: 'deepseek-model',
    inject: ['nano'],
    apply(ctx) {
      ctx.effect(() => ctx.nano.setModel({
        async complete(messages, tools) {
          const response = await client.chat.completions.create({
            model: process.env['DEEPSEEK_MODEL'] ?? 'deepseek-v4-flash',
            messages,
            tools,
          })
          const message = response.choices[0]?.message
          if (!message) throw new Error('DeepSeek returned no message.')
          return {
            content: message.content,
            toolCalls: (message.tool_calls ?? []).map((call) => {
              if (call.type !== 'function') throw new Error(`Unsupported tool: ${call.type}`)
              return {
                id: call.id,
                name: call.function.name,
                arguments: call.function.arguments,
              }
            }),
          }
        },
      }))
    },
  }
}

/** 循环插件：模型选择工具，代码执行工具，结果写回 messages。 */
export const agentLoopPlugin: CordisPlugin.Object<void> = {
  name: 'agent-loop',
  inject: ['nano'],
  apply(ctx) {
    ctx.effect(() => ctx.nano.setLoop(async (task) => {
      const messages: ChatCompletionMessageParam[] = [
        { role: 'system', content: 'You are a concise assistant. Use tools when needed.' },
        { role: 'user', content: task },
      ]

      for (let step = 0; step < 8; step += 1) {
        const reply = await ctx.nano.complete(messages)
        messages.push({
          role: 'assistant',
          content: reply.content,
          tool_calls: reply.toolCalls.map(call => ({
            id: call.id,
            type: 'function',
            function: { name: call.name, arguments: call.arguments },
          })),
        })

        if (reply.toolCalls.length === 0) {
          if (!reply.content) throw new Error('Model returned no text.')
          return reply.content
        }

        for (const call of reply.toolCalls) {
          let content: string
          try {
            content = await ctx.nano.execute(call.name, JSON.parse(call.arguments) as Args)
          } catch (error) {
            content = `Error: ${error instanceof Error ? error.message : String(error)}`
          }
          messages.push({ role: 'tool', tool_call_id: call.id, content })
        }
      }
      throw new Error('Agent exceeded 8 steps.')
    }))
  },
}

/** 工具插件：把一个普通 TypeScript 函数暴露给模型。 */
export const readFilePlugin: CordisPlugin.Object<void> = {
  name: 'read-file',
  inject: ['nano'],
  apply(ctx) {
    ctx.effect(() => ctx.nano.addTool({
      schema: {
        type: 'function',
        function: {
          name: 'read_file',
          description: 'Read a UTF-8 text file.',
          parameters: {
            type: 'object',
            properties: { path: { type: 'string' } },
            required: ['path'],
          },
        },
      },
      async execute(args) {
        return readFile(String(args['path']), 'utf8')
      },
    }))
  },
}
