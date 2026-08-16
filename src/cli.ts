#!/usr/bin/env node

import { Context } from 'cordis'
import { agentLoopPlugin, deepSeekPlugin, NanoRuntime, readFilePlugin } from './nano-deepseek-harness.js'

const task = process.argv.slice(2).join(' ')
  || 'Use read_file to read README.md, then summarize it in three sentences.'
const ctx = new Context()

try {
  // 组装顺序也就是推荐阅读顺序：内核 → 模型 → 工具 → 循环。
  await ctx.plugin(NanoRuntime)
  await ctx.plugin(deepSeekPlugin())
  await ctx.plugin(readFilePlugin())
  await ctx.plugin(agentLoopPlugin())

  // CLI 自己不理解 Tool Call，它只把任务交给 ctx.nano。
  process.stdout.write(`${await ctx.nano.run(task)}\n`)
} catch (error) {
  process.stderr.write(`${error instanceof Error ? error.stack : String(error)}\n`)
  process.exitCode = 1
} finally {
  // 无论成功还是失败，都卸载整棵 Cordis 插件树并执行 disposer。
  await ctx.fiber.dispose()
}
