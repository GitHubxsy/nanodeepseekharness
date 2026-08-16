#!/usr/bin/env node

import { Context } from 'cordis'
import { agentLoopPlugin, deepSeekPlugin, NanoRuntime, readFilePlugin } from './nano-deepseek-harness.js'

const task = process.argv.slice(2).join(' ')
  || 'Use read_file to read README.md, then summarize it in three sentences.'
const ctx = new Context()

try {
  await ctx.plugin(NanoRuntime)
  await ctx.plugin(deepSeekPlugin())
  await ctx.plugin(readFilePlugin())
  await ctx.plugin(agentLoopPlugin())
  process.stdout.write(`${await ctx.nano.run(task)}\n`)
} catch (error) {
  process.stderr.write(`${error instanceof Error ? error.stack : String(error)}\n`)
  process.exitCode = 1
} finally {
  await ctx.fiber.dispose()
}
