#!/usr/bin/env node

import { agentLoopPlugin, deepSeekPlugin, NanoHarness, readFilePlugin } from './nano-deepseek-harness.js'

const task = process.argv.slice(2).join(' ')
  || 'Use read_file to read README.md, then summarize it in three sentences.'

const harness = new NanoHarness()
  .use(deepSeekPlugin())
  .use(readFilePlugin())
  .use(agentLoopPlugin())

try {
  process.stdout.write(`${await harness.run(task)}\n`)
} catch (error) {
  process.stderr.write(`${error instanceof Error ? error.stack : String(error)}\n`)
  process.exitCode = 1
}
