# nanoDeepSeekHarness

一个不依赖 `deepseek-ai/deepseek-harness` 的最小 Agent Harness。

它只使用两个基础依赖：

- **Cordis**：Context、插件加载、依赖注入和生命周期；
- **OpenAI SDK**：调用 DeepSeek 的 OpenAI 兼容接口。

```text
Cordis Context
├── NanoRuntime       最小能力注册表
├── deepSeekPlugin    模型插件
├── readFilePlugin    工具插件
└── agentLoopPlugin   循环插件
```

核心仍然只负责组合，模型、工具和 Agent Loop 都是 Cordis Plugin。

> 这是独立的教学实现，不是 `deepseek-ai/deepseek-harness` 的精简发行版。

## 代码阅读顺序

核心实现集中在 [`src/nano-deepseek-harness.ts`](./src/nano-deepseek-harness.ts)，文件按下面的顺序分成五段：

1. `Tool / Model`：插件之间的最小协议；
2. `NanoRuntime`：只保存能力的 Cordis Service；
3. `deepSeekPlugin`：OpenAI SDK 与 DeepSeek 的适配层；
4. `agentLoopPlugin`：Tool Call 循环与消息回填；
5. `readFilePlugin`：一个普通文件读取函数。

命令行入口在 [`src/cli.ts`](./src/cli.ts)，它只负责组装插件、提交任务和释放 Context。

## 运行

要求 Node.js 22 或更高版本。

```bash
npm install
export DEEPSEEK_API_KEY='你的 API Key'
npm run dev -- '使用 read_file 读取 README.md，并用三句话概括'
```

默认模型是 `deepseek-v4-flash`，可以通过环境变量调整：

```bash
DEEPSEEK_MODEL='deepseek-v4-pro' npm run dev -- '你好'
```

## 最小代码

```ts
const ctx = new Context()

await ctx.plugin(NanoRuntime)
await ctx.plugin(deepSeekPlugin())
await ctx.plugin(readFilePlugin)
await ctx.plugin(agentLoopPlugin)

console.log(await ctx.nano.run('读取 README.md'))
await ctx.fiber.dispose()
```

一次完整循环只有四步：

```text
1. Model Plugin 返回 tool_calls
2. Agent Loop 找到 Tool Plugin
3. Tool 结果作为 tool message 写回上下文
4. Model Plugin 根据新上下文继续回答
```

为了保持最小，示例没有加入沙箱、权限确认、持久化和并发调度；`read_file` 会直接读取模型给出的路径。插件卸载时，它注册的能力会随 Cordis 生命周期撤销。

## 验证

```bash
npm run typecheck
npm test
npm run build
```

## 参考

- [Cordis](https://github.com/cordiverse/cordis)
- [OpenAI Node SDK](https://github.com/openai/openai-node)
- [DeepSeek Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion)
- [DeepSeek Tool Calls](https://api-docs.deepseek.com/guides/tool_calls)

## License

MIT
