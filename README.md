# nanoDeepSeekHarness

一个不依赖 DeepSeek Harness、Cordis 或 OpenAI SDK 的最小 Agent Harness。

运行时只有 Node.js 原生能力：

```text
NanoHarness
├── deepSeekPlugin   模型插件：调用 DeepSeek /chat/completions
├── readFilePlugin   工具插件：注册 read_file
└── agentLoopPlugin  循环插件：模型 → 工具 → 结果 → 模型
```

核心观点仍然是“一切皆插件”：`NanoHarness` 只负责组合，模型、工具和 Agent Loop 都由插件提供。

> 这是独立的教学实现，不是 `deepseek-ai/deepseek-harness` 的精简发行版。

## 最小运行

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
const harness = new NanoHarness()
  .use(deepSeekPlugin())
  .use(readFilePlugin())
  .use(agentLoopPlugin())

console.log(await harness.run('读取 README.md'))
```

一次完整循环只有四步：

```text
1. Model Plugin 返回 tool_calls
2. Agent Loop 找到 Tool Plugin
3. Tool 执行结果作为 tool message 写回上下文
4. Model Plugin 根据新上下文继续回答
```

`read_file` 只能访问启动目录以内的文件，`../` 路径逃逸会被拒绝。

## 验证

```bash
npm run typecheck
npm test
npm run build
```

## 官方 API

- [DeepSeek Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion)
- [DeepSeek Tool Calls](https://api-docs.deepseek.com/guides/tool_calls)

## License

MIT
