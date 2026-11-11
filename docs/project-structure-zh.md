# nanobot 项目目录结构说明

本文面向第一次阅读 nanobot 源码的人，说明仓库中主要目录和文件的职责，以及一次消息从入口到模型、工具、记忆和回复的大致流向。

## 总体定位

nanobot 是一个 Python 编写的轻量级 AI agent 框架，外层可以接 CLI、WebUI、Telegram、Discord、Slack、Email 等入口，核心是一个异步 agent loop。它会接收用户消息，读取会话上下文和配置，调用 LLM provider，按模型返回的 tool calls 执行工具，最后把结果发回对应 channel。

简化后的运行链路：

```text
用户/外部平台
  -> channels 或 CLI
  -> MessageBus
  -> AgentLoop
  -> AgentRunner
  -> Provider/LLM
  -> Tools/Memory/Session
  -> OutboundMessage
  -> 原 channel 返回
```

## 顶层目录

```text
.
├── nanobot/              Python 后端主体代码
├── webui/                React + TypeScript WebUI 源码
├── bridge/               TypeScript 桥接服务，例如 WhatsApp bridge
├── docs/                 项目文档
├── tests/                pytest 测试
├── scripts/              安装、环境和辅助脚本
├── images/               README 和文档图片资源
├── case/                 示例动图/案例资源
├── .agent/               给 AI coding agent 的设计、安全、注意事项文档
├── .github/              GitHub Actions、issue/PR 模板等
├── pyproject.toml        Python 包元数据、依赖、ruff/pytest 配置
├── Dockerfile            Docker 镜像构建文件
├── docker-compose.yml    Docker Compose 运行配置
└── README.md             项目入口文档
```

## Python 后端：`nanobot/`

`nanobot/` 是项目核心。它包含 CLI、配置、agent loop、provider、channel、工具、session、WebUI 后端接口等模块。

```text
nanobot/
├── agent/        Agent 核心运行逻辑
├── api/          OpenAI-compatible HTTP API
├── apps/         CLI Apps/扩展应用相关能力
├── audio/        音频处理辅助代码
├── bus/          异步消息总线和运行时事件
├── channels/     各聊天平台和 WebSocket channel
├── cli/          nanobot 命令行入口
├── command/      slash command 路由和内置命令
├── config/       配置 schema、加载和路径处理
├── cron/         定时任务服务
├── pairing/      私聊/发送者配对和授权
├── providers/    LLM、图片、转写 provider 实现
├── security/     启动安全检查
├── session/      会话、历史、目标状态和 WebUI turn 协调
├── skills/       内置 skill 定义
├── templates/    内置 prompt/heartbeat 模板
├── utils/        通用工具函数
├── web/          打包后的 WebUI 静态资源入口
├── webui/        WebUI 后端 API、设置、媒体、会话服务
├── nanobot.py    Python SDK 风格入口
├── __main__.py   支持 `python -m nanobot`
└── __init__.py
```

## Agent 核心：`nanobot/agent/`

这是项目最关键的运行时目录。

```text
nanobot/agent/
├── loop.py              AgentLoop，负责消费 inbound message、构建 turn、协调 session/hook/channel
├── runner.py            AgentRunner，负责和 LLM 多轮交互、处理 tool calls、流式输出
├── context.py           构建模型上下文
├── memory.py            会话记忆和 Dream 记忆整理
├── autocompact.py       上下文自动压缩
├── model_presets.py     模型 preset 解析
├── skills.py            加载内置 skills 到 agent 上下文
├── subagent.py          子 agent 派生执行
├── hook.py              生命周期 hook
├── progress_hook.py     进度事件 hook
├── cron_turns.py        定时任务触发的 agent turn
└── tools/               暴露给模型调用的工具集合
```

核心职责分工：

- `loop.py` 更偏“调度”：从消息总线拿消息，确定 session，准备上下文，调用 runner，发布回复。
- `runner.py` 更偏“模型对话循环”：调用 provider，解析模型输出，执行工具，再把工具结果送回模型，直到得到最终回答。
- `memory.py` 管理长期/会话记忆，包括 Dream 两阶段整理。
- `tools/` 是模型可以主动调用的能力边界。

## 工具系统：`nanobot/agent/tools/`

工具是 agent 能“做事”的主要来源。

```text
nanobot/agent/tools/
├── base.py              工具基类和统一接口
├── registry.py          工具注册表
├── loader.py            自动发现内置和插件工具
├── filesystem.py        读写文件、列目录、搜索文件
├── shell.py             命令执行工具
├── exec_session.py      长时间命令会话管理
├── apply_patch.py       文件补丁编辑
├── web.py               网页抓取
├── search.py            Web 搜索
├── mcp.py               MCP server 工具接入
├── cron.py              定时任务工具
├── long_task.py         长任务/持续目标工具
├── spawn.py             子 agent 工具
├── message.py           消息发送/用户交互工具
├── image_generation.py  图片生成工具封装
├── cli_apps.py          CLI Apps 工具
├── self.py              自检/自配置工具
├── sandbox.py           shell sandbox 抽象
└── schema.py            工具 schema 生成与校验
```

配置中的 `tools` 字段决定哪些工具启用，以及工具的安全限制。例如 `exec.enable` 控制命令执行，`restrictToWorkspace` 控制文件访问边界，`web.search` 控制搜索 provider。

## Provider 系统：`nanobot/providers/`

Provider 把不同模型服务统一成 agent 可用的接口。

```text
nanobot/providers/
├── base.py                    Provider 基类、消息结构、流式事件结构
├── registry.py                Provider 元数据注册表，例如 deepseek/openai/openrouter/ollama
├── factory.py                 根据配置创建实际 provider
├── fallback_provider.py       fallback 模型链
├── openai_compat_provider.py  OpenAI-compatible 后端，DeepSeek 等 provider 复用它
├── anthropic_provider.py      Anthropic 原生接口
├── azure_openai_provider.py   Azure OpenAI
├── bedrock_provider.py        AWS Bedrock
├── github_copilot_provider.py GitHub Copilot provider
├── openai_codex_provider.py   OpenAI Codex provider
├── image_generation.py        图片生成 provider
├── transcription.py           音频转写 provider
└── openai_responses/          OpenAI Responses API 相关实现
```

以 DeepSeek 为例，它在 `registry.py` 中注册为 `deepseek`，默认 endpoint 是 `https://api.deepseek.com`，底层走 `openai_compat_provider.py`。

## Channel 系统：`nanobot/channels/`

Channel 是外部入口/出口适配层。它们负责接收平台消息并转换成统一的 `InboundMessage`，也负责把 `OutboundMessage` 发回平台。

```text
nanobot/channels/
├── base.py        Channel 基类
├── manager.py     Channel 生命周期管理和自动发现
├── registry.py    Channel 注册
├── websocket.py   WebUI 使用的 WebSocket channel
├── telegram.py
├── discord.py
├── slack.py
├── email.py
├── feishu.py
├── dingtalk.py
├── weixin.py
├── wecom.py
├── whatsapp.py
├── qq.py
├── napcat.py
├── matrix.py
├── signal.py
├── mochat.py
└── msteams.py
```

你现在启动 `nanobot gateway` 后看到的：

```text
WebSocket server listening on ws://127.0.0.1:8765/
```

就是 `channels/websocket.py` 在工作。浏览器 WebUI 通过它和后端 agent 通信。

## 消息总线：`nanobot/bus/`

```text
nanobot/bus/
├── events.py          InboundMessage、OutboundMessage 等事件结构
├── queue.py           async MessageBus
├── progress.py        工具进度事件
└── runtime_events.py  turn/model/goal 等运行时事件
```

总线的作用是解耦 channel 和 agent core。channel 不直接调用模型，而是把消息发布到 bus；agent loop 从 bus 消费消息并处理。

## 配置系统：`nanobot/config/`

```text
nanobot/config/
├── schema.py   Pydantic 配置模型，定义完整 `config.json` 结构
├── loader.py   读取、合并、校验配置
├── paths.py    配置路径和 workspace 路径处理
└── __init__.py
```

默认配置文件位置：

```text
C:\Users\<用户名>\.nanobot\config.json
```

在 Linux/macOS 上对应：

```text
~/.nanobot/config.json
```

常见需要关注的字段：

- `providers`：API Key、API Base、provider 额外参数。
- `modelPresets`：命名模型配置。
- `agents.defaults`：默认 agent 行为、workspace、模型、上下文长度等。
- `channels`：开启 WebUI、Telegram、Slack 等入口。
- `tools`：控制文件、命令、Web、MCP、图片生成等工具。

## CLI：`nanobot/cli/`

```text
nanobot/cli/
├── commands.py  Typer CLI 命令入口
├── onboard.py   初始化/配置向导
├── stream.py    CLI 流式输出
├── models.py    CLI 内部模型
└── __init__.py
```

`pyproject.toml` 中把命令注册为：

```text
nanobot = "nanobot.cli.commands:app"
```

所以安装后可以直接运行：

```powershell
nanobot status
nanobot onboard --wizard
nanobot agent -m "你好"
nanobot gateway
```

## Gateway 和 WebUI 后端

`nanobot gateway` 会启动多个运行时组件：

- agent loop
- channel manager
- websocket channel
- health endpoint
- cron service
- Dream/heartbeat 等系统任务

WebUI 后端相关代码在：

```text
nanobot/webui/
├── ws_http.py              WebSocket channel 上挂载的 HTTP 处理
├── settings_api.py         WebUI 设置读写
├── settings_routes.py      设置路由
├── transcript.py           会话记录转换和读取
├── media_api.py            媒体上传/访问
├── media_gateway.py        媒体网关
├── file_preview.py         文件预览
├── workspaces.py           workspace 管理
├── session_list_index.py   会话列表索引
├── session_automations.py  会话自动化
├── token_usage.py          token 使用统计
├── mcp_presets_api.py      MCP preset API
├── skills_api.py           skill API
└── gateway_tokens.py       WebUI token 相关逻辑
```

注意：`nanobot/webui/` 是 Python 后端服务代码，不是 React 前端源码。

## React 前端：`webui/`

```text
webui/
├── src/
│   ├── App.tsx        WebUI 主应用
│   ├── main.tsx       React 入口
│   ├── globals.css    全局样式
│   ├── components/    UI 组件
│   ├── hooks/         React hooks
│   ├── lib/           WebSocket/API/工具函数
│   ├── providers/     React context providers
│   ├── i18n/          国际化
│   ├── types/         TypeScript 类型
│   ├── workers/       Web Worker
│   └── tests/         前端测试
├── package.json
├── vite.config.ts
├── tsconfig.json
├── tailwind.config.js
├── bun.lock
└── package-lock.json
```

开发前端时：

```powershell
cd webui
npm install
npm run dev
```

或使用仓库文档里的 Bun：

```powershell
cd webui
bun install
bun run dev
```

构建后产物会写入：

```text
nanobot/web/dist/
```

这就是 `nanobot gateway` 在 `http://127.0.0.1:8765` 返回浏览器页面时需要的静态文件。源码 editable 安装时，如果没有构建 WebUI，访问 `8765` 可能会看到 `Not Found`。

## Session、记忆和长目标

```text
nanobot/session/
├── manager.py            会话历史、上下文压缩、TTL 管理
├── goal_state.py         持续目标状态
├── turn_continuation.py  turn 续跑/中断处理
├── webui_turns.py        WebUI turn 状态协调
├── keys.py               session key 工具
└── __init__.py
```

相关数据通常落在：

```text
~/.nanobot/workspace/
```

和 `~/.nanobot/` 下的会话/记忆文件中。`agent/memory.py` 负责更具体的 memory 持久化和 Dream 整理。

## 定时任务：`nanobot/cron/`

```text
nanobot/cron/
├── service.py          cron 服务主逻辑
├── bound_runner.py     绑定 agent/session 的任务 runner
├── session_turns.py    定时任务触发 agent turn
├── session_delivery.py 任务消息投递
├── types.py            cron 数据结构
└── webui_metadata.py   WebUI 展示元数据
```

启动 gateway 后日志中的 heartbeat 和 dream 系统任务就由这里调度。

## Slash Command：`nanobot/command/`

```text
nanobot/command/
├── router.py   slash command 路由
├── builtin.py  内置命令处理
└── __init__.py
```

例如聊天里用于状态、模型切换、会话管理、目标管理的一些 `/xxx` 命令，会经过这里解析。

## OpenAI-compatible API：`nanobot/api/`

```text
nanobot/api/
└── server.py
```

提供类似 OpenAI API 的 HTTP 接口，例如：

- `/v1/chat/completions`
- `/v1/models`

适合把 nanobot 接到其他程序里，作为一个 OpenAI-compatible 服务使用。

## Skills 和模板

```text
nanobot/skills/       内置 skill 定义
nanobot/templates/    HEARTBEAT.md 等模板
```

Skills 会被加载进 agent 上下文，用于告诉模型某类任务应该怎么做，例如长目标、GitHub、图片生成等能力说明。

## Bridge：`bridge/`

```text
bridge/
├── src/
├── package.json
└── tsconfig.json
```

这里是 TypeScript 桥接服务，当前主要用于类似 WhatsApp 这类需要外部桥接进程的集成。`pyproject.toml` 中通过 `force-include` 把 `bridge` 打包进 Python wheel。

## 测试：`tests/`

`tests/` 按 Python 包结构组织，用 pytest 运行：

```powershell
pytest
```

常用开发检查：

```powershell
pytest
ruff check nanobot/
```

前端测试在 `webui/` 内：

```powershell
cd webui
npm run test
```

## 配置和启动相关文件

```text
pyproject.toml      Python 依赖、包构建、命令入口、ruff、pytest 配置
hatch_build.py      Python wheel 构建时的自定义逻辑
Dockerfile          容器镜像构建
docker-compose.yml  容器编排
scripts/install.ps1 Windows 安装脚本
scripts/install.sh  macOS/Linux 安装脚本
scripts/conda-env.ps1 本地新增的 conda 环境脚本
```

## 常见开发入口

后端本地开发：

```powershell
pip install -e ".[dev,api]"
nanobot onboard --wizard
nanobot agent -m "你好"
```

启动 WebUI 后端：

```powershell
nanobot gateway
```

开发 WebUI：

```powershell
cd webui
npm run dev
```

构建 WebUI 给 gateway 使用：

```powershell
cd webui
npm run build
```

## 阅读源码建议

如果只是想知道“消息怎么跑起来”，建议按这个顺序看：

1. `nanobot/cli/commands.py`：命令行入口怎么启动 agent/gateway。
2. `nanobot/channels/websocket.py`：WebUI channel 怎么接入。
3. `nanobot/bus/events.py` 和 `nanobot/bus/queue.py`：消息结构和消息总线。
4. `nanobot/agent/loop.py`：一次用户消息如何进入 agent core。
5. `nanobot/agent/runner.py`：模型调用和工具调用循环。
6. `nanobot/providers/registry.py` 和 `nanobot/providers/factory.py`：provider 如何从配置解析。
7. `nanobot/agent/tools/registry.py` 和具体工具文件：模型能调用哪些能力。
8. `nanobot/session/manager.py` 和 `nanobot/agent/memory.py`：会话历史、上下文压缩和记忆。

## 一句话总结

nanobot 的结构可以理解为：

```text
入口层 channels/cli/webui
  + 消息总线 bus
  + 核心 agent loop/runner
  + provider 适配层
  + tools 能力层
  + session/memory 状态层
  + config/schema 配置层
```

WebUI、Telegram、CLI 等只是不同入口；真正执行推理、工具调用、记忆和会话管理的是同一套 Python agent core。
