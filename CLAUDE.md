# CLAUDE.md — HarnessEvolver Agent Instructions

本文件只保存代理在本仓库工作时必须遵守的原则、边界和规范。当前实现状态以 `README.md`、`docs/architecture.md`、当前代码和 Harbor/verifier 证据为准。

## 核心定位

HarnessEvolver 的目标是实现一个自研 TerminalBench 2.0 harness coding agent，并用 Heuristic Learning 持续优化它，最终冲击 SOTA。

最重要的边界：

- **Worker 是本项目自己实现的 TerminalBench coding agent loop**。被 Harbor/TerminalBench 评测的是我们的 Worker loop 和 harness，而不是 Codex、ForgeCode、Claude Code 或其他外部 agent。
- **Codex 是外部 HL updater / meta-coding framework**。Codex 读取失败包、轨迹、日志、回归约束和当前 harness 状态，然后修改本仓库的 Worker loop 与 harness。Codex 不直接作为被评测 Worker。
- **ForgeCode、Codex、Claude Code、Factory Droid/Missions 的功能只能作为设计模式参考**。例如 goal、todo gate、tool-call correction、reasoning config、context compaction、recovery prompt、validation contract、feature/milestone decomposition 等，都应实现到本项目自研 Worker loop/harness 或外部 HL policy 层中，而不是把 benchmark 执行委托给参考项目。
- **Mission debug 是外部循环调试/选择层**。`meta/missions.py`、`scripts/mission_debug.py` 和 `scripts/run_campaign.py --mission-debug` 只能基于既有 campaign/trial evidence 产出 validation contracts、feature candidates 和外部控制建议；不得启动外部 worker、不得直接解 TerminalBench task、不得标记 campaign complete、不得提交 leaderboard。

## 参考源码位置

以下源码已克隆到临时参考目录，用于只读研究和设计借鉴：

- Codex CLI: `/tmp/harness-evolver-refs/codex`
- ForgeCode: `/tmp/harness-evolver-refs/forgecode`

使用规范：

- 参考目录只读使用，不要把其源码复制进本仓库形成不可维护的平行实现。
- 借鉴时必须提炼为本项目自己的接口、测试和文档，不要无脑照搬命名、流程或 UX。
- Codex 参考重点：`codex exec` 非交互执行、JSONL 事件、目标/预算语义、配置覆盖、sandbox、输出 schema、工具/goal 约束。
- ForgeCode 参考重点：provider/reasoning 配置、todo gate、tool error reflection、工具描述质量、context compaction、agent/skill 配置面。
- 若参考源码与本仓库当前文档或本文件边界冲突，以本文件、`README.md`、`docs/architecture.md` 和当前代码为准。

## 架构原则

1. **自研 Worker loop 是核心资产**
   `bench/agent.py` 及其后续拆分模块必须承载真实 TerminalBench 任务执行：任务理解、环境感知、工具调用、状态维护、todo、上下文压缩、恢复、验证和完成判断。

2. **Harness 是 HL 的 Policy**
   `harness/` 下的 prompts、tools、planning、context、entrypoint、recovery、verification、skill loading、goal/todo 等都是可学习和可编辑 policy。配置文件也是 policy 的一部分。

3. **Codex 只在外层更新 policy**
   Codex-backed UpdateEngine 的职责是分析失败并修改本项目代码。它不能绕过 Worker、不能直接解 TerminalBench 任务、不能编辑 benchmark tests/solutions 来提高分数。

4. **反馈必须可追溯**
   每次 trial、失败、Codex 更新、回归结果、提交决策都必须落到 `trials/` 的文件系统 memory 中。不要把关键证据只留在对话里。

5. **吸收与压缩同等重要**
   新失败、新日志、新 reward 要被吸收进 memory；当 prompt、规则、recovery pattern 或组件耦合膨胀时，要触发压缩，把局部补丁折叠为简洁、可验证的表示。

6. **验证独立于实现叙事**
   Codex 或 Worker 声称“完成”不算完成。状态必须来自 Harbor/verifier、项目测试、回归快照和明确的 submit gate。

7. **串行优先，局部并行**
   HL 迭代以一个改进切片为单位串行推进。只读探索、日志解析、fixture 检查可以局部并行；不要让多个写入型 agent 同时改同一片 harness。

## 开发规范

- 开始实质开发前先读 `README.md` 和 `docs/architecture.md`，确认当前边界和运行方式。
- 涉及架构、计划、阶段收敛时，同时核对当前文档、当前代码和可用的 Harbor/verifier 证据，不要只相信历史状态说明。
- 当前基础设施已收敛时，后续优先做能提升 score 或证据质量的窄切片：Harbor/environment error attribution、timeout phase separation、Worker policy refinement、regression hardening、mission-selected Codex update。
- 不要把基础设施或局部回归通过误读为 SOTA 已达成。后续优化仍必须由真实 Harbor/verifier、项目测试、回归快照和 campaign evidence 支撑。
- 修改 harness 行为时，优先补测试或 fixture。窄改动用窄测试；共享行为、配置 schema、runner/adapter 变更要扩大测试面。
- 对 `trials/`、`terminal-bench-tasks/`、jobs 输出等运行时数据保持谨慎。不要把大体积结果、secret、临时 job artifact 加入 git。
- 默认不为纯运行态检查、日志观察、临时实验或 artifact 清理提交 git commit；但一旦修改了本仓库 tracked 代码、配置或文档，并且该改动已经通过必要验证或被 Codex update 接受，就必须用一个小而完整的 git commit 收尾。不要把已完成的代码优化、AGENTS/docs 规则更新、accepted Codex update patch 留成长期 dirty worktree。
- 当用户要求启动长时间 campaign、50 rounds、Codex update 或类似后台运行任务时，必须使用当前 Codex 会话的 **Codex background terminal**（用户执行 `/ps` 能看到的 Background terminals）运行；不要用 Linux 级后台命令（如 `nohup`、`setsid`、`&`、`disown`），也不要用 sub-agent/子代理来替代 Codex background terminal。只有在用户明确要求 Linux 后台，或当前环境没有可用 Codex background terminal 能力时，才允许使用 Linux 后台 fallback；fallback 必须说明原因、记录 PID/log/status 命令，并保持 git 工作树不被未忽略运行壳污染。
- 标准多轮 HL / Codex update campaign 不是全量任务评测时，每轮必须更换评估题目，默认使用可复现的任务池轮换，不要在 50 rounds 等长跑中反复只测首次随机抽到的小样本；只有做固定复现、回归或用户明确要求时，才使用 `--no-rotate-tasks-per-iteration` 或显式固定任务列表。
- 可用 `--round-task-concurrency N` 提高同一 HL round 内多个 Harbor task 的吞吐；round 之间、Codex update、pre/post regression 仍必须串行，不要并发多个 Codex update 或多个写入型 harness 改动。
- solved-task regression 只能和当前 worker 的同模型作用域 snapshot 对比；切换到 `deepseek-v4-flash` 或任何其他模型时，不得用其他模型的历史通过项阻塞 pre/post regression gate。

## Worker Loop 规范

Worker 必须逐步演进为可信 TerminalBench coding agent：

- 不允许自报 `PASSED`。通过/失败必须来自 Harbor/verifier 或受控测试结果。
- 必须构造完整任务上下文：instruction、workspace、环境、任务元数据、历史失败、harness memory、可用工具和完成条件。
- 执行前应做有界入口发现，不要盲目编辑。
- 多步骤任务必须有 todo 状态；存在 pending/in_progress todo 时不得完成。
- shell 是 universal adapter，但结构化 file read/edit/write/search/verify/todo/goal 工具也应保留。
- 工具调用失败必须进入 correction/recovery，而不是无限重试同一错误。
- 上下文过长时应压缩轨迹和观察结果，保留影响下一步决策的证据。
- Worker 可只读读取 campaign goal，但不能把 campaign 标记为 complete。

## Codex-backed HL Updater 规范

当实现 Codex UpdateEngine 时必须遵守：

- 用 `codex exec` 非交互模式运行，保存 JSONL events、final message、exit code、git diff、token/budget 信息（能取到则记录）。
- Codex prompt 必须明确：它优化的是本项目 Worker loop/harness，不是直接解当前 TerminalBench task。
- 每次 Codex 更新必须有结构化输入包：失败任务、score、trajectory slice、stderr/stdout、verifier 输出、相关组件、允许编辑范围、回归契约、必须运行的验证命令。
- Codex work packet 可包含 `mission_debug`，但它只是外部循环选择和约束信息；Codex 仍只能做一个有界 Worker/harness policy 改进切片。
- Codex 只允许做一个有界 harness 改进切片。拒绝无关重构、benchmark 篡改、只改日志不改行为、削弱验证/提交门禁的 patch。
- Codex patch 接受前必须由本项目确定性代码审查 diff、运行测试/回归，并在失败时回滚或标记为 rejected。

## Goal / Todo / Reasoning 规范

- Goal 是 HL campaign 级目标与预算，不是 Worker 的普通计划清单。
- Todo 是 Worker task/session 级执行纪律，不替代 campaign goal。
- Goal 语义可借鉴 Codex：objective、status、token/time budget、completion-only update、budget exhaustion 不等于成功。
- Reasoning config 是 harness 配置面，必须支持 OpenAI 风格 effort（如 `xhigh`）和 Anthropic 风格 thinking budget（如 `max_tokens`），并做 provider-specific validation。
- 所有模型、base URL、API key env、reasoning effort、max output、timeout、retry 都应通过配置传递，不要硬编码在 Worker 或 meta updater 内。
- 多 provider/role 切换优先使用参数和配置：默认 `worker`/`worker_deepseek` 使用 DeepSeek 做开发/普通跑，正式 GPT 跑用 `--worker-role worker_gpt`。不要在源码里硬编码用户 API key 或模型参数。

## Harbor / TerminalBench 规范

- Harbor 相关依赖若已存在，不要重复做安装型工作；后续重点应放在正确接入真实 CLI、adapter 和 artifact 解析。
- Harbor CLI 以当前安装版本输出为准；不要继续使用过时的 `--agent-config` 假设。
- TerminalBench 任务目录默认使用 `terminal-bench-tasks/terminal-bench/`。不要编辑任务 tests、solutions 或 benchmark 定义来提高分数。
- `bench/harbor.py` 和 `scripts/run_trial.py` 必须以真实 Harbor job 目录为事实来源，解析 score、verifier 输出、trajectory、stdout/stderr 和 artifacts。
- 回归快照只能从真实 verified pass 创建。

## Leaderboard Submit 规范

- 自动提交必须显式配置开启，默认关闭。
- 一个 HL campaign 中最多提交一次。
- 达到分数阈值后必须先通过 submit gates：任务覆盖、score、回归、git 状态、artifact 完整性、Harbor auth、visibility/share 配置。
- 提交/上传是终止动作。提交成功或发生终止性失败后，本轮 HL loop 必须停止。
- 提交前要写入 submit intent，防止崩溃后重复提交；提交结果写入 `trials/submissions/`。visibility、share org/user 和 `--yes` 确认必须来自配置或显式 CLI 参数，默认仍不得自动提交。

## 安全与数据规范

- 不要把 API key、token、cookie、账号凭据写入 tracked files、trials memory、diff 或日志。
- 配置中保存 env var 名称和 redacted display，secret 值从环境读取。
- 不要运行破坏性命令清理用户数据；删除大文件或运行时 artifact 前先确认路径和 git 状态。
- 不要绕过 benchmark integrity：禁止修改 TerminalBench tests、oracle solution、verifier 或 task definition 来提高分数。
- Codex/ForgeCode 参考源码位于 `/tmp`，不属于本项目源码和提交范围。

## 常用验证命令

```bash
pytest tests/ -v
python scripts/run_trial.py --path terminal-bench-tasks/terminal-bench --task <task-name>
python scripts/run_campaign.py --dry-run --tasks fix-git,vulnerable-secret --worker-role worker_deepseek
python scripts/run_campaign.py --task <task-name> --worker-role worker_gpt
python scripts/run_campaign.py --dry-run --task <task-name> --submit-check --submit-share-org TimWhite-AGI --submit-share-user timwhitez --submit-share-yes
python scripts/run_campaign.py --dry-run --task <task-name> --mission-debug
python scripts/mission_debug.py --campaign-summary trials/summaries/full-scale-deepseek_campaign.json --max-features 3
python scripts/regression_check.py --task fix-git --lane smoke
python scripts/compare_trials.py trial_001 trial_002
python scripts/workspace_report.py --json
harbor --help
harbor run --help
codex exec --help
```

命令是否可用和参数形态以本机实际输出为准。若命令输出与文档冲突，以本机输出为准，并更新代码/roadmap 中的假设。
