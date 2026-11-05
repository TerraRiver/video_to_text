# video-to-text：抖音视频批量转 Markdown 文稿工具

基于 `uv` + **Playwright（持久化登录态）** + **阿里 SenseVoiceSmall (`funasr`)** + **LLM 保真校对** 构建的本地抖音视频批量转文字脚本工具。

## 特性

- **口令直粘**：支持直接将抖音 App 复制的整段分享口令粘贴到 `links.txt`，自动正则提取链接并去重。
- **持久化账号登录态**：通过 `uv run video-to-text login` 扫码登录一次，登录 Cookie 永久保存在本地 `.browser_profile/` 目录中，避免滑块验证码和未登录拦截。
- **免安装系统 `ffmpeg`**：内置 `av` (PyAV) 动态库，自动将视频音轨重采样为 `16kHz` 单声道音频并清理临时视频。
- **阿里 SenseVoiceSmall 本地加速**：自动检测 Apple Silicon (`mps`) / `cpu`，高精度中文口语识别并自带标点恢复。
- **大模型保真错别字校对（可选）**：结合视频标题、作者与 `#话题标签` 上下文，通过 OpenAI 兼容接口（DeepSeek / 通义千问 / Kimi / OpenAI / Ollama）修正同音错别字与专有名词，同时用 `<details>` 折叠块保留原始语音识别文本。
- **独立 Markdown 导出与断点续转**：每个视频生成独立的 `{作者}_{标题}_{ID后6位}.md`，自动跳过已转换作品，并支持对已生成的 `.md` 文件单独补充执行大模型校对。

## 快速上手

### 1. 首次扫码保存抖音登录态（推荐）

```bash
uv run video-to-text login
```

在弹出的浏览器窗口中使用手机抖音 App 扫码登录，登录成功后直接关闭浏览器窗口即可保存状态。

### 2. 配置大模型保真校对（可选）

复制 `.env.example` 为 `.env`，填入你的 OpenAI 兼容 API Key（如 DeepSeek 或通义千问）：

```bash
cp .env.example .env
```

编辑 `.env`：
```ini
LLM_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
```

### 3. 批量转换 `links.txt` 中的视频

将抖音分享口令粘贴到项目根目录的 `links.txt` 中，然后运行：

```bash
uv run video-to-text run links.txt -o ./output
```

- 如果已配置 `.env`，会在语音识别后自动执行大模型保真校对（可通过 `--no-llm` 临时关闭）。
- 也可以直接在命令行传入单条口令：
  ```bash
  uv run video-to-text run -u "3.56 复制打开抖音，看看【某某的作品】... https://v.douyin.com/xxxxx/"
  ```

### 4. 对已生成的 `.md` 文稿单独执行大模型校对

如果你之前已经生成了未校对的 `.md` 文稿，配置好 `.env` 后可直接运行：

```bash
uv run video-to-text check ./output
```

无需重新下载视频或重跑语音识别，即可原地升级为带 AI 校对版与 `<details>` 原始文本折叠块的文稿（加 `--force` 可强制重新校对）。
