# video-to-text：抖音视频批量转 Markdown 文稿工具

基于 `uv` + **Playwright（持久化登录态）** + **阿里 SenseVoiceSmall (`funasr`)** 构建的本地抖音视频批量转文字脚本工具。

## 特性

- **口令直粘**：支持直接将抖音 App 复制的整段分享口令粘贴到 `links.txt`，自动正则提取链接并去重。
- **持久化账号登录态**：通过 `uv run video-to-text login` 扫码登录一次，登录 Cookie 永久保存在本地 `.browser_profile/` 目录中，避免滑块验证码和未登录拦截。
- **免安装系统 `ffmpeg`**：内置 `av` (PyAV) 动态库，自动将视频音轨重采样为 `16kHz` 单声道音频并清理临时视频。
- **阿里 SenseVoiceSmall 本地加速**：自动检测 Apple Silicon (`mps`) / `cpu`，高精度中文口语识别并自带标点恢复。
- **独立 Markdown 导出与断点续转**：每个视频生成独立的 `{作者}_{标题}_{ID后6位}.md`，自动跳过已转换作品。

## 快速上手

### 1. 首次扫码保存抖音登录态（推荐）

```bash
uv run video-to-text login
```

在弹出的浏览器窗口中使用手机抖音 App 扫码登录，登录成功后直接关闭浏览器窗口即可保存状态。

### 2. 批量转换 `links.txt` 中的视频

将抖音分享口令粘贴到项目根目录的 `links.txt` 中，然后运行：

```bash
uv run video-to-text run links.txt -o ./output
```

### 3. 直接在命令行转换单条/多条口令

```bash
uv run video-to-text run -u "3.56 复制打开抖音，看看【某某的作品】... https://v.douyin.com/xxxxx/"
```

### 4. 常用参数

- `--output-dir` / `-o`：指定输出的 Markdown 文件夹（默认 `./output`）
- `--no-headless`：显示浏览器运行窗口（方便排查问题或手动过滑块）
- `--force` / `-f`：强制重新下载并转换已存在的文件
