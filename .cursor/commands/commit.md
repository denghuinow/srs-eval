# 生成并提交 Commit Message

请执行以下步骤：

1. **检查暂存区状态**
   - 运行 `git status` 查看暂存的文件
   - 运行 `git diff --cached` 查看暂存区的具体变更内容

2. **生成 Commit Message**
   - 根据暂存的变更内容，生成一个简洁的中文 commit message
   - **格式要求**：`[类型] 简短的中文描述`
   - **类型包括**：feat（新功能）、fix（修复）、docs（文档）、style（格式）、refactor（重构）、test（测试）、chore（构建/工具）
   - **要求**：
     - 使用中文描述
     - 简洁明了，不超过50字
     - 准确反映本次修改的核心内容
   - **示例**：
     - `feat: 添加用户注册功能`
     - `fix: 修复登录验证逻辑错误`
     - `docs: 更新README文档`
     - `refactor: 重构需求解析模块`

3. **展示生成的 Commit Message**
   - 将生成的 commit message 展示给用户
   - 等待用户确认

4. **执行提交**
   - 用户确认后，运行 `git commit -m "生成的commit message"` 执行提交
   - 如果用户拒绝，则取消操作

**重要提示**：
- 如果暂存区为空，提示用户先使用 `git add` 添加文件
- 生成的 commit message 必须使用中文
- 确保 commit message 格式正确：`[类型]: 描述` 或 `类型: 描述`

