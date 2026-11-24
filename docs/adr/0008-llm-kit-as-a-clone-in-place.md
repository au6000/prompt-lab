# ADR-0008: llm_kit は別のリポジトリのまま、フォルダそのものを clone として置く

状態: 採用 (2026-11-20)

## 背景
LLM を呼ぶ道具箱 llm_kit を、prompt-lab の一部として使いたいと考えました。
llm_kit には LangChain などの依存があり、prompt-lab の本体は numpy だけで動きます。
最初は自分の pyproject と検査を持つ下位プロジェクトとして入れました。そのため、`llm-kit/llm_kit/` の二重のフォルダと、2 組の管理のファイルができました。

## 決定
- GitHub の llm_kit のリポジトリの直下を、パッケージの中身そのものにします。`pyproject.toml` の `package-dir` で、直下を `llm_kit` に対応させます。
- prompt-lab の `llm_kit/` はそのリポジトリの clone にします。直したら、`llm_kit/` の中で commit と push をします。
- 依存の一覧は `llm_kit/pyproject.toml` の 1 か所に置き、prompt-lab の pyproject には入れません。
- `import llm_kit` だけでは LangChain を読まず、名前を使った時点で読みます。
- prompt-lab の最初のコミットでは、submodule として記録します。

## 理由
写しが 1 つなら、二重管理になりません。
遅らせて読む形なら、numpy だけの環境でも、リポジトリ直下の `python -m unittest` が llm_kit を import して落ちません。

## 捨てた案
- 下位プロジェクト `llm-kit/` に、自分の pyproject、検査、README を持たせる。二重のフォルダと 2 組の管理のファイルになります。
- prompt-lab の pyproject の追加の依存として、llm_kit の依存を並べる。同じ一覧が 2 か所になります。

## 引き受けること
`llm_kit/` の中で Python を動かすと、`types.py` が標準ライブラリを覆い隠します。検査は `llm_kit/` の親で流します。
`git add .` だけでは、llm_kit の中身が prompt-lab に入りません。

## 確かめ方
2026-11-20 に、作った wheel の中身と、新しい venv への 2 通りの入れ方 (wheel と `pip install -e`) を確かめました。
