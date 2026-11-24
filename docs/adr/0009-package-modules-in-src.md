# ADR-0009: パッケージ promptlab のモジュールは src/ に直接置く

状態: 採用 (2026-11-20、0.2.0)

## 背景
0.2.0 の本体は `prompt-lab/promptlab/` にあり、同じ名前が 2 段に並んでいました。

## 決定
- パッケージのモジュール (`cli.py`、`yardsticks.py` など) を `src/` に直接置きます。`src/promptlab/` のように、もう 1 段は作りません。
- `pyproject.toml` の `package-dir` で `src/` を `promptlab` に対応させます。配るのは `src/` の `.py` だけです。
- import の名前 (`promptlab`) とコマンド (`plab`) は変えません。
- 使う前に `pip install -e .` で入れます。

## 理由
同じ名前の 2 段のフォルダーが無くなります。
リポジトリの直下は文書とフォルダーだけのままです。
`src/` は、リポジトリの直下で Python を動かしても import の検索の道に入らないので、モジュールの名前 (`task`、`show`、`cli` など) がほかのパッケージの同じ名前を覆い隠しません。

## 捨てた案
- `promptlab/` のまま残す。名前の 2 段が残ります。
- リポジトリの直下をパッケージそのものにする ([ADR-0008](0008-llm-kit-as-a-clone-in-place.md) の llm_kit と同じ形)。2026-11-20 に試しました。直下にモジュール 19 個と文書が並び、直下で Python を動かすとモジュールの名前がほかとぶつかるおそれがありました。
- ふつうの src 形式の `src/promptlab/`。`promptlab` のフォルダーが残ります。
- パッケージの名前をコマンドと同じ `plab` に変える。import の名前が変わります。

## 引き受けること
`src/` という名前からは、中身がどのパッケージかが分かりません。CLAUDE.md の地図と pyproject の `package-dir` で示します。
リポジトリの中から `python -m promptlab` を動かすには、先に `pip install -e .` が要ります。

## 確かめ方
2026-11-20 に、作った wheel の中身、別のフォルダーからの `import promptlab`、2 つの環境の検査、`scripts/parity.py` による合成の例と FX の比較を確かめました。
