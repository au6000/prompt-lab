# prompt-lab

## このリポジトリが目指すこと

プロンプトを通して、判定モデル (まず Jev。状態と型つきの問いを渡すと確率を返す) が、あとから分かる結果について何ビット読み取れるか (V-information。考え方は `README.md` の冒頭) を、どの課題でも過学習せずに測るハーネスを、人に渡せる・公開できる形で保つ。
最初の使い道は「どれかに当てはまったら見送る」阻却事由の探索で、結果が 3 つ以上の値を取る課題も、1 行に問いが複数ある課題も、同じ物差しで測る。結果が 2 値で問いが 1 つの場合の数は、0.1.0 と完全に同じになる (`scripts/parity.py` で確かめる)。例外は学習の組の `fitted` の予想で、AUC を分割の中で測るので数が変わる (`docs/adr/0012-fitted-forecast-auc-within-folds.md`)。
時刻で分けた 学習・評価・本番、月ごとにまとめた交差検証、でたらめな答えで測った 0 の位置、評価の組を見る回数の上限と記録、固定した事前登録、決まりによる選択、1 回だけの本番を、道具の側で強制する。
本体 (パッケージ `promptlab`、モジュールは `src/`) の依存は numpy だけ (Python 3.10 以上)。LLM を呼ぶ道具箱 `llm_kit/` は別のリポジトリで、依存は `pip install -e "./llm_kit[openai,google]"` のように入れたときだけ入る。理由は `docs/protocol.md`、使い方は `README.md` (日本語) と `README.en.md` (英語)。

## ディレクトリ地図

| 場所 | 中身 |
|---|---|
| `src/` | パッケージ本体 `promptlab` のモジュール (`python -m promptlab` = `plab`)。pyproject の `package-dir` で `src/` を `promptlab` に対応させているので、`pip install -e .` で入れてから使う。`cli.py` コマンドの一覧と入口、`task.py` task.json を読んだ設定と既定値、`layout.py` `.plab/` の中の置き場所・ファイルの読み書き・エージェントごとのログ、`rows.py` 時刻での分け方と、測る行の順・重み・グループ・前半後半、`expr.py` 式の安全な評価、`questions.py` 問いの種類 (predicate・choice・score) ごとの決まりと答えの型、`prompt.py` プロンプトのファイル (0.1 の形も読む)・指紋・lint、`deciders.py` 答える側 (式を当てはめる machine、テスト用の fake、Gateway の判定モデル)、`gateway.py` Vercel AI Gateway の Decisions API (`/v1/decisions`) を標準ライブラリだけで呼ぶ部分 (鍵の読み込みと伏せ字、429 と 5xx の送り直し、402 での停止)、`answers.py` 答えの保管・予算・ledger、`yardsticks.py` 物差し (結果が 3 つ以上の値を取る場合も問いが複数ある場合も、2 値と同じ式で測る)、`nulls.py` 0 の位置、`study.py` 測る手順 (どのコマンドもここを通る)、`protocol.py` 事前登録の固定・段階ごとに許すコマンドの表・評価の確かめ・推薦・選択・本番・報告、`explore.py` 学習の組だけのコマンド、`survey.py` 因子ごとの係数の時間変化 (`plab survey`。期間ごとの傾き、平均とばらつき、学び直した利得。全部の組は閉じた研究でだけ読む。理論は README.md の 7 章、ADR-0013)、`show.py` 表示、`migrate.py` 0.1 の課題フォルダを今の形に直す (0.1 が払って得た答えと ledger も移す) |
| `tests/` | promptlab の検査 (ネットワークも鍵も使わない)。リポジトリ直下の `python -m unittest` で、`llm_kit/tests/` と一緒に流れる。ほとんどは `plab` のコマンドを通して確かめる。`helpers.py` が一時フォルダに小さな課題を作り、その場でコマンドを動かす。`Wire` はネットワークの代わりで、Gateway への要求に決めた返事を返す。`golden/v01_metrics.py` は 0.1.0 の `metrics.py` をそのまま写した物で、`test_yardsticks.py` が `yardsticks.py` の数をこれと照らす (書き換えない) |
| `scripts/` | `parity.py`: 課題のフォルダを写してコマンドを流し、`.plab/` に書かれた数をすべて取り出して、版どうしで比べる。`renames_since_0.1.json`: 0.1.0 から意図して変えた名前と置き場所の一覧 (理由つき)。`parity.py compare --renames` が読む |
| `examples/synthetic/` | 合成データの例。`make_data.py` (データを作る)、`task.json`・`prereg.md`・`prompts/`、`walkthrough.py` (鍵なしで最初から最後まで流す)。作ったデータと `.plab/` は git に入れない |
| `docs/` | `protocol.md`: 事例研究から得た 7 つの教訓と、ハーネスがそれぞれに何をするか。`adr/`: 後戻りしにくい設計の決定の記録 (1 件 1 ファイル、`adr/README.md` が一覧と書き方)。決定を変えるときは新しい ADR を足し、古い方を置き換え済みにする |
| `.claude/skills/` | リポジトリと一緒に配る Claude Code のスキル。`plab-search/SKILL.md` は、Opus が探す人としてプロンプトを書いて測り、直し続けて推すまでの手順 (ADR-0010)。`plab-survey/SKILL.md` は、研究を始める前に `plab survey` を流し、探索を始めるかを決める手順 (ADR-0013。数の定義は README.md の 7 章にだけ書く)。読んでよい物・使ってよいコマンド・上限は `plab brief` の指示にだけ書き、スキルには写さない |
| `.audit/` | git に入れない手元の記録。`baseline/` は 0.1.0 で取った parity の基準 (書き換えない)、`after/` は書き直した版で取った数、`backup-v0.1/` は 0.1.0 の写し |
| `llm_kit/` | LLM を呼ぶ道具箱 (LangChain の上。プロバイダの切り替え・構造化出力と送り直し・料金の集計、`config.py` に料金表と既定のモデル)。GitHub の au6000/llm_kit の clone そのもの (中に `.git` があり、直したらここで commit して push する)。prompt-lab はこれを submodule として記録し、使う llm_kit のコミットだけを持つ。llm_kit で commit したら、prompt-lab でも `llm_kit` が指すコミットを commit し直す。`--recurse-submodules` を付けずに clone すると中は空になるが、本体の検査はそのままで通る。自分の `pyproject.toml` を持ち、prompt-lab の pyproject には入れない。`tests/` はその検査で、依存が無い環境では丸ごと飛ばす。`import llm_kit` だけでは LangChain を読まない (`__init__.py`)。promptlab からはまだ呼んでいない。使い方は `llm_kit/README.md` |

課題のフォルダ (task.json・rows.jsonl・prereg.md・prompts/) はハーネスの外に置く。ハーネスはその中の `.plab/` にだけ書く。

## 決まり

- 鍵を画面・ログ・結果・リポジトリに出さない。鍵は環境変数 (`AI_GATEWAY_API_KEY` か、鍵のファイルを指す `PLAB_KEY_FILE`) でだけ渡す。
- 公開するので、追跡するファイルに手元の道・個人名・メールアドレス・実データを書かない。実データの課題はリポジトリの外に置く。
- 物差しの定義を変えるときは、事例研究の数との一致 (README の「鍵なしの手早い始め方」と英語版の「Quick start」、protocol の数) が崩れないかを確かめ、テストを足す。
- テストはネットワークなしで通ること。
- `README.md` (日本語) が元で、`README.en.md` はその訳。章立てと中身を揃え、`README.md` を直したら同じ作業の中で `README.en.md` も直す。
