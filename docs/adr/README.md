# 設計の決定の記録 (ADR)

後戻りしにくい設計の決定を、1 件 1 ファイルで残します。読み手はこのリポジトリのコードを直すエンジニアです。

決定を変えるときは、古い ADR を書き換えません。新しい番号で ADR を足し、古い方の「状態」を「置き換え済み (ADR-00NN)」に変えます。
1 つの ADR には背景、決定、理由、捨てた案、引き受けること、確かめ方を書きます。

0.1 のときからある手順の決まり (時刻で分けた組、0 の位置、評価の確かめの上限など) の理由は [protocol.md](../protocol.md) に、物差しの理論は [README.md](../../README.md) の冒頭にあります。

| 番号 | 決定 | 状態 |
|---|---|---|
| [0001](0001-measure-v-information.md) | 物差しの的は V-information | 採用 |
| [0002](0002-one-code-path-for-yardsticks.md) | 物差しは 1 本の計算で、2 値専用の計算を持たない | 採用 |
| [0003](0003-answers-as-distributions.md) | 答えは確率分布のまま持ち、断りは一様分布として数える | 採用 |
| [0004](0004-answer-store-keyed-by-what-the-decider-reads.md) | 答えの記録の鍵は、判定モデルが読む物から作る | 採用 |
| [0005](0005-decider-registry-and-decisions-api.md) | 判定モデルは表で引き、呼び先は /v1/decisions、モデルは事前登録で固定する | 採用 |
| [0006](0006-protocol-by-phase-table-and-lock.md) | 手順は段階の表と Lock で守る | 採用 |
| [0007](0007-one-shot-migration-from-0-1.md) | 0.1 との互換は一回きりの移し替えで扱う | 採用 |
| [0008](0008-llm-kit-as-a-clone-in-place.md) | llm_kit は別のリポジトリのまま、フォルダそのものを clone として置く | 採用 |
| [0009](0009-package-modules-in-src.md) | パッケージ promptlab のモジュールは src/ に直接置く | 採用 |
| [0010](0010-searcher-as-a-claude-code-skill.md) | 探す人は Claude Code のスキルとして動かし、promptlab に LLM を呼ぶコードを入れない | 採用 |
| [0011](0011-transfer-gain-keeps-the-training-direction.md) | 評価と本番の組では、学習の組で覚えた向きのまま測る利得も出す | 採用 |
| [0012](0012-fitted-forecast-auc-within-folds.md) | 学習の組の fitted の予想では、AUC を分割の中で測り、分割をまたいで行を比べる物差しを出さない | 採用 |
| [0013](0013-survey-of-coefficient-change.md) | 係数の時間変化は `plab survey` で因子ごとに測り、評価と本番の行は閉じた研究でだけ読む | 採用 |
