# LanguageTool sanity check

**Status: PASS** -- `CONFUSION_RULE_*` rules are firing, so the n-gram language model loaded correctly.

Without n-gram data LanguageTool starts normally and answers every request, but silently stops firing the confusion-set rules that resolve homophones from context. Those are central to this project, so this check gates Phase 1.

The probes marked *needs n-grams* were chosen by diffing this server against a control container started without `langtool_languageModel`: each one is caught only when the language model is present. That matters, because several classic German confusions -- das/dass among them -- are caught by hand-written rules that need no n-grams, and would give a false pass.

| Probe | Needs n-grams | Expected | Result | Rules fired |
|---|:--:|---|:--:|---|
| `Das hat mir fiel geholfen.` | yes | `viel` | ok | `CONFUSION_RULE_FIEL_VIEL` |
| `Ich habe ihm Garten gearbeitet.` | yes | `im` | ok | `CONFUSION_RULE_IHM_IM` |
| `Er hat mit das Buch gegeben.` | yes | `mir` | ok | `CONFUSION_RULE_MIT_MIR`, `PRAEP_DAT` |
| `Die Leere an der Universität ist gut.` | yes | `Lehre` | ok | `CONFUSION_RULE_LEERE_LEHRE` |
| `Ich hoffe dass du kommst.` | no | `,` | ok | `SUBJUNKTION_KOMMA` |
| `Ich weiß, das du morgen kommst.` | no | `dass` | ok | `RELATIVPRONOMEN_STATT_DASS` |
| `Das ist ein Fehlar.` | no | `Fehler` | ok | `GERMAN_SPELLER_RULE` |

## Server log evidence

```
(no matching line)
```
