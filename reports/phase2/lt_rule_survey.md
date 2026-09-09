# Phase 2: what LanguageTool calls our injected errors

2,346 damaged sentences carrying 3,492 injected errors. A match counts as being about an injected error if it overlaps that error's span, give or take 1 token -- a dropped comma is flagged on the words either side of the gap, not on the gap.

## Detection by injected type

LanguageTool finds **2,211 of 3,492** (63.3%) of the errors we planted.

| Injected type | Planted | Detected | Rate | Most common rule |
|---|---:|---:|---:|---|
| `R:SPELL` | 1,496 | 1,377 | 92.0% | `GERMAN_SPELLER_RULE` |
| `R:DET:FORM` | 901 | 480 | 53.3% | `DE_AGREEMENT` |
| `R:ADP` | 616 | 58 | 9.4% | `GERMAN_SPELLER_RULE` |
| `R:ORTH` | 318 | 262 | 82.4% | `GERMAN_SPELLER_RULE` |
| `M:DET` | 102 | 18 | 17.6% | `GERMAN_SPELLER_RULE` |
| `M:PUNCT` | 59 | 16 | 27.1% | `KOMMA_ZWISCHEN_HAUPT_UND_NEBENSATZ_2` |

A low rate is not a failure of the corruptor. It means LanguageTool cannot see that error, which is exactly where a trained model has something to add -- and it also means no template can be derived from a rule id there, so those explanations have to be written from the error type instead.

## Rule ids to write templates for

50 distinct rule ids fired. The top of this list is where the explanation coverage is: hand-writing the first hundred covers most of the volume, and the tail can be generated.

| Rule id | Fired | Message |
|---|---:|---|
| `GERMAN_SPELLER_RULE` | 1,812 | Möglicher Tippfehler gefunden. |
| `DE_AGREEMENT` | 357 | Möglicherweise passen das Nomen und die Wörter, die das Nomen beschreiben, grammatisch nic |
| `PRAEP_DAT` | 28 | Hier scheint „dem“ der korrekte Artikel zu sein. |
| `UPPERCASE_SENTENCE_START` | 10 | Dieser Satz fängt nicht mit einem großgeschriebenen Wort an. |
| `KOMMA_ZWISCHEN_HAUPT_UND_NEBENSATZ_2` | 10 | Hier sollte ein Komma eingefügt werden, wenn es sich um einen Haupt- und Nebensatz oder zw |
| `KOMMA_ZWISCHEN_HAUPT_UND_NEBENSATZ` | 9 | Hier sollte ein Komma eingefügt werden, wenn es sich bei dem hinteren Satzteil um einen Ne |
| `DE_MULTITOKEN_SPELLING_TWO` | 8 | Hier liegt möglicherweise ein Tippfehler vor. |
| `DEN_DEM` | 7 | Meinten Sie „dem“? (Alternativ prüfen Sie, ob ‚von‘ ersetzt werden muss.) |
| `DE_CASE` | 5 | Außer am Satzanfang werden nur Nomen und Eigennamen großgeschrieben. |
| `PRP_VER_PRGK` | 5 | Bitte prüfen Sie, ob „Reihen“ hier als Nomen gebraucht wird und daher großgeschrieben werd |
| `PRAEP_GEN` | 5 | Die Präposition ‚aufgrund‘ erfordert standardsprachlich den Genitiv. |
| `KONJUNKTION_DASS_DAS` | 5 | Meinten Sie möglicherweise die Konjunktion ‚dass‘? |
| `PRAEP_AKK` | 5 | Die Präposition ‚durch‘ erfordert in der Regel den Akkusativ. |
| `SIMPLE_AGREEMENT_MAS` | 4 | Möglicherweise fehlende grammatische Übereinstimmung von Artikel und Nomen. |
| `SUBJUNKTION_KOMMA_2` | 4 | Der von ‚dass‘ eingeleitete Nebensatz muss in der Regel mit einem Komma vom Hauptsatz getr |
| `SUBJUNKTION_KOMMA` | 3 | Der von ‚dass‘ eingeleitete Nebensatz muss in der Regel mit einem Komma vom Hauptsatz getr |
| `EINES_GENITIV` | 3 | Hier sollte vermutlich der Genitiv benutzt werden. |
| `DASS_MIT_VERB` | 3 | Meinten Sie den Artikel „Das“? |
| `DE_MULTITOKEN_SPELLING_THREE` | 2 | Hier liegt möglicherweise ein Tippfehler vor. |
| `F_ANSTATT_PH` | 2 | Möchten Sie die modernere Schreibweise „Fotografie“ verwenden? |
| `KOMMA_INFINITIVGRUPPEN` | 2 | Wenn es sich hier um eine Infinitivgruppe (‚zu‘ + Grundform) handelt, muss in der Regel ei |
| `ZWISCHEN_SEIT_BIS` | 2 | Ein Zeitraum sollte mit ‚zwischen X und Y‘ oder mit ‚von X bis Y‘ markiert werden. |
| `DE_UNPAIRED_QUOTES` | 2 | Zeichen ohne sein Gegenstück: ‚„‘ scheint zu fehlen |
| `IM_OSTEN` | 2 | Das Nomen „Westen“ wird großgeschrieben. |
| `DE_COMPOUNDS` | 2 | Dieses Wort wird zusammengeschrieben. |
| `ART_KLEINES_NOMEN` | 2 | Wenn es sich um ein Nomen handelt, muss es großgeschrieben werden. |
| `NOMEN_KLEIN` | 1 | Meinten Sie „Schnauben“? |
| `ERSTE_PERSON_SIN_OHNE_E` | 1 | Standardsprachlich ist die Verbform mit ‚e‘ eleganter. Zur besseren Lesbarkeit können Sie  |
| `DAS_DASS` | 1 | Meinten Sie „dass das“? |
| `ZUR_ABBITTE_ETC` | 1 | Meinten Sie das Nomen „Höhle“? |
| `ZUM_TEIL` | 1 | Das Nomen „Teil“ wird großgeschrieben. |
| `DASS_MIT_SUBJUNKTION` | 1 | Meinten Sie „das“? Die Konjunktion ‚dass‘ scheint an dieser Stelle nicht zu passen. |
| `ZEIT_SEINES_LEBENS` | 1 | Meinten Sie das Nomen „Zeit“? |
| `KOMMA_NACH_DIREKTER_REDE` | 1 | Wenn nach der direkten Rede ein Begleitsatz folgt, muss ein Komma gesetzt werden. |
| `INDIREKTE_FRAGE` | 1 | Indirekte Fragen werden mit Komma vom Hauptsatz abgetrennt. |
| `KOMMA_VOR_ERLAEUTERUNG` | 1 | Vor einer nachgestellten Erläuterung steht üblicherweise ein Komma. |
| `DE_VERBAGREEMENT` | 1 | Möglicherweise fehlende grammatische Übereinstimmung zwischen Subjekt (Er) und Prädikat (v |
| `ZUR_NEIGE_GEHEN` | 1 | Bitte prüfen Sie, ob „Neige“ hier als Nomen gebraucht wird und daher großgeschrieben werde |
| `EMPFOHLENE_GETRENNTSCHREIBUNG` | 1 | Empfehlung: Möchten Sie die Schreibweise „zu Hause“ verwenden? |
| `BEI_VERB` | 1 | Nach ‚bei‘ folgt normalerweise kein Verb. |
| `GERMAN_WORD_REPEAT_RULE` | 1 | Möglicher Tippfehler: ein Wort wird wiederholt |
| `KOSTEN_SUBST` | 1 | Das Nomen „Kosten“ wird großgeschrieben. |
| `JAEHRIG` | 1 | Meinten Sie „72-Jährige“? |
| `GROSSSCHREIBUNG_MAL` | 1 | Wenn „Mal“ hier ein Nomen (das Mal, die Male) oder eine Zähleinheit (ein einziges Mal) ist |
| `PRAEP_PLUS_VERB` | 1 | Das Verb „Lesen“ scheint hier nominalisiert verwendet zu werden und muss dann großgeschrie |
| `DE_SUBJECT_VERB_AGREEMENT` | 1 | Bitte prüfen, ob hier „waren“ stehen sollte. |
| `CONFUSION_RULE_VIEL_FIEL` | 1 | Bitte prüfen Sie, ob ‚fiel‘ (zu ‚fallen‘) hier das richtige Wort ist anstelle von ‚viel‘ ( |
| `DE_DATE_WEEKDAY_CURRENTYEAR` | 1 | Bezieht sich dieses Datum auf das aktuelle Jahr? Der 28. juli 2026 fällt nicht auf einen S |
| `COMMA_IN_FRONT_RELATIVE_CLAUSE` | 1 | Sowohl angehängte als auch eingeschobene Relativsätze werden durch Kommas vom Hauptsatz ge |
| `UEBER_EIN_MANGEL` | 1 | „durch“ erwartet den Akkusativ. Hier scheint „einen“ der korrekte Artikel zu sein. |

Those hundred account for 2,323 of 2,323 matches (100.0%).
