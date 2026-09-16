# Phase 2: the training set

76,442 examples: **19,237 real** learner sentences from Falko-MERLIN train, **10,000 rule-generated** ones, and **47,205 back-translated** ones. 17,204 need no correction (22.5%).

Iteration 1 was synthetic-only and made the model worse than the untrained baseline on 14 of 15 error types (`reports/phase3/evaluation.md`), so real learner errors became the primary signal and the corruptors the supporting one. Iteration 3 widened the corruptors from 8 error types to 17 and bought +0.0016, which is nothing (`reports/phase3/ablations.md`, section 5). The back-translated share is iteration 5's answer to why: a rule can only write the error types that are easy to write, and those are the ones the model was already good at. See `reports/phase3/headroom.md`.

## Invariants

| Check | Result |
|---|---|
| Damaged examples whose source equals their target | 0 |
| Edits with no explanation | 0 |
| Synthetic edits with no template | 0 |
| Sentences no corruptor could damage | 0 |

## Error density

The number iteration 1 got wrong: it produced nothing with three or more errors, and 39% of real learner sentences have them.

| Edits in a sentence | Share | Falko train |
|---|---:|---:|
| 0 | 22.5% | 22.1% |
| 1 | 24.3% | 21.2% |
| 2 | 18.3% | 17.5% |
| 3 or more | 34.9% | 39.2% |

Mean edits per sentence: **2.14** (Falko train: 2.55, iteration 1: 1.17).

## Error types

**56** distinct types, against 8 in iteration 1 and 54 in Falko train.

| Error type | Count | Share |
|---|---:|---:|
| `R:SPELL` | 22,313 | 13.6% |
| `R:DET:FORM` | 19,231 | 11.8% |
| `R:OTHER` | 16,052 | 9.8% |
| `R:ORTH` | 11,279 | 6.9% |
| `M:PUNCT` | 10,192 | 6.2% |
| `R:ADJ:FORM` | 7,764 | 4.7% |
| `R:ADP` | 7,414 | 4.5% |
| `R:NOUN:FORM` | 7,070 | 4.3% |
| `M:DET` | 5,779 | 3.5% |
| `U:PUNCT` | 5,743 | 3.5% |
| `R:NOUN` | 5,243 | 3.2% |
| `R:PUNCT` | 3,397 | 2.1% |
| `M:PRON` | 3,190 | 1.9% |
| `R:VERB:FORM` | 3,028 | 1.9% |
| `R:WO` | 3,010 | 1.8% |
| `R:MORPH` | 2,606 | 1.6% |
| `M:ADP` | 2,362 | 1.4% |
| `R:AUX:FORM` | 2,203 | 1.3% |
| `R:VERB` | 2,101 | 1.3% |
| `M:AUX` | 1,755 | 1.1% |

## Explanations

19,453 edits carry a rule-level explanation, from the corruptor that made them. 49,058 carry a type-level one, because a Falko edit records what the annotator changed and not why.

That split is the point of the mix: the real data covers all the error types and teaches what learner German looks like, and the synthetic data teaches how to explain the eight it can produce properly.

## 100 samples

The Phase 2 exit criterion: read these, and if more than five have a wrong correction or a wrong explanation, fix the pipeline before training on it.

**1.** *(learner)* `Viele Krippe und Kindergarten haben den Frauen ermöglicht , einen Beruf auszuüben und deshalb eine aktive Rolle in der Gesellschaft zu spielen .`
→ `Viele Krippen und Kindergärten haben den Frauen ermöglicht , einen Beruf auszuüben und deshalb eine aktive Rolle in der Gesellschaft zu spielen .`
  - *R:NOUN:FORM*: The noun is in the wrong form. German marks case and number on the noun itself, so it should be «Krippen».
  - *R:NOUN:FORM*: The noun is in the wrong form. German marks case and number on the noun itself, so it should be «Kindergärten».

**2.** *(backtranslation)* `Der Prozess könnte , nach Gerichtsangaben , bis zu acht Wochen zum Urteilsspruch dauern .`
→ `Der Prozess könnte nach Gerichtsangaben bis zu acht Wochen zum Urteilsspruch dauern .`
  - *U:PUNCT*: This punctuation mark does not belong here. German does not separate these parts of a sentence.
  - *U:PUNCT*: This punctuation mark does not belong here. German does not separate these parts of a sentence.

**3.** *(synthetic)* `Politisch , ist qilson Nesbitt nicht mehr Erscheinung getreten .`
→ `Politisch ist Wilson Nesbitt nicht mehr in Erscheinung getreten .`
  - *U:PUNCT*: German does not put a comma after a fronted phrase the way English does. The fronting is already marked by the verb standing second, so the comma has nothing left to do. Delete it. (Rat für deutsche Rechtschreibung, § 71)
  - *R:SPELL*: «qilson» is not a German word. It looks like a slip for «Wilson».
  - *M:ADP*: This phrase needs «in». Which preposition a German verb governs is not predictable from the English one -- «warten auf», «denken an» -- so it has to be learned with the verb rather than translated.

**4.** *(backtranslation)* `Die Mecklenburg wurde zunächst Gefangenenwohnschiff verwendet .`
→ `Die Mecklenburg wurde zunächst als Gefangenenwohnschiff verwendet .`
  - *M:ADP*: A preposition is missing. This construction needs «als».

**5.** *(backtranslation)* `Erst in den Verlauf des Nachmittags erwarten die Meteorologen auflockerungen mit höchstens 13 bis 17 Grad .`
→ `Erst im Verlauf des Nachmittags erwarten die Meteorologen Auflockerungen bei höchstens 13 bis 17 Grad .`
  - *R:ADP*: German prepositions are not chosen by translating the English one. This phrase takes «im» rather than «in».
  - *U:DET*: The article «den» does not belong here. Remove it.
  - *R:ORTH*: This is a convention of written German rather than a grammar rule. The correct written form here is «Auflockerungen».
  - *R:ADP*: German prepositions are not chosen by translating the English one. This phrase takes «bei» rather than «mit».

**6.** *(backtranslation)* `Ich habe auch dabei gelernt , dass Auszeiten wichtig sind und wieder der Lust zum Schaffen ankurbeln .`
→ `Ich habe dabei auch gelernt , dass Auszeiten wichtig sind und wieder die Lust zum Schaffen ankurbeln .`
  - *R:WO*: The word order is wrong. German fixes the position of the verb: second in a main clause, last in a subordinate one. It should read «dabei auch».
  - *R:DET:FORM*: The article has to match its noun in gender, case and number. Here that makes it «die», not «der».

**7.** *(synthetic)* `Nun plant die Armee auf Weisung von Regierungschef Benjamin Netanjahu auch in der Stadt eines offensive .`
→ `Nun plant die Armee auf Weisung von Regierungschef Benjamin Netanjahu auch in der Stadt eine Offensive .`
  - *R:DET:FORM*: The article has to agree with its noun in gender, number and case. «eines» does not fit here; the form the sentence needs is «eine».
  - *R:ORTH*: German capitalises every noun, not just names. «offensive» is a noun here, so it needs a capital letter: «Offensive». (Rat für deutsche Rechtschreibung, § 55)

**8.** *(backtranslation)* `MINSK , 9. Dezember ( BelTA ) - Belarus und Russland haben Übungen abgehalten , halten sie ab und werden sie auch weiter abhalten .`
→ `MINSK , 9. Dezember ( BelTA ) – Belarus und Russland haben Übungen abgehalten , halten sie ab und werden sie auch weiterhin abhalten .`
  - *R:PUNCT*: The wrong punctuation mark is used here. It should be «–».
  - *R:ADV*: «weiter» is not the right adverb here. Use «weiterhin».

**9.** *(backtranslation)* `Am 5. August 2004 wurde bekannt , dass sie eine Stelle als Jugend- und Kinderpastorin / i in der Hope Church in Wilton , Connecticut antreten werde .`
→ `Am 5. August 2004 wurde bekannt , dass sie eine Stelle als Jugend- und Kinderpastorin/-missionarin in der Hope Church in Wilton , Connecticut antreten werde .`
  - *R:OTHER*: «Kinderpastorin / i» is not the right word here. German uses «Kinderpastorin/-missionarin» in this context.

**10.** *(synthetic)* `Rechts auf der Haupthalle stehen die Schatzpagode ( 多宝塔 , Tahōtō ; 4 ) aus das Jahr 1645.`
→ `Rechts neben der Haupthalle steht die Schatzpagode ( 多宝塔 , Tahōtō ; 4 ) aus dem Jahr 1645.`
  - *R:ADP*: German prepositions are not chosen by translating the English one. This phrase takes «neben» rather than «auf».
  - *R:VERB:FORM*: «stehen» is the infinitive -- the form a dictionary lists. A finite verb has to agree with its subject, so this sentence needs «steht».
  - *R:DET:FORM*: The article has to agree with its noun in gender, number and case. «das» does not fit here; the form the sentence needs is «dem».

**11.** *(backtranslation)* `Damit , beispielweise , ließen sich die Stützen ausfahren , oder die Pneumatik-Pumpe betreiben .`
→ `Damit ließen sich beispielsweise die Stützen ausfahren oder die Pneumatik-Pumpe betreiben .`
  - *U:OTHER*: The word «, beispielweise ,» does not belong here. Remove it.
  - *M:ADV*: A adverb is missing here. Add «beispielsweise».
  - *U:PUNCT*: This punctuation mark does not belong here. German does not separate these parts of a sentence.

**12.** *(backtranslation)* `Die Erfolg hat ihr Leben verändert , war auch eine Schule für sie .`
→ `Der Erfolg hat ihr Leben verändert , war auch eine Schule für sie .`
  - *R:DET:FORM*: The article has to match its noun in gender, case and number. Here that makes it «Der», not «Die».

**13.** *(backtranslation)* `Heidrun Möller-Doll hat den Vorstand geverlaengt und wurde für 30 Jahre Ehrenamtliche Mitwirkung gewürdigt .`
→ `Heidrun Möller-Doll hat den Vorstand verlassen und wurde für 30 Jahre ehrenamtliche Mitarbeit gewürdigt .`
  - *R:VERB*: «geverlaengt» is not the right verb here. Use «verlassen».
  - *R:ORTH*: This is a convention of written German rather than a grammar rule. The correct written form here is «ehrenamtliche».
  - *R:NOUN*: «Mitwirkung» is not the right noun here. Use «Mitarbeit».

**14.** *(backtranslation)* `Nach den Sieg zeigte Nemo " wirklich dankbar " .`
→ `Nach dem Sieg zeigte sich Nemo " wirklich dankbar " .`
  - *R:DET:FORM*: The article has to match its noun in gender, case and number. Here that makes it «dem», not «den».
  - *M:PRON*: A pronoun is missing. German does not leave out the subject the way some languages do, so «sich» is needed here.

**15.** *(backtranslation)* `Bethge sah nicht in der Lage , seine Behauptungen zu belegen .`
→ `Bethge sah sich nicht in der Lage , seine Behauptung zu belegen .`
  - *M:PRON*: A pronoun is missing. German does not leave out the subject the way some languages do, so «sich» is needed here.
  - *R:NOUN:FORM*: The noun is in the wrong form. German marks case and number on the noun itself, so it should be «Behauptung».

**16.** *(backtranslation)* `Die CSU-Juristen warnten davor , dass diese Regelung auch " zu sachlich nicht gerechtfertigen Entlassungen von Täter " führen könnte , die weiter schwere Straftatbestände verwirklicht haben .`
→ `Die CSU-Juristen warnten davor , dass diese Regelung auch " zu sachlich nicht gerechtfertigten Entlassungen von Tätern " führen könnte , die weitere schwere Straftatbestände verwirklicht haben .`
  - *R:SPELL*: «gerechtfertigen» is not spelled correctly. It should be «gerechtfertigten».
  - *R:NOUN:FORM*: The noun is in the wrong form. German marks case and number on the noun itself, so it should be «Tätern».
  - *R:OTHER*: «weiter» is not the right word here. German uses «weitere» in this context.

**17.** *(backtranslation)* `Hier ist er ebenfalls Namensgeber des „ Matlockitgruppe “ mit der System-Nr .`
→ `Hier ist er ebenfalls Namensgeber der „ Matlockitgruppe “ mit der System-Nr .`
  - *R:DET:FORM*: The article has to match its noun in gender, case and number. Here that makes it «der», not «des».

**18.** *(backtranslation)* `Die Spiele fand zwischen dem 26 de Mai und 12 de Juni 1977 statt .`
→ `Die Spiele fanden zwischen dem 26. Mai und 12. Juni 1977 statt .`
  - *R:VERB:FORM*: The verb is in the wrong form. Check the tense, the person, and whether this clause needs an infinitive or a participle. It should be «fanden».
  - *R:OTHER*: «26» is not the right word here. German uses «26.» in this context.
  - *U:PNOUN*: The proper name «de» does not belong here. Remove it.
  - *R:OTHER*: «12» is not the right word here. German uses «12.» in this context.
  - *U:PNOUN*: The proper name «de» does not belong here. Remove it.

**19.** *(synthetic)* `Nun zeigt er sie an seiner Jubiläums-Ausstellung in aeinem Atelier im Bunt immer samstags und sonntags von 14 bis 18 Uhr vom 16. November bis 8. Dezember .`
→ `Nun zeigt er sie an seiner Jubiläums-Ausstellung in seinem Atelier im Bunt immer samstags und sonntags von 14 bis 18 Uhr vom 16. November bis 8. Dezember .`
  - *R:SPELL*: «aeinem» is not a German word. It looks like a slip for «seinem».

**20.** *(backtranslation)* `Dort bestätigte er die statistischen Leistungen der Vorjahren und wurde somit am Ende der Spielzeit 2017 / 18 als bester Torhüter der Liga geehrt und darüber hinaus im All-Star Team der KHL gewählt .`
→ `Dort bestätigte er die statistischen Leistungen der Vorjahre und wurde somit am Ende der Spielzeit 2017 / 18 als bester Torhüter der Liga geehrt und darüber hinaus ins All-Star Team der KHL gewählt .`
  - *R:NOUN:FORM*: The noun is in the wrong form. German marks case and number on the noun itself, so it should be «Vorjahre».
  - *R:ADP*: German prepositions are not chosen by translating the English one. This phrase takes «ins» rather than «im».

**21.** *(learner)* `Schwer zu sagen , ob die Studenten nach Bechelor weiterstudieren werden , ob nicht .`
→ `Schwer zu sagen , ob die Studenten nach dem Bachelor weiterstudieren werden oder nicht .`
  - *M:DET*: A German noun almost always needs an article, even where English drops it. Add «dem».
  - *R:SPELL*: «Bechelor» is not spelled correctly. It should be «Bachelor».
  - *U:PUNCT*: This punctuation mark does not belong here. German does not separate these parts of a sentence.
  - *R:OTHER*: «ob» is not the right word here. German uses «oder» in this context.

**22.** *(backtranslation)* `Doch in den kommenden Tagen haben die Menschen viel Zeit um zur Ruhe zu kommen , nachzudenken , mehr zu lesen ( auch über Politik ) und sich mit der Familie auszutauschen .`
→ `Doch in den kommenden Tagen haben die Menschen viel Zeit , um zur Ruhe zu kommen , nachzudenken , mehr zu lesen ( auch über Politik ) und sich mit der Familie auszutauschen .`
  - *M:PUNCT*: A punctuation mark is missing here. German punctuates by rule rather than by ear, and this sentence needs «,».

**23.** *(synthetic)* `Die CDU verzichtete im november 1990 endgültig auf dass nicht rechtsstaatlich erworbenem Vermögen das Ost-CDU und der DBD .`
→ `Die CDU verzichtete im November 1990 endgültig auf das nicht rechtsstaatlich erworbene Vermögen der Ost-CDU und der DBD .`
  - *R:ORTH*: German capitalises every noun, not just names. «november» is a noun here, so it needs a capital letter: «November». (Rat für deutsche Rechtschreibung, § 55)
  - *R:OTHER*: «das» and «dass» sound the same and do different jobs. «das» is an article or a relative pronoun and points at a thing; «dass» is a conjunction that opens a subordinate clause. Here the sentence needs «das».
  - *R:ADJ:FORM*: An adjective before a noun takes an ending that depends on the noun's gender, number and case, and on whether an article comes first. Here the adjective should read «erworbene», not «erworbenem».
  - *R:DET:FORM*: The article has to agree with its noun in gender, number and case. «das» does not fit here; the form the sentence needs is «der».

**24.** *(backtranslation)* `Sie sind die größte der fünf Inseln des Mar Menor Inseln , die alle sind unter Naturschutz .`
→ `Sie ist die größte der fünf Inseln des Mar Menor , die alle unter Naturschutz stehen .`
  - *R:AUX:FORM*: The auxiliary is in the wrong form. It should be «ist».
  - *U:NOUN*: The noun «Inseln» does not belong here. Remove it.
  - *U:AUX*: This auxiliary verb is not needed here. Remove «sind».
  - *M:VERB*: A verb is missing here. Add «stehen».

**25.** *(backtranslation)* `Außerdem wurde die Verwaltung verpflichtet , die Möblierung der Marktplatz deutlich zu verbessern , und im Sommer für ausreichend Schattend zu sorgen .`
→ `Außerdem wurde die Verwaltung verpflichtet , die Möblierung des Marktplatzes deutlich zu verbessern und im Sommer für ausreichend Schatten zu sorgen .`
  - *R:DET:FORM*: The article has to match its noun in gender, case and number. Here that makes it «des», not «der».
  - *R:NOUN:FORM*: The noun is in the wrong form. German marks case and number on the noun itself, so it should be «Marktplatzes».
  - *U:PUNCT*: This punctuation mark does not belong here. German does not separate these parts of a sentence.
  - *R:SPELL*: «Schattend» is not spelled correctly. It should be «Schatten».

**26.** *(backtranslation)* `Am 1. Mai 1919 wurde Čižnský mit einige Genossen verhaftet .`
→ `Am 1. Mai 1919 wurde Čižinský mit einigen Genossen verhaftet .`
  - *R:SPELL*: «Čižnský» is not spelled correctly. It should be «Čižinský».
  - *R:DET:FORM*: The article has to match its noun in gender, case and number. Here that makes it «einigen», not «einige».

**27.** *(backtranslation)* `Davon wurde auch rund 95 Millionen dollar öwerweisen .`
→ `Davon wurden rund 95 Millionen Dollar auch überwiesen .`
  - *R:AUX:FORM*: The auxiliary is in the wrong form. It should be «wurden».
  - *U:ADV*: The adverb «auch» does not belong here. Remove it.
  - *R:ORTH*: This is a convention of written German rather than a grammar rule. The correct written form here is «Dollar».
  - *R:OTHER*: «öwerweisen» is not the right word here. German uses «auch überwiesen» in this context.

**28.** *(backtranslation)* `Vilenviertel in der Beethovenallee 21. Das Gebäude wurde noch bis Jahr 2000 für diplomatische Zwecke genutzt .`
→ `Villenviertel in der Beethovenallee 21. Das Gebäude wurde noch bis zum Jahr 2000 für diplomatische Zwecke genutzt .`
  - *R:SPELL*: «Vilenviertel» is not spelled correctly. It should be «Villenviertel».
  - *M:ADP*: A preposition is missing. This construction needs «zum».

**29.** *(backtranslation)* `Im Juni 2023 beging die französische Behörden den gleichen Fehler .`
→ `Im Juni 2023 begingen die französischen Behörden den gleichen Fehler .`
  - *R:VERB*: «beging» is not the right verb here. Use «begingen».
  - *R:ADJ:FORM*: An adjective before a noun takes an ending that depends on the noun's gender, case and number, and on the article in front of it. It should read «französischen».

**30.** *(backtranslation)* `Lars Ricken äußerte , auf der Aktionärsversammlung am Montag , seine Überzeugung , dass das Heimspiel gegen den Rekordmeister gewonnen wird .`
→ `Lars Ricken äußerte auf der Aktionärsversammlung am Montag seine Überzeugung , dass das Heimspiel gegen den Rekordmeister gewonnen wird .`
  - *U:PUNCT*: This punctuation mark does not belong here. German does not separate these parts of a sentence.
  - *U:PUNCT*: This punctuation mark does not belong here. German does not separate these parts of a sentence.

**31.** *(backtranslation)* `Es entstand am ende des 12 Jahrhunderts und wurde 1447 erstmals urkundlich erörtert .`
→ `Es entstand am Ende des 12. Jahrhunderts und wurde 1447 erstmals urkundlich erwähnt .`
  - *R:ORTH*: This is a convention of written German rather than a grammar rule. The correct written form here is «Ende».
  - *R:OTHER*: «12» is not the right word here. German uses «12.» in this context.
  - *R:VERB*: «erörtert» is not the right verb here. Use «erwähnt».

**32.** *(backtranslation)* `In der Encyclopedia Britannica von 1990 verfasste er den Abschnitt über Geschichte der Mathematik im 17. und 18. jahrhundert .`
→ `In der Encyclopedia Britannica von 1990 verfasste er den Abschnitt über die Geschichte der Mathematik im 17. und 18. Jahrhundert .`
  - *M:DET*: A German noun almost always needs an article, even where English drops it. Add «die».
  - *R:ORTH*: This is a convention of written German rather than a grammar rule. The correct written form here is «Jahrhundert».

**33.** *(backtranslation)* `Nach Zahlung Geldstrafe , ließ man es frei wieder .`
→ `Nach Zahlung einer Geldstrafe ließ man sie wieder frei .`
  - *M:DET*: A German noun almost always needs an article, even where English drops it. Add «einer».
  - *U:PUNCT*: This punctuation mark does not belong here. German does not separate these parts of a sentence.
  - *R:PRON*: «es» is not the right pronoun here. Use «sie».
  - *R:WO*: The word order is wrong. German fixes the position of the verb: second in a main clause, last in a subordinate one. It should read «wieder frei».

**34.** *(backtranslation)* `Aus den thematischen Schwerpunkten entwickelten sich breitgefaßte Programme , vor allem nach dem Bau der Zimmereifachschule .`
→ `Aus den thematischen Schwerpunkten entwickelten sich breit gefächerte Programme , vor allem nach dem Bau der Zimmereifachschule .`
  - *R:OTHER*: «breitgefaßte» is not the right word here. German uses «breit gefächerte» in this context.

**35.** *(learner)* `Aber die Preisunterschied ist zwischen verschiedenen Bezirken sehr groß , deshalb ist es empfelenswert sich zuerst die Preisen zu erkundigen .`
→ `Aber der Preisunterschied ist zwischen verschiedenen Bezirken sehr groß , deshalb ist es empfehlenswert , sich zuerst nach den Preisen zu erkundigen .`
  - *R:DET:FORM*: The article has to match its noun in gender, case and number. Here that makes it «der», not «die».
  - *R:SPELL*: «empfelenswert» is not spelled correctly. It should be «empfehlenswert».
  - *M:PUNCT*: A punctuation mark is missing here. German punctuates by rule rather than by ear, and this sentence needs «,».
  - *M:ADP*: A preposition is missing. This construction needs «nach».
  - *R:DET:FORM*: The article has to match its noun in gender, case and number. Here that makes it «den», not «die».

**36.** *(learner)* `Denn haben Sie jehmals von einem Anwaltsbüro gehört , das ein Student ohne Universitätsabschluss angestellt hat ?`
→ `Denn haben Sie jemals von einem Anwaltsbüro gehört , das einen Studenten ohne Universitätsabschluss angestellt hat ?`
  - *R:SPELL*: «jehmals» is not spelled correctly. It should be «jemals».
  - *R:DET:FORM*: The article has to match its noun in gender, case and number. Here that makes it «einen», not «ein».
  - *R:NOUN:FORM*: The noun is in the wrong form. German marks case and number on the noun itself, so it should be «Studenten».

**37.** *(backtranslation)* `Der aus ihrer Sicht inhumane Umgang mit Sterbenden veranlasste sie , sich in der Versorgungs solcher Menschen zu engagieren .`
→ `Der aus ihrer Sicht inhumane Umgang mit Sterbenden veranlasste sie , sich in der Versorgung solcher Menschen zu engagieren .`
  - *R:SPELL*: «Versorgungs» is not spelled correctly. It should be «Versorgung».

**38.** *(learner)* `Ich brauche eine große Wagen für die Möbel .`
→ `Ich brauche einen großen Wagen für die Möbel .`
  - *R:DET:FORM*: The article has to match its noun in gender, case and number. Here that makes it «einen», not «eine».
  - *R:ADJ:FORM*: An adjective before a noun takes an ending that depends on the noun's gender, case and number, and on the article in front of it. It should read «großen».

**39.** *(learner)* `davon viele erfahrung gebrach .`
→ `was viel Erfahrung gebracht hat .`
  - *R:OTHER*: «davon» is not the right word here. German uses «was» in this context.
  - *R:MORPH*: «viel» is built from the same stem as «viele» but is the form this sentence needs -- a different ending, or a different part of speech.
  - *R:ORTH*: This is a convention of written German rather than a grammar rule. The correct written form here is «Erfahrung».
  - *R:SPELL*: «gebrach» is not spelled correctly. It should be «gebracht».
  - *M:AUX*: An auxiliary verb is missing. German builds its perfect and passive with «haben», «sein» or «werden», and this sentence needs «hat».

**40.** *(backtranslation)* `Sie zählt zu den Highlights unter den Winterblühern für die Wohnung - die Amaryllis , auch Ritterstern genannt .`
→ `Sie zählt zu den Highlights unter den Winterblühern für die Wohnung : die Amaryllis , auch Ritterstern genannt .`
  - *R:PUNCT*: The wrong punctuation mark is used here. It should be «:».

**41.** *(backtranslation)* `So durften z. B. Schösslingen außer der Herrschaft Falkenau weder verkauft noch verschenkt werden .`
→ `So durften z. B. Schösslinge außerhalb der Herrschaft Falkenau weder verkauft , noch verschenkt werden .`
  - *R:NOUN:FORM*: The noun is in the wrong form. German marks case and number on the noun itself, so it should be «Schösslinge».
  - *R:ADP*: German prepositions are not chosen by translating the English one. This phrase takes «außerhalb» rather than «außer».
  - *M:PUNCT*: A punctuation mark is missing here. German punctuates by rule rather than by ear, and this sentence needs «,».

**42.** *(backtranslation)* `Er ist auch zum Beitragen in Gletscherkunde bekannt .`
→ `Er ist auch für Beiträge in der Gletscherkunde bekannt .`
  - *R:ADP*: German prepositions are not chosen by translating the English one. This phrase takes «für» rather than «zum».
  - *R:NOUN*: «Beitragen» is not the right noun here. Use «Beiträge».
  - *M:DET*: A German noun almost always needs an article, even where English drops it. Add «der».

**43.** *(backtranslation)* `Die HNLMS Johan Maurits van Nassau während die Überfahrt wurde versenkt von deutschen Bomber .`
→ `Die HNLMS Johan Maurits van Nassau wurde während der Überfahrt von deutschen Bombern versenkt .`
  - *M:AUX*: An auxiliary verb is missing. German builds its perfect and passive with «haben», «sein» or «werden», and this sentence needs «wurde».
  - *R:DET:FORM*: The article has to match its noun in gender, case and number. Here that makes it «der», not «die».
  - *U:OTHER*: The word «wurde versenkt» does not belong here. Remove it.
  - *R:NOUN:FORM*: The noun is in the wrong form. German marks case and number on the noun itself, so it should be «Bombern».
  - *M:VERB*: A verb is missing here. Add «versenkt».

**44.** *(backtranslation)* `Architekt wird , zog aber Malerei vor .`
→ `Architekt werden , er zog aber die Malerei vor .`
  - *R:AUX:FORM*: The auxiliary is in the wrong form. It should be «werden».
  - *M:PRON*: A pronoun is missing. German does not leave out the subject the way some languages do, so «er» is needed here.
  - *M:DET*: A German noun almost always needs an article, even where English drops it. Add «die».

**45.** *(backtranslation)* `Sie helfen den guten Mächten , erhalten von diesen und von vielen Tieren Hilfe in einem langen , gefährlichen Weg und gewinnen dabei Selbstvertrauen von den .`
→ `Sie helfen den guten Mächten , erhalten von diesen und von vielen Tieren Hilfe auf einem langen , gefährlichen Weg und gewinnen dabei Selbstvertrauen .`
  - *R:ADP*: German prepositions are not chosen by translating the English one. This phrase takes «auf» rather than «in».
  - *U:OTHER*: The word «von den» does not belong here. Remove it.

**46.** *(backtranslation)* `Das Bundesverwaltungsgericht , das als zweite Instanz in Asylverfahren entscheidet , schätze zuletzt ebenfalls „ sehr labil " die Lage ein .`
→ `Das Bundesverwaltungsgericht , das als zweite Instanz in Asylverfahren entscheidet , schätzte die Lage zuletzt ebenfalls als „ sehr labil “ ein .`
  - *R:VERB*: «schätze» is not the right verb here. Use «schätzte».
  - *M:OTHER*: A word is missing here. Add «die Lage».
  - *M:ADP*: A preposition is missing. This construction needs «als».
  - *R:OTHER*: «" die Lage» is not the right word here. German uses «“» in this context.

**47.** *(backtranslation)* `Der Blattrand kann undifferenziert oder leicht gewellt , behaart bis fluhig oder zottig sein .`
→ `Der Blattrand kann undifferenziert oder leicht gewellt , behaart bis flaumig oder zottig sein .`
  - *R:SPELL*: «fluhig» is not spelled correctly. It should be «flaumig».

**48.** *(learner)* `Ich bin flexibl und eine Zusammenarbeit mit anderen macht mir viel Spaß .`
→ `Ich bin flexibel und eine Zusammenarbeit mit anderen macht mir viel Spaß .`
  - *R:SPELL*: «flexibl» is not spelled correctly. It should be «flexibel».

**49.** *(backtranslation)* `Nun zeigte , dass GPS Daten sind genug empfindlich , um atmosphärische Störungen aufzunehemen .`
→ `Nun zeigte sich , dass GPS-Daten empfindlich genug sind , um atmosphärische Störungen zu erfassen .`
  - *M:PRON*: A pronoun is missing. German does not leave out the subject the way some languages do, so «sich» is needed here.
  - *R:OTHER*: «GPS Daten» is not the right word here. German uses «GPS-Daten» in this context.
  - *U:AUX*: «sind» is in the wrong place in this sentence. German fixes word order more tightly than English does, so it has to move rather than stay where it is.
  - *R:WO*: The word order is wrong. German fixes the position of the verb: second in a main clause, last in a subordinate one. It should read «empfindlich genug».
  - *M:AUX*: «sind» is in the wrong place in this sentence. German fixes word order more tightly than English does, so it has to move rather than stay where it is.
  - *R:OTHER*: «aufzunehemen» is not the right word here. German uses «zu erfassen» in this context.

**50.** *(backtranslation)* `Bis zum Ausscheiden kam Aboukhlal in allen vier Partien zum Einsatz und absolvierte für die U17 in allem 12 Partien , in denen er fünf Tore gelangt .`
→ `Bis zum Ausscheiden kam Aboukhlal in allen vier Partien zum Einsatz und absolvierte für die U17 insgesamt 12 Partien , in denen ihm fünf Tore gelangen .`
  - *R:OTHER*: «in allem» is not the right word here. German uses «insgesamt» in this context.
  - *R:PRON*: «er» is not the right pronoun here. Use «ihm».
  - *R:VERB:FORM*: The verb is in the wrong form. Check the tense, the person, and whether this clause needs an infinitive or a participle. It should be «gelangen».

**51.** *(backtranslation)* `Wenn ich jung war , ich wollte schon 30 aufhören .`
→ `Als ich jung war , wollte ich schon mit 30 aufhören .`
  - *R:SCONJ*: «Wenn» is not the right subordinating conjunction here. Use «Als».
  - *R:WO*: The word order is wrong. German fixes the position of the verb: second in a main clause, last in a subordinate one. It should read «wollte ich».
  - *M:ADP*: A preposition is missing. This construction needs «mit».

**52.** *(backtranslation)* `Fast 43 Jahre war Lehner im Landesdienst .`
→ `Fast 43 Jahre stand Lehner im Landesdienst .`
  - *R:OTHER*: «war» is not the right word here. German uses «stand» in this context.

**53.** *(backtranslation)* `Die Pizzamen bekamen bei Familie , Freunden und Bekannten so gut an , dass er und seine Lebensgefährtin Anne Wiezorek beschlossen , einen Catering-Service mit den Pizzamen als wichtiger Bestandteil des Angebots zu gründen .`
→ `Die Pizzen kamen bei Familie , Freunden und Bekannten so gut an , dass er und seine Lebensgefährtin Anne Wiezorek beschlossen , einen Catering-Service mit den Pizzen als wichtigem Bestandteil des Angebots zu gründen .`
  - *R:SPELL*: «Pizzamen» is not spelled correctly. It should be «Pizzen».
  - *R:VERB*: «bekamen» is not the right verb here. Use «kamen».
  - *R:SPELL*: «Pizzamen» is not spelled correctly. It should be «Pizzen».
  - *R:ADJ:FORM*: An adjective before a noun takes an ending that depends on the noun's gender, case and number, and on the article in front of it. It should read «wichtigem».

**54.** *(backtranslation)* `Wenn der verlachte Bajazzo jedoch verstorben ist , gäbe es niemand mehr der stellvertretend für andere verlacht wird , und es würde das große Weinen beginnen .`
→ `Wenn der verlachte Bajazzo jedoch verstorben ist , gäbe es niemanden mehr , der stellvertretend für andere verlacht wird , und es würde das große Weinen beginnen .`
  - *R:PRON:FORM*: The pronoun «niemand» is in the wrong form. It should be «niemanden».
  - *M:PUNCT*: A punctuation mark is missing here. German punctuates by rule rather than by ear, and this sentence needs «,».

**55.** *(backtranslation)* `Der Name Nipperg wurde lange Zeit als Neuberg gedeutet , wird inzwischen aber auch als Trutzberg verstanden und meint wohl in jedem Fall die Burgberg .`
→ `Der Name Neipperg wurde lange Zeit als Neuberg gedeutet , wird inzwischen aber auch als Trutzberg verstanden und meint wohl in jedem Fall den Burgberg .`
  - *R:SPELL*: «Nipperg» is not spelled correctly. It should be «Neipperg».
  - *R:DET:FORM*: The article has to match its noun in gender, case and number. Here that makes it «den», not «die».

**56.** *(synthetic)* `RB keipzig holen in nainz einen souveränen 2:0-Sieg .`
→ `RB Leipzig holt in Mainz einen souveränen 2:0-Sieg .`
  - *R:SPELL*: «keipzig» is not a German word. It looks like a slip for «Leipzig».
  - *R:VERB:FORM*: «holen» is the infinitive -- the form a dictionary lists. A finite verb has to agree with its subject, so this sentence needs «holt».
  - *R:SPELL*: «nainz» is not a German word. It looks like a slip for «Mainz».

**57.** *(backtranslation)* `Der Wanderweg nach Siwy Zwornik führt längs des Hauptkamms des Tatra und der polnisch-slowakischen Grenze .`
→ `Der Wanderweg auf die Siwy Zwornik führt entlang des Hauptkamms der Tatra und der polnisch-slowakischen Grenze .`
  - *R:ADP*: German prepositions are not chosen by translating the English one. This phrase takes «auf» rather than «nach».
  - *M:DET*: A German noun almost always needs an article, even where English drops it. Add «die».
  - *R:OTHER*: «längs» is not the right word here. German uses «entlang» in this context.
  - *R:DET:FORM*: The article has to match its noun in gender, case and number. Here that makes it «der», not «des».

**58.** *(synthetic)* `Im Hafen von Stanley gibt es einen Lehrpfad entlang einer Reihe von Schiffswracks , die dort teilweise seit der ersten Hälfte des 19. Jahrhundert liegen .`
→ `Im Hafen von Stanley gibt es einen Lehrpfad entlang einer Reihe von Schiffswracks , die dort teilweise seit der ersten Hälfte des 19. Jahrhunderts liegen .`
  - *R:NOUN:FORM*: A masculine or neuter noun in the genitive takes an -s or -es ending, even though the article has already marked the case. Write «Jahrhunderts».

**59.** *(backtranslation)* `Die dargestellte Windmühle ist auch auf die Gesamdarstellung von New-Richmond von Zuckerberg aus vorhand .`
→ `Die dargestellte Windmühle ist auch auf der Gesamtdarstellung von Neu-Richmond vom Zuckerberg aus vorhanden .`
  - *R:DET:FORM*: The article has to match its noun in gender, case and number. Here that makes it «der», not «die».
  - *R:SPELL*: «Gesamdarstellung» is not spelled correctly. It should be «Gesamtdarstellung».
  - *R:OTHER*: «New-Richmond» is not the right word here. German uses «Neu-Richmond» in this context.
  - *R:ADP*: German prepositions are not chosen by translating the English one. This phrase takes «vom» rather than «von».
  - *R:SPELL*: «vorhand» is not spelled correctly. It should be «vorhanden».

**60.** *(learner)* `Du muss die Katze 3 Mal pro Tag futtern .`
→ `Du musst die Katze 3 Mal pro Tag füttern .`
  - *R:VERB:FORM*: The verb is in the wrong form. Check the tense, the person, and whether this clause needs an infinitive or a participle. It should be «musst».
  - *R:MORPH*: «füttern» is built from the same stem as «futtern» but is the form this sentence needs -- a different ending, or a different part of speech.

**61.** *(learner)* `Es ist eine kleine Stadt mit G .`
→ `Es ist eine kleine Stadt mit`
  - *U:OTHER*: The word «G .» does not belong here. Remove it.

**62.** *(backtranslation)* `Am Samstag ( 13.00 Uhr ) kommt Meister Bayern München zum Bundesliga Spitzenspiel ins Stadion am Brentanobad .`
→ `Am Samstag ( 13.00 Uhr ) kommt Meister Bayern München zum Bundesliga-Spitzenspiel ins Stadion am Brentanobad .`
  - *R:NOUN*: «Bundesliga Spitzenspiel» is not the right noun here. Use «Bundesliga-Spitzenspiel».

**63.** *(backtranslation)* `Er versucht auf diese Weise die Kontrolle über die Adelsfamilien zu erlangn , indem er für die jeweilige Farbe jeweils mehr Ritter im Burghof hat , als sein ( e ) Gegner .`
→ `Er versucht auf diese Weise , die Kontrolle über die Adelsfamilien zu erlangen , indem er für die jeweilige Farbe jeweils mehr Ritter am Burghof hat als sein ( e ) Gegner .`
  - *M:PUNCT*: «,» is in the wrong place in this sentence. German fixes word order more tightly than English does, so it has to move rather than stay where it is.
  - *R:SPELL*: «erlangn» is not spelled correctly. It should be «erlangen».
  - *R:ADP*: German prepositions are not chosen by translating the English one. This phrase takes «am» rather than «im».
  - *U:PUNCT*: «,» is in the wrong place in this sentence. German fixes word order more tightly than English does, so it has to move rather than stay where it is.

**64.** *(learner)* `Stadt Y ist eine schön Stadt .`
→ `Stadt Y ist eine schöne Stadt .`
  - *R:ADJ:FORM*: An adjective before a noun takes an ending that depends on the noun's gender, case and number, and on the article in front of it. It should read «schöne».

**65.** *(synthetic)* `Als ihr dies gelingt überschreitet sie willentlich die Grenze zwischen beiden Welten und rransferiert einen Teil des Akutos in die Parallelwelt .`
→ `Als ihr dies gelingt überschreitet sie willentlich die Grenze zwischen beiden Welten und transferiert einen Teil des Akutos in die Parallelwelt .`
  - *R:SPELL*: «rransferiert» is not a German word. It looks like a slip for «transferiert».

**66.** *(backtranslation)* `Zudem fanden hier die Studentengodessdienste der Katholischen Hochschulgemeind statt .`
→ `Zudem fanden hier die Studentengottesdienste der Katholischen Hochschulgemeinde statt .`
  - *R:SPELL*: «Studentengodessdienste» is not spelled correctly. It should be «Studentengottesdienste».
  - *R:SPELL*: «Hochschulgemeind» is not spelled correctly. It should be «Hochschulgemeinde».

**67.** *(backtranslation)* `Täglich sind die Mitarbeiterinnen des PBZ Scheiblingkirchen im Einsatz mit die Bewohnerinnen und Bewohner .`
→ `Täglich sind die Mitarbeiterinnen des PBZ Scheiblingkirchen für die Bewohnerinnen und Bewohner im Einsatz .`
  - *R:OTHER*: «im Einsatz mit» is not the right word here. German uses «für» in this context.
  - *M:OTHER*: A word is missing here. Add «im Einsatz».

**68.** *(backtranslation)* `Seitdem hat es sich viel getan : Weitere Sand- , sowie Allwetterplätze kamen hinzu .`
→ `Seitdem hat sich viel getan : Weitere Sand- , sowie Allwetterplätze kamen hinzu .`
  - *U:PRON*: This pronoun is not needed. German does not repeat it here, so remove «es».

**69.** *(backtranslation)* `Und was sagt Ruckdäschel selbst über sein WM Abenteuer ?`
→ `Und was sagt Ruckdäschel selbst über sein WM-Abenteuer ?`
  - *R:NOUN*: «WM Abenteuer» is not the right noun here. Use «WM-Abenteuer».

**70.** *(backtranslation)* `Entscheidend waren unsere unbewarbten Würfe und technische Fehler .`
→ `Entscheidend waren unsere unvorbereiteten Würfe und technischen Fehler .`
  - *R:ADJ*: «unbewarbten» is not the right adjective here. Use «unvorbereiteten».
  - *R:ADJ:FORM*: An adjective before a noun takes an ending that depends on the noun's gender, case and number, and on the article in front of it. It should read «technischen».

**71.** *(backtranslation)* `Niemand kann Bodo als Mörder vorstellen .`
→ `Niemand kann sich Bodo als Mörder vorstellen .`
  - *M:PRON*: A pronoun is missing. German does not leave out the subject the way some languages do, so «sich» is needed here.

**72.** *(synthetic)* `Im Jahr 2004 gaben sich die britischen Schauspieler im Londoner Stadtteil Richmond das hawort .`
→ `Im Jahr 2004 gaben sich die britischen Schauspieler im Londoner Stadtteil Richmond das Jawort .`
  - *R:SPELL*: «hawort» is not a German word. It looks like a slip for «Jawort».

**73.** *(backtranslation)* `Für den 88-Jährigen bedeutet die nunmehrige Entscheidung des Kremser Dreiersenats , dass zwar er in den Normalvollzug kommen soll , derzeit und bis zur Rechtskraft des Beschlusses aber in der Justizanstalt Stein im Maßnahmenvollzug bleibt .`
→ `Für den 88-Jährigen bedeutet die nunmehrige Entscheidung des Kremser Dreiersenats , dass er zwar in den Normalvollzug kommen soll , derzeit und bis zur Rechtskraft des Beschlusses aber in der Justizanstalt Stein im Maßnahmenvollzug bleibt .`
  - *R:WO*: The word order is wrong. German fixes the position of the verb: second in a main clause, last in a subordinate one. It should read «er zwar».

**74.** *(backtranslation)* `Die Stiftung will fundierte informationen auf Ökolandbau im Printmedien und im elektronische Medien zurverfügung stellen .`
→ `Die Stiftung will fundierte Informationen über den Ökolandbau in Printmedien und in elektronischen Medien zur Verfügung stellen .`
  - *R:ORTH*: This is a convention of written German rather than a grammar rule. The correct written form here is «Informationen».
  - *R:ADP*: German prepositions are not chosen by translating the English one. This phrase takes «über» rather than «auf».
  - *M:DET*: A German noun almost always needs an article, even where English drops it. Add «den».
  - *R:ADP*: German prepositions are not chosen by translating the English one. This phrase takes «in» rather than «im».
  - *R:ADP*: German prepositions are not chosen by translating the English one. This phrase takes «in» rather than «im».
  - *R:ADJ:FORM*: An adjective before a noun takes an ending that depends on the noun's gender, case and number, and on the article in front of it. It should read «elektronischen».
  - *R:ORTH*: This is a convention of written German rather than a grammar rule. The correct written form here is «zur Verfügung».

**75.** *(backtranslation)* `Beliebt waren die sogenannte Hausfesten , einen gab es im Frühjahr und das zweite im Herbst .`
→ `Beliebt , waren die sogenannten Hausfeste , eins gab es im Frühjahr und das zweite im Herbst .`
  - *M:PUNCT*: A punctuation mark is missing here. German punctuates by rule rather than by ear, and this sentence needs «,».
  - *R:ADJ:FORM*: An adjective before a noun takes an ending that depends on the noun's gender, case and number, and on the article in front of it. It should read «sogenannten».
  - *R:NOUN*: «Hausfesten» is not the right noun here. Use «Hausfeste».
  - *R:OTHER*: «einen» is not the right word here. German uses «eins» in this context.

**76.** *(backtranslation)* `Die Freiwillige Feuerwehr Matzing dürfte bundesweit die einzige Feuerwehr sein , die selbst Eigentümer ihres Feuerwehrhauses selbt ist .`
→ `Die Freiwillige Feuerwehr Matzing dürfte bundesweit die einzige Feuerwehr sein , die selbst Eigentümer ihres Feuerwehrhauses ist .`
  - *U:PNOUN*: The proper name «selbt» does not belong here. Remove it.

**77.** *(backtranslation)* `Die Obduktion der Gestorbenen ergab , dass es keine Hinweise auf eine grobe Gewalteinwirkung gab , die ursachlich für den Tod der Frau gewesen sein könnte .`
→ `Die Obduktion der Gestorbenen ergab , dass es keine Hinweise auf eine grobe Gewalteinwirkung gab , die ursächlich für den Tod der Frau gewesen sein könnte .`
  - *R:SPELL*: «ursachlich» is not spelled correctly. It should be «ursächlich».

**78.** *(learner)* `Das schlechteste ist , d Kinder diese Verhältnisse beobachten , und in spätere Jahren imitieren .`
→ `Das Schlechteste ist , dass Kinder diese Verhältnisse beobachten und in späteren Jahren imitieren .`
  - *R:ORTH*: This is a convention of written German rather than a grammar rule. The correct written form here is «Schlechteste».
  - *R:OTHER*: «d» is not the right word here. German uses «dass» in this context.
  - *U:PUNCT*: This punctuation mark does not belong here. German does not separate these parts of a sentence.
  - *R:ADJ:FORM*: An adjective before a noun takes an ending that depends on the noun's gender, case and number, and on the article in front of it. It should read «späteren».

**79.** *(backtranslation)* `Dort werde dem Spanier mitgeteilt , dass er in der kommenden Saison nicht mehr auf dem Barça-Bank sitzen werde .`
→ `Dort werde dem Spanier mitgeteilt , dass er in der kommenden Saison nicht mehr auf der Barça-Bank sitzen werde .`
  - *R:DET:FORM*: The article has to match its noun in gender, case and number. Here that makes it «der», not «dem».

**80.** *(backtranslation)* `Die Lebensdauer eines Rohrs betrug ungefähr 16.000 bis 20.000 Schuss .`
→ `Die Lebensdauer eines Rohres betrug ungefähr 16.000 bis 20.000 Schuss .`
  - *R:NOUN:FORM*: The noun is in the wrong form. German marks case and number on the noun itself, so it should be «Rohres».

**81.** *(learner)* `Jeder Fach hat viele Notizen und die Lehrer / innen kommt und reden über verschiedene Themen , das Werk die Studenten ist Notizen schreiben .`
→ `Jedes Fach hat viele Notizen und die LehrerInnen kommen und reden über verschiedene Themen , das Werk der Studenten ist Notizen schreiben .`
  - *R:DET:FORM*: The article has to match its noun in gender, case and number. Here that makes it «Jedes», not «Jeder».
  - *R:OTHER*: «Lehrer / innen kommt» is not the right word here. German uses «LehrerInnen kommen» in this context.
  - *R:DET:FORM*: The article has to match its noun in gender, case and number. Here that makes it «der», not «die».

**82.** *(backtranslation)* `Kein Mensch scherte sich um das gleichzeitig stattfinden Eröffnungsmatch der Fussball-Weltmeisterschaft .`
→ `Kein Mensch scherte sich um das gleichzeitig stattfindende Eröffnungsmatch der Fußball-Weltmeisterschaft .`
  - *R:ADJ*: «stattfinden» is not the right adjective here. Use «stattfindende».
  - *R:NOUN*: «Fussball-Weltmeisterschaft» is not the right noun here. Use «Fußball-Weltmeisterschaft».

**83.** *(learner)* `Vor man ein Haus oder Wohnung in Stadt X kaufen oder mieten , soll er erst die Distanz zwischen Haus und die Öffentlideverkehrsmittels beachten .`
→ `Bevor man ein Haus oder eine Wohnung in Stadt X kauft oder mietet , soll er erst die Distanz zwischen Haus und den öffentlichen Verkehrsmitteln beachten .`
  - *R:OTHER*: «Vor» is not the right word here. German uses «Bevor» in this context.
  - *M:DET*: A German noun almost always needs an article, even where English drops it. Add «eine».
  - *R:VERB:FORM*: The verb is in the wrong form. Check the tense, the person, and whether this clause needs an infinitive or a participle. It should be «kauft».
  - *R:VERB:FORM*: The verb is in the wrong form. Check the tense, the person, and whether this clause needs an infinitive or a participle. It should be «mietet».
  - *R:OTHER*: «die Öffentlideverkehrsmittels» is not the right word here. German uses «den öffentlichen Verkehrsmitteln» in this context.

**84.** *(learner)* `Daraufhin bekommt eine buchung für lärme .`
→ `Daraufhin bekommt man eine Buchung für Lärm .`
  - *M:PRON*: A pronoun is missing. German does not leave out the subject the way some languages do, so «man» is needed here.
  - *R:ORTH*: This is a convention of written German rather than a grammar rule. The correct written form here is «Buchung».
  - *R:MORPH*: «Lärm» is built from the same stem as «lärme» but is the form this sentence needs -- a different ending, or a different part of speech.

**85.** *(synthetic)* `Zunächst war eine Inbetriebnahme des Hauptgebäudes im Oktober 2009 zum 50-jährigen Bestehen der CCTV geplant doch konnte der Termin nicht eingehalten werden .`
→ `Zunächst war eine Inbetriebnahme des Hauptgebäudes im Oktober 2009 zum 50-jährigen Bestehen der CCTV geplant , doch konnte der Termin nicht eingehalten werden .`
  - *M:PUNCT*: German puts a comma before «sondern», «aber», «doch» and «jedoch» when they join two clauses. English usually does not, which is why this comma is easy to miss.

**86.** *(backtranslation)* `Seit 1998 regelt das Gesetz den heutigen Gesamtschul unterricht .`
→ `Seit 1998 regelt das Gesetz den heutigen Gesamtschulunterricht .`
  - *R:ORTH*: This is a convention of written German rather than a grammar rule. The correct written form here is «Gesamtschulunterricht».

**87.** *(backtranslation)* `Die SRG solle auf ihren Kernauftrag beschränken : Information , Bildung , Kultur .`
→ `Die SRG solle sich auf ihren Kernauftrag beschränken : Information , Bildung , Kultur .`
  - *M:PRON*: A pronoun is missing. German does not leave out the subject the way some languages do, so «sich» is needed here.

**88.** *(backtranslation)* `Die fünfbändiger Handschrift wurde von Antonio Maria Tasca kopiert , gleichfalls befindet sich im Archiv .`
→ `Die fünfbändige Handschrift wurde von Antonio Maria Tasca kopiert und befindet sich gleichfalls im Archiv .`
  - *R:ADJ:FORM*: An adjective before a noun takes an ending that depends on the noun's gender, case and number, and on the article in front of it. It should read «fünfbändige».
  - *R:OTHER*: «,» is not the right word here. German uses «und» in this context.
  - *R:WO*: The word order is wrong. German fixes the position of the verb: second in a main clause, last in a subordinate one. It should read «befindet sich gleichfalls».

**89.** *(backtranslation)* `Zwar galt die Credentes nicht als Mitglieder der Katharische Kirche , da sie die Consolamentum nicht erhalten hatten , aber das Melioramentum war ein Zeugnis dass die Credentes eines Tages die Consolamentum erhalten würden .`
→ `Zwar galten die Credentes nicht als Mitglieder der katharischen Kirche , da sie das Consolamentum nicht erhalten hatten , aber das Melioramentum war ein Zeugnis dafür , dass die Credentes eines Tages das Consolamentum erhalten würden .`
  - *R:VERB:FORM*: The verb is in the wrong form. Check the tense, the person, and whether this clause needs an infinitive or a participle. It should be «galten».
  - *R:ADJ:FORM*: An adjective before a noun takes an ending that depends on the noun's gender, case and number, and on the article in front of it. It should read «katharischen».
  - *R:DET:FORM*: The article has to match its noun in gender, case and number. Here that makes it «das», not «die».
  - *M:OTHER*: A word is missing here. Add «dafür ,».
  - *R:DET:FORM*: The article has to match its noun in gender, case and number. Here that makes it «das», not «die».

**90.** *(synthetic)* `bach erster angaben sterben 50 Menschen und 118 werden verletzt .`
→ `Nach ersten Angaben sterben 50 Menschen und 118 werden verletzt .`
  - *R:SPELL*: «bach» is not a German word. It looks like a slip for «Nach».
  - *R:ADJ:FORM*: An adjective before a noun takes an ending that depends on the noun's gender, number and case, and on whether an article comes first. Here the adjective should read «ersten», not «erster».
  - *R:ORTH*: German capitalises every noun, not just names. «angaben» is a noun here, so it needs a capital letter: «Angaben». (Rat für deutsche Rechtschreibung, § 55)

**91.** *(backtranslation)* `In Zusammenarbeit von Beamten mehrerer Polizeiinspektionen und Diebstahlsgruppe des Landeskriminalamtes Niederösterreich wurden die Männer schließlich ausgeforscht .`
→ `In Zusammenarbeit von Beamten mehrerer Polizeiinspektionen und der Diebstahlsgruppe des Landeskriminalamtes Niederösterreich wurden die Männer schließlich ausgeforscht .`
  - *M:DET*: A German noun almost always needs an article, even where English drops it. Add «der».

**92.** *(backtranslation)* `Sie bleiben mit dem Jude , als der geist die beide verfolgt und uberbringen will , und fahren mit ihm nach Flordivia .`
→ `Sie bleibt bei Jude , als der Geist die beiden verfolgt und umbringen will , und fährt mit ihm nach Florida .`
  - *R:VERB:FORM*: The verb is in the wrong form. Check the tense, the person, and whether this clause needs an infinitive or a participle. It should be «bleibt».
  - *R:ADP*: German prepositions are not chosen by translating the English one. This phrase takes «bei» rather than «mit».
  - *U:DET*: The article «dem» does not belong here. Remove it.
  - *R:ORTH*: This is a convention of written German rather than a grammar rule. The correct written form here is «Geist».
  - *R:PRON:FORM*: The pronoun «beide» is in the wrong form. It should be «beiden».
  - *R:SPELL*: «uberbringen» is not spelled correctly. It should be «umbringen».
  - *R:VERB:FORM*: The verb is in the wrong form. Check the tense, the person, and whether this clause needs an infinitive or a participle. It should be «fährt».
  - *R:SPELL*: «Flordivia» is not spelled correctly. It should be «Florida».

**93.** *(backtranslation)* `Donald Trump jr. äussert Bedenken über die höhe Frauenwahlbeteiligung .`
→ `Donald Trump Jr. äußert Bedenken über die hohe Frauenwahlbeteiligung .`
  - *R:ORTH*: This is a convention of written German rather than a grammar rule. The correct written form here is «Jr.».
  - *R:SPELL*: «äussert» is not spelled correctly. It should be «äußert».
  - *R:SPELL*: «höhe» is not spelled correctly. It should be «hohe».

**94.** *(backtranslation)* `Am Saisonende gewann er mit den Mannschaft den Landesmeistertitel .`
→ `Am Saisonende gewann er mit der Mannschaft den Landesmeistertitel .`
  - *R:DET:FORM*: The article has to match its noun in gender, case and number. Here that makes it «der», not «den».

**95.** *(learner)* `Das mann nicht als Student zur Uni mit der S-Bahn eine Stunde fahren muss , sondern nur d . a .`
→ `Dass man nicht als Student zur Uni mit der S-Bahn eine Stunde fahren muss , sondern nur`
  - *R:OTHER*: «Das» is not the right word here. German uses «Dass» in this context.
  - *R:OTHER*: «mann» is not the right word here. German uses «man» in this context.
  - *U:PNOUN*: The proper name «d» does not belong here. Remove it.
  - *U:PUNCT*: This punctuation mark does not belong here. German does not separate these parts of a sentence.
  - *U:OTHER*: The word «a» does not belong here. Remove it.
  - *U:PUNCT*: This punctuation mark does not belong here. German does not separate these parts of a sentence.

**96.** *(learner)* `Meiner Meinung nach , sollten die Staaten einige Maßnahmen treffen , um die Studenten praxisorientiert zu werden und dies wird zur Folge haben eine bessere Zukunft für alle ausgebildete Leute .`
→ `Meiner Meinung nach , sollten die Staaten einige Maßnahmen treffen , um für die Studenten praxisorientiert zu werden und dies wird eine bessere Zukunft für alle ausgebildeten Leute zur Folge haben .`
  - *M:ADP*: A preposition is missing. This construction needs «für».
  - *U:OTHER*: «zur Folge haben» is in the wrong place in this sentence. German fixes word order more tightly than English does, so it has to move rather than stay where it is.
  - *R:ADJ:FORM*: An adjective before a noun takes an ending that depends on the noun's gender, case and number, and on the article in front of it. It should read «ausgebildeten».
  - *M:OTHER*: «zur Folge haben» is in the wrong place in this sentence. German fixes word order more tightly than English does, so it has to move rather than stay where it is.

**97.** *(learner)* `Zum Schluss , trotz des Arbeitsbedigungen von bestimmten Berufen , oft leisten diese Individuelle nicht solche extrem hohe Gehälten .`
→ `Zum Schluss , oft leisten sich diese Individuellen trotz der Arbeitsbedingungen von bestimmten Berufen nicht solche extrem hohen Gehälter .`
  - *M:OTHER*: A word is missing here. Add «oft leisten sich diese Individuellen».
  - *R:DET:FORM*: The article has to match its noun in gender, case and number. Here that makes it «der», not «des».
  - *R:SPELL*: «Arbeitsbedigungen» is not spelled correctly. It should be «Arbeitsbedingungen».
  - *U:OTHER*: The word «, oft leisten diese Individuelle» does not belong here. Remove it.
  - *R:ADJ:FORM*: An adjective before a noun takes an ending that depends on the noun's gender, case and number, and on the article in front of it. It should read «hohen».
  - *R:SPELL*: «Gehälten» is not spelled correctly. It should be «Gehälter».

**98.** *(backtranslation)* `In die letzte zehn Jahren , die bevölkerung wachst knapp 2000 Köpf sind .`
→ `In den letzten zehn Jahren wuchs die Bevölkerung um knapp 2.000 Köpfe .`
  - *R:DET:FORM*: The article has to match its noun in gender, case and number. Here that makes it «den», not «die».
  - *R:ADJ:FORM*: An adjective before a noun takes an ending that depends on the noun's gender, case and number, and on the article in front of it. It should read «letzten».
  - *R:OTHER*: «,» is not the right word here. German uses «wuchs» in this context.
  - *R:ORTH*: This is a convention of written German rather than a grammar rule. The correct written form here is «Bevölkerung».
  - *R:OTHER*: «wachst» is not the right word here. German uses «um» in this context.
  - *R:OTHER*: «2000» is not the right word here. German uses «2.000» in this context.
  - *R:NOUN*: «Köpf» is not the right noun here. Use «Köpfe».
  - *U:AUX*: This auxiliary verb is not needed here. Remove «sind».

**99.** *(backtranslation)* `Ein Starter befand sich auf dem Deck , vor der Brücke , der zweite war vom Heck aus gesehen , hinter dem Hubschrauberhangar platziert .`
→ `Ein Starter befand sich auf dem Deck vor der Brücke , der zweite war , vom Heck aus gesehen , hinter dem Hubschrauberhangar platziert .`
  - *U:PUNCT*: «,» is in the wrong place in this sentence. German fixes word order more tightly than English does, so it has to move rather than stay where it is.
  - *M:PUNCT*: «,» is in the wrong place in this sentence. German fixes word order more tightly than English does, so it has to move rather than stay where it is.

**100.** *(learner)* `Mit freundlichen Grüssen Bewerbungsunte`
→ `Mit freundlichen Grüssen Bewerbungsunterlagen`
  - *R:SPELL*: «Bewerbungsunte» is not spelled correctly. It should be «Bewerbungsunterlagen».

