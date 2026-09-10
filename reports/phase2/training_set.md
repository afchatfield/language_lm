# Phase 2: the training set

29,237 examples: **19,237 real** learner sentences from Falko-MERLIN train and **10,000 synthetic** ones. 6,411 need no correction (21.9%).

Iteration 2 of the recipe. The first was synthetic-only and made the model worse than the untrained baseline on 14 of 15 error types (`reports/phase3/evaluation.md`), so real learner errors are the primary signal now and the corruptors are the supporting one.

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
| 0 | 21.9% | 22.1% |
| 1 | 20.9% | 21.2% |
| 2 | 18.1% | 17.5% |
| 3 or more | 39.1% | 39.2% |

Mean edits per sentence: **2.34** (Falko train: 2.55, iteration 1: 1.17).

## Error types

**54** distinct types, against 8 in iteration 1 and 54 in Falko train.

| Error type | Count | Share |
|---|---:|---:|
| `R:SPELL` | 11,833 | 17.3% |
| `R:DET:FORM` | 8,346 | 12.2% |
| `M:PUNCT` | 7,228 | 10.5% |
| `R:ORTH` | 6,467 | 9.4% |
| `R:OTHER` | 3,869 | 5.6% |
| `R:ADP` | 3,658 | 5.3% |
| `R:ADJ:FORM` | 3,512 | 5.1% |
| `M:DET` | 3,345 | 4.9% |
| `R:WO` | 2,130 | 3.1% |
| `R:NOUN:FORM` | 1,930 | 2.8% |
| `U:PUNCT` | 1,756 | 2.6% |
| `R:MORPH` | 1,358 | 2.0% |
| `M:PRON` | 1,177 | 1.7% |
| `R:VERB:FORM` | 963 | 1.4% |
| `R:PUNCT` | 872 | 1.3% |
| `M:AUX` | 782 | 1.1% |
| `R:AUX:FORM` | 663 | 1.0% |
| `U:AUX` | 599 | 0.9% |
| `M:ADP` | 587 | 0.9% |
| `U:PRON` | 556 | 0.8% |

## Explanations

19,480 edits carry a rule-level explanation, from the corruptor that made them. 49,058 carry a type-level one, because a Falko edit records what the annotator changed and not why.

That split is the point of the mix: the real data covers all the error types and teaches what learner German looks like, and the synthetic data teaches how to explain the eight it can produce properly.

## 100 samples

The Phase 2 exit criterion: read these, and if more than five have a wrong correction or a wrong explanation, fix the pipeline before training on it.

**1.** *(learner)* `Für die Familien mit Kinder ist Stadt Y eine gute Lösung .`
→ `Für die Familien mit Kindern ist Stadt Y eine gute Lösung .`
  - *R:NOUN:FORM*: The noun is in the wrong form. German marks case and number on the noun itself, so it should be «Kindern».

**2.** *(learner)* `Aber ich meine , dass diese immer existiert haben und dass es besser ist , wenn es für die Mutter kein Gefahr gibt , und dass dieses Argument nicht ehrlich ist .`
→ `Aber ich meine , dass diese immer existiert haben und dass es besser ist , wenn es für die Mutter keine Gefahr gibt , und dass dieses Argument nicht ehrlich ist .`
  - *R:DET:FORM*: The article has to match its noun in gender, case and number. Here that makes it «keine», not «kein».

**3.** *(learner)* `Liebe Dana , ich danke dir für deinen Brief Und schon am Anfang möchte ich dir gratulieren zur deiner bestandene Prüfung .`
→ `Liebe Dana , ich danke dir für deinen Brief und schon am Anfang möchte ich dir zur deiner bestandenen Prüfung gratulieren .`
  - *R:ORTH*: This is a convention of written German rather than a grammar rule. The correct written form here is «und».
  - *U:VERB*: «gratulieren» is in the wrong place in this sentence. German fixes word order more tightly than English does, so it has to move rather than stay where it is.
  - *R:ADJ:FORM*: An adjective before a noun takes an ending that depends on the noun's gender, case and number, and on the article in front of it. It should read «bestandenen».
  - *M:VERB*: «gratulieren» is in the wrong place in this sentence. German fixes word order more tightly than English does, so it has to move rather than stay where it is.

**4.** *(learner)* `Ich habe ein shenken für dich .`
→ `ich habe ein Geschenk für dich .`
  - *R:ORTH*: This is a convention of written German rather than a grammar rule. The correct written form here is «ich».
  - *R:SPELL*: «shenken» is not spelled correctly. It should be «Geschenk».

**5.** *(learner)* `Ich glaube , dass die Feminismus die Frauen einen Fenster geöffnet hat .`
→ `Ich glaube , dass der Feminismus den Frauen ein Fenster geöffnet hat .`
  - *R:DET:FORM*: The article has to match its noun in gender, case and number. Here that makes it «der», not «die».
  - *R:DET:FORM*: The article has to match its noun in gender, case and number. Here that makes it «den», not «die».
  - *R:DET:FORM*: The article has to match its noun in gender, case and number. Here that makes it «ein», not «einen».

**6.** *(synthetic)* `Westfassade wird von flachen , abgetreppten Strebepfeilern gegliedert .`
→ `Die Westfassade wird von flachen , abgetreppten Strebepfeilern gegliedert .`
  - *M:DET*: German nouns almost always need an article, even where English drops it. Add «Die» before the noun.

**7.** *(learner)* `5 , 53119 Bonn Seher geeharter Dammen und Herren !`
→ `5 , 53119 Bonn Sehr geehrte Damen und Herren !`
  - *R:OTHER*: «Seher» is not the right word here. German uses «Sehr» in this context.
  - *R:SPELL*: «geeharter» is not spelled correctly. It should be «geehrte».
  - *R:SPELL*: «Dammen» is not spelled correctly. It should be «Damen».

**8.** *(learner)* `Haben die Au-pair Mäddchen die gleiche Rechte wie die deutschen Au-pair ?`
→ `Haben die Au-pair-Mädchen die gleichen Rechte wie die deutschen Au-pair ?`
  - *R:OTHER*: «Au-pair Mäddchen» is not the right word here. German uses «Au-pair-Mädchen» in this context.
  - *R:ADJ:FORM*: An adjective before a noun takes an ending that depends on the noun's gender, case and number, and on the article in front of it. It should read «gleichen».

**9.** *(synthetic)* `Dieser ist laut Polizeiangaben ab 16.30 Uhr geplant startet in der Kampstrasse und führt über der B54 / Ruhrallee zum Stadion des BVB .`
→ `Dieser ist laut Polizeiangaben ab 16.30 Uhr geplant , startet in der Kampstraße und führt über die B54 / Ruhrallee zum Stadion des BVB .`
  - *M:PUNCT*: A comma belongs here. German uses them to mark off parts of a sentence -- an apposition, an insertion, a list -- more strictly than English does.
  - *R:SPELL*: After a long vowel or a diphthong German writes ß, not ss. «Kampstraße» keeps the ß because the vowel before it is long; ss would say the vowel is short. (Rat für deutsche Rechtschreibung, § 25)
  - *R:DET:FORM*: The article has to agree with its noun in gender, number and case. «der» does not fit here; the form the sentence needs is «die».

**10.** *(learner)* `Frauen und Männer müssen , wenn sie die gleiche Arbeit ausführen auch der gleiche Lohn bekommen .`
→ `Frauen und Männer müssen , wenn sie die gleiche Arbeit ausführen , auch den gleichen Lohn bekommen .`
  - *M:PUNCT*: A punctuation mark is missing here. German punctuates by rule rather than by ear, and this sentence needs «,».
  - *R:DET:FORM*: The article has to match its noun in gender, case and number. Here that makes it «den», not «der».
  - *R:ADJ:FORM*: An adjective before a noun takes an ending that depends on the noun's gender, case and number, and on the article in front of it. It should read «gleichen».

**11.** *(synthetic)* `Der Sitz des Bischofs lag auf der Festung Leibnitz außerhalb Diözesangebietes .`
→ `Der Sitz des Bischofs lag auf der Festung Leibnitz außerhalb des Diözesangebietes .`
  - *M:DET*: German nouns almost always need an article, even where English drops it. Add «des» before the noun.

**12.** *(synthetic)* `Am 23. februar 2016 erschien das Spiel für Xbox One und PlayStation 4. Ein Windows-Umsetzung folgte am 1. närz .`
→ `Am 23. Februar 2016 erschien das Spiel für Xbox One und PlayStation 4. Eine Windows-Umsetzung folgte am 1. März .`
  - *R:ORTH*: German capitalises every noun, not just names. «februar» is a noun here, so it needs a capital letter: «Februar». (Rat für deutsche Rechtschreibung, § 55)
  - *R:DET:FORM*: The article has to agree with its noun in gender, number and case. «Ein» does not fit here; the form the sentence needs is «Eine».
  - *R:SPELL*: «närz» is not a German word. It looks like a slip for «März».

**13.** *(learner)* `Diesen Fach fand ich unwichtig , da fast alle Informationen aus dem Internet stammten .`
→ `Dieses Fach fand ich unwichtig , da fast alle Informationen aus dem Internet stammten .`
  - *R:DET:FORM*: The article has to match its noun in gender, case and number. Here that makes it «Dieses», not «Diesen».

**14.** *(learner)* `Es gibt klar diesen illegalen Diktatur , Mugabe , den durch diesen Terror herrscht .`
→ `Es gibt klar diese illegale Diktatur , Mugabe , der durch diesen Terror herrscht .`
  - *R:DET:FORM*: The article has to match its noun in gender, case and number. Here that makes it «diese», not «diesen».
  - *R:ADJ:FORM*: An adjective before a noun takes an ending that depends on the noun's gender, case and number, and on the article in front of it. It should read «illegale».
  - *R:PRON:FORM*: The pronoun «den» is in the wrong form. It should be «der».

**15.** *(learner)* `Es ist ruhig und ganz liebe .`
→ `Es ist ruhig und ganz lieb .`
  - *R:MORPH*: «lieb» is built from the same stem as «liebe» but is the form this sentence needs -- a different ending, or a different part of speech.

**16.** *(learner)* `Sie fügen nicht nur den Menschen drumherum Schaden zu sondern auch sich selbst .`
→ `Sie fügen nicht nur den Menschen um sich Schaden zu , sondern auch sich selbst .`
  - *R:OTHER*: «drumherum» is not the right word here. German uses «um sich» in this context.
  - *M:PUNCT*: A punctuation mark is missing here. German punctuates by rule rather than by ear, and this sentence needs «,».

**17.** *(learner)* `Sie lohnen sich wohl irgendjemandem ( genauer gesagt :`
→ `Sie lohnen sich wohl für irgendjemanden ( genauer gesagt :`
  - *M:ADP*: A preposition is missing. This construction needs «für».
  - *R:MORPH*: «irgendjemanden» is built from the same stem as «irgendjemandem» but is the form this sentence needs -- a different ending, or a different part of speech.

**18.** *(learner)* `Aus grund , Deutsch verbesser .`
→ `Aus dem Grund , mein Deutsch zu verbessern .`
  - *M:DET*: A German noun almost always needs an article, even where English drops it. Add «dem».
  - *R:ORTH*: This is a convention of written German rather than a grammar rule. The correct written form here is «Grund».
  - *M:DET*: A German noun almost always needs an article, even where English drops it. Add «mein».
  - *M:PART*: A particle is missing here. Add «zu».
  - *R:MORPH*: «verbessern» is built from the same stem as «verbesser» but is the form this sentence needs -- a different ending, or a different part of speech.

**19.** *(learner)* `Heutzutage in einem flexiblen Arbeitsmarkt ist das allerdings leichter gesagt als getan .`
→ `Heutzutage ist das auf einem flexiblen Arbeitsmarkt allerdings leichter gesagt als getan .`
  - *M:AUX*: «ist» is in the wrong place in this sentence. German fixes word order more tightly than English does, so it has to move rather than stay where it is.
  - *M:PRON*: «das» is in the wrong place in this sentence. German fixes word order more tightly than English does, so it has to move rather than stay where it is.
  - *R:ADP*: German prepositions are not chosen by translating the English one. This phrase takes «auf» rather than «in».
  - *U:AUX*: «ist» is in the wrong place in this sentence. German fixes word order more tightly than English does, so it has to move rather than stay where it is.
  - *U:PRON*: «das» is in the wrong place in this sentence. German fixes word order more tightly than English does, so it has to move rather than stay where it is.

**20.** *(synthetic)* `Trotz ihrer Kinderlähmung , die sie an den Rollstuhl fesselte gründete sie ein Unternehmen das zu einer globalen Marke wurde .`
→ `Trotz ihrer Kinderlähmung , die sie an den Rollstuhl fesselte , gründete sie ein Unternehmen , das zu einer globalen Marke wurde .`
  - *M:PUNCT*: A comma belongs here. German uses them to mark off parts of a sentence -- an apposition, an insertion, a list -- more strictly than English does.
  - *M:PUNCT*: German separates a subordinate clause from its main clause with a comma. Unlike English, this is a rule rather than a matter of taste. Add the comma. (Rat für deutsche Rechtschreibung, § 74)

**21.** *(learner)* `Im Endejahr werde ich neue auto kaufen .`
→ `Am Ende des Jahres werde ich ein neues Auto kaufen .`
  - *R:ADP*: German prepositions are not chosen by translating the English one. This phrase takes «Am» rather than «Im».
  - *R:SPELL*: «Endejahr» is not spelled correctly. It should be «Ende».
  - *M:OTHER*: A word is missing here. Add «des Jahres».
  - *M:DET*: A German noun almost always needs an article, even where English drops it. Add «ein».
  - *R:ADJ:FORM*: An adjective before a noun takes an ending that depends on the noun's gender, case and number, and on the article in front of it. It should read «neues».
  - *R:ORTH*: This is a convention of written German rather than a grammar rule. The correct written form here is «Auto».

**22.** *(synthetic)* `Erste urkundliche Erwähnung findet der Ort im Jahr 1181 als „ ekessem “ ; über „ Eycse “ und „ Eyksen “ wandelte sich dem Name schließlich zu Eixe .`
→ `Erste urkundliche Erwähnung findet der Ort im Jahr 1181 als „ Ekessem “ ; über „ Eycse “ und „ Eyksen “ wandelte sich der Name schließlich zu Eixe .`
  - *R:ORTH*: German capitalises every noun, not just names. «ekessem» is a noun here, so it needs a capital letter: «Ekessem». (Rat für deutsche Rechtschreibung, § 55)
  - *R:DET:FORM*: The article has to agree with its noun in gender, number and case. «dem» does not fit here; the form the sentence needs is «der».

**23.** *(learner)* `Am anfang der Fernsehsendung " Friends " haben die weibliche Hauptfiguren weniger als die Männliche verdient .`
→ `Am Anfang der Fernsehsendung " Friends " haben die weiblichen Hauptfiguren weniger als die männlichen verdient .`
  - *R:ORTH*: This is a convention of written German rather than a grammar rule. The correct written form here is «Anfang».
  - *R:ADJ:FORM*: An adjective before a noun takes an ending that depends on the noun's gender, case and number, and on the article in front of it. It should read «weiblichen».
  - *R:MORPH*: «männlichen» is built from the same stem as «Männliche» but is the form this sentence needs -- a different ending, or a different part of speech.

**24.** *(synthetic)* `Hebel aelbst habe diese Veränderungen durchaus kritisch gesehen heißt es zwischen der Ankündigung die Hebelfreunde .`
→ `Hebel selbst habe diese Veränderungen durchaus kritisch gesehen , heißt es in der Ankündigung der Hebelfreunde .`
  - *R:SPELL*: «aelbst» is not a German word. It looks like a slip for «selbst».
  - *M:PUNCT*: A comma belongs here. German uses them to mark off parts of a sentence -- an apposition, an insertion, a list -- more strictly than English does.
  - *R:ADP*: German prepositions are not chosen by translating the English one. This phrase takes «in» rather than «zwischen».
  - *R:DET:FORM*: The article has to agree with its noun in gender, number and case. «die» does not fit here; the form the sentence needs is «der».

**25.** *(synthetic)* `siese messen mit geburt circa 17 cm .`
→ `Diese messen bei der Geburt circa 17 cm .`
  - *R:SPELL*: «siese» is not a German word. It looks like a slip for «Diese».
  - *R:ADP*: German prepositions are not chosen by translating the English one. This phrase takes «bei» rather than «mit».
  - *M:DET*: German nouns almost always need an article, even where English drops it. Add «der» before the noun.
  - *R:ORTH*: German capitalises every noun, not just names. «geburt» is a noun here, so it needs a capital letter: «Geburt». (Rat für deutsche Rechtschreibung, § 55)

**26.** *(synthetic)* `Dafur müssten verleger ihn qber erst einmal ausfindig machen .`
→ `Dafür müssten Verleger ihn aber erst einmal ausfindig machen .`
  - *R:SPELL*: The umlaut is part of the letter, not decoration on it. «Dafur» and «Dafür» are different words to a German reader. Write «Dafür».
  - *R:ORTH*: German capitalises every noun, not just names. «verleger» is a noun here, so it needs a capital letter: «Verleger». (Rat für deutsche Rechtschreibung, § 55)
  - *R:SPELL*: «qber» is not a German word. It looks like a slip for «aber».

**27.** *(synthetic)* `Auf der durchgehenden Fahrbahn zwischen Böblingen und whningen steht weiterhin jeweils eine Geradeausspur je Fahrtrichtung zur Verfügung .`
→ `Auf der durchgehenden Fahrbahn zwischen Böblingen und Ehningen steht weiterhin jeweils eine Geradeausspur je Fahrtrichtung zur Verfügung .`
  - *R:SPELL*: «whningen» is not a German word. It looks like a slip for «Ehningen».

**28.** *(learner)* `Man verliert alles was einem wichtig ist , Karriere , Famile und Freunde .`
→ `Man verliert alles , was einem wichtig ist : Karriere , Familie und Freunde .`
  - *M:PUNCT*: A punctuation mark is missing here. German punctuates by rule rather than by ear, and this sentence needs «,».
  - *R:PUNCT*: The wrong punctuation mark is used here. It should be «:».
  - *R:SPELL*: «Famile» is not spelled correctly. It should be «Familie».

**29.** *(synthetic)* `Elemente des Rundbogenstils , der Neugotik und Frührenaissance treten gemeinsam unter .`
→ `Elemente des Rundbogenstils , der Neugotik und der Frührenaissance treten gemeinsam auf .`
  - *M:DET*: German nouns almost always need an article, even where English drops it. Add «der» before the noun.
  - *R:ADP*: German prepositions are not chosen by translating the English one. This phrase takes «auf» rather than «unter».

**30.** *(learner)* `LIEBER JENS GLÜCKWÜNSCH`
→ `Lieber Jens , Glückwunsch`
  - *R:ORTH*: This is a convention of written German rather than a grammar rule. The correct written form here is «Lieber».
  - *R:ORTH*: This is a convention of written German rather than a grammar rule. The correct written form here is «Jens».
  - *R:OTHER*: «GLÜCKWÜNSCH» is not the right word here. German uses «, Glückwunsch» in this context.

**31.** *(learner)* `Ich brauche einen klein Wohnung im zentrum oder in der nehe von U-Bahn .`
→ `Ich brauche eine kleine Wohnung im Zentrum oder in der Nähe der U-Bahn .`
  - *R:DET:FORM*: The article has to match its noun in gender, case and number. Here that makes it «eine», not «einen».
  - *R:ADJ:FORM*: An adjective before a noun takes an ending that depends on the noun's gender, case and number, and on the article in front of it. It should read «kleine».
  - *R:ORTH*: This is a convention of written German rather than a grammar rule. The correct written form here is «Zentrum».
  - *R:OTHER*: «nehe» is not the right word here. German uses «Nähe» in this context.
  - *R:OTHER*: «von» is not the right word here. German uses «der» in this context.

**32.** *(synthetic)* `Ein Kind steht vor olakat des Deutschen kinderhilfswerks mit der Aufschrift „ kinderrechte “ .`
→ `Ein Kind steht vor einem Plakat des Deutschen Kinderhilfswerks mit der Aufschrift „ Kinderrechte “ .`
  - *M:DET*: German nouns almost always need an article, even where English drops it. Add «einem» before the noun.
  - *R:SPELL*: «olakat» is not a German word. It looks like a slip for «Plakat».
  - *R:ORTH*: German capitalises every noun, not just names. «kinderhilfswerks» is a noun here, so it needs a capital letter: «Kinderhilfswerks». (Rat für deutsche Rechtschreibung, § 55)
  - *R:ORTH*: German capitalises every noun, not just names. «kinderrechte» is a noun here, so it needs a capital letter: «Kinderrechte». (Rat für deutsche Rechtschreibung, § 55)

**33.** *(learner)* `Indem die Feministen gestrebt haben , haben sie die Lage für Frauen am Arbeitsplatz verbessert , jedoch könnte die Lage noch gleicher werden .`
→ `Indem die Feministen danach gestrebt haben , haben sie die Lage für Frauen am Arbeitsplatz verbessert , jedoch könnte die Lage noch gleicher werden .`
  - *M:ADV*: A adverb is missing here. Add «danach».

**34.** *(synthetic)* `Bis heute folgten mindestens 11 weitere unterschiedlich vollstandige skelettfunde .`
→ `Bis heute folgten mindestens 11 weitere , unterschiedlich vollständige Skelettfunde .`
  - *M:PUNCT*: A comma belongs here. German uses them to mark off parts of a sentence -- an apposition, an insertion, a list -- more strictly than English does.
  - *R:SPELL*: The umlaut is part of the letter, not decoration on it. «vollstandige» and «vollständige» are different words to a German reader. Write «vollständige».
  - *R:ORTH*: German capitalises every noun, not just names. «skelettfunde» is a noun here, so it needs a capital letter: «Skelettfunde». (Rat für deutsche Rechtschreibung, § 55)

**35.** *(learner)* `Eva Müller Winkelmann Zentralplatz 1 Stadt X 12345 Au-pair-Agentur Schultzt & Partner Regenstr .`
→ `Eva Müller Winkelmann Zentralplatz 1 Stadt X 12345 Au-pair-Agentur Schultz & Partner Regenstr .`
  - *R:SPELL*: «Schultzt» is not spelled correctly. It should be «Schultz».

**36.** *(synthetic)* `Er nissgönnt ihr diese Zustände nicht und würde sie ihr auch nicht wegnehmen wollen wenn die ginge .`
→ `Er missgönnt ihr diese Zustände nicht und würde sie ihr auch nicht wegnehmen wollen , wenn das ginge .`
  - *R:SPELL*: «nissgönnt» is not a German word. It looks like a slip for «missgönnt».
  - *M:PUNCT*: German separates a subordinate clause from its main clause with a comma. Unlike English, this is a rule rather than a matter of taste. Add the comma. (Rat für deutsche Rechtschreibung, § 74)
  - *R:DET:FORM*: The article has to agree with its noun in gender, number and case. «die» does not fit here; the form the sentence needs is «das».

**37.** *(learner)* `Die Arbeitsmarkt ist nicht so gut heute und so ist es wichtig dass man ein Diplom hat um die beste Chance zu haben .`
→ `Der Arbeitsmarkt ist nicht so gut heute und so ist es wichtig , dass man ein Diplom hat , um die beste Chance zu haben .`
  - *R:DET:FORM*: The article has to match its noun in gender, case and number. Here that makes it «Der», not «Die».
  - *M:PUNCT*: A punctuation mark is missing here. German punctuates by rule rather than by ear, and this sentence needs «,».
  - *M:PUNCT*: A punctuation mark is missing here. German punctuates by rule rather than by ear, and this sentence needs «,».

**38.** *(synthetic)* `Deswegen gaben wir bei allen Überlegungen jetzt schon darauf gedrängt , dass bei allfälligem Wegfall von Versorgungsleistungen an Einzelstandorten auch entsprechender Ersatz gesichert sein muss .`
→ `Deswegen haben wir bei allen Überlegungen jetzt schon darauf gedrängt , dass bei allfälligem Wegfall von Versorgungsleistungen an Einzelstandorten auch entsprechender Ersatz gesichert sein muss .`
  - *R:SPELL*: «gaben» is not a German word. It looks like a slip for «haben».

**39.** *(learner)* `Ich muss nicht nur in Deutschland unsere Produkt verkaufen , sondern auch in andere Lände , zum beispiel in Schweden , Frankreich , Russland .`
→ `Ich muss nicht nur in Deutschland unsere Produkt verkaufen , sondern auch in andere Länder , zum Beispiel in Schweden , Frankreich , Russland .`
  - *R:SPELL*: «Lände» is not spelled correctly. It should be «Länder».
  - *R:ORTH*: This is a convention of written German rather than a grammar rule. The correct written form here is «Beispiel».

**40.** *(learner)* `Auch nicht zu vergessen ist das Risiko vom Diebstahl ; etwas das nicht selten in Haftanstalten passiert .`
→ `Auch nicht zu vergessen ist das Risiko vom Diebstahl ; etwas , das nicht selten in Haftanstalten passiert .`
  - *M:PUNCT*: A punctuation mark is missing here. German punctuates by rule rather than by ear, and this sentence needs «,».

**41.** *(synthetic)* `Ab 2009 war sie mit Kleinst- und Nebenrollen im ungarischem fernsehen sowie in kurzfilmen vertreten .`
→ `Ab 2009 war sie mit Kleinst- und Nebenrollen im ungarischen Fernsehen sowie in Kurzfilmen vertreten .`
  - *R:ADJ:FORM*: An adjective before a noun takes an ending that depends on the noun's gender, number and case, and on whether an article comes first. Here the adjective should read «ungarischen», not «ungarischem».
  - *R:ORTH*: German capitalises every noun, not just names. «fernsehen» is a noun here, so it needs a capital letter: «Fernsehen». (Rat für deutsche Rechtschreibung, § 55)
  - *R:ORTH*: German capitalises every noun, not just names. «kurzfilmen» is a noun here, so it needs a capital letter: «Kurzfilmen». (Rat für deutsche Rechtschreibung, § 55)

**42.** *(learner)* `Es gibt jedoch leute , die die neuen studiengänge als eine negative Entwicklung behalten .`
→ `Es gibt jedoch Leute , die die neuen Studiengänge als eine negative Entwicklung ansehen .`
  - *R:ORTH*: This is a convention of written German rather than a grammar rule. The correct written form here is «Leute».
  - *R:ORTH*: This is a convention of written German rather than a grammar rule. The correct written form here is «Studiengänge».
  - *R:VERB*: «behalten» is not the right verb here. Use «ansehen».

**43.** *(learner)* `Wie heißen das Babys ?`
→ `Wie heißen die Babys ?`
  - *R:DET:FORM*: The article has to match its noun in gender, case and number. Here that makes it «die», not «das».

**44.** *(synthetic)* `Am sonnerstag haben Beamte bach eigenen angaben 14 verbotenes Waffen sichergestellt .`
→ `Am Donnerstag haben Beamte nach eigenen Angaben 14 verbotene Waffen sichergestellt .`
  - *R:SPELL*: «sonnerstag» is not a German word. It looks like a slip for «Donnerstag».
  - *R:SPELL*: «bach» is not a German word. It looks like a slip for «nach».
  - *R:ORTH*: German capitalises every noun, not just names. «angaben» is a noun here, so it needs a capital letter: «Angaben». (Rat für deutsche Rechtschreibung, § 55)
  - *R:ADJ:FORM*: An adjective before a noun takes an ending that depends on the noun's gender, number and case, and on whether an article comes first. Here the adjective should read «verbotene», not «verbotenes».

**45.** *(learner)* `Wenn man eventuell für die wirkliche Welt bereit sein will , sollte er auch lernen wie man Stress behandeln sollte .`
→ `Wenn man eventuell für die wirkliche Welt bereit sein will , sollte man auch lernen , wie man Stress behandeln sollte .`
  - *R:PRON*: «er» is not the right pronoun here. Use «man».
  - *M:PUNCT*: A punctuation mark is missing here. German punctuates by rule rather than by ear, and this sentence needs «,».

**46.** *(synthetic)* `Auch das Elektronik an Bord einer Mondkapsel wäre durch die schnellen teilchen gefahrdet .`
→ `Auch die Elektronik an Bord einer Mondkapsel wäre durch die schnellen Teilchen gefährdet .`
  - *R:DET:FORM*: The article has to agree with its noun in gender, number and case. «das» does not fit here; the form the sentence needs is «die».
  - *R:ORTH*: German capitalises every noun, not just names. «teilchen» is a noun here, so it needs a capital letter: «Teilchen». (Rat für deutsche Rechtschreibung, § 55)
  - *R:SPELL*: The umlaut is part of the letter, not decoration on it. «gefahrdet» and «gefährdet» are different words to a German reader. Write «gefährdet».

**47.** *(learner)* `Vielen Dank im Voraus Mit freundlichen Grüßen`
→ `Vielen Dank im Voraus . Mit freundlichen Grüßen`
  - *M:PUNCT*: A punctuation mark is missing here. German punctuates by rule rather than by ear, and this sentence needs «.».

**48.** *(learner)* `Die meisten möchten nur , dass die beiden Geschlechten gliech bezahlt würden , und dass Bestellten nachteilen Frauen nicht , falls sie schwanger werden .`
→ `Die meisten möchten nur , dass die beiden Geschlechter gleich bezahlt würden und dass Stellen Frauen nicht benachteiligen , falls sie schwanger werden .`
  - *R:SPELL*: «Geschlechten» is not spelled correctly. It should be «Geschlechter».
  - *R:SPELL*: «gliech» is not spelled correctly. It should be «gleich».
  - *U:PUNCT*: This punctuation mark does not belong here. German does not separate these parts of a sentence.
  - *R:OTHER*: «Bestellten nachteilen» is not the right word here. German uses «Stellen» in this context.
  - *M:VERB*: A verb is missing here. Add «benachteiligen».

**49.** *(synthetic)* `Es qurde vom 15. bis zum 13. Jahrhundert v. Chr. fur bestattungen verwendet .`
→ `Es wurde vom 15. bis zum 13. Jahrhundert v. Chr. für Bestattungen verwendet .`
  - *R:SPELL*: «qurde» is not a German word. It looks like a slip for «wurde».
  - *R:SPELL*: The umlaut is part of the letter, not decoration on it. «fur» and «für» are different words to a German reader. Write «für».
  - *R:ORTH*: German capitalises every noun, not just names. «bestattungen» is a noun here, so it needs a capital letter: «Bestattungen». (Rat für deutsche Rechtschreibung, § 55)

**50.** *(learner)* `Die Mehrheit von Krminellen sitzen wahrscheinlich im Knast oder werden irgendwann Zeit ins Gefängnis verbringen .`
→ `Die Mehrheit der Kriminellen sitzt wahrscheinlich im Knast oder wird irgendwann Zeit im Gefängnis verbringen .`
  - *R:OTHER*: «von» is not the right word here. German uses «der» in this context.
  - *R:SPELL*: «Krminellen» is not spelled correctly. It should be «Kriminellen».
  - *R:VERB:FORM*: The verb is in the wrong form. Check the tense, the person, and whether this clause needs an infinitive or a participle. It should be «sitzt».
  - *R:AUX:FORM*: The auxiliary is in the wrong form. It should be «wird».
  - *R:ADP*: German prepositions are not chosen by translating the English one. This phrase takes «im» rather than «ins».

**51.** *(learner)* `Stadt X , den 24 . September 2011 Sehr geehrte Damen und Herren , hiermit möchte ich mich über Ihren Au-pair-Agentur beschweren .`
→ `Stadt X , den 24 . September 2011 Sehr geehrte Damen und Herren , hiermit möchte ich mich über Ihre Au-pair-Agentur beschweren .`
  - *R:MORPH*: «Ihre» is built from the same stem as «Ihren» but is the form this sentence needs -- a different ending, or a different part of speech.

**52.** *(learner)* `Diese Maßnahmen beduetet , dass es ein großer Risk verhaftet zu sein gibt .`
→ `Diese Maßnahme bedeutet , dass es ein großes Risiko , verhaftet zu sein , gibt .`
  - *R:NOUN:FORM*: The noun is in the wrong form. German marks case and number on the noun itself, so it should be «Maßnahme».
  - *R:SPELL*: «beduetet» is not spelled correctly. It should be «bedeutet».
  - *R:ADJ:FORM*: An adjective before a noun takes an ending that depends on the noun's gender, case and number, and on the article in front of it. It should read «großes».
  - *R:OTHER*: «Risk» is not the right word here. German uses «Risiko ,» in this context.
  - *M:PUNCT*: A punctuation mark is missing here. German punctuates by rule rather than by ear, and this sentence needs «,».

**53.** *(synthetic)* `Aufgrund der weiteren Bebauung wurde einem eigenes Blockheizkraftwerk errichtet das 1982 seinen Betrieb aufnahm weitere qnlagen folgten .`
→ `Aufgrund der weiteren Bebauung wurde ein eigenes Blockheizkraftwerk errichtet , das 1982 seinen Betrieb aufnahm , weitere Anlagen folgten .`
  - *R:DET:FORM*: The article has to agree with its noun in gender, number and case. «einem» does not fit here; the form the sentence needs is «ein».
  - *M:PUNCT*: German separates a subordinate clause from its main clause with a comma. Unlike English, this is a rule rather than a matter of taste. Add the comma. (Rat für deutsche Rechtschreibung, § 74)
  - *M:PUNCT*: A comma belongs here. German uses them to mark off parts of a sentence -- an apposition, an insertion, a list -- more strictly than English does.
  - *R:SPELL*: «qnlagen» is not a German word. It looks like a slip for «Anlagen».

**54.** *(learner)* `Wenn das , was man getan hat , nicht so schlecht ist , ist die Strafe auch nicht so streng , wie eine kleine Geldstrafe oder ein paar Uhren oder Tage ins Gefängnis .`
→ `Wenn das , was man getan hat , nicht so schlecht ist , ist die Strafe auch nicht so streng , wie eine kleine Geldstrafe oder ein paar Uhren oder Tage im Gefängnis .`
  - *R:ADP*: German prepositions are not chosen by translating the English one. This phrase takes «im» rather than «ins».

**55.** *(learner)* `Sie weiß noch nich , was für eine Richtung gehen sollte .`
→ `Sie weiß noch nicht , in welche Richtung sie gehen sollte .`
  - *R:SPELL*: «nich» is not spelled correctly. It should be «nicht».
  - *U:PRON*: This pronoun is not needed. German does not repeat it here, so remove «was».
  - *R:ADP*: German prepositions are not chosen by translating the English one. This phrase takes «in» rather than «für».
  - *R:DET*: «eine» is not the right article here. Use «welche».
  - *M:PRON*: A pronoun is missing. German does not leave out the subject the way some languages do, so «sie» is needed here.

**56.** *(learner)* `Während dieses Aufenhaltes kann man bei einer Firma arbeiten und damit kann man praxisorienter werden .`
→ `Während dieses Aufenthaltes kann man bei einer Firma arbeiten und damit kann man praxisorientierter werden .`
  - *R:SPELL*: «Aufenhaltes» is not spelled correctly. It should be «Aufenthaltes».
  - *R:SPELL*: «praxisorienter» is not spelled correctly. It should be «praxisorientierter».

**57.** *(learner)* `Hier kann man also die Grenzen dieser Theorie sehen und zwar , wenn die Leute nicht " produktiv " sein können , werden sie nicht geholfen , weil es kein Beitrag für die Gesellschaft darstellt .`
→ `Hier kann man also die Grenzen dieser Theorie sehen , und zwar wird , wenn die Leute nicht " produktiv " sein können , ihnen nicht geholfen , weil es keinen Beitrag für die Gesellschaft darstellt .`
  - *M:PUNCT*: A punctuation mark is missing here. German punctuates by rule rather than by ear, and this sentence needs «,».
  - *M:AUX*: An auxiliary verb is missing. German builds its perfect and passive with «haben», «sein» or «werden», and this sentence needs «wird».
  - *U:AUX*: This auxiliary verb is not needed here. Remove «werden».
  - *R:PRON:FORM*: The pronoun «sie» is in the wrong form. It should be «ihnen».
  - *R:DET:FORM*: The article has to match its noun in gender, case and number. Here that makes it «keinen», not «kein».

**58.** *(learner)* `Was auch zu bedenken ist , dass wegen Feminismus Frauen nicht nur wahlen können , ob sie arbeiten wollen oder nicht .`
→ `Was auch zu bedenken ist , ist , dass wegen Feminismus Frauen nicht nur wählen können , ob sie arbeiten wollen oder nicht .`
  - *M:AUX*: An auxiliary verb is missing. German builds its perfect and passive with «haben», «sein» or «werden», and this sentence needs «ist».
  - *M:PUNCT*: A punctuation mark is missing here. German punctuates by rule rather than by ear, and this sentence needs «,».
  - *R:MORPH*: «wählen» is built from the same stem as «wahlen» but is the form this sentence needs -- a different ending, or a different part of speech.

**59.** *(learner)* `Man kann schneller aufsteigen und mit einem jungeren Alten .`
→ `Man kann schneller aufsteigen und mit einem jüngeren Alter .`
  - *R:SPELL*: «jungeren» is not spelled correctly. It should be «jüngeren».
  - *R:MORPH*: «Alter» is built from the same stem as «Alten» but is the form this sentence needs -- a different ending, or a different part of speech.

**60.** *(synthetic)* `Die Rio-Tinto-Aktie verlor hingegen eund zwei prozent .`
→ `Die Rio-Tinto-Aktie verlor hingegen rund zwei Prozent .`
  - *R:SPELL*: «eund» is not a German word. It looks like a slip for «rund».
  - *R:ORTH*: German capitalises every noun, not just names. «prozent» is a noun here, so it needs a capital letter: «Prozent». (Rat für deutsche Rechtschreibung, § 55)

**61.** *(learner)* `Es wäre Parktisch , wenn Sie einen Schweiz Bürger mit haben .`
→ `Es wäre praktisch , wenn Sie einen Schweizer Bürger mithaben .`
  - *R:SPELL*: «Parktisch» is not spelled correctly. It should be «praktisch».
  - *R:OTHER*: «Schweiz» is not the right word here. German uses «Schweizer» in this context.
  - *R:ORTH*: This is a convention of written German rather than a grammar rule. The correct written form here is «mithaben».

**62.** *(synthetic)* `Seit dpa-Informationen anstrebt dies der Deutsche Fußball-Bund aber nicht .`
→ `Nach dpa-Informationen strebt dies der Deutsche Fußball-Bund aber nicht an .`
  - *R:ADP*: German prepositions are not chosen by translating the English one. This phrase takes «Nach» rather than «Seit».
  - *R:WO*: This verb is separable. In a main clause the prefix detaches and moves to the end, so it is «strebt dies der Deutsche Fußball-Bund aber nicht an» rather than one word.

**63.** *(learner)* `Ich denke die Arbeitslösigkeitproblem spielt eine große Rolle in diese Problem .`
→ `Ich denke , die Arbeitslosigkeitsprobleme spielen eine große Rolle bei diesem Problem .`
  - *M:PUNCT*: A punctuation mark is missing here. German punctuates by rule rather than by ear, and this sentence needs «,».
  - *R:SPELL*: «Arbeitslösigkeitproblem» is not spelled correctly. It should be «Arbeitslosigkeitsprobleme».
  - *R:VERB:FORM*: The verb is in the wrong form. Check the tense, the person, and whether this clause needs an infinitive or a participle. It should be «spielen».
  - *R:ADP*: German prepositions are not chosen by translating the English one. This phrase takes «bei» rather than «in».
  - *R:DET:FORM*: The article has to match its noun in gender, case and number. Here that makes it «diesem», not «diese».

**64.** *(learner)* `Wenn Sie mir die Gelegenheit geben werden , dann kann ich auch meine persönlichen Eigenschaften beweisen wie z . B .`
→ `Wenn Sie mir die Gelegenheit geben werden , dann kann ich auch meine persönlichen Eigenschaften beweisen , wie z . B .`
  - *M:PUNCT*: A punctuation mark is missing here. German punctuates by rule rather than by ear, and this sentence needs «,».

**65.** *(learner)* `Wenn ein junger Mann oder eine junge Frau eine Arbeitsstelle sucht , ist es wichtig , dass er einige ähnliche Erfahrungen als Student hat , wie die anderen Mitarbeiter , mit den er mitmachen muss .`
→ `Wenn ein junger Mann oder eine junge Frau eine Arbeitsstelle sucht , ist es wichtig , dass er einige ähnliche Erfahrungen als Studenten haben wie die anderen Mitarbeiter , mit denen er mitmachen müssen .`
  - *R:NOUN:FORM*: The noun is in the wrong form. German marks case and number on the noun itself, so it should be «Studenten».
  - *R:AUX:FORM*: The auxiliary is in the wrong form. It should be «haben».
  - *U:PUNCT*: This punctuation mark does not belong here. German does not separate these parts of a sentence.
  - *R:PRON:FORM*: The pronoun «den» is in the wrong form. It should be «denen».
  - *R:VERB:FORM*: The verb is in the wrong form. Check the tense, the person, and whether this clause needs an infinitive or a participle. It should be «müssen».

**66.** *(learner)* `Sie stecken im einem Teufelkreis .`
→ `Sie stecken in einem Teufelskreis .`
  - *R:ADP*: German prepositions are not chosen by translating the English one. This phrase takes «in» rather than «im».
  - *R:SPELL*: «Teufelkreis» is not spelled correctly. It should be «Teufelskreis».

**67.** *(learner)* `Für alleinstehende empfehle ich ein Wohnung zu mieten .`
→ `Für Alleinstehende empfehle ich eine Wohnung zu mieten .`
  - *R:ORTH*: This is a convention of written German rather than a grammar rule. The correct written form here is «Alleinstehende».
  - *R:DET:FORM*: The article has to match its noun in gender, case and number. Here that makes it «eine», not «ein».

**68.** *(learner)* `Ich habe die Wesel Zeitung ´ Ihre Wohnung gelesen .`
→ `Ich habe in der Weseler Zeitung Ihre Wohnung gelesen .`
  - *M:ADP*: A preposition is missing. This construction needs «in».
  - *R:DET:FORM*: The article has to match its noun in gender, case and number. Here that makes it «der», not «die».
  - *R:SPELL*: «Wesel» is not spelled correctly. It should be «Weseler».
  - *U:VERB*: The verb «´» does not belong here. Remove it.

**69.** *(synthetic)* `Zwei Nordic Walker laufen am Rondell in iberhof durch den Schnee .`
→ `Zwei Nordic Walker laufen am Rondell in Oberhof durch den Schnee .`
  - *R:SPELL*: «iberhof» is not a German word. It looks like a slip for «Oberhof».

**70.** *(learner)* `Sehr geherteDamen und Herren , ich suche eine Wohnung , mit drei Zimmer und eine Kleine Balkon .`
→ `Sehr geehrte Damen und Herren , ich suche eine Wohnung mit drei Zimmern und einem kleinen Balkon .`
  - *R:OTHER*: «geherteDamen» is not the right word here. German uses «geehrte Damen» in this context.
  - *U:PUNCT*: This punctuation mark does not belong here. German does not separate these parts of a sentence.
  - *R:NOUN:FORM*: The noun is in the wrong form. German marks case and number on the noun itself, so it should be «Zimmern».
  - *R:DET:FORM*: The article has to match its noun in gender, case and number. Here that makes it «einem», not «eine».
  - *R:ADJ:FORM*: An adjective before a noun takes an ending that depends on the noun's gender, case and number, and on the article in front of it. It should read «kleinen».

**71.** *(synthetic)* `In dem Gebäude befand sich zudem auch ein Kindergarten , der in den 1980en Jahren in die Tangstedter Landstraße 152 zog .`
→ `In dem Gebäude befand sich zudem auch ein Kindergarten , der in den 1980er Jahren in die Tangstedter Landstraße 152 zog .`
  - *R:ADJ:FORM*: An adjective before a noun takes an ending that depends on the noun's gender, number and case, and on whether an article comes first. Here the adjective should read «1980er», not «1980en».

**72.** *(synthetic)* `Meine Eltern sind ja unentwegt umgezogen seit uns Kindern .`
→ `Meine Eltern sind ja unentwegt umgezogen mit uns Kindern .`
  - *R:ADP*: German prepositions are not chosen by translating the English one. This phrase takes «mit» rather than «seit».

**73.** *(learner)* `Ich glaube fast alle sind damit einverstanden .`
→ `Ich glaube , fast alle sind damit einverstanden .`
  - *M:PUNCT*: A punctuation mark is missing here. German punctuates by rule rather than by ear, and this sentence needs «,».

**74.** *(synthetic)* `Wie konnen sich Astronauten künftig praktisch und günstig hinter der erdtrabanten fortbewegen ?`
→ `Wie können sich Astronauten künftig praktisch und günstig auf dem Erdtrabanten fortbewegen ?`
  - *R:SPELL*: The umlaut is part of the letter, not decoration on it. «konnen» and «können» are different words to a German reader. Write «können».
  - *R:ADP*: German prepositions are not chosen by translating the English one. This phrase takes «auf» rather than «hinter».
  - *R:DET:FORM*: The article has to agree with its noun in gender, number and case. «der» does not fit here; the form the sentence needs is «dem».
  - *R:ORTH*: German capitalises every noun, not just names. «erdtrabanten» is a noun here, so it needs a capital letter: «Erdtrabanten». (Rat für deutsche Rechtschreibung, § 55)

**75.** *(synthetic)* `In fast allen Sorten haben wir zum Beispiel den Anteil vor Soße und Stückchen erhöht .`
→ `In fast allen Sorten haben wir zum Beispiel den Anteil an Soße und Stückchen erhöht .`
  - *R:ADP*: German prepositions are not chosen by translating the English one. This phrase takes «an» rather than «vor».

**76.** *(synthetic)* `unternehmen zuschießt beim regulären Tarif von 33 Cent pro Minute etwas für seine Mitarbeiter .`
→ `Das Unternehmen schießt beim regulären Tarif von 33 Cent pro Minute etwas für seine Mitarbeiter zu .`
  - *M:DET*: German nouns almost always need an article, even where English drops it. Add «Das» before the noun.
  - *R:ORTH*: German capitalises every noun, not just names. «unternehmen» is a noun here, so it needs a capital letter: «Unternehmen». (Rat für deutsche Rechtschreibung, § 55)
  - *R:WO*: This verb is separable. In a main clause the prefix detaches and moves to the end, so it is «schießt beim regulären Tarif von 33 Cent pro Minute etwas für seine Mitarbeiter zu» rather than one word.

**77.** *(learner)* `ist Praxiserfahrung wirklich nötig bei aller Studiengänge ?`
→ `" Ist Praxiserfahrung wirklich nötig bei allen Studiengängen ? "`
  - *M:PUNCT*: A punctuation mark is missing here. German punctuates by rule rather than by ear, and this sentence needs «" Ist».
  - *R:DET:FORM*: The article has to match its noun in gender, case and number. Here that makes it «allen», not «aller».
  - *R:NOUN:FORM*: The noun is in the wrong form. German marks case and number on the noun itself, so it should be «Studiengängen».
  - *M:PUNCT*: A punctuation mark is missing here. German punctuates by rule rather than by ear, and this sentence needs «"».

**78.** *(learner)* `Wir hatten Magaret Thatchet , leider , und jetzt sind Politikerin sieht als schlecht .`
→ `Wir hatten Magaret Thatchet , leider , und jetzt sind Politikerinnen als schlecht angesehen .`
  - *R:NOUN:FORM*: The noun is in the wrong form. German marks case and number on the noun itself, so it should be «Politikerinnen».
  - *U:VERB*: The verb «sieht» does not belong here. Remove it.
  - *M:VERB*: A verb is missing here. Add «angesehen».

**79.** *(learner)* `Sie wollten vor allem das Recht auf Ausbildung und Wahlrecht gewinnen .`
→ `Sie wollten vor allem das Recht auf Ausbildung und das Wahlrecht gewinnen .`
  - *M:DET*: A German noun almost always needs an article, even where English drops it. Add «das».

**80.** *(synthetic)* `Neben wechselnder Belastung gält das knoten aber nur begrenzt .`
→ `Unter wechselnder Belastung hält der Knoten aber nur begrenzt .`
  - *R:ADP*: German prepositions are not chosen by translating the English one. This phrase takes «Unter» rather than «Neben».
  - *R:SPELL*: «gält» is not a German word. It looks like a slip for «hält».
  - *R:DET:FORM*: The article has to agree with its noun in gender, number and case. «das» does not fit here; the form the sentence needs is «der».
  - *R:ORTH*: German capitalises every noun, not just names. «knoten» is a noun here, so it needs a capital letter: «Knoten». (Rat für deutsche Rechtschreibung, § 55)

**81.** *(learner)* `Es fing an notwendig zu sein , dass auch die Frauen arbeiten mussten .`
→ `Es fing an , notwendig zu sein , dass auch die Frauen arbeiten mussten .`
  - *M:PUNCT*: A punctuation mark is missing here. German punctuates by rule rather than by ear, and this sentence needs «,».

**82.** *(learner)* `Das heißt , für die Ausländer , die eine Wohnung oder Haus suchen , kann es schwerig sein .`
→ `Das heißt für die Ausländer , die eine Wohnung oder ein Haus suchen , kann es schwerig sein .`
  - *U:PUNCT*: This punctuation mark does not belong here. German does not separate these parts of a sentence.
  - *M:DET*: A German noun almost always needs an article, even where English drops it. Add «ein».

**83.** *(learner)* `Die Einstellungen der meisten Menschen sind von Kindheit an festgestellt ,`
→ `Die Einstellungen der meisten Menschen sind von Kindheit an festgestellt ;`
  - *R:PUNCT*: The wrong punctuation mark is used here. It should be «;».

**84.** *(learner)* `man studiert oberflächlich viele Themenbereiche in eine kürzer Zeit - greift sich nicht mit der material so tief oder analytisch ein . Geringerer Wahlmöglichkeiten in Bezug auf Fachbereiche ,`
→ `Man studiert oberflächlich viele Themenbereiche in einer kurzen Zeit , greift nicht in das Material so tief oder analytisch ein . Geringere Wahlmöglichkeiten in Bezug auf Fachbereiche :`
  - *R:ORTH*: This is a convention of written German rather than a grammar rule. The correct written form here is «Man».
  - *R:DET:FORM*: The article has to match its noun in gender, case and number. Here that makes it «einer», not «eine».
  - *R:ADJ:FORM*: An adjective before a noun takes an ending that depends on the noun's gender, case and number, and on the article in front of it. It should read «kurzen».
  - *R:PUNCT*: The wrong punctuation mark is used here. It should be «,».
  - *U:PRON*: This pronoun is not needed. German does not repeat it here, so remove «sich».
  - *R:ADP*: German prepositions are not chosen by translating the English one. This phrase takes «in» rather than «mit».
  - *R:DET:FORM*: The article has to match its noun in gender, case and number. Here that makes it «das», not «der».
  - *R:ORTH*: This is a convention of written German rather than a grammar rule. The correct written form here is «Material».
  - *R:ADJ:FORM*: An adjective before a noun takes an ending that depends on the noun's gender, case and number, and on the article in front of it. It should read «Geringere».
  - *R:PUNCT*: The wrong punctuation mark is used here. It should be «:».

**85.** *(synthetic)* `rrainer Lohrbach cermisst den Spieldisziplin in seinem team .`
→ `Trainer Lohrbach vermisst die Spieldisziplin in seinem Team .`
  - *R:SPELL*: «rrainer» is not a German word. It looks like a slip for «Trainer».
  - *R:SPELL*: «cermisst» is not a German word. It looks like a slip for «vermisst».
  - *R:DET:FORM*: The article has to agree with its noun in gender, number and case. «den» does not fit here; the form the sentence needs is «die».
  - *R:ORTH*: German capitalises every noun, not just names. «team» is a noun here, so it needs a capital letter: «Team». (Rat für deutsche Rechtschreibung, § 55)

**86.** *(learner)* `101 65185 Wiesbaden Interesse für eine Au-pair-Stelle Sehr geehrte Damen und Herren , ich habe Ihre Anzeige über Vermittlung Au-pair-Stellen gelesen .`
→ `101 65185 Wiesbaden Interesse an einer Au-pair-Stelle Sehr geehrte Damen und Herren , ich habe Ihre Anzeige für die Vermittlung von Au-pair-Stellen gelesen .`
  - *R:ADP*: German prepositions are not chosen by translating the English one. This phrase takes «an» rather than «für».
  - *R:DET:FORM*: The article has to match its noun in gender, case and number. Here that makes it «einer», not «eine».
  - *R:ADP*: German prepositions are not chosen by translating the English one. This phrase takes «für» rather than «über».
  - *M:DET*: A German noun almost always needs an article, even where English drops it. Add «die».
  - *M:ADP*: A preposition is missing. This construction needs «von».

**87.** *(synthetic)* `Dazu gehört natürlich Russland das nicht eingeladen wurde weil es wollte dies gar nicht – das machten russisches Vertreter der Schweiz gegenüber mehrfach klar .`
→ `Dazu gehört natürlich Russland , das nicht eingeladen wurde , weil es dies gar nicht wollte – das machten russische Vertreter der Schweiz gegenüber mehrfach klar .`
  - *M:PUNCT*: German separates a subordinate clause from its main clause with a comma. Unlike English, this is a rule rather than a matter of taste. Add the comma. (Rat für deutsche Rechtschreibung, § 74)
  - *M:PUNCT*: German separates a subordinate clause from its main clause with a comma. Unlike English, this is a rule rather than a matter of taste. Add the comma. (Rat für deutsche Rechtschreibung, § 74)
  - *R:WO*: In a subordinate clause the finite verb goes to the very end, which is not where English puts it. The clause should read «es dies gar nicht wollte».
  - *R:ADJ:FORM*: An adjective before a noun takes an ending that depends on the noun's gender, number and case, and on whether an article comes first. Here the adjective should read «russische», not «russisches».

**88.** *(learner)* `Wir erwarten , dass in diesem Moment , dem wir unseren Grad bekommen , dass wir ganz vorbereitet auf der " wirkliche " Welt sein werden .`
→ `Wir erwarten , dass in diesem Moment , in dem wir unseren Grad bekommen , wir ganz vorbereitet auf der " wirklichen " Welt sein werden .`
  - *M:ADP*: A preposition is missing. This construction needs «in».
  - *U:SCONJ*: The subordinating conjunction «dass» does not belong here. Remove it.
  - *R:VERB:FORM*: The verb is in the wrong form. Check the tense, the person, and whether this clause needs an infinitive or a participle. It should be «wirklichen».

**89.** *(learner)* `Wir haben nach Österrich , Polen , Stadt X gefahren .`
→ `Wir sind nach Österreich , Polen , Stadt X gefahren .`
  - *R:AUX*: «haben» is not the right auxiliary verb here. Use «sind».
  - *R:SPELL*: «Österrich» is not spelled correctly. It should be «Österreich».

**90.** *(learner)* `Ich habe Ihre Agentur genommen um meine Reise zu organisieren , Sie haben mir erzählen , dass ich in eine Familie in Stadt Z wäre .`
→ `Ich habe Ihre Agentur genommen , um meine Reise zu organisieren , Sie haben mir erzählen , dass ich in einer Familie in Stadt Z wäre .`
  - *M:PUNCT*: A punctuation mark is missing here. German punctuates by rule rather than by ear, and this sentence needs «,».
  - *R:DET:FORM*: The article has to match its noun in gender, case and number. Here that makes it «einer», not «eine».

**91.** *(synthetic)* `ayrische stellung sei saraufhin angegriffen und viermal getroffen worden .`
→ `Die syrische Stellung sei daraufhin angegriffen und viermal getroffen worden .`
  - *M:DET*: German nouns almost always need an article, even where English drops it. Add «Die» before the noun.
  - *R:SPELL*: «ayrische» is not a German word. It looks like a slip for «syrische».
  - *R:ORTH*: German capitalises every noun, not just names. «stellung» is a noun here, so it needs a capital letter: «Stellung». (Rat für deutsche Rechtschreibung, § 55)
  - *R:SPELL*: «saraufhin» is not a German word. It looks like a slip for «daraufhin».

**92.** *(synthetic)* `Der Kriegsschiff wurde 1898 bei Schichau in Deutschland gebaut und 1916 unter Hongkong zur verschrottung verkauft .`
→ `Das Kriegsschiff wurde 1898 bei Schichau in Deutschland gebaut und 1916 in Hongkong zur Verschrottung verkauft .`
  - *R:DET:FORM*: The article has to agree with its noun in gender, number and case. «Der» does not fit here; the form the sentence needs is «Das».
  - *R:ADP*: German prepositions are not chosen by translating the English one. This phrase takes «in» rather than «unter».
  - *R:ORTH*: German capitalises every noun, not just names. «verschrottung» is a noun here, so it needs a capital letter: «Verschrottung». (Rat für deutsche Rechtschreibung, § 55)

**93.** *(synthetic)* `qber er allein sei jein heilsbringer und „ er ist quch kein Bayern-Gesicht “ .`
→ `Aber er allein sei kein Heilsbringer und „ er ist auch kein Bayern-Gesicht “ .`
  - *R:SPELL*: «qber» is not a German word. It looks like a slip for «Aber».
  - *R:SPELL*: «jein» is not a German word. It looks like a slip for «kein».
  - *R:ORTH*: German capitalises every noun, not just names. «heilsbringer» is a noun here, so it needs a capital letter: «Heilsbringer». (Rat für deutsche Rechtschreibung, § 55)
  - *R:SPELL*: «quch» is not a German word. It looks like a slip for «auch».

**94.** *(learner)* `Es wird immer Problemen in unserer Gesellschaft , die wir nicht helfen können .`
→ `Es wird immer Probleme in unserer Gesellschaft geben , denen wir nicht helfen können .`
  - *R:NOUN:FORM*: The noun is in the wrong form. German marks case and number on the noun itself, so it should be «Probleme».
  - *M:VERB*: A verb is missing here. Add «geben».
  - *R:PRON:FORM*: The pronoun «die» is in the wrong form. It should be «denen».

**95.** *(learner)* `Wenn man sich so früh auf " Gastfreundschaft " beschränkt , kann es sehr schwer sein , später seinen Beruf zu wechseln , ohne dass er irgendwelche anerkannten Abschlüsse mit sich bringt .`
→ `Wenn man sich so früh auf " Gastfreundschaft " beschränkt , kann es sehr schwer sein , später seinen Beruf zu wechseln , ohne dass man irgendwelche anerkannten Abschlüsse mit sich bringt .`
  - *R:PRON*: «er» is not the right pronoun here. Use «man».

**96.** *(learner)* `Ich wollte doch anderen Beruf haben ! "`
→ `Ich wollte doch einen anderen Beruf haben ! "`
  - *M:DET*: A German noun almost always needs an article, even where English drops it. Add «einen».

**97.** *(synthetic)* `Auf Wien studierte er Germanistik und ohilosophie .`
→ `In Wien studierte er Germanistik und Philosophie .`
  - *R:ADP*: German prepositions are not chosen by translating the English one. This phrase takes «In» rather than «Auf».
  - *R:SPELL*: «ohilosophie» is not a German word. It looks like a slip for «Philosophie».

**98.** *(learner)* `Es wird immer teuerer sich in einer Universität anzumelden , weil die Studentenfees von der Staat fast jedes Jahr erhöht werden .`
→ `Es wird immer teurer , sich an einer Universität anzumelden , weil die Studiengebühren vom Staat fast jedes Jahr erhöht werden .`
  - *R:ADJ:FORM*: An adjective before a noun takes an ending that depends on the noun's gender, case and number, and on the article in front of it. It should read «teurer».
  - *M:PUNCT*: A punctuation mark is missing here. German punctuates by rule rather than by ear, and this sentence needs «,».
  - *R:ADP*: German prepositions are not chosen by translating the English one. This phrase takes «an» rather than «in».
  - *R:SPELL*: «Studentenfees» is not spelled correctly. It should be «Studiengebühren».
  - *R:ADP*: German prepositions are not chosen by translating the English one. This phrase takes «vom» rather than «von».
  - *U:DET*: The article «der» does not belong here. Remove it.

**99.** *(synthetic)* `Elektrische Geräte wie Pumpen , Steuerungen Dosierungsanlagen , Beleuchtung usw. sollten immer von einem Fachmann installiert werden .`
→ `Elektrische Geräte wie Pumpen , Steuerungen , Dosierungsanlagen , Beleuchtung usw. sollten immer von einem Fachmann installiert werden .`
  - *M:PUNCT*: A comma belongs here. German uses them to mark off parts of a sentence -- an apposition, an insertion, a list -- more strictly than English does.

**100.** *(synthetic)* `Sie zeigen oft eine ruhige persönlichkeit , weshalb sie oft auf den wrsten Blick ziemlich distanziert wirken .`
→ `Sie zeigen oft eine ruhige Persönlichkeit , weshalb sie oft auf den ersten Blick ziemlich distanziert wirken .`
  - *R:ORTH*: German capitalises every noun, not just names. «persönlichkeit» is a noun here, so it needs a capital letter: «Persönlichkeit». (Rat für deutsche Rechtschreibung, § 55)
  - *R:SPELL*: «wrsten» is not a German word. It looks like a slip for «ersten».

