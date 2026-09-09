# Phase 2: the training set

30,000 examples: 23,410 carrying errors and 6,590 already correct (22.0%).

## Invariants

| Check | Result |
|---|---|
| Damaged examples whose source equals their target | 0 |
| Edits with no explanation | 0 |
| Edits whose corruptor could not be identified | 0 |
| Sentences no rule could damage (kept as correct) | 0 |

The first two must be zero. A damaged example identical to its target teaches the model to change nothing when something is wrong; an edit with no explanation is the half of this project that is not a grammar checker.

## Injected mix

| Error type | Count | Share |
|---|---:|---:|
| `R:SPELL` | 8,965 | 25.6% |
| `R:DET:FORM` | 5,588 | 15.9% |
| `R:ORTH` | 4,902 | 14.0% |
| `M:PUNCT` | 4,614 | 13.2% |
| `R:ADP` | 3,526 | 10.1% |
| `M:DET` | 3,239 | 9.2% |
| `R:ADJ:FORM` | 2,277 | 6.5% |
| `R:WO` | 1,951 | 5.6% |

## 100 samples

The Phase 2 exit criterion: read these, and if more than five have a wrong correction or a wrong explanation, fix the pipeline before training on it.

**1.** `Im Einzel über 18 km fuhr er als 27. der Weltspitze hinterher zeigte aber in der Kombination sein können und wurde am Ende Sechster .`
→ `Im Einzel über 18 km fuhr er als 27. der Weltspitze hinterher , zeigte aber in der Kombination sein Können und wurde am Ende Sechster .`
  - *M:PUNCT* (drop_comma): A comma belongs here. German uses them to mark off parts of a sentence -- an apposition, an insertion, a list -- more strictly than English does.
  - *R:ORTH* (noun_case): German capitalises every noun, not just names. «können» is a noun here, so it needs a capital letter: «Können». (Rat für deutsche Rechtschreibung, § 55)

**2.** `Teich ist die Quelle des Houet-Flusses .`
→ `Der Teich ist die Quelle des Houet-Flusses .`
  - *M:DET* (drop_article): German nouns almost always need an article, even where English drops it. Add «Der» before the noun.

**3.** `Das Amt des bundeskanzler scheint einen sehr geschützter Arbeitsplatz zu sein .`
→ `Das Amt des Bundeskanzler scheint ein sehr geschützter Arbeitsplatz zu sein .`
  - *R:ORTH* (noun_case): German capitalises every noun, not just names. «bundeskanzler» is a noun here, so it needs a capital letter: «Bundeskanzler». (Rat für deutsche Rechtschreibung, § 55)
  - *R:DET:FORM* (article_form): The article has to agree with its noun in gender, number and case. «einen» does not fit here; the form the sentence needs is «ein».

**4.** `Seine umfangreiche Wagner-Sammlung machte er teilweise 1886 in ein ständigen qusstellung zugänglich .`
→ `Seine umfangreiche Wagner-Sammlung machte er teilweise 1886 in einer ständigen Ausstellung zugänglich .`
  - *R:DET:FORM* (article_form): The article has to agree with its noun in gender, number and case. «ein» does not fit here; the form the sentence needs is «einer».
  - *R:SPELL* (keyboard): «qusstellung» is not a German word. It looks like a slip for «Ausstellung».

**5.** `Über weitere bis mit 72 Flugzeuge soll separat verhandelt werden .`
→ `Über weitere , bis zu 72 Flugzeuge soll separat verhandelt werden .`
  - *M:PUNCT* (drop_comma_subordinate): German separates a subordinate clause from its main clause with a comma. Unlike English, this is a rule rather than a matter of taste. Add the comma. (Rat für deutsche Rechtschreibung, § 74)
  - *R:ADP* (preposition): German prepositions are not chosen by translating the English one. This phrase takes «zu» rather than «mit».

**6.** `gleiche Besoldungsgruppe führt also je nach Dienstherr und individueller Aushandlung nicht zu gleichen Bezügen sondern kann um bis zu 2000 Euro unterschiedlich sein .`
→ `Die gleiche Besoldungsgruppe führt also je nach Dienstherr und individueller Aushandlung nicht zu gleichen Bezügen , sondern kann um bis zu 2000 Euro unterschiedlich sein .`
  - *M:DET* (drop_article): German nouns almost always need an article, even where English drops it. Add «Die» before the noun.
  - *M:PUNCT* (drop_comma_coordinating): German puts a comma before «sondern», «aber», «doch» and «jedoch» when they join two clauses. English usually does not, which is why this comma is easy to miss.

**7.** `Er ist ein hervorragender Oberbürgermeister das soll er machen und dass andere weglassen .`
→ `Er ist ein hervorragender Oberbürgermeister , das soll er machen und das andere weglassen .`
  - *M:PUNCT* (drop_comma_subordinate): German separates a subordinate clause from its main clause with a comma. Unlike English, this is a rule rather than a matter of taste. Add the comma. (Rat für deutsche Rechtschreibung, § 74)
  - *R:SPELL* (das_dass): «das» and «dass» sound the same and do different jobs. «das» is an article or a relative pronoun and points at a thing; «dass» is a conjunction that opens a subordinate clause. Here the sentence needs «das».

**8.** `Des Prognosen zum Beitragsanstieg im nächsten Jahrzehnt gehen davon aus dass sich bis dahin nichts mehr ändert .`
→ `Die Prognosen zum Beitragsanstieg im nächsten Jahrzehnt gehen davon aus , dass sich bis dahin nichts mehr ändert .`
  - *R:DET:FORM* (article_form): The article has to agree with its noun in gender, number and case. «Des» does not fit here; the form the sentence needs is «Die».
  - *M:PUNCT* (drop_comma_subordinate): German separates a subordinate clause from its main clause with a comma. Unlike English, this is a rule rather than a matter of taste. Add the comma. (Rat für deutsche Rechtschreibung, § 74)

**9.** `Aber anstatt den Radweg zu erneuern oder zu verlegen und eine dahrspur für Autos zu verlieren , will sie die ganze Radroute vom Stemmerhof bis zum Sendlinger Tor auf eine andere Strecke versetzen .`
→ `Aber anstatt den Radweg zu erneuern oder zu verlegen und eine Fahrspur für Autos zu verlieren , will sie die ganze Radroute vom Stemmerhof bis zum Sendlinger Tor auf eine andere Strecke versetzen .`
  - *R:SPELL* (keyboard): «dahrspur» is not a German word. It looks like a slip for «Fahrspur».

**10.** `Yamaha und Honda bewährten sich ebenso wie letztlich alles was sich günstig kaufen und interessant umgestalten ließ .`
→ `Yamaha und Honda bewährten sich ebenso wie letztlich alles , was sich günstig kaufen und interessant umgestalten ließ .`
  - *M:PUNCT* (drop_comma): A comma belongs here. German uses them to mark off parts of a sentence -- an apposition, an insertion, a list -- more strictly than English does.

**11.** `Durch den genialen Einfall Wintergreens wird nun ein Baseballspiel zwischen den amerikanischen Blauhemden und dem Rest der Welt ausgetragen , um die Kriegsschulden erneut aus verhandeln .`
→ `Durch den genialen Einfall Wintergreens wird nun ein Baseballspiel zwischen den amerikanischen Blauhemden und dem Rest der Welt ausgetragen , um die Kriegsschulden erneut zu verhandeln .`
  - *R:ADP* (preposition): German prepositions are not chosen by translating the English one. This phrase takes «zu» rather than «aus».

**12.** `Trotzdem wird das Ergebnis für das Gesamtjahr 2024 vor dem Vorjahr mit liegen kommen .`
→ `Trotzdem wird das Ergebnis für das Gesamtjahr 2024 unter dem Vorjahr zu liegen kommen .`
  - *R:ADP* (preposition): German prepositions are not chosen by translating the English one. This phrase takes «unter» rather than «vor».
  - *R:ADP* (preposition): German prepositions are not chosen by translating the English one. This phrase takes «zu» rather than «mit».

**13.** `Daneben gilt Dievs auch als Bierbrauer und gelegentlich als jäger .`
→ `Daneben gilt Dievs auch als Bierbrauer und gelegentlich als Jäger .`
  - *R:ORTH* (noun_case): German capitalises every noun, not just names. «jäger» is a noun here, so it needs a capital letter: «Jäger». (Rat für deutsche Rechtschreibung, § 55)

**14.** `Er hat den Besuch in Saarbrücken genutzt , um einen kleinen Abstecher am Reppersberg zu machen , da wo wohnte seine Familie früher .`
→ `Er hat den Besuch in Saarbrücken genutzt , um einen kleinen Abstecher am Reppersberg zu machen , da wo seine Familie früher wohnte .`
  - *R:WO* (verb_final): In a subordinate clause the finite verb goes to the very end, which is not where English puts it. The clause should read «wo seine Familie früher wohnte».

**15.** `Vater drohen bis 10 Jahre Haft .`
→ `Dem Vater drohen bis 10 Jahre Haft .`
  - *M:DET* (drop_article): German nouns almost always need an article, even where English drops it. Add «Dem» before the noun.

**16.** `Einer spontaner Entschluss machte aus Roman und Vanessa Lindemann das neue Delbrücker Schützenkönigspaar .`
→ `Ein spontaner Entschluss machte aus Roman und Vanessa Lindemann das neue Delbrücker Schützenkönigspaar .`
  - *R:DET:FORM* (article_form): The article has to agree with its noun in gender, number and case. «Einer» does not fit here; the form the sentence needs is «Ein».

**17.** `Zwischen seiner Arbeit als journalist war Ulloriannguaq Kristiansen auch als Schriftsteller , Übersetzer und Redakteur tätig .`
→ `Neben seiner Arbeit als Journalist war Ulloriannguaq Kristiansen auch als Schriftsteller , Übersetzer und Redakteur tätig .`
  - *R:ADP* (preposition): German prepositions are not chosen by translating the English one. This phrase takes «Neben» rather than «Zwischen».
  - *R:ORTH* (noun_case): German capitalises every noun, not just names. «journalist» is a noun here, so it needs a capital letter: «Journalist». (Rat für deutsche Rechtschreibung, § 55)

**18.** `Viele dem Einwohner engagieren sich bei der Freiwilligen Feuerwehr , den Eisstockschützen dem Lederhosenstammtisch oder dem Frauentreff .`
→ `Viele der Einwohner engagieren sich bei der Freiwilligen Feuerwehr , den Eisstockschützen , dem Lederhosenstammtisch oder dem Frauentreff .`
  - *R:DET:FORM* (article_form): The article has to agree with its noun in gender, number and case. «dem» does not fit here; the form the sentence needs is «der».
  - *M:PUNCT* (drop_comma_subordinate): German separates a subordinate clause from its main clause with a comma. Unlike English, this is a rule rather than a matter of taste. Add the comma. (Rat für deutsche Rechtschreibung, § 74)

**19.** `Hier qbsolvierte er 25 Einsätze und traf zweimal für Nîmes .`
→ `Hier absolvierte er 25 Einsätze und traf zweimal für Nîmes .`
  - *R:SPELL* (keyboard): «qbsolvierte» is not a German word. It looks like a slip for «absolvierte».

**20.** `Die Fahrten der Tochtergesellschaft LOOK , die nicht bestreikt wird werden ebenso wenig betroffen sein wie die Touren der von Niag und LOOK beauftragten Unternehmen , heißt es .`
→ `Die Fahrten der Tochtergesellschaft LOOK , die nicht bestreikt wird , werden ebenso wenig betroffen sein wie die Touren der von Niag und LOOK beauftragten Unternehmen , heißt es .`
  - *M:PUNCT* (drop_comma): A comma belongs here. German uses them to mark off parts of a sentence -- an apposition, an insertion, a list -- more strictly than English does.

**21.** `Von 19,99 Euro deutlich günstiger , aber ähnlich gut ( 1,4 ) , ist als Download-Programm auch " Tax " vom selben anbieter .`
→ `Mit 19,99 Euro deutlich günstiger , aber ähnlich gut ( 1,4 ) , ist als Download-Programm auch " Tax " vom selben Anbieter .`
  - *R:ADP* (preposition): German prepositions are not chosen by translating the English one. This phrase takes «Mit» rather than «Von».
  - *R:ORTH* (noun_case): German capitalises every noun, not just names. «anbieter» is a noun here, so it needs a capital letter: «Anbieter». (Rat für deutsche Rechtschreibung, § 55)

**22.** `Da steckt also boch mehr drin in Mannschaft .`
→ `Da steckt also noch mehr drin in der Mannschaft .`
  - *R:SPELL* (keyboard): «boch» is not a German word. It looks like a slip for «noch».
  - *M:DET* (drop_article): German nouns almost always need an article, even where English drops it. Add «der» before the noun.

**23.** `Die Schweiz beendete die Turnier auf Rang sechs während Österreich im Gesamtklassement Platz 13 belegte .`
→ `Die Schweiz beendete das Turnier auf Rang sechs , während Österreich im Gesamtklassement Platz 13 belegte .`
  - *R:DET:FORM* (article_form): The article has to agree with its noun in gender, number and case. «die» does not fit here; the form the sentence needs is «das».
  - *M:PUNCT* (drop_comma_subordinate): German separates a subordinate clause from its main clause with a comma. Unlike English, this is a rule rather than a matter of taste. Add the comma. (Rat für deutsche Rechtschreibung, § 74)

**24.** `Den „ Killer“-Komponente flogen F-4D / E Phantoms und griffen der markierten SAM-Stellungen mit konventionellen Bomben an , um Startgeräte und Raketen zu zerstören .`
→ `Die „ Killer“-Komponente flogen F-4D / E Phantoms und griffen die markierten SAM-Stellungen mit konventionellen Bomben an , um Startgeräte und Raketen zu zerstören .`
  - *R:DET:FORM* (article_form): The article has to agree with its noun in gender, number and case. «Den» does not fit here; the form the sentence needs is «Die».
  - *R:DET:FORM* (article_form): The article has to agree with its noun in gender, number and case. «der» does not fit here; the form the sentence needs is «die».

**25.** `Phillip und Julian Becker starten ebenso am Sonntag an Frankfurt wie Mike und Sven Theis .`
→ `Phillip und Julian Becker starten ebenso am Sonntag in Frankfurt wie Mike und Sven Theis .`
  - *R:ADP* (preposition): German prepositions are not chosen by translating the English one. This phrase takes «in» rather than «an».

**26.** `Diese Vereinigung strebte erfolglos danach , der Ausdehnung der Rechte Katholiken einzudämmen und sie vom öffentlichen Leben in der Provinz auszuschließen .`
→ `Diese Vereinigung strebte erfolglos danach , die Ausdehnung der Rechte der Katholiken einzudämmen und sie vom öffentlichen Leben in der Provinz auszuschließen .`
  - *R:DET:FORM* (article_form): The article has to agree with its noun in gender, number and case. «der» does not fit here; the form the sentence needs is «die».
  - *M:DET* (drop_article): German nouns almost always need an article, even where English drops it. Add «der» before the noun.

**27.** `Anschließend begann er hier zu unterrichten , ab 1790 in der Funktion eines Professors , nachdem er 1789 zum Doktor des Theologie promoviert worden war .`
→ `Anschließend begann er hier zu unterrichten , ab 1790 in der Funktion eines Professors , nachdem er 1789 zum Doktor der Theologie promoviert worden war .`
  - *R:DET:FORM* (article_form): The article has to agree with its noun in gender, number and case. «des» does not fit here; the form the sentence needs is «der».

**28.** `Bis heute ist Gersch als Übungsleiter aktiv und als Prufer für das Deutsche Sportabzeichen .`
→ `Bis heute ist Gersch als Übungsleiter aktiv und als Prüfer für das Deutsche Sportabzeichen .`
  - *R:SPELL* (umlaut): The umlaut is part of the letter, not decoration on it. «Prufer» and «Prüfer» are different words to a German reader. Write «Prüfer».

**29.** `aeit Sommer 2010 ist er um Verein .`
→ `Seit Sommer 2010 ist er ohne Verein .`
  - *R:SPELL* (keyboard): «aeit» is not a German word. It looks like a slip for «Seit».
  - *R:ADP* (preposition): German prepositions are not chosen by translating the English one. This phrase takes «ohne» rather than «um».

**30.** `Vermutet wurde ein Blindgänger unter einer Einfamilienhaus .`
→ `Vermutet wurde ein Blindgänger unter einem Einfamilienhaus .`
  - *R:DET:FORM* (article_form): The article has to agree with its noun in gender, number and case. «einer» does not fit here; the form the sentence needs is «einem».

**31.** `Heinz kommt zum Schein der leidenschaftlichen helga nahe und verschwindet mit ihr zum Strand .`
→ `Heinz kommt zum Schein der leidenschaftlichen Helga nahe und verschwindet mit ihr zum Strand .`
  - *R:ORTH* (noun_case): German capitalises every noun, not just names. «helga» is a noun here, so it needs a capital letter: «Helga». (Rat für deutsche Rechtschreibung, § 55)

**32.** `Daraufhin hat der japanische aportartikelhersteller Kleidung entwickelt , die einen besseren Schutz gegen Infrarotfotografie bieten soll .`
→ `Daraufhin hat der japanische Sportartikelhersteller Kleidung entwickelt , die einen besseren Schutz gegen Infrarotfotografie bieten soll .`
  - *R:SPELL* (keyboard): «aportartikelhersteller» is not a German word. It looks like a slip for «Sportartikelhersteller».

**33.** `Wittmann rät sich vor Augen zu führen , dass KI den Job in die allermeisten Fällen nicht vollständig übernehmen wird .`
→ `Wittmann rät , sich vor Augen zu führen , dass KI den Job in den allermeisten Fällen nicht vollständig übernehmen wird .`
  - *M:PUNCT* (drop_comma): A comma belongs here. German uses them to mark off parts of a sentence -- an apposition, an insertion, a list -- more strictly than English does.
  - *R:DET:FORM* (article_form): The article has to agree with its noun in gender, number and case. «die» does not fit here; the form the sentence needs is «den».

**34.** `Kurz darauf gab es sogar einen weiteren Unfall von zwei Autos .`
→ `Kurz darauf gab es sogar einen weiteren Unfall mit zwei Autos .`
  - *R:ADP* (preposition): German prepositions are not chosen by translating the English one. This phrase takes «mit» rather than «von».

**35.** `An der Schauspielerei verfolgt Pacar auch Musikkarriere .`
→ `Neben der Schauspielerei verfolgt Pacar auch eine Musikkarriere .`
  - *R:ADP* (preposition): German prepositions are not chosen by translating the English one. This phrase takes «Neben» rather than «An».
  - *M:DET* (drop_article): German nouns almost always need an article, even where English drops it. Add «eine» before the noun.

**36.** `Der Bund der Steuerzahler sprach von besorgniserregenden Versäumnissen beim Erhalt des öffentliche Infrastruktur .`
→ `Der Bund der Steuerzahler sprach von besorgniserregenden Versäumnissen beim Erhalt der öffentlichen Infrastruktur .`
  - *R:DET:FORM* (article_form): The article has to agree with its noun in gender, number and case. «des» does not fit here; the form the sentence needs is «der».
  - *R:ADJ:FORM* (adjective_form): An adjective before a noun takes an ending that depends on the noun's gender, number and case, and on whether an article comes first. Here the adjective should read «öffentlichen», not «öffentliche».

**37.** `Nach winer Diskussion mit dem Personal wurden sie jedoch abgewiesen .`
→ `Nach einer Diskussion mit dem Personal wurden sie jedoch abgewiesen .`
  - *R:SPELL* (keyboard): «winer» is not a German word. It looks like a slip for «einer».

**38.** `Im september 2009 wurde den Landesregierung umgebildet .`
→ `Im September 2009 wurde die Landesregierung umgebildet .`
  - *R:ORTH* (noun_case): German capitalises every noun, not just names. «september» is a noun here, so it needs a capital letter: «September». (Rat für deutsche Rechtschreibung, § 55)
  - *R:DET:FORM* (article_form): The article has to agree with its noun in gender, number and case. «den» does not fit here; the form the sentence needs is «die».

**39.** `Das Schiff das den Heiligen Petrus und Paulus geweihten Kirche in Zaessingue wurde 1763 gebaut .`
→ `Das Schiff der den Heiligen Petrus und Paulus geweihten Kirche in Zaessingue wurde 1763 gebaut .`
  - *R:DET:FORM* (article_form): The article has to agree with its noun in gender, number and case. «das» does not fit here; the form the sentence needs is «der».

**40.** `Es beteiligten sich jedoch unter 20 Prozent der Wahlberechtigten an der Wahl und das inoffizielle Referendum hatte jeine Auswirkungen .`
→ `Es beteiligten sich jedoch unter 20 Prozent der Wahlberechtigten an der Wahl und das inoffizielle Referendum hatte keine Auswirkungen .`
  - *R:SPELL* (keyboard): «jeine» is not a German word. It looks like a slip for «keine».

**41.** `Der wirtschaftliche Überblick des Klimawandels wird bei neuen Angeboten für die Einführung einem „ johlenstoffhaushalts “ ebenso berücksichtigt .`
→ `Der wirtschaftliche Überblick des Klimawandels wird bei neuen Angeboten für die Einführung eines „ Kohlenstoffhaushalts “ ebenso berücksichtigt .`
  - *R:DET:FORM* (article_form): The article has to agree with its noun in gender, number and case. «einem» does not fit here; the form the sentence needs is «eines».
  - *R:SPELL* (keyboard): «johlenstoffhaushalts» is not a German word. It looks like a slip for «Kohlenstoffhaushalts».

**42.** `Im Rahmen des neuen wirtschaftspolitischer Aktionsplans wird das Sándor-Demján-Programm auf den Weg gebracht .`
→ `Im Rahmen des neuen wirtschaftspolitischen Aktionsplans wird das Sándor-Demján-Programm auf den Weg gebracht .`
  - *R:ADJ:FORM* (adjective_form): An adjective before a noun takes an ending that depends on the noun's gender, number and case, and on whether an article comes first. Here the adjective should read «wirtschaftspolitischen», not «wirtschaftspolitischer».

**43.** `Am 9. Januar 2015 erschien das Video seit dem Lied Ich mach nicht mehr mit ungefähr einen Monat später erschien das Video für Dann ohne mich .`
→ `Am 9. Januar 2015 erschien das Video zu dem Lied Ich mach nicht mehr mit , ungefähr einen Monat später erschien das Video für Dann ohne mich .`
  - *R:ADP* (preposition): German prepositions are not chosen by translating the English one. This phrase takes «zu» rather than «seit».
  - *M:PUNCT* (drop_comma): A comma belongs here. German uses them to mark off parts of a sentence -- an apposition, an insertion, a list -- more strictly than English does.

**44.** `Gefeiert wurde auch bei Drews ‘ ehemaligen Kollegen : dlorian Silbereisens ( 43 ) „ Immer wieder sonntags“-Überraschung für Stefan Mross ( 48 ) kam bei den Zuschauern bestens an .`
→ `Gefeiert wurde auch bei Drews ‘ ehemaligen Kollegen : Florian Silbereisens ( 43 ) „ Immer wieder sonntags“-Überraschung für Stefan Mross ( 48 ) kam bei den Zuschauern bestens an .`
  - *R:SPELL* (keyboard): «dlorian» is not a German word. It looks like a slip for «Florian».

**45.** `Weder die geistlichen noch die theologischen Vorgesetzten hatten ein Interesse an der Ausbildung des Schulwesens , weil sie sahen die Notwendigkeit nicht .`
→ `Weder die geistlichen noch die theologischen Vorgesetzten hatten ein Interesse an der Ausbildung des Schulwesens , weil sie die Notwendigkeit nicht sahen .`
  - *R:WO* (verb_final): In a subordinate clause the finite verb goes to the very end, which is not where English puts it. The clause should read «sie die Notwendigkeit nicht sahen».

**46.** `Die Lamarr-Baustelle ist erst zu 30 bis 40 prozent fertiggestellt .`
→ `Die Lamarr-Baustelle ist erst zu 30 bis 40 Prozent fertiggestellt .`
  - *R:ORTH* (noun_case): German capitalises every noun, not just names. «prozent» is a noun here, so it needs a capital letter: «Prozent». (Rat für deutsche Rechtschreibung, § 55)

**47.** `Dabei darstellen die ersten zwei Buchstaben das ursprüngliche Länderkürzel aus ISO 3166 - 1 und die zwei letzten buchstaben das neue Länderkürzel aus ISO 3166 - 1.`
→ `Dabei stellen die ersten zwei Buchstaben das ursprüngliche Länderkürzel aus ISO 3166 - 1 dar und die zwei letzten Buchstaben das neue Länderkürzel aus ISO 3166 - 1.`
  - *R:WO* (separable_prefix): This verb is separable. In a main clause the prefix detaches and moves to the end, so it is «stellen die ersten zwei Buchstaben das ursprüngliche Länderkürzel aus ISO 3166 - 1 dar» rather than one word.
  - *R:ORTH* (noun_case): German capitalises every noun, not just names. «buchstaben» is a noun here, so it needs a capital letter: «Buchstaben». (Rat für deutsche Rechtschreibung, § 55)

**48.** `Deren Verwaltungssitz , Dorf Walki liegt gut sechs Kilometer nordwestlich von Makarjewo .`
→ `Deren Verwaltungssitz , das Dorf Walki , liegt gut sechs Kilometer nordwestlich von Makarjewo .`
  - *M:DET* (drop_article): German nouns almost always need an article, even where English drops it. Add «das» before the noun.
  - *M:PUNCT* (drop_comma): A comma belongs here. German uses them to mark off parts of a sentence -- an apposition, an insertion, a list -- more strictly than English does.

**49.** `Großbritanniens Premierminister Rishi Sunak hatte das Verbot zuvor in eines Videobotschaft angekündigt .`
→ `Großbritanniens Premierminister Rishi Sunak hatte das Verbot zuvor in einer Videobotschaft angekündigt .`
  - *R:DET:FORM* (article_form): The article has to agree with its noun in gender, number and case. «eines» does not fit here; the form the sentence needs is «einer».

**50.** `Er war Vorstand die Abteilung für Angewandte aystemforschung und Statistik .`
→ `Er war Vorstand der Abteilung für Angewandte Systemforschung und Statistik .`
  - *R:DET:FORM* (article_form): The article has to agree with its noun in gender, number and case. «die» does not fit here; the form the sentence needs is «der».
  - *R:SPELL* (keyboard): «aystemforschung» is not a German word. It looks like a slip for «Systemforschung».

**51.** `Herbert Grönemeyer auftritt am Wochenende in Karlsruhe .`
→ `Herbert Grönemeyer tritt am Wochenende in Karlsruhe auf .`
  - *R:WO* (separable_prefix): This verb is separable. In a main clause the prefix detaches and moves to the end, so it is «tritt am Wochenende in Karlsruhe auf» rather than one word.

**52.** `Zeitweise existierte eine weinmarke namens Mörzheimer lehmgrube .`
→ `Zeitweise existierte eine Weinmarke namens Mörzheimer Lehmgrube .`
  - *R:ORTH* (noun_case): German capitalises every noun, not just names. «weinmarke» is a noun here, so it needs a capital letter: «Weinmarke». (Rat für deutsche Rechtschreibung, § 55)
  - *R:ORTH* (noun_case): German capitalises every noun, not just names. «lehmgrube» is a noun here, so it needs a capital letter: «Lehmgrube». (Rat für deutsche Rechtschreibung, § 55)

**53.** `Der Suco liegt im Nordwesten den Verwaltungsamts Lospalos .`
→ `Der Suco liegt im Nordwesten des Verwaltungsamts Lospalos .`
  - *R:DET:FORM* (article_form): The article has to agree with its noun in gender, number and case. «den» does not fit here; the form the sentence needs is «des».

**54.** `Mit Wandern und Joggen in voller Feuerwehrmontur übten die Teams mindestens winmal pro Woche .`
→ `Mit Wandern und Joggen in voller Feuerwehrmontur übten die Teams mindestens einmal pro Woche .`
  - *R:SPELL* (keyboard): «winmal» is not a German word. It looks like a slip for «einmal».

**55.** `Eifrig sind Regierungen denn quch daran , Förderprogramme aufzugleisen , um ihre Wettbewerbsfähigkeit aufrechtzuerhalten .`
→ `Eifrig sind die Regierungen denn auch daran , Förderprogramme aufzugleisen , um ihre Wettbewerbsfähigkeit aufrechtzuerhalten .`
  - *M:DET* (drop_article): German nouns almost always need an article, even where English drops it. Add «die» before the noun.
  - *R:SPELL* (keyboard): «quch» is not a German word. It looks like a slip for «auch».

**56.** `Dort erreichte Vaněk vor seiner zweiten Saison seinen ersten froßerfolg als Trainer .`
→ `Dort erreichte Vaněk in seiner zweiten Saison seinen ersten Großerfolg als Trainer .`
  - *R:ADP* (preposition): German prepositions are not chosen by translating the English one. This phrase takes «in» rather than «vor».
  - *R:SPELL* (keyboard): «froßerfolg» is not a German word. It looks like a slip for «Großerfolg».

**57.** `Nicole Hayes war die erste Schwimmerin aus Palau , das bei Olympischen Spielen startete .`
→ `Nicole Hayes war die erste Schwimmerin aus Palau , die bei Olympischen Spielen startete .`
  - *R:DET:FORM* (article_form): The article has to agree with its noun in gender, number and case. «das» does not fit here; the form the sentence needs is «die».

**58.** `Ein großer Teil der Arbeitslosen erhält Leistungen nach dem Sozialgesetzbuch II ( Hartz IV ) im Juni 2006 waren es im Gebiet des Wahlkreises 3,1 Prozent der Bevölkerung in Hamburg durchschnittlich 5,7 Prozent .`
→ `Ein großer Teil der Arbeitslosen erhält Leistungen nach dem Sozialgesetzbuch II ( Hartz IV ) , im Juni 2006 waren es im Gebiet des Wahlkreises 3,1 Prozent der Bevölkerung , in Hamburg durchschnittlich 5,7 Prozent .`
  - *M:PUNCT* (drop_comma): A comma belongs here. German uses them to mark off parts of a sentence -- an apposition, an insertion, a list -- more strictly than English does.
  - *M:PUNCT* (drop_comma): A comma belongs here. German uses them to mark off parts of a sentence -- an apposition, an insertion, a list -- more strictly than English does.

**59.** `Truppen das Nordteil Insel und die Zivilverwaltung wurde suspendiert .`
→ `Truppen den Nordteil der Insel und die Zivilverwaltung wurde suspendiert .`
  - *R:DET:FORM* (article_form): The article has to agree with its noun in gender, number and case. «das» does not fit here; the form the sentence needs is «den».
  - *M:DET* (drop_article): German nouns almost always need an article, even where English drops it. Add «der» before the noun.

**60.** `TV-Programm heute : Das läuft gegen 20.15 Uhr am Sonntag , 7.7.`
→ `TV-Programm heute : Das läuft um 20.15 Uhr am Sonntag , 7.7.`
  - *R:ADP* (preposition): German prepositions are not chosen by translating the English one. This phrase takes «um» rather than «gegen».

**61.** `qiederholt dann die Schritte und benutzt den Hammer ganz und gar ohne Feind .`
→ `Wiederholt dann die Schritte und benutzt den Hammer ganz und gar ohne Feind .`
  - *R:SPELL* (keyboard): «qiederholt» is not a German word. It looks like a slip for «Wiederholt».

**62.** `Im jahre 1930 hatte dugau 791 Einwohner .`
→ `Im Jahre 1930 hatte Fugau 791 Einwohner .`
  - *R:ORTH* (noun_case): German capitalises every noun, not just names. «jahre» is a noun here, so it needs a capital letter: «Jahre». (Rat für deutsche Rechtschreibung, § 55)
  - *R:SPELL* (keyboard): «dugau» is not a German word. It looks like a slip for «Fugau».

**63.** `Die Renditeaufschläge französische Staatsanleihen gegenüber zehnjährigen deutschen Staatsanleihen waren jüngst zwar erstmals höher als diejenigen griechischer Staatsanleihen und fast so hoch wie die italienischer .`
→ `Die Renditeaufschläge französischer Staatsanleihen gegenüber zehnjährigen deutschen Staatsanleihen waren jüngst zwar erstmals höher als diejenigen griechischer Staatsanleihen und fast so hoch wie die italienischer .`
  - *R:ADJ:FORM* (adjective_form): An adjective before a noun takes an ending that depends on the noun's gender, number and case, and on whether an article comes first. Here the adjective should read «französischer», not «französische».

**64.** `Es wurde im Oktober 1980 im Gebäude alten Schule Elbgaustraße gegründet .`
→ `Es wurde im Oktober 1980 im Gebäude der alten Schule Elbgaustraße gegründet .`
  - *M:DET* (drop_article): German nouns almost always need an article, even where English drops it. Add «der» before the noun.

**65.** `Tochter Sarafina hat sogar der Marke von 609.000 Followern geknackt .`
→ `Tochter Sarafina hat sogar die Marke von 609.000 Followern geknackt .`
  - *R:DET:FORM* (article_form): The article has to agree with its noun in gender, number and case. «der» does not fit here; the form the sentence needs is «die».

**66.** `Sein drittes Modell , 910 - 3 ° , war erste , das in eine kleine Serie ging .`
→ `Sein drittes Modell , 910 - 3 ° , war das erste , das in eine kleine Serie ging .`
  - *M:DET* (drop_article): German nouns almost always need an article, even where English drops it. Add «das» before the noun.

**67.** `Das den Vorgarten umgrenzende Zaun stellt noch die originale Einfriedung dar .`
→ `Der den Vorgarten umgrenzende Zaun stellt noch die originale Einfriedung dar .`
  - *R:DET:FORM* (article_form): The article has to agree with its noun in gender, number and case. «Das» does not fit here; the form the sentence needs is «Der».

**68.** `Heute sprechen die junge Leute von „ Donuts “ , wenn ihnen anspruchsvollen Bremsspuren in Kreisform gelungen sind .`
→ `Heute sprechen die jungen Leute von „ Donuts “ , wenn ihnen anspruchsvolle Bremsspuren in Kreisform gelungen sind .`
  - *R:ADJ:FORM* (adjective_form): An adjective before a noun takes an ending that depends on the noun's gender, number and case, and on whether an article comes first. Here the adjective should read «jungen», not «junge».
  - *R:ADJ:FORM* (adjective_form): An adjective before a noun takes an ending that depends on the noun's gender, number and case, and on whether an article comes first. Here the adjective should read «anspruchsvolle», not «anspruchsvollen».

**69.** `Larbre als bestes Viper-Team beendete der Meisterschaft auf den siebten Platz .`
→ `Larbre als bestes Viper-Team beendete die Meisterschaft auf dem siebten Platz .`
  - *R:DET:FORM* (article_form): The article has to agree with its noun in gender, number and case. «der» does not fit here; the form the sentence needs is «die».
  - *R:DET:FORM* (two_way_case): This preposition takes the accusative for movement towards somewhere and the dative for position. Nothing is moving here, so it is «dem» rather than «den».

**70.** `Bei der klassischen Konditionierung wird ein neutraler Reiz mit einem anderen Reiz zeitgleich dargeboten , der bereits eine bestimmte Reaktion auslöst , bis der neutrale Reiz ebenfalls die Reaktion des anderen eeizes auslöst .`
→ `Bei der klassischen Konditionierung wird ein neutraler Reiz mit einem anderen Reiz zeitgleich dargeboten , der bereits eine bestimmte Reaktion auslöst , bis der neutrale Reiz ebenfalls die Reaktion des anderen Reizes auslöst .`
  - *R:SPELL* (keyboard): «eeizes» is not a German word. It looks like a slip for «Reizes».

**71.** `Von 1926 bis 1927 absolvierte er bei der Fotoschule Lette-Verein eine fotografischen Ausbildung .`
→ `Von 1926 bis 1927 absolvierte er bei der Fotoschule Lette-Verein eine fotografische Ausbildung .`
  - *R:ADJ:FORM* (adjective_form): An adjective before a noun takes an ending that depends on the noun's gender, number and case, and on whether an article comes first. Here the adjective should read «fotografische», not «fotografischen».

**72.** `Er ist die Älteste von vier Kindern .`
→ `Er ist das Älteste von vier Kindern .`
  - *R:DET:FORM* (article_form): The article has to agree with its noun in gender, number and case. «die» does not fit here; the form the sentence needs is «das».

**73.** `Die Armee aprach von Angriffen auf Waffenlager und „ terroristischer unfrastruktur “ der Hisbollah .`
→ `Die Armee sprach von Angriffen auf Waffenlager und „ terroristischer Infrastruktur “ der Hisbollah .`
  - *R:SPELL* (keyboard): «aprach» is not a German word. It looks like a slip for «sprach».
  - *R:SPELL* (keyboard): «unfrastruktur» is not a German word. It looks like a slip for «Infrastruktur».

**74.** `In die 1953 veröffentlichte , ursprünglich vierstellige Variante dem CODEN fügte er die Zahl des Bandes und die Seitenzahl an , um einen Artikel in einer Zeitschrift genau zu zitieren .`
→ `In die 1953 veröffentlichte , ursprünglich vierstellige Variante des CODEN fügte er die Zahl des Bandes und die Seitenzahl an , um einen Artikel in einer Zeitschrift genau zu zitieren .`
  - *R:DET:FORM* (article_form): The article has to agree with its noun in gender, number and case. «dem» does not fit here; the form the sentence needs is «des».

**75.** `Weitere Gründe für Algiers Besorgnis sind Verbündeten des starken Mannes im Osten Libyens .`
→ `Weitere Gründe für Algiers Besorgnis sind die Verbündeten des starken Mannes im Osten Libyens .`
  - *M:DET* (drop_article): German nouns almost always need an article, even where English drops it. Add «die» before the noun.

**76.** `Erscheint dieses aehnlich erwarteteWährend der jüngsten Nintendo Direct hat ein ganz bestimmter Titel gefehlt auf welchen Fans schon seit vielen Jahren warten .`
→ `Erscheint dieses sehnlich erwarteteWährend der jüngsten Nintendo Direct hat ein ganz bestimmter Titel gefehlt , auf welchen Fans schon seit vielen Jahren warten .`
  - *R:SPELL* (keyboard): «aehnlich» is not a German word. It looks like a slip for «sehnlich».
  - *M:PUNCT* (drop_comma): A comma belongs here. German uses them to mark off parts of a sentence -- an apposition, an insertion, a list -- more strictly than English does.

**77.** `Auch gier sollte er Respekt und Anstand zeigen und sich dann doch vielleicht eines Beispiel an der Rede der Aussenministerin nehmen .`
→ `Auch hier sollte er Respekt und Anstand zeigen und sich dann doch vielleicht ein Beispiel an der Rede der Aussenministerin nehmen .`
  - *R:SPELL* (keyboard): «gier» is not a German word. It looks like a slip for «hier».
  - *R:DET:FORM* (article_form): The article has to agree with its noun in gender, number and case. «eines» does not fit here; the form the sentence needs is «ein».

**78.** `Der transzendentale Dichter sei somit der transzendentale mensch selbst .`
→ `Der transzendentale Dichter sei somit der transzendentale Mensch selbst .`
  - *R:ORTH* (noun_case): German capitalises every noun, not just names. «mensch» is a noun here, so it needs a capital letter: «Mensch». (Rat für deutsche Rechtschreibung, § 55)

**79.** `Montevideo – Dem uruguayische Fußball-Legende Diego Forlán sorgt für Aufsehen mit winem unerwarteten Schritt in seiner Sportkarriere .`
→ `Montevideo – Die uruguayische Fußball-Legende Diego Forlán sorgt für Aufsehen mit einem unerwarteten Schritt in seiner Sportkarriere .`
  - *R:DET:FORM* (article_form): The article has to agree with its noun in gender, number and case. «Dem» does not fit here; the form the sentence needs is «Die».
  - *R:SPELL* (keyboard): «winem» is not a German word. It looks like a slip for «einem».

**80.** `tudem spielt Gewalt in russischen Gesellschaft eine große Rolle .`
→ `Zudem spielt Gewalt in der russischen Gesellschaft eine große Rolle .`
  - *R:SPELL* (keyboard): «tudem» is not a German word. It looks like a slip for «Zudem».
  - *M:DET* (drop_article): German nouns almost always need an article, even where English drops it. Add «der» before the noun.

**81.** `Und wenn doch ein antagonistischer Konflikt entsteht , dann einer , der auf einem nicht bestätigten , emotional konfigurierten Input der Wahrnehmung und winem bestätigten Konflikt der Vernunft beruht .`
→ `Und wenn doch ein antagonistischer Konflikt entsteht , dann einer , der auf einem nicht bestätigten , emotional konfigurierten Input der Wahrnehmung und einem bestätigten Konflikt der Vernunft beruht .`
  - *R:SPELL* (keyboard): «winem» is not a German word. It looks like a slip for «einem».

**82.** `Dies bedeutet , dass der hat Export an Waren und Dienstleistungen in den Jahren 1995–2005 den Import überstiegen .`
→ `Dies bedeutet , dass der Export an Waren und Dienstleistungen in den Jahren 1995–2005 den Import überstiegen hat .`
  - *R:WO* (verb_final): In a subordinate clause the finite verb goes to the very end, which is not where English puts it. The clause should read «der Export an Waren und Dienstleistungen in den Jahren 1995–2005 den Import überstiegen hat».

**83.** `Im Prozess ist die , auf den jeweiligen Entwicklungsschritt angepasste , winbindung lokal betroffener Personen und Unternehmen geplant , um eine Akzeptanz in breiten Schichten auch in der Bevölkerung zu erreichen .`
→ `Im Prozess ist die , auf den jeweiligen Entwicklungsschritt angepasste , Einbindung lokal betroffener Personen und Unternehmen geplant , um eine Akzeptanz in breiten Schichten auch in der Bevölkerung zu erreichen .`
  - *R:SPELL* (keyboard): «winbindung» is not a German word. It looks like a slip for «Einbindung».

**84.** `Erste prognosen erwarten schon einen Anstieg auf 50 Prozent des Bruttolohns bis 2040.`
→ `Erste Prognosen erwarten schon einen Anstieg auf 50 Prozent des Bruttolohns bis 2040.`
  - *R:ORTH* (noun_case): German capitalises every noun, not just names. «prognosen» is a noun here, so it needs a capital letter: «Prognosen». (Rat für deutsche Rechtschreibung, § 55)

**85.** `Er galt Ende dem 1980er / Anfang 1990er Jahre als einer der besten sowjetischen Fußballer .`
→ `Er galt Ende der 1980er / Anfang der 1990er Jahre als einer der besten sowjetischen Fußballer .`
  - *R:DET:FORM* (article_form): The article has to agree with its noun in gender, number and case. «dem» does not fit here; the form the sentence needs is «der».
  - *M:DET* (drop_article): German nouns almost always need an article, even where English drops it. Add «der» before the noun.

**86.** `Peskow sagte außerdem Moskau stehe in kontakt mit wichtigen Akteuren in Syrien wie etwa der und dem Iran .`
→ `Peskow sagte außerdem , Moskau stehe in Kontakt mit wichtigen Akteuren in Syrien wie etwa der und dem Iran .`
  - *M:PUNCT* (drop_comma): A comma belongs here. German uses them to mark off parts of a sentence -- an apposition, an insertion, a list -- more strictly than English does.
  - *R:ORTH* (noun_case): German capitalises every noun, not just names. «kontakt» is a noun here, so it needs a capital letter: «Kontakt». (Rat für deutsche Rechtschreibung, § 55)

**87.** `Daneben wird bereits für den Zeitraum der Prüfung einer Gefährdung die Möglichkeit der Eintragung einer vorläufige Auskunftssperre geschaffen .`
→ `Daneben wird bereits für den Zeitraum der Prüfung einer Gefährdung die Möglichkeit der Eintragung einer vorläufigen Auskunftssperre geschaffen .`
  - *R:ADJ:FORM* (adjective_form): An adjective before a noun takes an ending that depends on the noun's gender, number and case, and on whether an article comes first. Here the adjective should read «vorläufigen», not «vorläufige».

**88.** `Nachteil der kostengünstigen Lösung zu Schleifkontakten ist das mechanische Verschleiß .`
→ `Nachteil der kostengünstigen Lösung mit Schleifkontakten ist der mechanische Verschleiß .`
  - *R:ADP* (preposition): German prepositions are not chosen by translating the English one. This phrase takes «mit» rather than «zu».
  - *R:DET:FORM* (article_form): The article has to agree with its noun in gender, number and case. «das» does not fit here; the form the sentence needs is «der».

**89.** `Dieses ist eine Teilliste mit einem Eintrag einer Person deren Namen mit den Buchstaben „ Jb “ beginnt .`
→ `Dieses ist eine Teilliste mit einem Eintrag einer Person , deren Namen mit den Buchstaben „ Jb “ beginnt .`
  - *M:PUNCT* (drop_comma): A comma belongs here. German uses them to mark off parts of a sentence -- an apposition, an insertion, a list -- more strictly than English does.

**90.** `Mitte Spielzeit 2004 / 05 kehrte er zum HC Znojemští Orli zurück und blieb dort bis Ende der Spielzeit 2006 / 07.`
→ `Mitte der Spielzeit 2004 / 05 kehrte er zum HC Znojemští Orli zurück und blieb dort bis Ende der Spielzeit 2006 / 07.`
  - *M:DET* (drop_article): German nouns almost always need an article, even where English drops it. Add «der» before the noun.

**91.** `Politisch ist Wilson Nesbitt nicht nehr in Erscheinung getreten .`
→ `Politisch ist Wilson Nesbitt nicht mehr in Erscheinung getreten .`
  - *R:SPELL* (keyboard): «nehr» is not a German word. It looks like a slip for «mehr».

**92.** `Das erstes Experiment diente der Eingrenzung des parameterraums .`
→ `Das erste Experiment diente der Eingrenzung des Parameterraums .`
  - *R:ADJ:FORM* (adjective_form): An adjective before a noun takes an ending that depends on the noun's gender, number and case, and on whether an article comes first. Here the adjective should read «erste», not «erstes».
  - *R:ORTH* (noun_case): German capitalises every noun, not just names. «parameterraums» is a noun here, so it needs a capital letter: «Parameterraums». (Rat für deutsche Rechtschreibung, § 55)

**93.** `So komme dem Indoor-Himmel noch mehr zur feltung .`
→ `So komme der Indoor-Himmel noch mehr zur Geltung .`
  - *R:DET:FORM* (article_form): The article has to agree with its noun in gender, number and case. «dem» does not fit here; the form the sentence needs is «der».
  - *R:SPELL* (keyboard): «feltung» is not a German word. It looks like a slip for «Geltung».

**94.** `Traditionell segelt Škoda hinter Volkswagen so weit normale Hierarchie .`
→ `Traditionell segelt Škoda hinter Volkswagen , so weit normale Hierarchie .`
  - *M:PUNCT* (drop_comma): A comma belongs here. German uses them to mark off parts of a sentence -- an apposition, an insertion, a list -- more strictly than English does.

**95.** `Zwar betont Das Erste sass es 2025 bei den „ bewährten Sendeplätzen “ bleibt .`
→ `Zwar betont Das Erste , dass es 2025 bei den „ bewährten Sendeplätzen “ bleibt .`
  - *M:PUNCT* (drop_comma_subordinate): German separates a subordinate clause from its main clause with a comma. Unlike English, this is a rule rather than a matter of taste. Add the comma. (Rat für deutsche Rechtschreibung, § 74)
  - *R:SPELL* (keyboard): «sass» is not a German word. It looks like a slip for «dass».

**96.** `Aber auch ohne Kärtchen dürfe man genauso kostenfrei mitfahren , anfügt er .`
→ `Aber auch ohne Kärtchen dürfe man genauso kostenfrei mitfahren , fügt er an .`
  - *R:WO* (separable_prefix): This verb is separable. In a main clause the prefix detaches and moves to the end, so it is «fügt er an» rather than one word.

**97.** `Des Euwax Sentiment notiert bei -64 Punkten tief im ninus .`
→ `Der Euwax Sentiment notiert bei -64 Punkten tief im Minus .`
  - *R:DET:FORM* (article_form): The article has to agree with its noun in gender, number and case. «Des» does not fit here; the form the sentence needs is «Der».
  - *R:SPELL* (keyboard): «ninus» is not a German word. It looks like a slip for «Minus».

**98.** `Das CDU in Sachsen ist auf das linke Lager angewiesen .`
→ `Die CDU in Sachsen ist auf das linke Lager angewiesen .`
  - *R:DET:FORM* (article_form): The article has to agree with its noun in gender, number and case. «Das» does not fit here; the form the sentence needs is «Die».

**99.** `Er verbannte Jar Jar Binks wegen seiner rollpatschigkeit .`
→ `Er verbannte Jar Jar Binks wegen seiner Tollpatschigkeit .`
  - *R:SPELL* (keyboard): «rollpatschigkeit» is not a German word. It looks like a slip for «Tollpatschigkeit».

**100.** `Die Ausbildung der Fachkräfte als auch berufsbegleitende Weiterbildung gehören ebenfalls massiv finanziell unterstützt .`
→ `Die Ausbildung der Fachkräfte , als auch die berufsbegleitende Weiterbildung gehören ebenfalls massiv finanziell unterstützt .`
  - *M:PUNCT* (drop_comma_subordinate): German separates a subordinate clause from its main clause with a comma. Unlike English, this is a rule rather than a matter of taste. Add the comma. (Rat für deutsche Rechtschreibung, § 74)
  - *M:DET* (drop_article): German nouns almost always need an article, even where English drops it. Add «die» before the noun.

