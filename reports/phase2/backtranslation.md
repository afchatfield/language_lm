# Phase 2: back-translated errors

47,205 pairs kept of 50,000 generated, from a generator trained on Falko-MERLIN train read backwards. The argument for this over the hand-written corruptors is in `reports/phase3/headroom.md`.

## Settings

| | |
|---|---|
| adapter | `checkpoints/corrupter-de` |
| sampling | temperature 1.0, top-p 0.95 |
| seed | 20260909 |
| clean sentences | 50,000 |
| divergence threshold | 0.6667 (99% of real learner pairs) |

## The funnel

| Outcome | Pairs | Share |
|---|---:|---:|
| kept | 47,205 | 94.4% |
| copied when asked for errors | 2,175 | 4.3% |
| more damaged than real learner German | 360 | 0.7% |
| asked for none, damaged anyway | 236 | 0.5% |
| length ratio out of range | 21 | 0.0% |
| degenerate repetition | 3 | 0.0% |

## Density

10,767 of the kept pairs need no correction (22.8%); Falko-MERLIN train is 22.1%.

| Edits | Asked for | Actually made |
|---:|---:|---:|
| 0 | 10,767 | 10,767 |
| 1 | 8,851 | 11,005 |
| 2 | 8,584 | 8,153 |
| 3 | 6,201 | 5,836 |
| 4 | 4,484 | 4,082 |
| 5 | 2,937 | 2,851 |
| 6 | 1,886 | 1,817 |
| 7 | 1,204 | 1,161 |
| 8 | 851 | 632 |
| 9 | 492 | 400 |
| 10+ | 948 | 501 |

Mean edits a sentence: **2.30** (Falko train 2.55, the rule-based corpus 2.34).

## Samples

- asked for 2
  - clean: `Diese erste erfolgreiche Überlandfahrt trägt wesentlich dazu bei , die noch bestehenden Vorbehalte von potentiellen Kunden zu zerstreuen und ermöglicht in der Folge den wirtschaftlichen Erfolg der Firma .`
  - damaged: `Diese erste erfolgreiche Überlandfahrt trägt wesentlich dazu bei , die noch bestehenden Vorabehaltungen von potentiellen Kunden zu zerstreuen und ermöglicht in der Folge den wirtschaftlichen Erfolg der Firma .`
- asked for 0
  - clean: `Friedrich Schiller wurde durch ein Maskengedicht Amalie von Imhoffs an die Herzogin auf ihr poetisches Talent aufmerksam und ermutigte sie zusammen mit Johann Wolfgang von Goethe zu ernsthaften Auseinandersetzungen mit der Literatur .`
  - damaged: `Friedrich Schiller wurde durch ein Maskengedicht Amalie von Imhoffs an die Herzogin auf ihr poetisches Talent aufmerksam und ermutigte sie zusammen mit Johann Wolfgang von Goethe zu ernsthaften Auseinandersetzungen mit der Literatur .`
- asked for 4
  - clean: `So sei selbst beim Hochwasserereignis Anfang Juni , als die Seepromenade in Bad Wiessee komplett überflutet worden sei , und auch bei den jüngsten Hochwasserereignissen kein Wasser in die Baugrube eingedrungen .`
  - damaged: `So sei selbst beim Hochwasserereignis anfängen Juni , als die Seepromenade in Bad Wiessee komplett überflutet worden sei , und auch bei den jüngsten Hochwasserereignissen kein Wasser in die Baugrube eingedrungen .`
- asked for 1
  - clean: `Die Erzeugerpreise , ein Vorbote der Verbraucherpreise , waren im Januar im Vergleich zum Vormonat um 0,3 Prozent gestiegen , während befragte Analysten lediglich 0,1 Prozent erwartet hatten .`
  - damaged: `Die Erzeugerpreise , ein Vorbote der Verbraucherpreise waren im Januar im Vergleich zum Vormonat um 0,3 Prozent gestiegen , während befragte Analysten lediglich 0,1 Prozent erwartet hatten .`
- asked for 6
  - clean: `Erst im Zug zurück nach Berlin treffen die beiden sich wieder .`
  - damaged: `Erst im ZUG zurück nach Berlin Treffen die Beiden sich wecheisch .`
- asked for 1
  - clean: `Dieser Vortrag erschien noch im gleichen Jahr im Schulverein in Druck .`
  - damaged: `Dieser Vortrag erschien noch im gleichen Jahr im Schulverein in druck .`
- asked for 2
  - clean: `Die Flugtests begannen 1999 , die Einsatzreife wurde im Jahr 2000 erreicht .`
  - damaged: `Die Flugtests begannen 1999 die Einsatzreife wurde im Jahr 2000 erreicht .`
- asked for 5
  - clean: `Zu Beginn der zweiten Jahreshälfte fiel die Teuerungsrate erstmals wieder unter 3 Prozent .`
  - damaged: `Zum Beginn den zweiten Jahrenhälfte fiel die Teuerungsrate erstmals wieder unter 3 Prozent .`
- asked for 0
  - clean: `Die Konzerne , die sich gegen die weitere Verschärfung der Regeln aussprechen , fürchten gleich doppelte Folgen .`
  - damaged: `Die Konzerne , die sich gegen die weitere Verschärfung der Regeln aussprechen , fürchten gleich doppelte Folgen .`
- asked for 1
  - clean: `In § 2.02 Model Penal Code wurden diese klassischen Vorsatzformen aufgegeben .`
  - damaged: `In § 2.02 Model Penal Code , wurden diese klassischen Vorsatzformen aufgegeben .`
- asked for 5
  - clean: `Als Georgina und Giulia wieder ins Camp zurückkehren , geht der Zoff zwischen Elena und Georgina in die nächste Runde .`
  - damaged: `Als Georgina und Giulia in der Camp wieder zuruck kehren geht den Zoff zwischen Elena und Georgina in nächste Runde .`
- asked for 0
  - clean: `Hauptimportgüter sind Konsumgüter und Rohölprodukte , wobei hier die wichtigsten Herkunftsländer Frankreich und China sind .`
  - damaged: `Hauptimportgüter sind Konsumgüter und Rohölprodukte , wobei hier die wichtigsten Herkunftsländer Frankreich und China sind .`
- asked for 7
  - clean: `Das West-Nil-Virus stammt ursprünglich aus Afrika und wurde erstmals 1937 im West-Nil-Distrikt in Uganda festgestellt .`
  - damaged: `das West-Nil Virus stamme ursprunglich aus Afrika und wurde erstmals in 1937 im West Nilandistrikt in Uganda festgestellt .`
- asked for 0
  - clean: `Meterhohe Schuttberge türmen sich im Schweizer Örtchen Brienz .`
  - damaged: `Meterhohe Schuttberge türmen sich im Schweizer Örtchen Brienz .`
- asked for 5
  - clean: `Nachdem der Spieler einen Namen und einen Seed angegeben hatte , wurde er in eine anhand des Seeds generierte Welt gesetzt .`
  - damaged: `Nachdem der Spieler einen Name und ein Seed angegeben hatte , wurde er in einer anhand des Seeds generierten Welt gesetzt .`
- asked for 10
  - clean: `Einige Republikaner im Kongress begannen , die Chancen von Gaetz auf seine Bestätigung als Minister in Frage zu stellen .`
  - damaged: `Einigen Republikaner in der Kongres begannen die Chancen Gaetz von seiner Bestätigung als ein Minister in Frage zu stellen .`
- asked for 3
  - clean: `Viele Betreiber haben ihre Türen schon längst wieder für Besucher geöffnet .`
  - damaged: `Viele Betreiber haben ihre Tueren schon lang wieder fur Besucher geoffnet .`
- asked for 5
  - clean: `Im Februar 1924 gab Seeckt die diktatorischen Vollmachten , die er von Ebert erhalten hatte , wieder ab .`
  - damaged: `Im Februar 1924 gab Seeckt die diktatorischen Vollmachten auf , die er von Ebert sich erhalten hatte .`
- asked for 1
  - clean: `Nachdem die Königin ihren Sohn in der hässlichen Ausführung gesehen hat , bietet ihr die Hexe an , ihn zu erlösen .`
  - damaged: `Nachdem die Königin ihren Sohn in der hässlichen Ausführung gesehen hat , bietet ihr die Hexe an ihm zu erlösen .`
- asked for 0
  - clean: `Das Hinterende ist sehr weit gerundet , ebenso das Vorderende .`
  - damaged: `Das Hinterende ist sehr weit gerundet , ebenso das Vorderende .`
- asked for 0
  - clean: `Das Trainingsfeld Ost bleibt ein Naturrasenplatz und soll saniert werden .`
  - damaged: `Das Trainingsfeld Ost bleibt ein Naturrasenplatz und soll saniert werden .`
- asked for 3
  - clean: `Das Klinikum ist an der notärztlichen Versorgung des Landkreises beteiligt .`
  - damaged: `das Klinikum ist an der nichtärztlicher Versorung des Landkreises beteiligt .`
- asked for 3
  - clean: `Die Wahl eines Ministerpräsidenten ohne eigene Mehrheit könnte eine Vorlage für den nächsten Akt sein .`
  - damaged: `Der Wahl eines Ministerpräsidenten ohne eigene Mehrheit könnte ein Vorlage für das nächste Akt sein .`
- asked for 1
  - clean: `Die dritte Frau des letzten Eigentümers , dessen zweite Frau das letzte Mitglied der Familie Thellusson war , lebt heute auf dem Anwesen in einem Haus , das für den Chefgärtner gebaut wurde .`
  - damaged: `Die dritte Frau des letzten Eigentümers , deren zweite Frau das letzte Mitglied der Familie Thellusson war , lebt heute auf dem Anwesen in einem Haus , das für den Chefgärtner gebaut wurde .`
- asked for 1
  - clean: `Seine Abschlussarbeit dieses Seminars war das Drehbuch „ Harold und Maude “ über die Liebe eines 18-Jährigen zu einer 80-Jährigen .`
  - damaged: `Seine Abschlussarbeit , dieses Seminars war das Drehbuch „ Harold und Maude “ über die Liebe eines 18-Jährigen zu einer 80-Jährigen .`
- asked for 1
  - clean: `Auch einige andere bedeutende , mit den jeweiligen Kaisern in Verbindung stehende Personen ( wie Kaiserinnen ) werden aufgeführt .`
  - damaged: `Auch einige andere bedeutende , mit den jeweiligen Kaisern verknüpfte Personen ( wie Kaiserinnen ) werden aufgeführt .`
- asked for 2
  - clean: `Die Ecken der finiten Elemente heißen Knoten .`
  - damaged: `Die Ecken der endlichen Elementen heissen Knoten .`
- asked for 2
  - clean: `Er färbt sich das Haar schwarz , klebt sich eher schlecht als recht einen Schnurrbart an und übt auf Dänisch zu radebrechen .`
  - damaged: `Er färbt sich den Haar schwarz , klebt sich eher schlecht als recht ein Schnurrbart an und übt auf Dänisch zu radebrechen .`
- asked for 1
  - clean: `Sie wissen doch sonst immer alles , vor allem alles besser !`
  - damaged: `Sie wissen doch sonst immer alles , vor allem besser ! ! !`
- asked for 7
  - clean: `Es besteht die Möglichkeit der Weiterbildung zur Pflegefachassistenz .`
  - damaged: `Es besteht die Möglichkeit weiterbildung für Pflegebedacht .`
