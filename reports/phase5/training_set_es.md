# Phase 5: the Spanish training set

97,473 examples: **37,473 real** learner sentences from COWS-L2H train, and **60,000 rule-generated** ones. 32,173 need no correction (33.0%).

Built to German's iteration-2 state -- real data primary, synthetic supporting -- not to every later German iteration. In particular there is no back-translated third source: German's own (iteration 5) measured *negative* against the bar its first successful run set (`reports/phase3/headroom.md`), so it is not assumed Spanish wants one before a first real+synthetic run exists to compare against.

## Invariants

| Check | Result |
|---|---|
| Damaged examples whose source equals their target | 0 |
| Edits with no explanation | 0 |
| Synthetic edits with no template | 0 |
| Sentences no corruptor could damage | 0 |

## Error density

| Edits in a sentence | Share | COWS-L2H train |
|---|---:|---:|
| 0 | 33.0% | 33.0% |
| 1 | 19.9% | 25.0% |
| 2 | 16.4% | 16.1% |
| 3 or more | 30.7% | 25.9% |

Mean edits per sentence: **1.70**.

## Error types

**51** distinct types, against 13 the synthetic rules alone can make and 51 in the real COWS-L2H histogram (`reports/phase5/error_type_histogram_es.md`).

| Error type | Count | Share |
|---|---:|---:|
| `R:ORTH` | 31,737 | 19.2% |
| `R:SPELL` | 18,176 | 11.0% |
| `M:ADP` | 14,030 | 8.5% |
| `R:DET:FORM` | 13,875 | 8.4% |
| `R:ADP` | 13,419 | 8.1% |
| `R:VERB:FORM` | 12,008 | 7.2% |
| `R:OTHER` | 10,806 | 6.5% |
| `M:DET` | 10,744 | 6.5% |
| `U:PRON` | 7,187 | 4.3% |
| `R:ADJ:FORM` | 6,029 | 3.6% |
| `M:PRON` | 2,855 | 1.7% |
| `M:PUNCT` | 2,616 | 1.6% |
| `R:AUX` | 2,497 | 1.5% |
| `U:DET` | 1,924 | 1.2% |
| `R:VERB` | 1,847 | 1.1% |
| `R:NOUN` | 1,745 | 1.1% |
| `R:AUX:FORM` | 1,333 | 0.8% |
| `U:OTHER` | 1,214 | 0.7% |
| `M:OTHER` | 1,049 | 0.6% |
| `U:ADP` | 1,017 | 0.6% |

## Explanations

100,727 edits carry a rule-level explanation, from the corruptor that made them. 64,909 carry a type-level one, because a COWS-L2H edit records what changed and not why.

## 100 samples

The same exit criterion Phase 2 used: read these, and if more than five have a wrong correction or a wrong explanation, fix the pipeline before training on it.

**1.** *(synthetic)* `Lo confirmaba como una condición normalizada en los hoteles de Japón , pues añadía que " probablemente los goteles de lujo ( españoles ) tengan pijama " , pero que el alojamiento que había escogido ella no ofrecía .`
→ `Lo confirmaba como una condición normalizada en los hoteles de Japón , pues añadía que " probablemente los hoteles de lujo ( españoles ) tengan pijama " , pero que el alojamiento que había escogido ella no lo ofrecía .`
  - *R:SPELL*: «goteles» is not a Spanish word. It looks like a slip for «hoteles».
  - *M:PRON*: An object pronoun is missing here. Spanish does not leave it out the way some constructions in English do; add «lo».

**2.** *(synthetic)* `Todas ellas interpretan tambien los canciones a apertura y cierre de el episodios .`
→ `Todas ellas interpretan también las canciones de apertura y cierre de los episodios .`
  - *R:ORTH*: The stress mark is part of the word, not decoration on it. «tambien» and «también» are written differently for a reason -- Spanish marks stress in writing precisely where it would otherwise be unpredictable. Write «también».
  - *R:DET:FORM*: The article has to agree with its noun in gender and number. «los» does not fit here; the form the sentence needs is «las».
  - *R:ADP*: Spanish prepositions are not chosen by translating the English one. This phrase takes «de» rather than «a».
  - *R:DET:FORM*: The article has to agree with its noun in gender and number. «el» does not fit here; the form the sentence needs is «los».

**3.** *(learner)* `Angelina Jolie trabaja en Africa .`
→ `Angelina Jolie trabaja en África .`
  - *R:ORTH*: This is a convention of written Spanish rather than a grammar rule. The correct written form here is «África».

**4.** *(learner)* `También me encanta cuando vamos a downtown * CITY * para bailar en el discotheque .`
→ `También me encanta cuando vamos a downtown * CITY * para bailar en la discoteca .`
  - *R:DET:FORM*: The article has to agree with its noun in gender and number. Here that makes it «la», not «el».
  - *R:SPELL*: «discotheque» is not spelled correctly. It should be «discoteca».

**5.** *(learner)* `Fui todos los días al bosque mágico , pero solo en ese tiempo específico o de lo contrario no va funcionar .`
→ `Fui todos los días al bosque mágico , pero solo en ese tiempo específico o de lo contrario no iba a funcionar .`
  - *R:OTHER*: «va» is not the right word here. Spanish uses «iba a» in this context.

**6.** *(synthetic)* `Mostaza va un paso más allá y recuerda cuál fue estrategia de Mariano Rajoy cuando Podemos presentar unas mocion contra él .`
→ `Mostaza va un paso más allá y recuerda cuál fue la estrategia de Mariano Rajoy cuando Podemos presentó una moción contra él .`
  - *M:DET*: Spanish nouns almost always need an article, even where English drops it. Add «la» before the noun.
  - *R:VERB:FORM*: «presentar» is the infinitive -- the form a dictionary lists. A finite verb has to agree with its subject, so this sentence needs «presentó».
  - *R:DET:FORM*: The article has to agree with its noun in gender and number. «unas» does not fit here; the form the sentence needs is «una».
  - *R:ORTH*: The stress mark is part of the word, not decoration on it. «mocion» and «moción» are written differently for a reason -- Spanish marks stress in writing precisely where it would otherwise be unpredictable. Write «moción».

**7.** *(learner)* `También me gusta los EE.UU pero aquí personalmente tengo muchas responsabilidades y aveces me siento ahogada .`
→ `También me gusta los EE.UU , pero aquí personalmente tengo muchas responsabilidades y a veces me siento ahogado .`
  - *M:PUNCT*: A punctuation mark is missing here. Spanish punctuates by rule rather than by ear -- including marking a question or exclamation at the opening «¿» or «¡» as well as the close, which English does not -- and this sentence needs «,».
  - *R:ORTH*: This is a convention of written Spanish rather than a grammar rule. The correct written form here is «a veces».
  - *R:ADJ:FORM*: An adjective has to agree with the noun it describes in gender and number. It should read «ahogado».

**8.** *(learner)* `Nos mirábamos unos a otros y tenían miedo de lo que acabamos de escuchar .`
→ `Nos mirábamos unos a otros y teníamos miedo de lo que acabábamos de escuchar .`
  - *R:VERB:FORM*: The verb is in the wrong form. Check the tense, the person, and whether the sentence needs the indicative or the subjunctive. It should be «teníamos».
  - *R:VERB*: «acabamos» is not the right verb here. Spanish uses «acabábamos» in this context.

**9.** *(synthetic)* `En 1936 pasa a la enseñanza cursos wspeciales de cuadros de mando en esta misma qcademia .`
→ `En 1936 pasa a la enseñanza de cursos especiales de cuadros de mando en esta misma academia .`
  - *M:ADP*: This phrase needs «de». Which preposition a Spanish verb governs is often not predictable from the English one -- «pensar en», «soñar con» -- so it has to be learned with the verb rather than translated.
  - *R:SPELL*: «wspeciales» is not a Spanish word. It looks like a slip for «especiales».
  - *R:SPELL*: «qcademia» is not a Spanish word. It looks like a slip for «academia».

**10.** *(synthetic)* `Para persuadir a Gomm , Treves le había enseñado recitar algunas frases corteses .`
→ `Para persuadir a Gomm , Treves le había enseñado a recitar algunas frases corteses .`
  - *M:ADP*: This phrase needs «a». Which preposition a Spanish verb governs is often not predictable from the English one -- «pensar en», «soñar con» -- so it has to be learned with the verb rather than translated.

**11.** *(synthetic)* `No existe un lugar más apropiado para entregarse a las aensaciones que brinda el contacto que el mercado navideño de Liberec .`
→ `No existe un lugar más apropiado para entregarse a las sensaciones que brinda el contacto que el mercado navideño de Liberec .`
  - *R:SPELL*: «aensaciones» is not a Spanish word. It looks like a slip for «sensaciones».

**12.** *(synthetic)* `Aunque el Tesla Cyberquad ha sido todo un éxito en Norteamerica , a mercados como el español aun no se había dejado ver .`
→ `Aunque el Tesla Cyberquad ha sido todo un éxito en Norteamérica , en mercados como el español aún no se había dejado ver .`
  - *R:ORTH*: The stress mark is part of the word, not decoration on it. «Norteamerica» and «Norteamérica» are written differently for a reason -- Spanish marks stress in writing precisely where it would otherwise be unpredictable. Write «Norteamérica».
  - *R:ADP*: Spanish prepositions are not chosen by translating the English one. This phrase takes «en» rather than «a».
  - *R:ORTH*: The stress mark is part of the word, not decoration on it. «aun» and «aún» are written differently for a reason -- Spanish marks stress in writing precisely where it would otherwise be unpredictable. Write «aún».

**13.** *(synthetic)* `“ el dolor que genera el inflación al conjunto de la sociedad , sobre todo en los sectores populares , lo entendemos y da muchísima bronca que suceda en nuestro gobierno ” .`
→ `“ el dolor que genera la inflación al conjunto de la sociedad , sobre todo en los sectores populares , lo entendemos y da muchísima bronca que suceda en nuestro gobierno ” .`
  - *R:DET:FORM*: The article has to agree with its noun in gender and number. «el» does not fit here; the form the sentence needs is «la».

**14.** *(learner)* `Ella le gusta la película se llama " the thunderbirds " porque ella piense es una película muy buena .`
→ `A ella le gusta la película que se llama " The Thunderbirds " porque piensa que es una película muy buena .`
  - *M:ADP*: A preposition is missing. This construction needs «A ella».
  - *M:PRON*: A pronoun is missing. This sentence needs «que».
  - *R:ORTH*: This is a convention of written Spanish rather than a grammar rule. The correct written form here is «The Thunderbirds».
  - *U:PRON*: Spanish verb endings already show who the subject is. «ella» is not needed here -- keep a subject pronoun only when the sentence is deliberately contrasting or emphasising who is doing the action.
  - *R:OTHER*: «piense» is not the right word here. Spanish uses «piensa» in this context.
  - *M:SCONJ*: A subordinating conjunction is missing here. Add «que».

**15.** *(synthetic)* `El secretario de Presidencia , Ákvaro Delgado , estuvo presente este lunes en la plaza Mártires Chicago por el Día de los Trabajadores y destaco que está " un acto de respeto y tolerancia " hacerse presente en un ámbito como el de esta jornada .`
→ `El secretario de Presidencia , Álvaro Delgado , estuvo presente este lunes en la plaza Mártires de Chicago por el Día de los Trabajadores y destacó que es " un acto de respeto y tolerancia " hacerse presente en un ámbito como el de esta jornada .`
  - *R:SPELL*: «Ákvaro» is not a Spanish word. It looks like a slip for «Álvaro».
  - *M:ADP*: This phrase needs «de». Which preposition a Spanish verb governs is often not predictable from the English one -- «pensar en», «soñar con» -- so it has to be learned with the verb rather than translated.
  - *R:ORTH*: The stress mark is part of the word, not decoration on it. «destaco» and «destacó» are written differently for a reason -- Spanish marks stress in writing precisely where it would otherwise be unpredictable. Write «destacó».
  - *R:AUX*: Spanish has two verbs for English "to be", and they are not interchangeable. This is «es», not «está».

**16.** *(synthetic)* `Con estas mujeres no solos mueren estos cantares sino también toda su cosmología , nosotros perdemos parte de nuestro pasado y por lo tanto de nuestra identidad .`
→ `Con estas mujeres no solo mueren estos cantares sino también toda su cosmología , perdemos parte de nuestro pasado y por lo tanto de nuestra identidad .`
  - *R:ADJ:FORM*: An adjective has to agree with the noun it describes in gender and number. Here it should read «solo», not «solos».
  - *U:PRON*: Spanish verb endings already say who the subject is, so the pronoun is not needed here the way it would be in English. Remove «nosotros» -- or keep it only if the sentence is deliberately emphasising or contrasting who is doing the action.

**17.** *(synthetic)* `Se supone que mujer no alcanzo a trabar la puerta de la casa y que el delincuente , ya arma en mano , no tuvo problemas en imponer su fuerza física .`
→ `Se supone que la mujer no alcanzó a trabar la puerta de la casa y que el delincuente , ya arma en mano , no tuvo problemas en imponer su fuerza física .`
  - *M:DET*: Spanish nouns almost always need an article, even where English drops it. Add «la» before the noun.
  - *R:ORTH*: The stress mark is part of the word, not decoration on it. «alcanzo» and «alcanzó» are written differently for a reason -- Spanish marks stress in writing precisely where it would otherwise be unpredictable. Write «alcanzó».

**18.** *(synthetic)* `Jose Arpino Vega es una las víctimas que desapareció en « Boiso Lanza » .`
→ `José Arpino Vega es una de las víctimas que desapareció en « Boiso Lanza » .`
  - *R:ORTH*: The stress mark is part of the word, not decoration on it. «Jose» and «José» are written differently for a reason -- Spanish marks stress in writing precisely where it would otherwise be unpredictable. Write «José».
  - *M:ADP*: This phrase needs «de». Which preposition a Spanish verb governs is often not predictable from the English one -- «pensar en», «soñar con» -- so it has to be learned with the verb rather than translated.

**19.** *(learner)* `Por la noche , le dieron una alfombra rota y una sábana desigual como cama y cobertor .`
→ `Por la noche , le dieron una alfombra rota y una sábana desigual , como cama y cobertor .`
  - *M:PUNCT*: A punctuation mark is missing here. Spanish punctuates by rule rather than by ear -- including marking a question or exclamation at the opening «¿» or «¡» as well as the close, which English does not -- and this sentence needs «,».

**20.** *(synthetic)* `Buchanan llegó a la presidencia con 65 años , además de ser el último nacido en el siglo XVIII , fue el último mandatario antes la Guerra de Secesión y el única presidente que permaneció soltero toda su vida .`
→ `Buchanan llegó a la presidencia con 65 años , además de ser el último nacido en el siglo XVIII , fue el último mandatario antes de la Guerra de Secesión y el único presidente que permaneció soltero toda su vida .`
  - *M:ADP*: This phrase needs «de». Which preposition a Spanish verb governs is often not predictable from the English one -- «pensar en», «soñar con» -- so it has to be learned with the verb rather than translated.
  - *R:ADJ:FORM*: An adjective has to agree with the noun it describes in gender and number. Here it should read «único», not «única».

**21.** *(synthetic)* `Es una postura reiterada en muchos otros de sus retratos o figuras de fantasia .`
→ `Es una postura reiterada en muchos otros de sus retratos o figuras de fantasía .`
  - *R:ORTH*: The stress mark is part of the word, not decoration on it. «fantasia» and «fantasía» are written differently for a reason -- Spanish marks stress in writing precisely where it would otherwise be unpredictable. Write «fantasía».

**22.** *(learner)* `Él es muy privado sobre su familia .`
→ `Es muy reservado sobre su familia .`
  - *U:PRON*: Spanish verb endings already show who the subject is. «Él es» is not needed here -- keep a subject pronoun only when the sentence is deliberately contrasting or emphasising who is doing the action.
  - *R:ADJ*: «privado» is not the right adjective here. Use «reservado».

**23.** *(synthetic)* `El secretario general de Junts , Jordi Turull , ha planteado este lunes a fuerzas independentistas concurrir con una lista unitaria a las elecciones generales del proximo 23 de julio .`
→ `El secretario general de Junts , Jordi Turull , ha planteado este lunes a las fuerzas independentistas concurrir con una lista unitaria a las elecciones generales del próximo 23 de julio .`
  - *M:DET*: Spanish nouns almost always need an article, even where English drops it. Add «las» before the noun.
  - *R:ORTH*: The stress mark is part of the word, not decoration on it. «proximo» and «próximo» are written differently for a reason -- Spanish marks stress in writing precisely where it would otherwise be unpredictable. Write «próximo».

**24.** *(synthetic)* `Para lo menos hasta dines de 1969 el desarrollo alcanzado por este aparato militar fue escaso .`
→ `Por lo menos hasta fines de 1969 el desarrollo alcanzado por este aparato militar fue escaso .`
  - *R:ADP*: Spanish prepositions are not chosen by translating the English one. This phrase takes «Por» rather than «Para».
  - *R:SPELL*: «dines» is not a Spanish word. It looks like a slip for «fines».

**25.** *(synthetic)* `El direccion de Monitoreo de Eventos Adversos de Ecuador descarto en un informe que hubiera oeligro de un tsunami .`
→ `La dirección de Monitoreo de Eventos Adversos de Ecuador descartó en un informe que hubiera peligro de un tsunami .`
  - *R:DET:FORM*: The article has to agree with its noun in gender and number. «El» does not fit here; the form the sentence needs is «La».
  - *R:ORTH*: The stress mark is part of the word, not decoration on it. «direccion» and «dirección» are written differently for a reason -- Spanish marks stress in writing precisely where it would otherwise be unpredictable. Write «dirección».
  - *R:ORTH*: The stress mark is part of the word, not decoration on it. «descarto» and «descartó» are written differently for a reason -- Spanish marks stress in writing precisely where it would otherwise be unpredictable. Write «descartó».
  - *R:SPELL*: «oeligro» is not a Spanish word. It looks like a slip for «peligro».

**26.** *(learner)* `Benedict Cumberbatch esta casado a Sophie Hunter en 2015 .`
→ `Benedict Cumberbatch está casado con Sophie Hunter en 2015 .`
  - *R:ORTH*: This is a convention of written Spanish rather than a grammar rule. The correct written form here is «está».
  - *R:ADP*: Spanish prepositions are not chosen by translating the English one. This phrase takes «con» rather than «a».

**27.** *(learner)* `No quiero caminar en el puente , pero me gusta como bonito lo es .`
→ `No quiero caminar en el puente , pero me gusta lo bonito que es .`
  - *U:SCONJ*: The subordinating conjunction «como» does not belong here. Remove it.
  - *R:WO*: This needs to be «lo bonito» rather than «bonito lo».
  - *M:SCONJ*: A subordinating conjunction is missing here. Add «que».

**28.** *(synthetic)* `El primer equipo que vi jugar fue a América contra un equipo español y desde ahí me yo hice americanista .`
→ `El primer equipo que vi jugar fue a América contra un equipo español y desde ahí me hice americanista .`
  - *U:PRON*: Spanish verb endings already say who the subject is, so the pronoun is not needed here the way it would be in English. Remove «yo» -- or keep it only if the sentence is deliberately emphasising or contrasting who is doing the action.

**29.** *(synthetic)* `Sylvia hacer aparecer una cadena de plata , y colgando a esta una miniatura del Enterprise .`
→ `Sylvia hace aparecer una cadena de plata , y colgando de ésta una miniatura del Enterprise .`
  - *R:VERB:FORM*: «hacer» is the infinitive -- the form a dictionary lists. A finite verb has to agree with its subject, so this sentence needs «hace».
  - *R:ADP*: Spanish prepositions are not chosen by translating the English one. This phrase takes «de» rather than «a».
  - *R:ORTH*: The stress mark is part of the word, not decoration on it. «esta» and «ésta» are written differently for a reason -- Spanish marks stress in writing precisely where it would otherwise be unpredictable. Write «ésta».

**30.** *(learner)* `Ella es actriz .`
→ `Es actriz .`
  - *U:PRON*: Spanish verb endings already show who the subject is. «Ella es» is not needed here -- keep a subject pronoun only when the sentence is deliberately contrasting or emphasising who is doing the action.

**31.** *(learner)* `El es rápido .`
→ `Es rápido .`
  - *U:DET*: This article does not belong here. Remove «El es».

**32.** *(synthetic)* `La carretera N-006a es la principal vía de acceso a Las Herrerías Valcarce y es la que comunica esta localidad con la capital del municipio al que pertenece : Vega de Valcarce .`
→ `La carretera N-006a es la principal vía de acceso a Las Herrerías de Valcarce y es la que comunica esta localidad con la capital del municipio al que pertenece : Vega de Valcarce .`
  - *M:ADP*: This phrase needs «de». Which preposition a Spanish verb governs is often not predictable from the English one -- «pensar en», «soñar con» -- so it has to be learned with the verb rather than translated.

**33.** *(synthetic)* `El gobierno está formado desde 2010 por unas coalición de centro-izquierda de 35 concejales de 3 partidos diferentes : Socialdemócrata , Verde e Izquierda ( con 29 , 4 y dos concejales , respectivamente ) .`
→ `El gobierno está formado desde 2010 por una coalición de centro-izquierda de 35 concejales de 3 partidos diferentes : Socialdemócrata , Verde e Izquierda ( con 29 , 4 y dos concejales , respectivamente ) .`
  - *R:DET:FORM*: The article has to agree with its noun in gender and number. «unas» does not fit here; the form the sentence needs is «una».

**34.** *(synthetic)* `La Balón de Oro , en las primera con peligro que tuvo , disparo ajustado con la zurda y puso el 1-0 .`
→ `La Balón de Oro , en la primera con peligro que tuvo , disparó ajustado con la zurda y puso el 1-0 .`
  - *R:DET:FORM*: The article has to agree with its noun in gender and number. «las» does not fit here; the form the sentence needs is «la».
  - *R:ORTH*: The stress mark is part of the word, not decoration on it. «disparo» and «disparó» are written differently for a reason -- Spanish marks stress in writing precisely where it would otherwise be unpredictable. Write «disparó».

**35.** *(synthetic)* `Aprenderemos sobre las diferentes especies de aves del paraíso y los lugares donde se pueden encontrar , así como su estatus conservación y los esfuerzos para proteger estas hermosas aves .`
→ `Aprenderemos sobre las diferentes especies de aves del paraíso y los lugares donde se pueden encontrar , así como su estatus de conservación y los esfuerzos para proteger estas hermosas aves .`
  - *M:ADP*: This phrase needs «de». Which preposition a Spanish verb governs is often not predictable from the English one -- «pensar en», «soñar con» -- so it has to be learned with the verb rather than translated.

**36.** *(synthetic)* `Aunque el pingüinos de las Snares actualmente no están amenazados , son clasificados como especie vulnerable .`
→ `Aunque los pingüinos de las Snares actualmente no están amenazados , están clasificados como especie vulnerable .`
  - *R:DET:FORM*: The article has to agree with its noun in gender and number. «el» does not fit here; the form the sentence needs is «los».
  - *R:AUX*: Spanish has two verbs for English "to be", and they are not interchangeable. This is «están», not «son».

**37.** *(learner)* `Fue una historia horrible porque durante mi vacacion , tuve que preocuparme con mi carro que necesitaba ayuda .`
→ `Fue una historia horrible porque durante mis vacaciones , tuve que preocuparme de mi carro que necesitaba ayuda .`
  - *R:DET:FORM*: The article has to agree with its noun in gender and number. Here that makes it «mis», not «mi».
  - *R:SPELL*: «vacacion» is not spelled correctly. It should be «vacaciones».
  - *R:ADP*: Spanish prepositions are not chosen by translating the English one. This phrase takes «de» rather than «con».

**38.** *(learner)* `Otras veces salimos al centro de la ciudad en el que comimos a los restaurantes y las heladerías .`
→ `Otras veces salíamos al centro de la ciudad en el que comíamos en los restaurantes y en las heladerías .`
  - *R:VERB*: «salimos» is not the right verb here. Spanish uses «salíamos» in this context.
  - *R:VERB*: «comimos» is not the right verb here. Spanish uses «comíamos» in this context.
  - *R:ADP*: Spanish prepositions are not chosen by translating the English one. This phrase takes «en» rather than «a».
  - *M:ADP*: A preposition is missing. This construction needs «en».

**39.** *(learner)* `Ahora , * FIRST_NAME * no pierde habilidades rápidamente y es la más contenta persona conozco .`
→ `Ahora , * FIRST_NAME * no pierde habilidades rápidamente y es la persona más contenta conozco .`
  - *R:WO*: This needs to be «persona más contenta» rather than «más contenta persona».

**40.** *(synthetic)* `En ese momento la Colegiata fue abandonada y los canónigos desaparecer .`
→ `En ese momento la Colegiata fue abandonada y los canónigos desaparecieron .`
  - *R:VERB:FORM*: «desaparecer» is the infinitive -- the form a dictionary lists. A finite verb has to agree with its subject, so this sentence needs «desaparecieron».

**41.** *(learner)* `Era muy frustrante al principio pero se hizo muy provechoso con el tiempo .`
→ `Era muy frustrante al principio , pero resultó muy provechoso con el tiempo .`
  - *M:PUNCT*: A punctuation mark is missing here. Spanish punctuates by rule rather than by ear -- including marking a question or exclamation at the opening «¿» or «¡» as well as the close, which English does not -- and this sentence needs «,».
  - *R:OTHER*: «se hizo» is not the right word here. Spanish uses «resultó» in this context.

**42.** *(learner)* `Tengo * PLACE * años , pero se siento que soy una niña .`
→ `Tengo * PLACE * años , pero siento que soy una niña .`
  - *U:PRON*: Spanish verb endings already show who the subject is. «se» is not needed here -- keep a subject pronoun only when the sentence is deliberately contrasting or emphasising who is doing the action.

**43.** *(synthetic)* `Este factor incluir cualquier cosa que pueda comprometer conocimiento .`
→ `Este factor incluye cualquier cosa que pueda comprometer el conocimiento .`
  - *R:VERB:FORM*: «incluir» is the infinitive -- the form a dictionary lists. A finite verb has to agree with its subject, so this sentence needs «incluye».
  - *M:DET*: Spanish nouns almost always need an article, even where English drops it. Add «el» before the noun.

**44.** *(synthetic)* `Finalmente , Knight le enseña a Mitch como disfrutarse a si mismo y vivir en campus sin " quemarse " .`
→ `Finalmente , Knight le enseña a Mitch como disfrutarse a sí mismo y vivir en el campus sin " quemarse " .`
  - *R:ORTH*: The stress mark is part of the word, not decoration on it. «si» and «sí» are written differently for a reason -- Spanish marks stress in writing precisely where it would otherwise be unpredictable. Write «sí».
  - *M:DET*: Spanish nouns almost always need an article, even where English drops it. Add «el» before the noun.

**45.** *(synthetic)* `Habian permanecido casados trece años , sin duda , mas exitosas de la carrera militar de Bonaparte .`
→ `Habían permanecido casados trece años , sin duda , los más exitosos de la carrera militar de Bonaparte .`
  - *R:ORTH*: The stress mark is part of the word, not decoration on it. «Habian» and «Habían» are written differently for a reason -- Spanish marks stress in writing precisely where it would otherwise be unpredictable. Write «Habían».
  - *M:DET*: Spanish nouns almost always need an article, even where English drops it. Add «los» before the noun.
  - *R:ORTH*: The stress mark is part of the word, not decoration on it. «mas» and «más» are written differently for a reason -- Spanish marks stress in writing precisely where it would otherwise be unpredictable. Write «más».
  - *R:ADJ:FORM*: An adjective has to agree with the noun it describes in gender and number. Here it should read «exitosos», not «exitosas».

**46.** *(learner)* `* FIRST_NAME * graduado colegio en tres años con un grado en financia .`
→ `* FIRST_NAME * se graduó del colegio hace tres años con un grado en finanzas .`
  - *R:OTHER*: «graduado» is not the right word here. Spanish uses «se graduó del» in this context.
  - *R:OTHER*: «en» is not the right word here. Spanish uses «hace» in this context.
  - *R:NOUN*: «financia» is not the right noun here. Spanish uses «finanzas» in this context.

**47.** *(learner)* `Yo no estoy contenta .`
→ `No estoy contenta .`
  - *U:PRON*: Spanish verb endings already show who the subject is. «Yo no» is not needed here -- keep a subject pronoun only when the sentence is deliberately contrasting or emphasising who is doing the action.

**48.** *(learner)* `Ahora , ellos tienen A+'s porque ellos todos pasaron mucho tiempo con estudiar antes de examen .`
→ `Están muy felices y están entusiasmados por el español .`
  - *R:OTHER*: «Ahora , ellos tienen A+'s porque ellos todos pasaron» is not the right word here. Spanish uses «Están» in this context.
  - *R:MORPH*: This needs to be «muy» rather than «mucho».
  - *R:OTHER*: «tiempo» is not the right word here. Spanish uses «felices» in this context.
  - *R:OTHER*: «con» is not the right word here. Spanish uses «y» in this context.
  - *R:OTHER*: «estudiar» is not the right word here. Spanish uses «están» in this context.
  - *R:OTHER*: «antes» is not the right word here. Spanish uses «entusiasmados por» in this context.
  - *R:OTHER*: «de» is not the right word here. Spanish uses «el» in this context.
  - *R:NOUN*: «examen» is not the right noun here. Spanish uses «español» in this context.

**49.** *(learner)* `Es interesante porque ella es vegana y hice yoga y por esta razón cuando visite * CITY * es muy divertido visitar restaurantes diferentes que tienen comidas veganos .`
→ `Es interesante porque ella es vegana y yo hice yoga y , por esta razón , cuando visito * CITY * es muy divertido visitar restaurantes diferentes que tienen comidas veganas .`
  - *M:PRON*: A pronoun is missing. This sentence needs «yo».
  - *M:PUNCT*: A punctuation mark is missing here. Spanish punctuates by rule rather than by ear -- including marking a question or exclamation at the opening «¿» or «¡» as well as the close, which English does not -- and this sentence needs «,».
  - *M:PUNCT*: A punctuation mark is missing here. Spanish punctuates by rule rather than by ear -- including marking a question or exclamation at the opening «¿» or «¡» as well as the close, which English does not -- and this sentence needs «,».
  - *R:OTHER*: «visite» is not the right word here. Spanish uses «visito» in this context.
  - *R:ADJ*: «veganos» is not the right adjective here. Use «veganas».

**50.** *(learner)* `juego el baloncesto .`
→ `Juego al baloncesto .`
  - *R:ORTH*: This is a convention of written Spanish rather than a grammar rule. The correct written form here is «Juego».
  - *R:OTHER*: «el» is not the right word here. Spanish uses «al» in this context.

**51.** *(learner)* `Todas las clases , nosotros juegan fútbol por divertido y para mejorar nuestros saludes .`
→ `Todas las clases , jugamos al fútbol para divertirnos y para mejorar nuestra salud .`
  - *R:OTHER*: «nosotros juegan» is not the right word here. Spanish uses «jugamos al» in this context.
  - *R:ADP*: Spanish prepositions are not chosen by translating the English one. This phrase takes «para» rather than «por».
  - *R:OTHER*: «divertido» is not the right word here. Spanish uses «divertirnos» in this context.
  - *R:DET:FORM*: The article has to agree with its noun in gender and number. Here that makes it «nuestra», not «nuestros».
  - *R:NOUN:FORM*: The noun is in the wrong form. Spanish marks gender and number on the noun itself, so it should be «salud».

**52.** *(synthetic)* `Nominados a los oremios BAFTA 2023 : Ana de Armas , candidata a mejor actriz por ' Blonde ' Academia Britanica de Cine ha anunciado los nominados a premios BAFTA 2023 .`
→ `Nominados a los premios BAFTA 2023 : Ana de Armas , candidata a mejor actriz por ' Blonde ' La Academia Británica de Cine ha anunciado los nominados a los premios BAFTA 2023 .`
  - *R:SPELL*: «oremios» is not a Spanish word. It looks like a slip for «premios».
  - *M:DET*: Spanish nouns almost always need an article, even where English drops it. Add «La» before the noun.
  - *R:ORTH*: The stress mark is part of the word, not decoration on it. «Britanica» and «Británica» are written differently for a reason -- Spanish marks stress in writing precisely where it would otherwise be unpredictable. Write «Británica».
  - *M:DET*: Spanish nouns almost always need an article, even where English drops it. Add «los» before the noun.

**53.** *(synthetic)* `Conoce cómo estará tiempo en Dallas , Texas , para hoy sábado 30 de septiembre .`
→ `Conoce cómo estará el tiempo en Dallas , Texas , para hoy sábado 30 de septiembre .`
  - *M:DET*: Spanish nouns almost always need an article, even where English drops it. Add «el» before the noun.

**54.** *(synthetic)* `En marzo , Scarlett Camberos denunciar en redes sociales a un hombre que ha creado perfiles falsos suyos Instagram y rwitter su foto y nombre .`
→ `En marzo , Scarlett Camberos denunció en redes sociales a un hombre que ha creado perfiles falsos suyos en Instagram y Twitter con su foto y nombre .`
  - *R:VERB:FORM*: «denunciar» is the infinitive -- the form a dictionary lists. A finite verb has to agree with its subject, so this sentence needs «denunció».
  - *M:ADP*: This phrase needs «en». Which preposition a Spanish verb governs is often not predictable from the English one -- «pensar en», «soñar con» -- so it has to be learned with the verb rather than translated.
  - *R:SPELL*: «rwitter» is not a Spanish word. It looks like a slip for «Twitter».
  - *M:ADP*: This phrase needs «con». Which preposition a Spanish verb governs is often not predictable from the English one -- «pensar en», «soñar con» -- so it has to be learned with the verb rather than translated.

**55.** *(learner)* `El puede ser un de mayores cómicos en historia .`
→ `Puede ser uno de mayores cómicos en la historia .`
  - *U:DET*: This article does not belong here. Remove «El puede».
  - *R:PRON:FORM*: This pronoun is in the wrong form. It should be «uno».
  - *M:DET*: A Spanish noun almost always needs an article, even where English drops it. Add «la».

**56.** *(synthetic)* `No cometas los mismos errores que cometiste en el pasado , está momento que tú asumas tus consecuencias .`
→ `No cometas los mismos errores que cometiste en el pasado , es momento que asumas tus consecuencias .`
  - *R:AUX*: Spanish has two verbs for English "to be", and they are not interchangeable. This is «es», not «está».
  - *U:PRON*: Spanish verb endings already say who the subject is, so the pronoun is not needed here the way it would be in English. Remove «tú» -- or keep it only if the sentence is deliberately emphasising or contrasting who is doing the action.

**57.** *(learner)* `Se fue y instamente me sentí triste .`
→ `Se fue e instantáneamente me sentí triste .`
  - *R:CONJ*: «y» is not the right conjunction here. Use «e».
  - *R:SPELL*: «instamente» is not spelled correctly. It should be «instantáneamente».

**58.** *(learner)* `Espero que si tenga hijos , ellos tengan una experiencia como eso de mi hermana y yo .`
→ `Espero que , si tengo hijos , tengan una experiencia como esa de mi hermana y yo .`
  - *M:PUNCT*: A punctuation mark is missing here. Spanish punctuates by rule rather than by ear -- including marking a question or exclamation at the opening «¿» or «¡» as well as the close, which English does not -- and this sentence needs «,».
  - *R:VERB:FORM*: The verb is in the wrong form. Check the tense, the person, and whether the sentence needs the indicative or the subjunctive. It should be «tengo».
  - *U:PRON*: Spanish verb endings already show who the subject is. «ellos» is not needed here -- keep a subject pronoun only when the sentence is deliberately contrasting or emphasising who is doing the action.
  - *R:MORPH*: This needs to be «esa» rather than «eso».

**59.** *(synthetic)* `Además , algunos temen que referendos puedan ser manipulados por lideres políticos o utilizados para promover agendas divisivas .`
→ `Además , algunos temen que los referendos puedan ser manipulados por líderes políticos o utilizados para promover agendas divisivas .`
  - *M:DET*: Spanish nouns almost always need an article, even where English drops it. Add «los» before the noun.
  - *R:ORTH*: The stress mark is part of the word, not decoration on it. «lideres» and «líderes» are written differently for a reason -- Spanish marks stress in writing precisely where it would otherwise be unpredictable. Write «líderes».

**60.** *(synthetic)* `Prosperi quedó de segundo lugar al obtener 4,75% del escrutinio ( 28.153 votos ) , de acuerdo con las resultado preliminar .`
→ `Prosperi quedó de segundo lugar al obtener 4,75% del escrutinio ( 28.153 votos ) , de acuerdo con el resultado preliminar .`
  - *R:DET:FORM*: The article has to agree with its noun in gender and number. «las» does not fit here; the form the sentence needs is «el».

**61.** *(synthetic)* `Un oficial del Estado Mayor observa que esos ataques en noche y con bruma son demasiados frecuentes como para ser casualidad .`
→ `Un oficial del Estado Mayor observa que esos ataques en la noche y con bruma son demasiados frecuentes como para ser casualidad .`
  - *M:DET*: Spanish nouns almost always need an article, even where English drops it. Add «la» before the noun.

**62.** *(synthetic)* `El Cabildo a La Gomera mantiene abierto el olazo de aolicitud de becas para estudiantes de Erasmus , Sicue o cualquier otro programa de movilidad europeo .`
→ `El Cabildo de La Gomera mantiene abierto el plazo de solicitud de becas para estudiantes de Erasmus , Sicue o cualquier otro programa de movilidad europeo .`
  - *R:ADP*: Spanish prepositions are not chosen by translating the English one. This phrase takes «de» rather than «a».
  - *R:SPELL*: «olazo» is not a Spanish word. It looks like a slip for «plazo».
  - *R:SPELL*: «aolicitud» is not a Spanish word. It looks like a slip for «solicitud».

**63.** *(learner)* `Está relajada .`
→ `Estoy relajada .`
  - *R:AUX:FORM*: The auxiliary is in the wrong form. It should be «Estoy».

**64.** *(synthetic)* `« Las condiciones xlimáticas no eran ideales » , dijo los teniente comodoro Jeb Slick , copiloto de la misión , quien destacó que el rescate se logró en gran medida gracias a « la excelente ética de trabajo y la dedicacion al entrenamiento » los guardacostas .`
→ `« Las condiciones climáticas no eran ideales » , dijo el teniente comodoro Jeb Slick , copiloto de la misión , quien destacó que el rescate se logró en gran medida gracias a « la excelente ética de trabajo y la dedicación al entrenamiento » de los guardacostas .`
  - *R:SPELL*: «xlimáticas» is not a Spanish word. It looks like a slip for «climáticas».
  - *R:DET:FORM*: The article has to agree with its noun in gender and number. «los» does not fit here; the form the sentence needs is «el».
  - *R:ORTH*: The stress mark is part of the word, not decoration on it. «dedicacion» and «dedicación» are written differently for a reason -- Spanish marks stress in writing precisely where it would otherwise be unpredictable. Write «dedicación».
  - *M:ADP*: This phrase needs «de». Which preposition a Spanish verb governs is often not predictable from the English one -- «pensar en», «soñar con» -- so it has to be learned with the verb rather than translated.

**65.** *(learner)* `Yo conocido él para diez mesas .`
→ `Yo le conozco a él desde hace diez meses .`
  - *M:PRON*: A pronoun is missing. This sentence needs «le».
  - *R:OTHER*: «conocido» is not the right word here. Spanish uses «conozco» in this context.
  - *M:ADP*: A preposition is missing. This construction needs «a».
  - *R:OTHER*: «para» is not the right word here. Spanish uses «desde hace» in this context.
  - *R:NOUN*: «mesas» is not the right noun here. Spanish uses «meses» in this context.

**66.** *(learner)* `Ella fue bajo y joven así que toma un poco pero , después de un rato me llegó y me dio la flor con su brazo extendido .`
→ `Era baja y joven , así que tomó un poco pero , después de un rato , me alcanzó y me dio la flor con su brazo extendido .`
  - *R:OTHER*: «Ella fue» is not the right word here. Spanish uses «Era» in this context.
  - *R:ADJ:FORM*: An adjective has to agree with the noun it describes in gender and number. It should read «baja».
  - *M:PUNCT*: A punctuation mark is missing here. Spanish punctuates by rule rather than by ear -- including marking a question or exclamation at the opening «¿» or «¡» as well as the close, which English does not -- and this sentence needs «,».
  - *R:VERB:FORM*: The verb is in the wrong form. Check the tense, the person, and whether the sentence needs the indicative or the subjunctive. It should be «tomó».
  - *M:PUNCT*: A punctuation mark is missing here. Spanish punctuates by rule rather than by ear -- including marking a question or exclamation at the opening «¿» or «¡» as well as the close, which English does not -- and this sentence needs «,».
  - *R:VERB*: «llegó» is not the right verb here. Spanish uses «alcanzó» in this context.

**67.** *(learner)* `Cuando ella quería irse el chico disculpó y la besó .`
→ `Cuando ella quería irse el chico se disculpó y la besó .`
  - *M:PRON*: A pronoun is missing. This sentence needs «se».

**68.** *(synthetic)* `El procedimiento consiste en una inyeccion que se aplicar semanalmente , a diferencia de otros farmacos que se inoculan de forma diaria .`
→ `El procedimiento consiste en una inyección que se aplica semanalmente , a diferencia de otros fármacos que se inoculan de forma diaria .`
  - *R:ORTH*: The stress mark is part of the word, not decoration on it. «inyeccion» and «inyección» are written differently for a reason -- Spanish marks stress in writing precisely where it would otherwise be unpredictable. Write «inyección».
  - *R:VERB:FORM*: «aplicar» is the infinitive -- the form a dictionary lists. A finite verb has to agree with its subject, so this sentence needs «aplica».
  - *R:ORTH*: The stress mark is part of the word, not decoration on it. «farmacos» and «fármacos» are written differently for a reason -- Spanish marks stress in writing precisely where it would otherwise be unpredictable. Write «fármacos».

**69.** *(learner)* `Una persona importante en mi vida es mi mentor de la música y amiga , * FIRST_NAME * .`
→ `Una persona importante en mi vida es mi mentora en la música y amiga , * FIRST_NAME * .`
  - *R:NOUN*: «mentor» is not the right noun here. Spanish uses «mentora» in this context.
  - *R:ADP*: Spanish prepositions are not chosen by translating the English one. This phrase takes «en» rather than «de».

**70.** *(synthetic)* `Otro elemento de relevancia lo es el Parque Lineal de Martin Peña , el cual se reconocio mediante unos pared ondulante de cristal usadas en ambas entradas .`
→ `Otro elemento de relevancia lo es el Parque Lineal de Martín Peña , el cual se reconoció mediante una pared ondulante de cristal usada en ambas entradas .`
  - *R:ORTH*: The stress mark is part of the word, not decoration on it. «Martin» and «Martín» are written differently for a reason -- Spanish marks stress in writing precisely where it would otherwise be unpredictable. Write «Martín».
  - *R:ORTH*: The stress mark is part of the word, not decoration on it. «reconocio» and «reconoció» are written differently for a reason -- Spanish marks stress in writing precisely where it would otherwise be unpredictable. Write «reconoció».
  - *R:DET:FORM*: The article has to agree with its noun in gender and number. «unos» does not fit here; the form the sentence needs is «una».
  - *R:ADJ:FORM*: An adjective has to agree with the noun it describes in gender and number. Here it should read «usada», not «usadas».

**71.** *(synthetic)* `realización premiada de el Sección en Competencia para Cortometrajes Entrerrianos , fue para Pescantora , de Fabio Marcelo Bonell .`
→ `La realización premiada de la Sección en Competencia para Cortometrajes Entrerrianos , fue para Pescantora , de Fabio Marcelo Bonell .`
  - *M:DET*: Spanish nouns almost always need an article, even where English drops it. Add «La» before the noun.
  - *R:DET:FORM*: The article has to agree with its noun in gender and number. «el» does not fit here; the form the sentence needs is «la».

**72.** *(synthetic)* `En cualquier caso , este descenso en los margenes de la actividad productiva y comercializadora « ha sido compensada parcialmente con el buen desempeño de la unidad de flota durante el año » , wxplican desde la empresa .`
→ `En cualquier caso , este descenso en los márgenes de la actividad productiva y comercializadora « ha sido compensada parcialmente con el buen desempeño de la unidad de flota durante el año » , explican desde la empresa .`
  - *R:ORTH*: The stress mark is part of the word, not decoration on it. «margenes» and «márgenes» are written differently for a reason -- Spanish marks stress in writing precisely where it would otherwise be unpredictable. Write «márgenes».
  - *R:SPELL*: «wxplican» is not a Spanish word. It looks like a slip for «explican».

**73.** *(synthetic)* `El plantel argentino se alzar el título al vencer a todos sus xontendientes .`
→ `El plantel argentino se alzó con el título al vencer a todos sus contendientes .`
  - *R:VERB:FORM*: «alzar» is the infinitive -- the form a dictionary lists. A finite verb has to agree with its subject, so this sentence needs «alzó».
  - *M:ADP*: This phrase needs «con». Which preposition a Spanish verb governs is often not predictable from the English one -- «pensar en», «soñar con» -- so it has to be learned with the verb rather than translated.
  - *R:SPELL*: «xontendientes» is not a Spanish word. It looks like a slip for «contendientes».

**74.** *(synthetic)* `El nunca existió , nunca estuvo en ninguna parte de la casa , ni cantando en el bar , él era parte de los memoria de la joven y solo estuvo ahí porque ella imagino .`
→ `Él nunca existió , nunca estuvo en ninguna parte de la casa , ni cantando en el bar , él era parte de la memoria de la joven y solo estuvo ahí porque ella lo imaginó .`
  - *R:ORTH*: The stress mark is part of the word, not decoration on it. «El» and «Él» are written differently for a reason -- Spanish marks stress in writing precisely where it would otherwise be unpredictable. Write «Él».
  - *R:DET:FORM*: The article has to agree with its noun in gender and number. «los» does not fit here; the form the sentence needs is «la».
  - *M:PRON*: An object pronoun is missing here. Spanish does not leave it out the way some constructions in English do; add «lo».
  - *R:ORTH*: The stress mark is part of the word, not decoration on it. «imagino» and «imaginó» are written differently for a reason -- Spanish marks stress in writing precisely where it would otherwise be unpredictable. Write «imaginó».

**75.** *(learner)* `Hoy pienso que mi espanol todavia no es suficente , y por un parte si es que no hay praticado el lengua por mucho tiempo pero por el otro mano es porque no tiengo confidencia en yo misma .`
→ `Hoy pienso que mi español todavía no es suficientemente bueno , y por una parte sí es porque no he practicado durante mucho tiempo , pero por otro lado es porque no tengo confianza en mí misma .`
  - *R:SPELL*: «espanol» is not spelled correctly. It should be «español».
  - *R:ORTH*: This is a convention of written Spanish rather than a grammar rule. The correct written form here is «todavía».
  - *R:OTHER*: «suficente» is not the right word here. Spanish uses «suficientemente bueno» in this context.
  - *R:DET:FORM*: The article has to agree with its noun in gender and number. Here that makes it «una», not «un».
  - *R:ORTH*: This is a convention of written Spanish rather than a grammar rule. The correct written form here is «sí».
  - *R:SCONJ*: «que» is not the right subordinating conjunction here. Use «porque».
  - *R:AUX:FORM*: The auxiliary is in the wrong form. It should be «he».
  - *R:SPELL*: «praticado» is not spelled correctly. It should be «practicado».
  - *R:OTHER*: «el lengua por» is not the right word here. Spanish uses «durante» in this context.
  - *M:PUNCT*: A punctuation mark is missing here. Spanish punctuates by rule rather than by ear -- including marking a question or exclamation at the opening «¿» or «¡» as well as the close, which English does not -- and this sentence needs «,».
  - *U:DET*: This article does not belong here. Remove «el».
  - *R:NOUN*: «mano» is not the right noun here. Spanish uses «lado» in this context.
  - *R:VERB:FORM*: The verb is in the wrong form. Check the tense, the person, and whether the sentence needs the indicative or the subjunctive. It should be «tengo».
  - *R:NOUN*: «confidencia» is not the right noun here. Spanish uses «confianza» in this context.
  - *R:PRON:FORM*: This pronoun is in the wrong form. It should be «mí».

**76.** *(learner)* `Ojalá que esté el mismo este trimestre .`
→ `Ojalá que sea lo mismo este trimestre .`
  - *R:AUX*: Spanish has two verbs for English "to be", «ser» and «estar», and they are not interchangeable. This is «sea», not «esté».
  - *R:OTHER*: «el» is not the right word here. Spanish uses «lo» in this context.

**77.** *(learner)* `Ella tiene una voz muy singular que la distingue de otros artistas y siempre es fácil reconocerla .`
→ `Tiene una voz muy singular que la distingue de otros artistas y siempre es fácil reconocerla .`
  - *U:PRON*: Spanish verb endings already show who the subject is. «Ella tiene» is not needed here -- keep a subject pronoun only when the sentence is deliberately contrasting or emphasising who is doing the action.

**78.** *(learner)* `Por suerte mi profesor todavía estaba en clase .`
→ `Por suerte mi profesora todavía estaba en clase .`
  - *R:NOUN*: «profesor» is not the right noun here. Spanish uses «profesora» in this context.

**79.** *(synthetic)* `El verde y el verdadero Albert unen sus fuerzas para destruir el aparato y desactivar las armas biologicas que se han utilizado en un ataque contra una ciudad cercana .`
→ `El verde y el verdadero Albert unen sus fuerzas para destruir el aparato y desactivar las armas biológicas que se han utilizado en un ataque contra una ciudad cercana .`
  - *R:ORTH*: The stress mark is part of the word, not decoration on it. «biologicas» and «biológicas» are written differently for a reason -- Spanish marks stress in writing precisely where it would otherwise be unpredictable. Write «biológicas».

**80.** *(synthetic)* `Su primer gol en la temporada le significó la salvacion matemática su grupo .`
→ `Su primer gol en la temporada le significó la salvación matemática a su grupo .`
  - *R:ORTH*: The stress mark is part of the word, not decoration on it. «salvacion» and «salvación» are written differently for a reason -- Spanish marks stress in writing precisely where it would otherwise be unpredictable. Write «salvación».
  - *M:ADP*: This phrase needs «a». Which preposition a Spanish verb governs is often not predictable from the English one -- «pensar en», «soñar con» -- so it has to be learned with the verb rather than translated.

**81.** *(learner)* `Hoy en día , yo puedo decir , con orgullo , que admiro a mi madre por su coraje , su valentía , y por su voluntad se sacrificar todo por mi hermana y yo .`
→ `Hoy en día , yo puedo decir , con orgullo , que admiro a mi madre por su coraje , su valentía , y por su voluntad de sacrificar todo por mi hermana y por mí .`
  - *R:OTHER*: «se» is not the right word here. Spanish uses «de» in this context.
  - *M:ADP*: A preposition is missing. This construction needs «por».
  - *R:PRON:FORM*: This pronoun is in the wrong form. It should be «mí».

**82.** *(learner)* `Mi gusta ver Netflix , comprando con mi amigas y hermanos , y ir a la playa .`
→ `Me gusta ver Netflix , comprar con mi amigas y hermanos e ir a la playa .`
  - *R:OTHER*: «Mi» is not the right word here. Spanish uses «Me» in this context.
  - *R:VERB:FORM*: The verb is in the wrong form. Check the tense, the person, and whether the sentence needs the indicative or the subjunctive. It should be «comprar».
  - *U:PUNCT*: This punctuation mark does not belong here.
  - *R:CONJ*: «y» is not the right conjunction here. Use «e».

**83.** *(synthetic)* `Encontramos en ellos vendedores de todos los rincones del mundo , haciendo hincapié en un pabellón central donde se encuentran los comerciantes con los minerales más espectaculares , lo que soler llamar : zona VIP .`
→ `Encontramos en ellos vendedores de todos los rincones del mundo , haciendo hincapié en un pabellón central donde se encuentran los comerciantes con los minerales más espectaculares , lo que solemos llamar : zona VIP .`
  - *R:VERB:FORM*: «soler» is the infinitive -- the form a dictionary lists. A finite verb has to agree with its subject, so this sentence needs «solemos».

**84.** *(learner)* `Pero el gente de North Dakota es mas alto como Wyoming .`
→ `Pero la gente de North Dakota es más alta que en Wyoming .`
  - *R:DET:FORM*: The article has to agree with its noun in gender and number. Here that makes it «la», not «el».
  - *R:ORTH*: This is a convention of written Spanish rather than a grammar rule. The correct written form here is «más».
  - *R:ADJ:FORM*: An adjective has to agree with the noun it describes in gender and number. It should read «alta».
  - *R:SCONJ*: «como» is not the right subordinating conjunction here. Use «que».
  - *M:ADP*: A preposition is missing. This construction needs «en».

**85.** *(synthetic)* `Una de los edificios que más llama la atención en el identificado como 10 de la que sobresale su obra escultural a un jaguar ( símbolo de autoridad de realeza maya ) Artículo “ El Rastrojón ” la nueva atraccion del mundo maya en Copán Ruinas .`
→ `Una de los edificios que más llama la atención en el identificado como 10 de la que sobresale su obra escultural de un jaguar ( símbolo de autoridad de la realeza maya ) Artículo “ El Rastrojón ” la nueva atracción del mundo maya en Copán Ruinas .`
  - *R:ADP*: Spanish prepositions are not chosen by translating the English one. This phrase takes «de» rather than «a».
  - *M:DET*: Spanish nouns almost always need an article, even where English drops it. Add «la» before the noun.
  - *R:ORTH*: The stress mark is part of the word, not decoration on it. «atraccion» and «atracción» are written differently for a reason -- Spanish marks stress in writing precisely where it would otherwise be unpredictable. Write «atracción».

**86.** *(synthetic)* `Sin esta frase indirecta , Georgina confirmó que su relación con el futbolista portugués seguir igual bien que siempre .`
→ `Con esta frase indirecta , Georgina confirmó que su relación con el futbolista portugués sigue igual de bien que siempre .`
  - *R:ADP*: Spanish prepositions are not chosen by translating the English one. This phrase takes «Con» rather than «Sin».
  - *R:VERB:FORM*: «seguir» is the infinitive -- the form a dictionary lists. A finite verb has to agree with its subject, so this sentence needs «sigue».
  - *M:ADP*: This phrase needs «de». Which preposition a Spanish verb governs is often not predictable from the English one -- «pensar en», «soñar con» -- so it has to be learned with the verb rather than translated.

**87.** *(synthetic)* `Lo que no toman en cuenta es que dentro de ese bosque acecha asesino al que le agradan los campistas , y estos pronto empezarán caer uno por uno .`
→ `Lo que no toman en cuenta es que dentro de ese bosque acecha un asesino al que le agradan los campistas , y estos pronto empezarán a caer uno por uno .`
  - *M:DET*: Spanish nouns almost always need an article, even where English drops it. Add «un» before the noun.
  - *M:ADP*: This phrase needs «a». Which preposition a Spanish verb governs is often not predictable from the English one -- «pensar en», «soñar con» -- so it has to be learned with the verb rather than translated.

**88.** *(synthetic)* `Del otro lado tampoco escatimarían en criticas , ya que el fundador de Apple lo trató de sinvergüenza .`
→ `Del otro lado tampoco escatimarían en críticas , ya que el fundador de Apple lo trató de sinvergüenza .`
  - *R:ORTH*: The stress mark is part of the word, not decoration on it. «criticas» and «críticas» are written differently for a reason -- Spanish marks stress in writing precisely where it would otherwise be unpredictable. Write «críticas».

**89.** *(learner)* `" De repente , todo se hace clic .`
→ `Si usted la hubiera pedido amablemente , tal vez yo miraría para otro lado .`
  - *R:OTHER*: «"» is not the right word here. Spanish uses «Si» in this context.
  - *R:OTHER*: «De repente» is not the right word here. Spanish uses «usted la hubiera pedido amablemente» in this context.
  - *M:NOUN*: A noun is missing here. Add «tal vez».
  - *R:PRON*: «todo se» is not the right pronoun here. Use «yo».
  - *R:OTHER*: «hace clic» is not the right word here. Spanish uses «miraría para otro lado» in this context.

**90.** *(synthetic)* `Es considerado uno de el « Padres de la Medicina Canadiense » tanto en investigación médica como en educación .`
→ `Es considerado uno de los « Padres de la Medicina Canadiense » tanto en investigación médica como en educación .`
  - *R:DET:FORM*: The article has to agree with its noun in gender and number. «el» does not fit here; the form the sentence needs is «los».

**91.** *(synthetic)* `No saben que wstán siendo observados fuera de la cerca las prisión por una figura desconocida .`
→ `No saben que están siendo observados desde fuera de la cerca de la prisión por una figura desconocida .`
  - *R:SPELL*: «wstán» is not a Spanish word. It looks like a slip for «están».
  - *M:ADP*: This phrase needs «desde». Which preposition a Spanish verb governs is often not predictable from the English one -- «pensar en», «soñar con» -- so it has to be learned with the verb rather than translated.
  - *M:ADP*: This phrase needs «de». Which preposition a Spanish verb governs is often not predictable from the English one -- «pensar en», «soñar con» -- so it has to be learned with the verb rather than translated.
  - *R:DET:FORM*: The article has to agree with its noun in gender and number. «las» does not fit here; the form the sentence needs is «la».

**92.** *(synthetic)* `Gritos de angustia y agonía se mezclar con nuestro propio y enemigo enganchado de ametralladora .`
→ `Gritos de angustia y agonía se mezclaron con nuestro propio y enemigo enganchado de ametralladora .`
  - *R:VERB:FORM*: «mezclar» is the infinitive -- the form a dictionary lists. A finite verb has to agree with its subject, so this sentence needs «mezclaron».

**93.** *(synthetic)* `El cortometraje ya está en proceso de montaje y edición , cuenta con el apoyo de Aragón Televisión -ahora es Natalia Martínez quien ha asumido la tarea de tender puentes con el cine aragonés- y se estrenara en 2024 .`
→ `El cortometraje ya está en proceso de montaje y edición , cuenta con el apoyo de Aragón Televisión -ahora es Natalia Martínez quien ha asumido la tarea de tender puentes con el cine aragonés- y se estrenará en 2024 .`
  - *R:ORTH*: The stress mark is part of the word, not decoration on it. «estrenara» and «estrenará» are written differently for a reason -- Spanish marks stress in writing precisely where it would otherwise be unpredictable. Write «estrenará».

**94.** *(synthetic)* `El xomedor y la sala en lectura de los jueces también están en esta planta .`
→ `El comedor y la sala de lectura de los jueces también están en esta planta .`
  - *R:SPELL*: «xomedor» is not a Spanish word. It looks like a slip for «comedor».
  - *R:ADP*: Spanish prepositions are not chosen by translating the English one. This phrase takes «de» rather than «en».

**95.** *(synthetic)* `“ Esto significa que las emisoras donde trabajan no se transmitió ninguna información sobre el evento electoral ” , expresó Ipys .`
→ `“ Esto significa que desde las emisoras donde trabajan no se transmitió ninguna información sobre el evento electoral ” , expresó Ipys .`
  - *M:ADP*: This phrase needs «desde». Which preposition a Spanish verb governs is often not predictable from the English one -- «pensar en», «soñar con» -- so it has to be learned with the verb rather than translated.

**96.** *(synthetic)* `El nalestar en la resaca no la dejó dormir mas , así que antes de salir , bebió mucha agua , aunque no tuviera sed .`
→ `El malestar de la resaca no la dejó dormir más , así que antes de salir , bebió mucha agua , aunque no tuviera sed .`
  - *R:SPELL*: «nalestar» is not a Spanish word. It looks like a slip for «malestar».
  - *R:ADP*: Spanish prepositions are not chosen by translating the English one. This phrase takes «de» rather than «en».
  - *R:ORTH*: The stress mark is part of the word, not decoration on it. «mas» and «más» are written differently for a reason -- Spanish marks stress in writing precisely where it would otherwise be unpredictable. Write «más».

**97.** *(synthetic)* `Ahora , su proximo desafío será retar al serbio Novak Djokovic , actual ' Maestro ' , en la aemifinales .`
→ `Ahora , su próximo desafío será retar al serbio Novak Djokovic , actual ' Maestro ' , en las semifinales .`
  - *R:ORTH*: The stress mark is part of the word, not decoration on it. «proximo» and «próximo» are written differently for a reason -- Spanish marks stress in writing precisely where it would otherwise be unpredictable. Write «próximo».
  - *R:DET:FORM*: The article has to agree with its noun in gender and number. «la» does not fit here; the form the sentence needs is «las».
  - *R:SPELL*: «aemifinales» is not a Spanish word. It looks like a slip for «semifinales».

**98.** *(synthetic)* `El impacto potencial en la aalud de las PFC no es clara .`
→ `El impacto potencial en la salud de los PFC no está claro .`
  - *R:SPELL*: «aalud» is not a Spanish word. It looks like a slip for «salud».
  - *R:DET:FORM*: The article has to agree with its noun in gender and number. «las» does not fit here; the form the sentence needs is «los».
  - *R:AUX*: Spanish has two verbs for English "to be", and they are not interchangeable. This is «está», not «es».
  - *R:ADJ:FORM*: An adjective has to agree with the noun it describes in gender and number. Here it should read «claro», not «clara».

**99.** *(learner)* `Actualmente trabajo en una firma de abogados médicos , donde proceso achivos de compensación laboral .`
→ `Actualmente trabajo en una firma de abogados médicos , donde proceso archivos de compensación laboral .`
  - *R:SPELL*: «achivos» is not spelled correctly. It should be «archivos».

**100.** *(synthetic)* `Los colores del amor dar sentido de la existencia .`
→ `Los colores del amor le dan sentido a la existencia .`
  - *M:PRON*: An object pronoun is missing here. Spanish does not leave it out the way some constructions in English do; add «le».
  - *R:VERB:FORM*: «dar» is the infinitive -- the form a dictionary lists. A finite verb has to agree with its subject, so this sentence needs «dan».
  - *R:ADP*: Spanish prepositions are not chosen by translating the English one. This phrase takes «a» rather than «de».

