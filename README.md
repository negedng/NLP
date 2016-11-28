# Natural Language Processing
## Training Project Laboratory(VIAUAL00) at BME-VIK AUT
For more details: 
* Course at AUT: https://www.aut.bme.hu/Course/Temalabor
* Course at VIK: https://portal.vik.bme.hu/kepzes/targyak/VIAUAL00
* NPL: https://www.aut.bme.hu/Task/16-17-osz/Termeszetes-nyelvfeldolgozas

## Exercises
###1. feladat: Karakter és szószámlálás
- Készítsetek gyakorisági listát magyar szöveg 1. karaktereiről, 2. szavairól, ahol a szavakat úgy definiáljuk, hogy ami whitespace-szel van elválasztva (ez egy nagyon elnagyolt definíció, majd javítunk rajta).
- Adatforrás: https://dumps.wikimedia.org/huwikibooks/20160920/huwikibooks-20160920-pages-meta-current.xml.bz2
Használjátok ezt a dumpot, még akkor is, ha megjelenik újabb, különben nem összehasonlíthatóak az eredményeitek egymással.
A Mediawiki markup eltávolítható az Autocorpus nevű eszközzel: http://mpacula.com/autocorpus/
A wiki-textify modul kell nektek.
- A kimenetet írjátok fájlba csökkenő gyakorisági sorrendben. Soronként egy szó és annak gyakorisága szerepeljen tabbal elválasztva, pl:

    a\<TAB>123
      
    az\<TAB>34

- Kétféle megoldást várok:

  1. Csak beépített típusokat használsz, 

  2. Standard libraryből lehet importálni (tipp: collections modul).

- Opcionálisan lehet vizualizálni a gyakoriságokat és azok hisztogramját (ld. matplotlib).

###2. feladat: Szövegtisztítás
Végezz egyszerű szövegtisztítást az alábbi lépések végrehajtásával:

1. Kisbetűsítés

2. Központozás eltávolítása (minden, ami a string.punctuation-ben van törölhető). Ez nem egy veszteségmentes változtatás, de most túl fogjuk élni. Majd később lesz jobb megoldásunk.

3. Számok helyettesítése egy közös tokennel (_NUM_).

Készítsd el a karakter- és szógyakoriságokat a "tisztított" adaton.

Gondolkodtató kérdés: milyen normalizálásokat alkalmaznál még?

###3. feladat: Long tail eloszlás
A szavak gyakoriságánál azt tapasztaltuk, hogy az eloszlás egy ú.n. long-tailed eloszlás (direkt nem fordítottam le magyarra), ami természetesen gondot okoz a legtöbb NLP algoritmusnak, hiszen végtelen memóriát fel tudna emészteni, miközben nagyon zajossá teszi a szótárat. Egyik lehetséges megoldás az, hogy a ritkán előforduló szavakról feltételezzük, hogy nem adnak hozzá hasznos információt a szöveghez, ezért ezeket helyettesíthetjük egy közös tokennel.

Készíts függvényt, amely a bemeneti szövegben lecseréli a paraméterként kapott küszöbértéknél (rare_threshold) kevesebbszer előforduló szavakat egy közös tokenre (rare_token).
A paramétereknek legyen alapértelmezett értékük (rare_threshold legyen 5, rare_token legyen _RARE_), amit a hívó felüldefiniálhat.

Gondolkodtató kérdés: hogyan lehetne a rare_thresholdot "okosabban" meghatározni?

###4. feladat: Szótárfedés

Szótárfedés vizsgálata a magyar, és az angol korpuszon

### Végső feladat
Készíts szótárfedést vizsgáló függvényt vagy osztályt. 

A bemenet két korpusz, az egyikből szótár készül, a másikon vizsgáljuk a szótár fedését. A kimenet a lefedett szavak aránya.Végezz kíséleteket tisztított és tisztítatlan korpuszokkal is felhasználva a korábban készített egyszerű szövegtisztítót.Futtasd le a kíséleteket többféle szótármérettel és ábrázold az eredményeket oszlopdiagramon. A szótárméret 100...1000000 között logaritmikus skálán mozogjon.

A korpuszok: Webcorpus többféle nyelvre, Europarl, ItWAC.

Nyelvek: finn, olasz, francia és dán
