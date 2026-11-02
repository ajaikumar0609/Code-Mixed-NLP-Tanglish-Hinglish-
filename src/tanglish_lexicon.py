"""Hand-curated resources for Romanized Tamil (Tanglish).

These lists encode the project's annotation convention:

* ``TAMIL_WORDS``   - romanized Tamil words/particles (label ``TA``), including
  common spelling variants and Tamil words that collide with English spellings
  (``than``, ``ah``, ``mama`` ...).
* ``NAMED_ENTITIES`` - people, teams, brands and acronyms (label ``OTHER``).
* ``EN_SLANG``      - informal English chat spellings (label ``EN``).
* ``VARIANT_GROUPS`` - spelling variants -> preferred form, used by the normalizer.
* ``EN_SLANG_EXPANSIONS`` - English chat abbreviations -> standard English.

The curation was driven by the most frequent tokens of Tanglish-Corpus-185k.
"""


def _words(block: str) -> set[str]:
    return set(block.split())


TAMIL_WORDS = _words("""
naan nan naa na nee ni neenga ninga nenga neega nega naanga nanga naam naama namma nama namba nanum naanum nanba
avan ava avar avaru avanga avangala avanuku avana avara ivan ivanga ivar ivaru ivalo ivlo evlo evalo avlo avalo
enaku enakku ennaku ennakku unaku unakku unakkum ungaluku ungalukku ungala ungaloda unga un en enna yenna ena yena
ennada ennoda enoda unnoda unna ennai engaluku namaku namakku engala yen yean yenda yenga
itha ithu idhu idha athu adhu adha atha antha andha intha indha entha endha ethu edhu etha edho etho ethuku edhuku
ethana ithuku ithula idhula idhuku athula adhula athuku adhuku adutha
yaru yaaru yarum yaarum yaaruku yaaruku evvalavu eppadi epdi epadi appadi apdi apdiye ippadi ipdi ipadi
ipo ippo ipa ippa epo eppo apo appo apram aprm apram aprom appuram inga inge anga ange enga ingaye engaye
mela keela munnadi pinnadi pinnaadi veliya ulla kulla mele
iruku irukku irukum irukkum iruka irukka irukan iruken irukinga irukeenga irukanga irukaru irukura irukalam irukanum
irukom irukaa irukaanga irunthu irundhu iruntha irundha irunthalum irunthaalum irukalaam irukena irukkena
irruku eruku erukku erukum eruka erukkum erukaru irku irukkuthu irukuthu irukkuma irukuma erukkum irukeen
illa illai ila illama illana illaya illaiya ilama
panna pana pannu pannunga panunga pannuga panuga pannitu pannittu panitu pannanum pananum pannalam pannala panala
panra pandra panren pandren panringa pandringa panranga pandrathu panrathu pannum panum pannuvom pannuven
panni pani paniye pannita pannitaan panninaan panniyum pandratha pannura pandra pannuren pannudhu
poi po pona poga pogum poguthu poitu poittu poda podu podunga potu pottu potta pota podra podhu pothum pothu pora poranga
poi poitaan ponga ponathu pogalam pogathe pogatha podhum
vara varum varuthu varudhu varathu varala varanum varuma vantha vandha vanthu vandhu vaanga vanga vaa va vanthuru
varen varuven varuvom varalam varudhu varuvaan varuvaanga vandhaachu vandhuduchu vantheen varuvaaru
solla sollu sollunga solunga sonna solli sollitu sollittu solra soldra solluga sollala solluven sollren solren
sollalam sollathinga sollatha sollanum sollama sollapora soldrathu solrathu
paakura paakkura paatha patha pathu paathu paathutu pathutu paaka paka paaru paru paarunga parunga pakka paakka paarthu
paathen paathiya paathiya paakalam paarkalam paakanum paakama paaththu pathutu paathuten
theriyum theriyala therila theriyathu theriyadhu theriyama therium therla theriyuma theriyudhu theriyalaye
puriyala puriyuthu purinjathu purila puriyuma
mudiyala mudiyathu mudiyum mudila mudiyadhu mudiyaathu mudinjathu mudiyuma mudiyalana mudincha
venum vendam venam vena vendaam vendaam vendiyathu venumna venuma
kudunga kuduthu kudutha kudathu koodathu kudu koduthu kodutha kodunga kuduthaan kudukalam
kandipa kandippa kandipaa kandipaa kandippaa kandipaaga
kooda kuda kooda koodave kudave
mattum matum mattume matume
mathiri madhiri maathiri mathri madhri maadhiri maari mari mathri maathri
neraya neraiya neraya konjam konja konjum konjamaa sila palar pala romba rombha rompa mikavum
periya perusa chinna chinnadha nalla nala nallaa mokka semma sema semmaa vera verum summa suma
oru onnu onu onnum onnumey rendu irandu moonu munu naalu aaru ezhu ettu onbathu pathu patthu nooru aayiram
oda oda ooda ooda udane udaney
la laye layum lam laam lamum lum kum ku kku ukku ah aa aah ahh ha eh oh uh dei dai da di daa ya yaa yaaa
pa ma ka ra ta ne nu nuu nu nna nna thaan than thaa tha dhan dhaan dhana thana thane athan dha
um ae ey ye yo yov yow yaar
aana ana aanal aanaal aanalum aanalum aagum agum aaguthu aguthu aagala agala aachu achu aachi aagi aaga aaganum
anna annan akka amma appa thambi thangachi thangai ponnu ponnunga paiyan pasanga pasangala payan payanga
makkal aalu aalunga aal mama machan machi mapla maapla maapillai machaan dei thalaiva thala thalapathy
ayya amma ammaa appaa thatha paati ammama chithi chithappa mama
padam padathula padathukku veedu veetla veettula vela velai vaela kaasu kasu panam kaasuku neram naal naala naalaiku
nalaiku naalaikku inniku innaiku innaikku inniki innikku indru indha innum inum inumum
kalam ulagam nadu oor peru perum vaarthai varthai paatu paattu pattu kadhai kathai kelvi
ama aama seri sari serii sarii seriya sariya nalladhu nallathu nalladha
ponga poga paaru kelunga kelu keta ketta ketkaathinga ketkala
kitta kita kittaye kitte
nu sollu nu ketta
thappu thappa thappaa sari
pesa pesu pesunga pesi pesitu pesura pesalam pesathinga pesatha pesama pesuringa
eduthu eduthaan edukka edutha edu edukkum
vachi vachu vechu vechi vechuruken vachurukken vaichu vaikka vaiyunga
kondu kondu kooda kondu vanthu
vittu vidunga vida vidu vitta vidama vittuten
therinju theriyuthu theriyudhu terinjuthu
pidikkum pudikum pudikala pidikala pudichiruku pidichiruku pidikum pidichu pudichu
kashtama kastama kashtam kastam sandhosham santhosam santhoshama
sollitaanga solliruken sollirukken
kavalai kavalaipadathinga
nalla irukku
ellam ellaam elam ella ellarum ellame ellaarum yellam yella yellarum ellaathukum elaam
oruthan oruthar oruthi orey ore ore onnaa
enakkum unakkum avanukum avalukum
konjam konjama
neengalum neengal ungalukkum
ippodhu ippodhaiku ipodhu
ivlo ivlova evlova
thevai theva thevaiyilla thevaiyila
nadakkum nadakkuthu nadanthuchu nadakka
tharkuri tharkuri
paavam pavam
pudhu pudhusu pudhusa
kekkaama
oru nal
varuvar
muttu muttal mutta
punda pundai sunni thevdiya otha thayoli loosu loosuthanama kena kenayan
olunga olungaa olungaaga
sirippu sirika sirichen
azhaga alaga azhagu azhagaana azhagana
puthusu
unmaiya unmai unmaiyave nijama nijamave
parava paravala paravayilla paravaillai
kalakku kalakkal
mudinjidum mudinjidhu mudinjuthu
vaanga vaanganum vaangi vangi vaangina
poduvanga poduvaanga podalam podalama
irukkaanga irukanga irukkanga irukkingala irukingala
pochu pochi poiduchu poiduchi poyiduchu
panninga panninga paninga
enga ingaye
kekum ketkum ketkura
theriyaama
nammala namala nammaloda
eppavume eppovum eppavum eppothum
ellame
aprom apro appram
aama aamam
oru vaati vaati thadava thadavai
varaikum varaikkum varai
thavira thavirathu
mathathu
chinnatha
kodi koodi
thirumba thirumbi thirumbavum
pesaama
sutham
perusu
kuraiyala kuraiya kurachu kammi kammiya
adhigam athigama athigam
ippadiyae
yaaravathu yaravathu
yedhukku yethuku yedhuku
kaaranam
ithanala athanala ithanaal
adhanala
apdiye
nalladhan
vaazhga vazhga vazhthukal vaazhthukkal
nandri nanri nandri
vanakkam vanakam
""")

# People, teams, brands, handles, acronyms: labelled OTHER by convention.
NAMED_ENTITIES = _words("""
csk rcb mi kkr srh dc pbks lsg gt rr ipl bcci icc odi t20 wtc
dmk admk aiadmk tvk ntk bjp inc pmk mdmk dmdk ttv
vijay ajith rajini rajinikanth kamal dhoni kohli rohit gill ashwin jadeja pandya sanju surya stalin eps ops seeman
udhayanidhi vijayakanth annamalai modi rahul trump putin jk steffi irfan madan pdogg ash askmrtt mrtt gp
iphone samsung oneplus realme redmi xiaomi oppo vivo iqoo poco moto motorola nokia apple google youtube instagram insta
whatsapp facebook twitter netflix amazon flipkart swiggy zomato jio airtel vodafone bsnl tesla ford toyota honda
chennai madras coimbatore madurai trichy salem tamilnadu india delhi mumbai bangalore bengaluru kerala
tamil tamilan tamizh tamizhan telugu hindi english kannada malayalam
ifa gpay paytm phonepe android ios windows linux
ai gb tb mb kb ram ssd fps hd uhd 4k 5g 4g 3g amoled lcd led usb bgm vfx cgi
""")

# Informal English chat spellings treated as English.
EN_SLANG = _words("""
pls plz plss plzz pleeze thx thnx thanx tq ty bro bruh broo brooo bhai sis bcoz bcz coz cuz
b4 gud nyc msg frnd frnds tmrw tmr wat wats whr wen abt bt luv idk imo imho tbh btw lol lmao omg wtf ngl irl fyi
ok okay okk k kk hmm hmmm ur u r yr yrs im dont cant didnt doesnt wont isnt wasnt thats whats
vdo vedio vid vids reel reels pic pics sub subs subscribe
gonna wanna gotta lemme dunno
ikr rn asap dm pm am fb yt
""")

# --- normalisation resources -------------------------------------------------

# preferred form -> variants (all lower case). Used by the Tanglish normalizer.
VARIANT_GROUPS: dict[str, str] = {
    "naan": "nan",
    "nee": "ni",
    "neenga": "nenga ninga neega nega nenga",
    "romba": "rombha rompa",
    "nalla": "nala nallaa",
    "enna": "yenna ena yena",
    "enakku": "enaku ennaku ennakku",
    "unakku": "unaku",
    "ungalukku": "ungaluku",
    "inniku": "innaiku innaikku inniki innikku",
    "ippo": "ipo ippa ipa",
    "appo": "apo",
    "eppo": "epo",
    "eppadi": "epdi epadi",
    "appadi": "apdi",
    "ippadi": "ipdi ipadi",
    "thaan": "than dhan dhaan",
    "athu": "adhu",
    "ithu": "idhu",
    "itha": "idha",
    "atha": "adha",
    "antha": "andha",
    "intha": "indha",
    "entha": "endha",
    "edhu": "ethu",
    "irukku": "iruku irruku eruku erukku irku",
    "irukkum": "irukum erukum",
    "illa": "ila illai",
    "pannunga": "panunga pannuga panuga",
    "panna": "pana",
    "pannitu": "panitu pannittu",
    "pannanum": "pananum",
    "panra": "pandra",
    "vanthu": "vandhu",
    "vantha": "vandha",
    "varuthu": "varudhu",
    "sollunga": "solunga solluga",
    "paathu": "pathu",
    "paatha": "patha",
    "paaka": "paka",
    "paaru": "paru",
    "paarunga": "parunga",
    "theriyala": "therila therla",
    "theriyathu": "theriyadhu",
    "mudiyala": "mudila",
    "vendam": "vendaam venam",
    "kandippa": "kandipa kandipaa",
    "kudunga": "kodunga",
    "kooda": "kuda",
    "mattum": "matum mattume",
    "mathiri": "madhiri maathiri mathri madhri maadhiri",
    "neraya": "neraiya",
    "konjam": "konja konjum",
    "semma": "sema semmaa",
    "seri": "sari serii sarii",
    "aama": "ama",
    "aprom": "apram appuram aprm",
    "venum": "venumm",
    "yaaru": "yaru",
    "yaarum": "yarum",
    "evlo": "evalo evvalavu",
    "ivlo": "ivalo",
    "avlo": "avalo",
    "inga": "inge",
    "anga": "ange",
    "ethuku": "edhuku yedhukku yethuku yedhuku",
    "athuku": "adhuku",
    "ithuku": "idhuku",
    "thambi": "thambii",
    "vela": "velai",
    "thala": "thalaiva",
}

# English chat abbreviation -> standard English (applied to EN-tagged tokens).
EN_SLANG_EXPANSIONS: dict[str, str] = {
    "frndss": "friends", "frndz": "friends", "frnz": "friends", "pls": "please", "plz": "please", "plss": "please", "plzz": "please", "pleeze": "please",
    "thx": "thanks", "thnx": "thanks", "thanx": "thanks", "tq": "thank you",
    "u": "you", "ur": "your", "r": "are", "yr": "your", "bcoz": "because", "bcz": "because",
    "coz": "because", "cuz": "because", "b4": "before", "gud": "good", "nyc": "nice",
    "msg": "message", "frnd": "friend", "frnds": "friends", "tmrw": "tomorrow", "tmr": "tomorrow",
    "wat": "what", "whr": "where", "wen": "when", "abt": "about", "bt": "but", "luv": "love",
    "vdo": "video", "vedio": "video", "vid": "video", "vids": "videos", "pic": "picture",
    "pics": "pictures", "dis": "this", "dat": "that", "im": "i'm", "dont": "don't",
    "cant": "can't", "didnt": "didn't", "doesnt": "doesn't", "wont": "won't", "isnt": "isn't",
    "wasnt": "wasn't", "thats": "that's", "whats": "what's", "brooo": "bro", "bruh": "bro",
    "okk": "ok", "kk": "ok", "k": "ok", "hv": "have", "wil": "will", "gr8": "great",
    "gn": "good night", "gm": "good morning", "tc": "take care", "bday": "birthday",
    "smthng": "something", "sry": "sorry", "srry": "sorry", "thnks": "thanks", "ty": "thanks",
}
