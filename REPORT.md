VecFrameStack fungerar som AI-modellens korttidsminne genom att den inte bara ser den aktuella kamerabilden, utan även de tre föregående bilderna i en sekvens. I din kod är n_stack=4 inställt, vilket innebär att varje gång AI:n ska fatta ett beslut så analyserar den ett "paket" av fyra bilder efter varandra.

Detta är helt avgörande för att modellen ska kunna förstå dynamik och rörelse; en enskild bild visar bara var linjen är, men fyra bilder i rad avslöjar hur snabbt bilen rör sig och i vilken vinkel den närmar sig en kurva. Utan detta minne skulle AI:n vara "blind" för tid och ha mycket svårare att parera fördröjningar eller beräkna hur hårt den behöver svänga för att hålla sig kvar på banan i högre hastigheter.

Utan test_model.py skulle du köra i blindo. Det är här du upptäcker om modellen har blivit "övertränad" (att den bara kan köra de banor den sett förut) eller om den faktiskt har lärt sig konceptet att följa en linje oavsett hur den svänger. Att se bilen i GUI-läge avslöjar också om den har tendenser att "vobbla", vilket kan betyda att du behöver höja straffet för snabba styrförändringar i environment.py.

model.py definierar arkitekturen för bilens intelligens genom en CustomCNN. Den tar in de staplade kamerabilderna och skickar dem genom sex olika lager av faltningar (convolutions). Dessa lager fungerar som filter som gradvis extraherar allt mer avancerad information – från enkla linjer och kanter till banans kurvatur och bilens position. Genom att använda features_dim=512 skapas en mycket bred "tankebana" som gör att AI:n kan lagra detaljerade representationer av körsituationen innan de skickas vidare till den slutgiltiga linjära beslutsenheten

train.py ansvarar för att driva inlärningsprocessen genom Reinforcement Learning (förstärkningsinlärning). Skriptet använder algoritmen PPO (Proximal Policy Optimization) och kör 16 simuleringar parallellt för att samla in data extremt snabbt. Den kopplar samman din CustomCNN-hjärna med simulatorn och använder ett belöningssystem för att stegvis förbättra bilens förmåga att hålla sig på banan. Genom att använda automatiska sparfiler (callbacks) säkerställer skriptet att den bästa versionen av modellen alltid sparas, även om träningen skulle avbrytas.

# Rapport: Autonom RC-bil med AI-styrning
**Projektmedlemmar:** Emil Carlsson Edman, Fredrik Lam, Martin Gustafsson, Jenny Skoglund

---

## 1. Syfte
Detta projekt beskriver utvecklingen av en självkörande RC-bil som navigerar efter linjer med hjälp av artificiell intelligens. Genom att kombinera avancerad datorsimulering (**Digital Twin**) med praktisk tillämpning på en Raspberry Pi baserad plattform, har målet varit att skapa en modell som är både snabb, stabil och robust mot störningar.

---

## 2. Teknisk lösning

### AI-modell & Arkitektur
Vi använder **Reinforcement Learning** (förstärkningsinlärning) via algoritmen **PPO (Proximal Policy Optimization)**. 
* **Hjärnan (`model.py`):** Ett 6-lagers skräddarsytt neuralt nätverk (**CustomCNN**) med 512 dimensioner i feature-lagret. Detta gör att AI:n kan extrahera komplexa visuella mönster.
* **Korttidsminne (`VecFrameStack`):** Genom att stacka 4 bilder ($n\_stack=4$) får modellen förmågan att uppfatta tid och rörelse, vilket är avgörande för att hantera hastighet och kurvtagning.

### Hårdvara
* **Chassi:** WL Toys 144001.
* **Dator:** Raspberry Pi 5.
* **Styrning:** PCA9685 PWM-driver för högprecision i styrservo.
* **Kamera:** Raspberry Pi Modul 3 (Vidvinkel).

---

## 3. Utvecklingsprocessen: Från V1 till V14

Utvecklingen skedde iterativt genom 14 versioner för att överbrygga gapet mellan simulering och verklighet (**Sim-to-Real**).

* **V1–V4: Grunden.** Uppbyggnad av en Digital Twin i PyBullet med korrekt hjulbas (23cm). Introduktion av latenssimulering (3-frame buffer) för att kompensera för trådlös fördröjning.
* **V5–V8: Robusthet.** Implementering av **Domain Randomization**. Vi insåg tidigt att modellen var känslig för ljus. Vi adderade "Visual Chaos" (skuggor, blänk och brus) och belönade modellen när den ignorerade dessa.
* **V9–V11: Geometri.** Justering av kamerans FOV till 65 grader och fixering av monteringen. Vi implementerade "Steering Boost" för att klara de skarpaste kurvorna.
* **V12–V14: Förfining.** Fokus på att eliminera sick sack-körning. Genom **Reward Shaping** infördes en `smoothing_penalty` som straffar hastiga styrförändringar. 

---

## 4. Utmaningar och lösningar

* **Sim-to-Real Gap:** Verkligheten är aldrig så ren som en simulator. Lösningen blev extrem *Domain Randomization* i `environment.py` där vi simulerade allt från suddiga linjer till mekaniskt glapp i styrstagen.
* **Färgvariationer:** Tidiga modeller misslyckades på grund av skillnader i hur kameran tolkade färg (Yellow vs Cyan). Detta löstes genom en `BGR_FLIP`-fix i mjukvaran och träning i gråskala/färg-agnostiska miljöer.
* **Instabilitet:** Genom att använda `test_model.py` kunde vi visuellt identifiera när modellen blev "övertränad". Detta hjälpte oss att justera vår Learning Rate till $1 \times 10^{-5}$ för en mer stabil inlärningskurva.

---

## 5. Resultat & Slutsats

Version 14 representerar projektets slutpunkt. Bilen kan nu:
1.  **Konsekvent** följa banan även under svåra ljusförhållanden.
2.  **Hantera** skarpa kurvor tack vare optimerad FOV och "Steering Boost".
3.  **Köra mjukt** utan det vobbliga beteende som präglade tidigare versioner.

Projektet visar att det med relativt enkla medel går att skapa ett avancerat autonomt system genom att lägga stor vikt vid simuleringsmiljöns realism och belöningsfunktionens utformning.

---

### Bilaga: Kodstruktur
* **`environment.py`**: Den digitala tvillingen och reglerna för träning.
* **`model.py`**: CNN-arkitekturen (512 dim).
* **`train.py`**: Träningsmotorn (PPO, 16 parallella miljöer).
* **`utils.py`**: Bildbehandling och ban-matematik (Splines).
* **`test_model.py`**: Kvalitetssäkring och visuell verifiering.