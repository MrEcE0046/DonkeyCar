# Autonom RC-bil 
**Projektmedlemmar:** Emil Carlsson Edman, Fredrik Lam, Martin Gustafsson, Jenny Skoglund

---

## 1. Syfte
Detta projekt beskriver utvecklingen av en självkörande RC-bil som navigerar efter linjer med hjälp av artificiell intelligens. Genom att kombinera avancerad datorsimulering (**Digital Twin**) med praktisk tillämpning på en Raspberry Pi baserad plattform, har målet varit att skapa en modell som är både snabb, stabil och robust mot störningar.

---

## 2. Teknisk lösning

### AI-modell & Arkitektur
**Reinforcement Learning** används via algoritmen **PPO (Proximal Policy Optimization)**. 

### Hårdvara
* **Chassi:** WL Toys 144001.
* **Dator:** Raspberry Pi 5.
* **Styrning:** PCA9685 PWM-driver för högprecision i styrservo.
* **Kamera:** Raspberry Pi Modul 3 (Vidvinkel).

### Mjukvaruarkitektur och dataflöde
Systemet bygger på en sammanhängande mjukvarupipeline som sträcker sig från simulering till fysisk exekvering. Processen är uppdelad i tre huvudfaser: träning, verifiering och inferens.
Inlärningsprocessen drivs av algoritmen PPO (Proximal Policy Optimization). För att maximera effektiviteten används SubprocVecEnv, vilket möjliggör körning av 16 parallella simuleringsmiljöer. Detta accelererar datainsamlingen och stabiliserar policyns uppdateringar. Learning Rate har succissivt sänkts under projektets gång och hamnade på 0.00001 som verkade ge bäst resultat till slut. Miljön fungerar som en brygga mellan fysikmotorn PyBullet och RL-algoritmen. Genom så kallad Reward Shaping tilldelas agenten belöningar baserat på centrering, vinkel i förhållande till banan och mjukhet i styrutslag, medan avvikelser och kollisioner leder till bestraffningar.
Beslutsfattandet centraliseras i ett skräddarsytt CNN. Nätverket är uppbyggt av sex konvolutionella lager som fungerar som hierarkiska filter. De första lagren extraherar primitiva former såsom kanter och linjer, medan de djupare lagren tolkar komplexa mönster som kurvatur och banans räckvidd. Den extraherade informationen komprimeras till en feature vector med 512 dimensioner. Denna breda representation möjliggör för modellen att lagra nyanserade detaljer om körsituationen innan datan skickas till det slutgiltiga linjära lagret för beslut om styrvinkel.
För att bibehålla medvetenhet används VecFrameStack med parametern $n\_stack=4$. Detta innebär att modellen inte fattar beslut baserat på en enskild stillbild, utan på en sekvens av de fyra senaste observationerna. Denna metod är bra för att systemet ska kunna beräkna fordonets hastighet och vinkelhastighet. Före analys genomgår varje bild en förbehandlingsprocess vilken inkluderar:

* **Beskäring:**: Oviktig information såsom horisonten avlägsnas för att minska beräkningsbördan.
* **Normalisering:**: Pixelvärden skalas om till intervallet [0, 1] för stabilare gradienter.
* **Kanalformatering:**: Bildmatrisen transformeras till formatet (Channels, Height, Width) för kompatibilitet med PyTorch.

Efter avslutad träning utvärderas modellen genom test_model.py. Detta steg är avgörande för att identifiera eventuell överträning (overfitting), där modellen memorerat specifika banor istället för att lära sig generella navigeringsprinciper. Genom visuell verifiering i en grafisk miljö analyseras bilens beteende, särskilt avseende stabilitet och förmågan att parera systemlatens, innan modellen exporteras till det fysiska fordonet.
I den slutliga fasen exporteras den tränade modellen till formatet ONNX för att möjliggöra högpresterande inferens på Raspberry Pi 5. Skriptet main.py ansvarar för att läsa in live-bilder från kameran, genomföra samma förbehandling som i simulatorn och reglera styrservot via PCA9685 komponenten i realtid. Genom att matcha simulatorns kameravinkel och latens i den fysiska miljön säkerställs att den inlärda policyn kan appliceras framgångsrikt i verkligheten.

---

## 3. Utvecklingsprocessen: Från V1 till V14

Utvecklingen skedde iterativt genom 14 versioner för att överbrygga gapet mellan simulering och verklighet (**Sim-to-Real**).

* **V1–V4: Grunden.** Uppbyggnad av en Digital Twin i PyBullet med korrekt hjulbas (23cm). Introduktion av latenssimulering (3-frame buffer) för att kompensera för trådlös fördröjning.
* **V5–V8: Robusthet.** Implementering av **Domain Randomization**. Initiala utvärderingar identifierade en signifikant korrelation mellan modellens precision och omgivningens ljus. För att reducera denna känslighet implementerades en utökad Domain Randomization i simulatorn, där parametrar för dynamisk ljussättning och reflexer varierades stokastiskt under träningsfasen. I syfte att stärka modellens förmåga att separera signal från brus adderades komplexa ljuseffekter i simulatorn. Denna metodik innebar att agenten tränades i att extrahera banans position ur en miljö med hög visuell komplexitet, där framgångsrik filtrering av störningsmoment direkt korrelerade med positiva belöningssignaler.
* **V9–V11: Geometri.** Justering av kamerans FOV till 65 grader och fixering av monteringen. För att hantera snäva kurvor implementerades en förstärkning av styrutslaget, vilket möjliggjorde fullt utnyttjande av fordonets mekaniska svängradie.
* **V12–V14: Förfining.** Fokus på att eliminera sick sack-körning. Genom **Reward Shaping** infördes en `smoothing_penalty` som straffar hastiga styrförändringar. 

---

## 4. Utmaningar och lösningar

* **Sim-to-Real Gap:** I syfte att minimera Sim-to-Real-gapet introducerades avancerade störningsmoment. Detta inkluderade simulering av visuella artefakter och mekaniska toleranser, vilket ökade modellens robusthet vid fysisk drift.
* **Färgvariationer:** Tidiga modeller misslyckades på grund av skillnader i hur kameran tolkade färg (Yellow vs Cyan). Detta löstes genom en `BGR_FLIP`-fix i mjukvaran och träning i gråskala/färg-agnostiska miljöer.
* **Instabilitet:** För att motverka identifierad överträning och instabilitet i policyn nyttjades test_model.py för kvalitativ granskning av agentens beteende. Genom att begränsa Learning Rate till $1 \times 0.00001$ stabiliserades optimeringsprocessen, vilket minimerade risken för att modellen memorerade specifika simuleringsdata

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